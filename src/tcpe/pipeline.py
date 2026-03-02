"""Phase 15 end-to-end pipeline orchestration with checkpointing and resume."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from tcpe.anndata_schema import AnnDataSchemaError
from tcpe.baselines import BASELINE_RANDOM_GRN_UNS_KEY, BASELINE_RESULTS_UNS_KEY
from tcpe.causal import CausalConfig, CausalError, CausalModule
from tcpe.config import ConfigLoadError, ConfigValidationError, TCPEConfig
from tcpe.dataset_loaders import DatasetLoaderError
from tcpe.evaluation import (
    EVALUATION_REPORT_UNS_KEY,
    EvaluationConfig,
    EvaluationError,
    EvaluationModule,
    EvaluationReport,
)
from tcpe.ingestion import IngestionModule
from tcpe.preprocessing import PreprocessingConfig, PreprocessingError
from tcpe.runtime.run_context import ArtifactLayout
from tcpe.transport import (
    OTTransportConfig,
    OTTransportStrategy,
    TransportError,
    TransportInputError,
    TransportModule,
    TransportTrainingData,
)

PipelineStep = Literal["ingest", "preprocess", "embed", "train", "causal", "evaluate", "card"]
PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    "ingest",
    "preprocess",
    "embed",
    "train",
    "causal",
    "evaluate",
    "card",
)
PIPELINE_CHECKPOINT_SCHEMA_VERSION = "pipeline_checkpoint_v1"
PIPELINE_ARTIFACT_MANIFEST_SCHEMA_VERSION = "pipeline_artifact_manifest_v1"
PIPELINE_RUN_REPORT_SCHEMA_VERSION = "pipeline_run_report_v1"


class PipelineError(RuntimeError):
    """Base class for pipeline execution errors."""


class PipelineResumeError(PipelineError):
    """Raised when pipeline resume state is missing or malformed."""


class PipelineCompletionError(PipelineError):
    """Raised when pipeline completion criteria are not satisfied."""


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for deterministic Phase 15 local pipeline runs."""

    dataset_id: str = "synthetic"
    synthetic_n_cells: int = 120
    synthetic_n_genes: int = 60
    synthetic_n_perturbations: int = 8
    dataset_source_uris: dict[str, str] = field(default_factory=dict)
    seed: int = 42
    transport_latent_dim: int = 32
    transport_hidden_dim: int = 96
    transport_epochs: int = 5
    transport_batch_size: int = 48
    transport_learning_rate: float = 2e-3
    transport_sinkhorn_weight: float = 0.05
    causal_max_hvgs: int = 40
    causal_bootstrap_iterations: int = 0
    stop_after_step: PipelineStep | None = None

    def __post_init__(self) -> None:
        if self.synthetic_n_cells <= 0:
            raise ValueError("synthetic_n_cells must be positive.")
        if self.synthetic_n_genes <= 1:
            raise ValueError("synthetic_n_genes must be greater than 1.")
        if self.synthetic_n_perturbations < 2:
            raise ValueError("synthetic_n_perturbations must be at least 2.")
        for key, value in self.dataset_source_uris.items():
            if str(key).strip() == "":
                raise ValueError("dataset_source_uris keys must be non-empty.")
            if str(value).strip() == "":
                raise ValueError("dataset_source_uris values must be non-empty paths/URIs.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.transport_latent_dim <= 0:
            raise ValueError("transport_latent_dim must be positive.")
        if self.transport_hidden_dim <= 0:
            raise ValueError("transport_hidden_dim must be positive.")
        if self.transport_epochs <= 0:
            raise ValueError("transport_epochs must be positive.")
        if self.transport_batch_size <= 0:
            raise ValueError("transport_batch_size must be positive.")
        if self.transport_learning_rate <= 0:
            raise ValueError("transport_learning_rate must be positive.")
        if self.transport_sinkhorn_weight < 0:
            raise ValueError("transport_sinkhorn_weight must be non-negative.")
        if self.causal_max_hvgs <= 1:
            raise ValueError("causal_max_hvgs must be greater than 1.")
        if self.causal_bootstrap_iterations < 0:
            raise ValueError("causal_bootstrap_iterations must be non-negative.")
        if self.stop_after_step is not None and self.stop_after_step not in PIPELINE_STEPS:
            raise ValueError(f"Unsupported stop_after_step '{self.stop_after_step}'.")


@dataclass
class PipelineCheckpoint:
    """Persisted checkpoint for step-level resume."""

    schema_version: str
    run_id: str
    completed_steps: list[PipelineStep] = field(default_factory=list)
    step_artifacts: dict[PipelineStep, dict[str, Any]] = field(default_factory=dict)
    updated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["completed_steps"] = [str(step) for step in self.completed_steps]
        payload["step_artifacts"] = {
            str(step): artifact
            for step, artifact in self.step_artifacts.items()
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineCheckpoint:
        completed_raw = payload.get("completed_steps", [])
        if not isinstance(completed_raw, list):
            raise PipelineResumeError("Checkpoint `completed_steps` must be a list.")
        completed_steps: list[PipelineStep] = []
        for step in completed_raw:
            step_str = str(step).strip()
            if step_str not in PIPELINE_STEPS:
                raise PipelineResumeError(f"Checkpoint has unknown step '{step_str}'.")
            completed_steps.append(cast(PipelineStep, step_str))

        artifacts_raw = payload.get("step_artifacts", {})
        if not isinstance(artifacts_raw, dict):
            raise PipelineResumeError("Checkpoint `step_artifacts` must be an object.")
        step_artifacts: dict[PipelineStep, dict[str, Any]] = {}
        for step_name, artifact in artifacts_raw.items():
            step_str = str(step_name).strip()
            if step_str not in PIPELINE_STEPS:
                continue
            if isinstance(artifact, dict):
                step_artifacts[cast(PipelineStep, step_str)] = cast(dict[str, Any], artifact)

        return cls(
            schema_version=str(payload.get("schema_version", "")),
            run_id=str(payload.get("run_id", "")),
            completed_steps=completed_steps,
            step_artifacts=step_artifacts,
            updated_at_utc=str(payload.get("updated_at_utc", datetime.now(UTC).isoformat())),
        )


@dataclass(frozen=True)
class PipelineRunResult:
    """Top-level result payload for one pipeline run."""

    run_id: str
    status: Literal["completed", "partial"]
    completed_steps: list[PipelineStep]
    skipped_steps: list[PipelineStep]
    resumed_from_checkpoint: bool
    checkpoint_path: Path
    run_report_path: Path
    artifact_manifest_path: Path | None
    artifact_bundle_dir: Path | None
    step_artifacts: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "completed_steps": [str(step) for step in self.completed_steps],
            "skipped_steps": [str(step) for step in self.skipped_steps],
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "checkpoint_path": str(self.checkpoint_path),
            "run_report_path": str(self.run_report_path),
            "artifact_manifest_path": (
                None if self.artifact_manifest_path is None else str(self.artifact_manifest_path)
            ),
            "artifact_bundle_dir": (
                None if self.artifact_bundle_dir is None else str(self.artifact_bundle_dir)
            ),
            "step_artifacts": self.step_artifacts,
        }


@dataclass
class PipelineExecutionError(PipelineError):
    """Classified pipeline execution error with stage-level context."""

    step: PipelineStep
    category: Literal["data", "config", "model", "infra", "unknown"]
    message: str


class PipelineModule:
    """Phase 15 orchestration module for end-to-end deterministic local runs."""

    def status(self) -> str:
        return "phase15_pipeline_ready"

    def run(
        self,
        *,
        config: TCPEConfig,
        layout: ArtifactLayout,
        run_id: str,
        pipeline_config: PipelineConfig | None = None,
        resume: bool = False,
    ) -> PipelineRunResult:
        selected = pipeline_config if pipeline_config is not None else PipelineConfig()
        layout.run_root.mkdir(parents=True, exist_ok=True)
        layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        layout.reports_dir.mkdir(parents=True, exist_ok=True)
        layout.metadata_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = layout.metadata_dir / "pipeline_checkpoint.json"
        run_report_path = layout.metadata_dir / "pipeline_run_report.json"

        checkpoint, resumed_from_checkpoint = self._load_or_initialize_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            resume=resume,
        )
        skipped_steps: list[PipelineStep] = []

        for step in PIPELINE_STEPS:
            if step in checkpoint.completed_steps:
                skipped_steps.append(step)
                continue

            try:
                artifacts = self._execute_step(
                    step=step,
                    config=config,
                    layout=layout,
                    checkpoint=checkpoint,
                    pipeline_config=selected,
                )
            except Exception as exc:
                category = _classify_exception(exc)
                failed_payload: dict[str, Any] = {
                    "schema_version": PIPELINE_RUN_REPORT_SCHEMA_VERSION,
                    "run_id": run_id,
                    "status": "failed",
                    "failed_step": step,
                    "error_category": category,
                    "error_message": str(exc),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
                run_report_path.write_text(
                    json.dumps(failed_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                raise PipelineExecutionError(
                    step=step,
                    category=category,
                    message=str(exc),
                ) from exc

            checkpoint.step_artifacts[step] = artifacts
            checkpoint.completed_steps.append(step)
            checkpoint.updated_at_utc = datetime.now(UTC).isoformat()
            checkpoint_path.write_text(
                json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

            if selected.stop_after_step == step:
                partial_payload: dict[str, Any] = {
                    "schema_version": PIPELINE_RUN_REPORT_SCHEMA_VERSION,
                    "run_id": run_id,
                    "status": "partial",
                    "completed_steps": [str(item) for item in checkpoint.completed_steps],
                    "skipped_steps": [str(item) for item in skipped_steps],
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
                run_report_path.write_text(
                    json.dumps(partial_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                return PipelineRunResult(
                    run_id=run_id,
                    status="partial",
                    completed_steps=list(checkpoint.completed_steps),
                    skipped_steps=skipped_steps,
                    resumed_from_checkpoint=resumed_from_checkpoint,
                    checkpoint_path=checkpoint_path,
                    run_report_path=run_report_path,
                    artifact_manifest_path=None,
                    artifact_bundle_dir=None,
                    step_artifacts={
                        str(step_name): dict(payload)
                        for step_name, payload in checkpoint.step_artifacts.items()
                    },
                )

        self._enforce_completion_criteria(checkpoint=checkpoint)
        artifact_manifest_path = self._write_artifact_manifest(
            run_id=run_id,
            checkpoint=checkpoint,
            metadata_dir=layout.metadata_dir,
        )
        artifact_bundle_dir = self._export_cloud_handoff_bundle(
            artifact_manifest_path=artifact_manifest_path,
            checkpoint=checkpoint,
            reports_dir=layout.reports_dir,
        )

        completed_payload: dict[str, Any] = {
            "schema_version": PIPELINE_RUN_REPORT_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "completed",
            "completed_steps": [str(item) for item in checkpoint.completed_steps],
            "skipped_steps": [str(item) for item in skipped_steps],
            "artifact_manifest_path": str(artifact_manifest_path),
            "artifact_bundle_dir": str(artifact_bundle_dir),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        run_report_path.write_text(
            json.dumps(completed_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return PipelineRunResult(
            run_id=run_id,
            status="completed",
            completed_steps=list(checkpoint.completed_steps),
            skipped_steps=skipped_steps,
            resumed_from_checkpoint=resumed_from_checkpoint,
            checkpoint_path=checkpoint_path,
            run_report_path=run_report_path,
            artifact_manifest_path=artifact_manifest_path,
            artifact_bundle_dir=artifact_bundle_dir,
            step_artifacts={
                str(step_name): dict(payload)
                for step_name, payload in checkpoint.step_artifacts.items()
            },
        )

    def _execute_step(
        self,
        *,
        step: PipelineStep,
        config: TCPEConfig,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        if step == "ingest":
            return self._step_ingest(config=config, layout=layout, pipeline_config=pipeline_config)
        if step == "preprocess":
            return self._step_preprocess(
                layout=layout,
                checkpoint=checkpoint,
                pipeline_config=pipeline_config,
            )
        if step == "embed":
            return self._step_embed(layout=layout, checkpoint=checkpoint)
        if step == "train":
            return self._step_train(
                layout=layout,
                checkpoint=checkpoint,
                pipeline_config=pipeline_config,
            )
        if step == "causal":
            return self._step_causal(
                layout=layout,
                checkpoint=checkpoint,
                pipeline_config=pipeline_config,
            )
        if step == "evaluate":
            return self._step_evaluate(
                layout=layout,
                checkpoint=checkpoint,
                pipeline_config=pipeline_config,
            )
        if step == "card":
            return self._step_card(layout=layout, checkpoint=checkpoint)
        raise PipelineError(f"Unsupported pipeline step '{step}'.")

    def _step_ingest(
        self,
        *,
        config: TCPEConfig,
        layout: ArtifactLayout,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        module = IngestionModule()
        if pipeline_config.dataset_id == "synthetic":
            bundle = module.build_synthetic_dataset(
                n_cells=pipeline_config.synthetic_n_cells,
                n_genes=pipeline_config.synthetic_n_genes,
                n_perturbations=pipeline_config.synthetic_n_perturbations,
                seed=pipeline_config.seed,
            )
            adata = bundle.adata
            source_type = "synthetic"
            dataset_name = "synthetic"
        else:
            source_uri = pipeline_config.dataset_source_uris.get(pipeline_config.dataset_id)
            loaded = module.load_dataset(
                pipeline_config.dataset_id,
                cache_dir=config.paths.cache_root,
                source_uri=source_uri,
                force_refresh=False,
            )
            adata = loaded.adata
            source_type = "loader"
            dataset_name = loaded.dataset_id

        adata.uns["dataset_name"] = dataset_name
        path = layout.checkpoints_dir / "step_ingest_adata.h5ad"
        adata.write_h5ad(path)
        return {
            "adata_path": str(path),
            "dataset_id": dataset_name,
            "source_type": source_type,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
        }

    def _step_preprocess(
        self,
        *,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        adata = _load_adata_from_step(checkpoint=checkpoint, step="ingest")
        module = IngestionModule()
        result = module.preprocess_adata(
            adata,
            config=PreprocessingConfig(
                seed=pipeline_config.seed,
                min_counts_per_cell=1,
                min_genes_per_cell=1,
                min_cells_per_gene=1,
            ),
        )
        path = layout.checkpoints_dir / "step_preprocess_adata.h5ad"
        result.adata.write_h5ad(path)
        return {
            "adata_path": str(path),
            "n_cells_after_qc": int(result.n_cells_after_qc),
            "n_genes_after_qc": int(result.n_genes_after_qc),
            "n_hvgs_selected": int(result.n_hvgs_selected),
            "hvg_fallback_used": bool(result.hvg_fallback_used),
        }

    def _step_embed(
        self,
        *,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
    ) -> dict[str, Any]:
        adata = _load_adata_from_step(checkpoint=checkpoint, step="preprocess")
        module = IngestionModule()
        sequence_meta = module.annotate_sequence_embeddings(adata)
        cell_meta = module.annotate_cell_state_embeddings(adata)
        path = layout.checkpoints_dir / "step_embed_adata.h5ad"
        adata.write_h5ad(path)
        return {
            "adata_path": str(path),
            "sequence_embedding_dim": int(sequence_meta["embedding_dim"]),
            "cell_state_embedding_dim": int(cell_meta["embedding_dim"]),
            "sequence_provider": str(sequence_meta["provider_name"]),
            "cell_state_provider": str(cell_meta["provider_name"]),
        }

    def _step_train(
        self,
        *,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        adata = _load_adata_from_step(checkpoint=checkpoint, step="embed")
        evaluator = EvaluationModule()
        baseline_suite = evaluator.run_baselines(
            adata,
            expression_layer="normalized_log1p",
            seed=pipeline_config.seed,
            persist_to_anndata=True,
        )

        training_data = TransportTrainingData.from_anndata(
            adata,
            expression_layer="normalized_log1p",
            source_policy="control_mean",
        )
        ot_config = OTTransportConfig(
            input_dim=training_data.source_expression.shape[1],
            sequence_embedding_dim=training_data.sequence_embedding.shape[1],
            cell_state_embedding_dim=training_data.cell_state_embedding.shape[1],
            latent_dim=pipeline_config.transport_latent_dim,
            hidden_dim=pipeline_config.transport_hidden_dim,
            learning_rate=pipeline_config.transport_learning_rate,
            n_epochs=pipeline_config.transport_epochs,
            batch_size=pipeline_config.transport_batch_size,
            sinkhorn_weight=pipeline_config.transport_sinkhorn_weight,
        )

        transport = TransportModule()
        strategy, fit_result = transport.train_ot(
            baseline_suite=baseline_suite,
            training_data=training_data,
            config=ot_config,
        )
        checkpoint_path = strategy.save(layout.checkpoints_dir / "transport_model.ckpt")
        fit_summary_path = layout.reports_dir / "transport_fit_summary.json"
        fit_summary_path.write_text(
            json.dumps(fit_result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        baseline_suite_path = layout.reports_dir / "baseline_suite_results.json"
        baseline_suite_path.write_text(
            json.dumps(baseline_suite.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Store baseline payload as JSON artifact instead of `.uns` object-lists for h5ad safety.
        adata.uns.pop(BASELINE_RESULTS_UNS_KEY, None)
        adata.uns.pop(BASELINE_RANDOM_GRN_UNS_KEY, None)

        adata_path = layout.checkpoints_dir / "step_train_adata.h5ad"
        adata.write_h5ad(adata_path)
        return {
            "adata_path": str(adata_path),
            "transport_checkpoint_path": str(checkpoint_path),
            "transport_fit_summary_path": str(fit_summary_path),
            "baseline_suite_json_path": str(baseline_suite_path),
            "transport_variant": strategy.variant_name,
            "baseline_schema_version": baseline_suite.schema_version,
        }

    def _step_causal(
        self,
        *,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        adata = _load_adata_from_step(checkpoint=checkpoint, step="train")
        module = CausalModule()
        result = module.fit(
            adata,
            config=CausalConfig(
                max_hvgs=pipeline_config.causal_max_hvgs,
                bootstrap_iterations=pipeline_config.causal_bootstrap_iterations,
                bootstrap_seed=pipeline_config.seed,
            ),
            persist_to_anndata=True,
        )
        export_paths = module.export_graph_artifacts(
            result.graph,
            output_dir=layout.reports_dir / "causal",
            file_prefix="causal_graph",
        )

        adata_path = layout.checkpoints_dir / "step_causal_adata.h5ad"
        adata.write_h5ad(adata_path)
        return {
            "adata_path": str(adata_path),
            "causal_adjacency_npz_path": str(export_paths.adjacency_npz_path),
            "causal_edge_table_csv_path": str(export_paths.edge_table_csv_path),
            "causal_metadata_json_path": str(export_paths.metadata_json_path),
            "causal_schema_version": result.graph.schema_version,
        }

    def _step_evaluate(
        self,
        *,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
        pipeline_config: PipelineConfig,
    ) -> dict[str, Any]:
        adata = _load_adata_from_step(checkpoint=checkpoint, step="causal")
        train_artifacts = checkpoint.step_artifacts.get("train")
        if train_artifacts is None:
            raise PipelineResumeError("Missing `train` artifacts for evaluate step.")
        checkpoint_path = _require_path(train_artifacts, key="transport_checkpoint_path")
        baseline_suite_path = _require_path(train_artifacts, key="baseline_suite_json_path")
        baseline_payload = json.loads(baseline_suite_path.read_text(encoding="utf-8"))
        if not isinstance(baseline_payload, dict):
            raise PipelineResumeError("Baseline suite artifact must be a JSON object.")

        strategy = OTTransportStrategy.load(checkpoint_path)
        training_data = TransportTrainingData.from_anndata(
            adata,
            expression_layer="normalized_log1p",
            source_policy="control_mean",
        )
        prediction = strategy.predict_distribution(
            source_expression=training_data.source_expression,
            sequence_embedding=training_data.sequence_embedding,
            cell_state_embedding=training_data.cell_state_embedding,
        )

        evaluator = EvaluationModule()
        report = evaluator.evaluate_predictions(
            adata,
            prediction_mean=prediction.mean,
            prediction_variance=prediction.variance,
            baseline_suite=cast(dict[str, Any], baseline_payload),
            config=EvaluationConfig(
                seed=pipeline_config.seed,
                model_name="ot_sinkhorn",
                model_version="phase15_pipeline",
            ),
            persist_to_anndata=True,
        )

        evaluation_dir = layout.reports_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        report_path = evaluation_dir / "evaluation_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        prediction_path = evaluation_dir / "transport_prediction.npz"
        np.savez_compressed(
            prediction_path,
            prediction_mean=prediction.mean,
            prediction_variance=prediction.variance,
        )

        # Store evaluator payload as JSON artifact for robust h5ad serialization.
        adata.uns.pop(EVALUATION_REPORT_UNS_KEY, None)
        adata_path = layout.checkpoints_dir / "step_evaluate_adata.h5ad"
        adata.write_h5ad(adata_path)
        return {
            "adata_path": str(adata_path),
            "evaluation_report_json_path": str(report_path),
            "prediction_npz_path": str(prediction_path),
        }

    def _step_card(
        self,
        *,
        layout: ArtifactLayout,
        checkpoint: PipelineCheckpoint,
    ) -> dict[str, Any]:
        adata = _load_adata_from_step(checkpoint=checkpoint, step="evaluate")
        evaluate_artifacts = checkpoint.step_artifacts.get("evaluate")
        if evaluate_artifacts is None:
            raise PipelineResumeError("Missing `evaluate` artifacts for card step.")
        report_path = _require_path(evaluate_artifacts, key="evaluation_report_json_path")
        report = _load_evaluation_report(report_path)

        evaluator = EvaluationModule()
        artifacts = evaluator.generate_model_card(
            report,
            output_dir=layout.reports_dir / "model_card",
            file_prefix="model_card",
        )
        evaluator.assert_run_complete(artifacts)
        adata.uns["model_card_artifacts"] = artifacts.to_dict()
        adata_path = layout.checkpoints_dir / "step_card_adata.h5ad"
        adata.write_h5ad(adata_path)

        return {
            "adata_path": str(adata_path),
            "model_card_json_path": str(artifacts.json_path),
            "model_card_markdown_path": str(artifacts.markdown_path),
            "model_card_schema_version": artifacts.schema_version,
        }

    def _load_or_initialize_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        run_id: str,
        resume: bool,
    ) -> tuple[PipelineCheckpoint, bool]:
        if resume:
            if not checkpoint_path.exists():
                raise PipelineResumeError(
                    f"Cannot resume: checkpoint does not exist at {checkpoint_path}."
                )
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise PipelineResumeError("Checkpoint payload root must be an object.")
            checkpoint = PipelineCheckpoint.from_dict(cast(dict[str, Any], payload))
            if checkpoint.run_id != run_id:
                raise PipelineResumeError(
                    f"Checkpoint run_id '{checkpoint.run_id}' does not match requested '{run_id}'."
                )
            if checkpoint.schema_version != PIPELINE_CHECKPOINT_SCHEMA_VERSION:
                raise PipelineResumeError(
                    f"Unsupported checkpoint schema version '{checkpoint.schema_version}'."
                )
            return checkpoint, True

        checkpoint = PipelineCheckpoint(
            schema_version=PIPELINE_CHECKPOINT_SCHEMA_VERSION,
            run_id=run_id,
        )
        checkpoint_path.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return checkpoint, False

    def _enforce_completion_criteria(self, *, checkpoint: PipelineCheckpoint) -> None:
        missing = [step for step in PIPELINE_STEPS if step not in checkpoint.completed_steps]
        if missing:
            raise PipelineCompletionError(
                "Pipeline completion criteria failed. Missing steps: " + ", ".join(missing)
            )
        card_artifacts = checkpoint.step_artifacts.get("card", {})
        if "model_card_json_path" not in card_artifacts:
            raise PipelineCompletionError(
                "Pipeline completion criteria failed: model card JSON artifact is missing."
            )
        if "model_card_markdown_path" not in card_artifacts:
            raise PipelineCompletionError(
                "Pipeline completion criteria failed: model card Markdown artifact is missing."
            )

    def _write_artifact_manifest(
        self,
        *,
        run_id: str,
        checkpoint: PipelineCheckpoint,
        metadata_dir: Path,
    ) -> Path:
        required_artifacts = _collect_required_artifacts(checkpoint.step_artifacts)
        manifest = {
            "schema_version": PIPELINE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "required_artifacts": required_artifacts,
            "step_artifacts": {
                str(step): dict(payload)
                for step, payload in checkpoint.step_artifacts.items()
            },
        }
        manifest_path = metadata_dir / "pipeline_artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def _export_cloud_handoff_bundle(
        self,
        *,
        artifact_manifest_path: Path,
        checkpoint: PipelineCheckpoint,
        reports_dir: Path,
    ) -> Path:
        bundle_dir = reports_dir / "cloud_handoff_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        required_artifacts = _collect_required_artifacts(checkpoint.step_artifacts)
        copied: dict[str, str] = {}
        for artifact_name, path_str in required_artifacts.items():
            source_path = Path(path_str)
            if not source_path.exists():
                raise PipelineCompletionError(
                    f"Required artifact does not exist for bundling: {source_path}"
                )
            destination = bundle_dir / source_path.name
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)
            copied[artifact_name] = str(destination)

        manifest_target = bundle_dir / artifact_manifest_path.name
        if artifact_manifest_path.resolve() != manifest_target.resolve():
            shutil.copy2(artifact_manifest_path, manifest_target)

        bundle_manifest = {
            "schema_version": "cloud_handoff_bundle_v1",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "copied_artifacts": copied,
            "pipeline_manifest_path": str(manifest_target),
        }
        (bundle_dir / "cloud_handoff_bundle_manifest.json").write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return bundle_dir


def _collect_required_artifacts(
    step_artifacts: dict[PipelineStep, dict[str, Any]],
) -> dict[str, str]:
    required_map: dict[str, tuple[PipelineStep, str]] = {
        "ingest_adata": ("ingest", "adata_path"),
        "preprocess_adata": ("preprocess", "adata_path"),
        "embed_adata": ("embed", "adata_path"),
        "transport_checkpoint": ("train", "transport_checkpoint_path"),
        "transport_fit_summary": ("train", "transport_fit_summary_path"),
        "baseline_suite_json": ("train", "baseline_suite_json_path"),
        "causal_adjacency_npz": ("causal", "causal_adjacency_npz_path"),
        "causal_edge_table_csv": ("causal", "causal_edge_table_csv_path"),
        "causal_metadata_json": ("causal", "causal_metadata_json_path"),
        "evaluation_report_json": ("evaluate", "evaluation_report_json_path"),
        "prediction_npz": ("evaluate", "prediction_npz_path"),
        "model_card_json": ("card", "model_card_json_path"),
        "model_card_markdown": ("card", "model_card_markdown_path"),
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for output_name, (step, key) in required_map.items():
        step_payload = step_artifacts.get(step)
        if step_payload is None:
            missing.append(f"{output_name} (missing step `{step}`)")
            continue
        value = step_payload.get(key)
        if not isinstance(value, str) or value.strip() == "":
            missing.append(f"{output_name} (missing key `{key}`)")
            continue
        resolved[output_name] = value
    if missing:
        raise PipelineCompletionError(
            "Artifact manifest generation failed due to missing required outputs:\n- "
            + "\n- ".join(missing)
        )
    return resolved


def _load_evaluation_report(path: Path) -> EvaluationReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineResumeError("Evaluation report payload root must be an object.")
    if "metrics" not in payload or "baseline_comparison" not in payload:
        raise PipelineResumeError("Evaluation report payload is missing required keys.")
    return EvaluationReport(
        schema_version=str(payload.get("schema_version", "evaluation_report_v1")),
        metrics=cast(dict[str, float], payload.get("metrics", {})),
        baseline_comparison=cast(list[dict[str, Any]], payload.get("baseline_comparison", [])),
        failure_modes=cast(list[dict[str, Any]], payload.get("failure_modes", [])),
        metadata=cast(dict[str, Any], payload.get("metadata", {})),
    )


def _load_adata_from_step(*, checkpoint: PipelineCheckpoint, step: PipelineStep) -> Any:
    payload = checkpoint.step_artifacts.get(step)
    if payload is None:
        raise PipelineResumeError(f"Missing checkpoint artifacts for step '{step}'.")
    adata_path = _require_path(payload, key="adata_path")
    return _require_anndata().read_h5ad(adata_path)


def _require_path(payload: dict[str, Any], *, key: str) -> Path:
    raw = payload.get(key)
    if not isinstance(raw, str) or raw.strip() == "":
        raise PipelineResumeError(f"Checkpoint payload missing path key `{key}`.")
    path = Path(raw)
    if not path.exists():
        raise PipelineResumeError(f"Expected checkpoint artifact path does not exist: {path}")
    return path


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - runtime dependency behavior.
        raise PipelineError("`anndata` is required for pipeline execution.") from exc
    return ad


def _classify_exception(
    exc: Exception,
) -> Literal["data", "config", "model", "infra", "unknown"]:
    if isinstance(exc, (ConfigLoadError, ConfigValidationError, ValueError)):
        return "config"
    if isinstance(exc, (DatasetLoaderError, PreprocessingError, AnnDataSchemaError, KeyError)):
        return "data"
    if isinstance(exc, (TransportError, TransportInputError, CausalError, EvaluationError)):
        return "model"
    if isinstance(exc, (OSError, IOError, MemoryError)):
        return "infra"
    return "unknown"
