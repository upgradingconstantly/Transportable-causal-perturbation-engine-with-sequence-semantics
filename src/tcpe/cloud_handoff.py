"""Phase 17 cloud handoff planning, transfer scaffolds, and resume seeding."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from tcpe.config import TCPEConfig
from tcpe.pipeline import (
    PIPELINE_CHECKPOINT_SCHEMA_VERSION,
    PIPELINE_STEPS,
    PipelineCheckpoint,
    PipelineStep,
)
from tcpe.runtime.run_context import ArtifactLayout

CLOUD_HANDOFF_SCHEMA_VERSION = "phase17_cloud_handoff_v1"
ENVIRONMENT_SPEC_SCHEMA_VERSION = "phase17_environment_spec_v1"
RUN_PRESET_SCHEMA_VERSION = "phase17_run_preset_v1"
BUDGET_LOG_SCHEMA_VERSION = "phase17_budget_log_v1"
ARTIFACT_SYNC_SCHEMA_VERSION = "phase17_artifact_sync_v1"
SPOT_RESUME_SCHEMA_VERSION = "phase17_spot_resume_v1"
PREPROCESSED_SEED_SCHEMA_VERSION = "phase17_preprocessed_seed_v1"

ArtifactProvider = Literal["github", "huggingface", "zenodo"]
ArtifactDirection = Literal["push", "pull"]


class CloudHandoffError(RuntimeError):
    """Raised when phase-17 handoff planning or seeding fails."""


@dataclass(frozen=True)
class Phase17DatasetSpec:
    """Concrete dataset-level cloud handoff constraints."""

    dataset_key: str
    display_name: str
    source_uri: str
    processed_h5ad_name: str
    hvg_gene_target: int = 5000
    expected_size_gb_min: int = 10
    expected_size_gb_max: int = 15

    def __post_init__(self) -> None:
        if str(self.dataset_key).strip() == "":
            raise ValueError("dataset_key must be non-empty.")
        if str(self.display_name).strip() == "":
            raise ValueError("display_name must be non-empty.")
        if str(self.source_uri).strip() == "":
            raise ValueError("source_uri must be non-empty.")
        if str(self.processed_h5ad_name).strip() == "":
            raise ValueError("processed_h5ad_name must be non-empty.")
        if self.hvg_gene_target <= 0:
            raise ValueError("hvg_gene_target must be positive.")
        if self.expected_size_gb_min <= 0:
            raise ValueError("expected_size_gb_min must be positive.")
        if self.expected_size_gb_max < self.expected_size_gb_min:
            raise ValueError("expected_size_gb_max must be >= expected_size_gb_min.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_datasets() -> tuple[Phase17DatasetSpec, ...]:
    base_uri = "https://plus.figshare.com/articles/dataset/20029387"
    return (
        Phase17DatasetSpec(
            dataset_key="replogle_k562_gwps",
            display_name="Replogle K562_gwps",
            source_uri=f"{base_uri}#K562_gwps",
            processed_h5ad_name="replogle_k562_gwps_hvg5000.h5ad",
        ),
        Phase17DatasetSpec(
            dataset_key="k562_essential",
            display_name="K562_essential",
            source_uri=f"{base_uri}#K562_essential",
            processed_h5ad_name="k562_essential_hvg5000.h5ad",
        ),
        Phase17DatasetSpec(
            dataset_key="rpe1",
            display_name="RPE1",
            source_uri=f"{base_uri}#RPE1",
            processed_h5ad_name="rpe1_hvg5000.h5ad",
        ),
    )


@dataclass(frozen=True)
class CloudHandoffConfig:
    """Confirmed infra contract for phase-17 cloud handoff."""

    oracle_vm_shape: str = "VM.Standard2.4"
    oracle_vm_os: str = "Ubuntu 22.04"
    oracle_vm_ocpus: int = 4
    oracle_vm_ram_gb: int = 60
    oracle_region: str = "us-phoenix-1"
    oracle_namespace: str = "axwtnfeahsmg"
    oracle_bucket: str = "tcpe-datasets"
    oracle_bucket_versioning_enabled: bool = True
    kaggle_gpu_type: str = "T4"
    kaggle_gpu_count: int = 2
    kaggle_ram_gb: int = 30
    kaggle_working_dir_limit_gb: int = 20
    hvg_gene_target: int = 5000
    hvg_payload_size_gb_min: int = 10
    hvg_payload_size_gb_max: int = 15
    checkpoint_sync_interval_minutes: int = 15
    kaggle_max_runtime_minutes: int = 480
    oracle_max_runtime_minutes: int = 720
    oracle_auto_stop_after_minutes: int = 715
    planned_budget_usd: float | None = None
    budget_warning_fraction: float = 0.8
    datasets: tuple[Phase17DatasetSpec, ...] = field(default_factory=_default_datasets)

    def __post_init__(self) -> None:
        if str(self.oracle_vm_shape).strip() == "":
            raise ValueError("oracle_vm_shape must be non-empty.")
        if str(self.oracle_vm_os).strip() == "":
            raise ValueError("oracle_vm_os must be non-empty.")
        if self.oracle_vm_ocpus <= 0:
            raise ValueError("oracle_vm_ocpus must be positive.")
        if self.oracle_vm_ram_gb <= 0:
            raise ValueError("oracle_vm_ram_gb must be positive.")
        if str(self.oracle_region).strip() == "":
            raise ValueError("oracle_region must be non-empty.")
        if str(self.oracle_namespace).strip() == "":
            raise ValueError("oracle_namespace must be non-empty.")
        if str(self.oracle_bucket).strip() == "":
            raise ValueError("oracle_bucket must be non-empty.")
        if str(self.kaggle_gpu_type).strip() == "":
            raise ValueError("kaggle_gpu_type must be non-empty.")
        if self.kaggle_gpu_count <= 0:
            raise ValueError("kaggle_gpu_count must be positive.")
        if self.kaggle_ram_gb <= 0:
            raise ValueError("kaggle_ram_gb must be positive.")
        if self.kaggle_working_dir_limit_gb <= 0:
            raise ValueError("kaggle_working_dir_limit_gb must be positive.")
        if self.hvg_gene_target <= 0:
            raise ValueError("hvg_gene_target must be positive.")
        if self.hvg_payload_size_gb_min <= 0:
            raise ValueError("hvg_payload_size_gb_min must be positive.")
        if self.hvg_payload_size_gb_max < self.hvg_payload_size_gb_min:
            raise ValueError("hvg_payload_size_gb_max must be >= hvg_payload_size_gb_min.")
        if self.checkpoint_sync_interval_minutes <= 0:
            raise ValueError("checkpoint_sync_interval_minutes must be positive.")
        if self.kaggle_max_runtime_minutes <= 0:
            raise ValueError("kaggle_max_runtime_minutes must be positive.")
        if self.oracle_max_runtime_minutes <= 0:
            raise ValueError("oracle_max_runtime_minutes must be positive.")
        if self.oracle_auto_stop_after_minutes <= 0:
            raise ValueError("oracle_auto_stop_after_minutes must be positive.")
        if self.oracle_auto_stop_after_minutes > self.oracle_max_runtime_minutes:
            raise ValueError(
                "oracle_auto_stop_after_minutes must not exceed oracle_max_runtime_minutes."
            )
        if self.planned_budget_usd is not None and self.planned_budget_usd < 0:
            raise ValueError("planned_budget_usd must be non-negative when provided.")
        if not 0.0 < self.budget_warning_fraction < 1.0:
            raise ValueError("budget_warning_fraction must be in (0, 1).")
        if len(self.datasets) == 0:
            raise ValueError("At least one dataset must be configured for phase 17.")


@dataclass(frozen=True)
class CloudResourcePreset:
    """One platform-level resource preset."""

    preset_name: str
    platform: Literal["oracle_vm", "kaggle"]
    operating_system: str
    role: str
    cpu_cores: int
    ram_gb: int
    gpu_type: str | None = None
    gpu_count: int = 0
    working_dir_limit_gb: int | None = None
    region: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase17RunPreset:
    """A concrete command preset for the Oracle/Kaggle split."""

    preset_name: str
    executor: Literal["oracle_vm", "kaggle"]
    description: str
    command: str
    writes_checkpoint: bool
    expected_next_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactTransferPlan:
    """One upload/download command plan."""

    provider: ArtifactProvider
    direction: ArtifactDirection
    local_path: str
    remote_ref: str
    command: tuple[str, ...]
    notes: str

    @property
    def shell_command(self) -> str:
        return shlex.join(self.command)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["shell_command"] = self.shell_command
        return payload


@dataclass(frozen=True)
class SpotResumeSimulation:
    """Expected resume behavior after an interruption."""

    interrupted_after_step: str
    resume_from_step: str | None
    completed_steps: list[str]
    checkpoint_recovered: bool
    resume_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CloudHandoffResult:
    """Full phase-17 planning output."""

    schema_version: str
    generated_at_utc: str
    run_id: str
    dry_run: bool
    output_dir: Path | None
    environment_specs: dict[str, dict[str, Any]]
    resource_presets: list[CloudResourcePreset]
    run_presets: list[Phase17RunPreset]
    artifact_transfers: list[ArtifactTransferPlan]
    artifact_smoke_tests: list[dict[str, Any]]
    checkpoint_policy: dict[str, Any]
    budget_log: dict[str, Any]
    spot_resume_simulation: SpotResumeSimulation
    datasets: list[Phase17DatasetSpec]
    preprocessed_seed: dict[str, Any] | None
    report_json_path: Path | None = None
    runbook_markdown_path: Path | None = None
    environment_spec_paths: dict[str, Path] = field(default_factory=dict)
    run_preset_path: Path | None = None
    budget_log_path: Path | None = None
    smoke_test_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "run_id": self.run_id,
            "dry_run": self.dry_run,
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "environment_specs": self.environment_specs,
            "resource_presets": [item.to_dict() for item in self.resource_presets],
            "run_presets": [item.to_dict() for item in self.run_presets],
            "artifact_transfers": [item.to_dict() for item in self.artifact_transfers],
            "artifact_smoke_tests": [dict(item) for item in self.artifact_smoke_tests],
            "checkpoint_policy": dict(self.checkpoint_policy),
            "budget_log": dict(self.budget_log),
            "spot_resume_simulation": self.spot_resume_simulation.to_dict(),
            "datasets": [item.to_dict() for item in self.datasets],
            "preprocessed_seed": (
                None if self.preprocessed_seed is None else dict(self.preprocessed_seed)
            ),
            "report_json_path": (
                None if self.report_json_path is None else str(self.report_json_path)
            ),
            "runbook_markdown_path": (
                None if self.runbook_markdown_path is None else str(self.runbook_markdown_path)
            ),
            "environment_spec_paths": {
                key: str(path) for key, path in self.environment_spec_paths.items()
            },
            "run_preset_path": None if self.run_preset_path is None else str(self.run_preset_path),
            "budget_log_path": None if self.budget_log_path is None else str(self.budget_log_path),
            "smoke_test_path": None if self.smoke_test_path is None else str(self.smoke_test_path),
        }


class CloudHandoffModule:
    """Phase-17 operational handoff planner and checkpoint seeder."""

    def status(self) -> str:
        return "phase17_cloud_handoff_ready"

    def run(
        self,
        *,
        config: TCPEConfig,
        layout: ArtifactLayout,
        run_id: str,
        handoff_config: CloudHandoffConfig | None = None,
        preprocessed_h5ad_path: str | Path | None = None,
        dataset_label: str = "external_hvg_h5ad",
        budget_spent_usd: float = 0.0,
        simulate_interruption_after: PipelineStep = "train",
        dry_run: bool = False,
    ) -> CloudHandoffResult:
        if budget_spent_usd < 0:
            raise ValueError("budget_spent_usd must be non-negative.")
        if str(dataset_label).strip() == "":
            raise ValueError("dataset_label must be non-empty.")
        if simulate_interruption_after not in PIPELINE_STEPS:
            raise ValueError(
                f"Unsupported simulate_interruption_after '{simulate_interruption_after}'."
            )

        selected = handoff_config if handoff_config is not None else CloudHandoffConfig()
        generated_at = datetime.now(UTC).isoformat()
        output_dir = None if dry_run else (layout.metadata_dir / "phase17_cloud_handoff")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        environment_specs = self._build_environment_specs(config=config, selected=selected)
        resource_presets = self._build_resource_presets(selected=selected)
        run_presets = self._build_run_presets(
            config=config,
            run_id=run_id,
            preprocessed_h5ad_path=preprocessed_h5ad_path,
            dataset_label=dataset_label,
        )
        checkpoint_policy = self._build_checkpoint_policy(
            selected=selected,
            run_id=run_id,
        )
        budget_log = self._build_budget_log(selected=selected, budget_spent_usd=budget_spent_usd)
        seed_payload = None
        if preprocessed_h5ad_path is not None and not dry_run:
            seed_payload = self._seed_preprocessed_resume_bundle(
                layout=layout,
                run_id=run_id,
                preprocessed_h5ad_path=Path(preprocessed_h5ad_path),
                dataset_label=dataset_label,
                output_dir=cast(Path, output_dir),
            )
        artifact_target = self._resolve_transfer_target(
            output_dir=output_dir,
            layout=layout,
        )
        artifact_transfers = self._build_artifact_transfers(
            run_id=run_id,
            local_target=artifact_target,
        )
        smoke_tests = self._run_artifact_smoke_tests(artifact_transfers)
        spot_resume = self._simulate_spot_resume(
            run_id=run_id,
            layout=layout,
            interrupted_after=simulate_interruption_after,
            checkpoint_seeded=seed_payload is not None,
        )

        report_json_path: Path | None = None
        runbook_path: Path | None = None
        environment_spec_paths: dict[str, Path] = {}
        run_preset_path: Path | None = None
        budget_log_path: Path | None = None
        smoke_test_path: Path | None = None

        if output_dir is not None:
            for key, payload in environment_specs.items():
                path = output_dir / f"{key}.yaml"
                path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
                environment_spec_paths[key] = path

            run_preset_path = output_dir / "run_presets.json"
            run_preset_path.write_text(
                json.dumps(
                    {
                        "schema_version": RUN_PRESET_SCHEMA_VERSION,
                        "generated_at_utc": generated_at,
                        "run_id": run_id,
                        "run_presets": [item.to_dict() for item in run_presets],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            budget_log_path = output_dir / "budget_log.json"
            budget_log_path.write_text(
                json.dumps(budget_log, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            smoke_test_path = output_dir / "artifact_smoke_tests.json"
            smoke_test_path.write_text(
                json.dumps(
                    {
                        "schema_version": ARTIFACT_SYNC_SCHEMA_VERSION,
                        "generated_at_utc": generated_at,
                        "results": smoke_tests,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            runbook_path = output_dir / "phase17_tmux_runbook.md"
            runbook_path.write_text(
                _render_runbook(
                    run_id=run_id,
                    config=config,
                    selected=selected,
                    run_presets=run_presets,
                    checkpoint_policy=checkpoint_policy,
                    budget_log=budget_log,
                    spot_resume=spot_resume,
                ),
                encoding="utf-8",
            )

            report_json_path = output_dir / "phase17_cloud_handoff_plan.json"

        result = CloudHandoffResult(
            schema_version=CLOUD_HANDOFF_SCHEMA_VERSION,
            generated_at_utc=generated_at,
            run_id=run_id,
            dry_run=dry_run,
            output_dir=output_dir,
            environment_specs=environment_specs,
            resource_presets=resource_presets,
            run_presets=run_presets,
            artifact_transfers=artifact_transfers,
            artifact_smoke_tests=smoke_tests,
            checkpoint_policy=checkpoint_policy,
            budget_log=budget_log,
            spot_resume_simulation=spot_resume,
            datasets=list(selected.datasets),
            preprocessed_seed=seed_payload,
            report_json_path=report_json_path,
            runbook_markdown_path=runbook_path,
            environment_spec_paths=environment_spec_paths,
            run_preset_path=run_preset_path,
            budget_log_path=budget_log_path,
            smoke_test_path=smoke_test_path,
        )

        if report_json_path is not None:
            report_json_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return result

    def _build_environment_specs(
        self,
        *,
        config: TCPEConfig,
        selected: CloudHandoffConfig,
    ) -> dict[str, dict[str, Any]]:
        oracle_spec = {
            "schema_version": ENVIRONMENT_SPEC_SCHEMA_VERSION,
            "platform": "oracle_vm",
            "os": selected.oracle_vm_os,
            "shape": selected.oracle_vm_shape,
            "ocpus": selected.oracle_vm_ocpus,
            "ram_gb": selected.oracle_vm_ram_gb,
            "region": selected.oracle_region,
            "tmux_required": True,
            "roles": [
                "raw_data_download",
                "cpu_preprocessing",
                "causal",
                "evaluation",
                "publication",
            ],
            "object_storage": {
                "namespace": selected.oracle_namespace,
                "bucket": selected.oracle_bucket,
                "versioning_enabled": selected.oracle_bucket_versioning_enabled,
            },
            "artifact_root": str(config.paths.artifact_root),
        }
        kaggle_spec = {
            "schema_version": ENVIRONMENT_SPEC_SCHEMA_VERSION,
            "platform": "kaggle",
            "gpu_type": selected.kaggle_gpu_type,
            "gpu_count": selected.kaggle_gpu_count,
            "ram_gb": selected.kaggle_ram_gb,
            "working_dir_limit_gb": selected.kaggle_working_dir_limit_gb,
            "roles": ["embedding", "transport_training"],
            "accepted_inputs": {
                "hvg_gene_target": selected.hvg_gene_target,
                "expected_payload_size_gb_min": selected.hvg_payload_size_gb_min,
                "expected_payload_size_gb_max": selected.hvg_payload_size_gb_max,
            },
        }
        return {
            "oracle_vm_environment": oracle_spec,
            "kaggle_environment": kaggle_spec,
        }

    def _build_resource_presets(self, *, selected: CloudHandoffConfig) -> list[CloudResourcePreset]:
        return [
            CloudResourcePreset(
                preset_name="oracle_vm_cpu_jobs",
                platform="oracle_vm",
                operating_system=selected.oracle_vm_os,
                role="Raw download, preprocessing, causal, evaluation",
                cpu_cores=selected.oracle_vm_ocpus,
                ram_gb=selected.oracle_vm_ram_gb,
                region=selected.oracle_region,
                extra={
                    "shape": selected.oracle_vm_shape,
                    "bucket": selected.oracle_bucket,
                    "namespace": selected.oracle_namespace,
                    "tmux_required": True,
                },
            ),
            CloudResourcePreset(
                preset_name="kaggle_t4_x2_training",
                platform="kaggle",
                operating_system="Kaggle Linux",
                role="Embedding and neural transport training",
                cpu_cores=2,
                ram_gb=selected.kaggle_ram_gb,
                gpu_type=selected.kaggle_gpu_type,
                gpu_count=selected.kaggle_gpu_count,
                working_dir_limit_gb=selected.kaggle_working_dir_limit_gb,
                extra={
                    "hvg_gene_target": selected.hvg_gene_target,
                    "payload_size_gb_range": [
                        selected.hvg_payload_size_gb_min,
                        selected.hvg_payload_size_gb_max,
                    ],
                },
            ),
        ]

    def _build_run_presets(
        self,
        *,
        config: TCPEConfig,
        run_id: str,
        preprocessed_h5ad_path: str | Path | None,
        dataset_label: str,
    ) -> list[Phase17RunPreset]:
        h5ad_arg = (
            shlex.quote(str(Path(preprocessed_h5ad_path)))
            if preprocessed_h5ad_path is not None
            else shlex.quote("<path-to-hvg5000.h5ad>")
        )
        dataset_arg = shlex.quote(dataset_label)
        cloud_seed = (
            f"python -m tcpe cloud --env {config.environment} --run-id {shlex.quote(run_id)} "
            f"--preprocessed-h5ad {h5ad_arg} --dataset-label {dataset_arg}"
        )
        kaggle_train = (
            f"python -m tcpe pipeline --env kaggle --run-id {shlex.quote(run_id)} "
            "--resume --stop-after train"
        )
        oracle_resume = (
            f"python -m tcpe pipeline --env local --run-id {shlex.quote(run_id)} --resume"
        )
        return [
            Phase17RunPreset(
                preset_name="oracle_seed_resume_bundle",
                executor="oracle_vm",
                description=(
                    "Convert an externally preprocessed HVG-selected h5ad into a phase-15 "
                    "resume-ready checkpoint layout."
                ),
                command=cloud_seed,
                writes_checkpoint=True,
                expected_next_step="embed",
            ),
            Phase17RunPreset(
                preset_name="kaggle_gpu_train_only",
                executor="kaggle",
                description="Resume from the seeded checkpoint and stop immediately after train.",
                command=kaggle_train,
                writes_checkpoint=True,
                expected_next_step="causal",
            ),
            Phase17RunPreset(
                preset_name="oracle_finalize_cpu_stages",
                executor="oracle_vm",
                description="Resume the same run after Kaggle and finish causal/evaluate/card.",
                command=oracle_resume,
                writes_checkpoint=True,
                expected_next_step=None,
            ),
        ]

    def _build_checkpoint_policy(
        self,
        *,
        selected: CloudHandoffConfig,
        run_id: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": SPOT_RESUME_SCHEMA_VERSION,
            "checkpoint_contract": {
                "pipeline_checkpoint_file": "metadata/pipeline_checkpoint.json",
                "required_stage_files": [
                    "checkpoints/step_preprocess_adata.h5ad",
                    "metadata/pipeline_checkpoint.json",
                ],
                "checkpoint_sync_interval_minutes": selected.checkpoint_sync_interval_minutes,
                "persist_after_every_completed_step": True,
            },
            "kaggle_policy": {
                "max_runtime_minutes": selected.kaggle_max_runtime_minutes,
                "run_mode": "resume_with_stop_after_train",
                "sync_back_to_oracle_bucket_after": ["embed", "train"],
            },
            "oracle_policy": {
                "max_runtime_minutes": selected.oracle_max_runtime_minutes,
                "auto_stop_after_minutes": selected.oracle_auto_stop_after_minutes,
                "auto_stop_command": f"sudo shutdown -h +{selected.oracle_auto_stop_after_minutes}",
                "tmux_required": True,
            },
            "failure_recovery": {
                "authoritative_storage": (
                    f"oci://{selected.oracle_namespace}/{selected.oracle_bucket}/runs/{run_id}/"
                ),
                "resume_rule": (
                    "Always resume with the same run_id after restoring the full run layout."
                ),
            },
        }

    def _build_budget_log(
        self,
        *,
        selected: CloudHandoffConfig,
        budget_spent_usd: float,
    ) -> dict[str, Any]:
        log = {
            "schema_version": BUDGET_LOG_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "planned_budget_usd": selected.planned_budget_usd,
            "spent_budget_usd": float(budget_spent_usd),
        }
        planned = selected.planned_budget_usd
        if planned is None:
            log["status"] = "tracking_only"
            log["remaining_budget_usd"] = None
            log["warning_threshold_usd"] = None
            return log

        warning_threshold = planned * selected.budget_warning_fraction
        remaining = planned - budget_spent_usd
        if budget_spent_usd >= planned:
            status = "over_limit"
        elif budget_spent_usd >= warning_threshold:
            status = "warning"
        else:
            status = "within_limit"
        log["status"] = status
        log["remaining_budget_usd"] = float(remaining)
        log["warning_threshold_usd"] = float(warning_threshold)
        return log

    def _seed_preprocessed_resume_bundle(
        self,
        *,
        layout: ArtifactLayout,
        run_id: str,
        preprocessed_h5ad_path: Path,
        dataset_label: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        if not preprocessed_h5ad_path.exists():
            raise CloudHandoffError(f"Preprocessed h5ad does not exist: {preprocessed_h5ad_path}")

        layout.run_root.mkdir(parents=True, exist_ok=True)
        layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        layout.metadata_dir.mkdir(parents=True, exist_ok=True)

        target_path = layout.checkpoints_dir / "step_preprocess_adata.h5ad"
        if preprocessed_h5ad_path.resolve() != target_path.resolve():
            shutil.copy2(preprocessed_h5ad_path, target_path)

        checkpoint = PipelineCheckpoint(
            schema_version=PIPELINE_CHECKPOINT_SCHEMA_VERSION,
            run_id=run_id,
            completed_steps=[
                cast(PipelineStep, "ingest"),
                cast(PipelineStep, "preprocess"),
            ],
            step_artifacts={
                cast(PipelineStep, "ingest"): {
                    "adata_path": str(target_path),
                    "dataset_id": dataset_label,
                    "source_type": "phase17_preprocessed_seed",
                },
                cast(PipelineStep, "preprocess"): {
                    "adata_path": str(target_path),
                    "dataset_id": dataset_label,
                    "n_hvgs_selected": 5000,
                    "seeded_by_phase17": True,
                },
            },
        )
        checkpoint_path = layout.metadata_dir / "pipeline_checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        seed_payload = {
            "schema_version": PREPROCESSED_SEED_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "dataset_label": dataset_label,
            "source_h5ad_path": str(preprocessed_h5ad_path),
            "seeded_h5ad_path": str(target_path),
            "checkpoint_path": str(checkpoint_path),
            "completed_steps": ["ingest", "preprocess"],
        }
        seed_manifest_path = output_dir / "preprocessed_seed_manifest.json"
        seed_manifest_path.write_text(
            json.dumps(seed_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            **seed_payload,
            "seed_manifest_path": str(seed_manifest_path),
        }

    def _resolve_transfer_target(
        self,
        *,
        output_dir: Path | None,
        layout: ArtifactLayout,
    ) -> Path:
        if output_dir is not None:
            return output_dir / "phase17_cloud_handoff_plan.json"
        return layout.metadata_dir / "phase17_cloud_handoff_plan.json"

    def _build_artifact_transfers(
        self,
        *,
        run_id: str,
        local_target: Path,
    ) -> list[ArtifactTransferPlan]:
        github_ref = f"tcpe-{run_id}"
        hf_ref = f"tcpe/{run_id}"
        zenodo_ref = run_id
        return [
            build_artifact_transfer_plan(
                provider="github",
                direction="push",
                local_path=local_target,
                remote_ref=github_ref,
            ),
            build_artifact_transfer_plan(
                provider="github",
                direction="pull",
                local_path=local_target,
                remote_ref=github_ref,
            ),
            build_artifact_transfer_plan(
                provider="huggingface",
                direction="push",
                local_path=local_target,
                remote_ref=hf_ref,
            ),
            build_artifact_transfer_plan(
                provider="huggingface",
                direction="pull",
                local_path=local_target,
                remote_ref=hf_ref,
            ),
            build_artifact_transfer_plan(
                provider="zenodo",
                direction="push",
                local_path=local_target,
                remote_ref=zenodo_ref,
            ),
            build_artifact_transfer_plan(
                provider="zenodo",
                direction="pull",
                local_path=local_target,
                remote_ref=zenodo_ref,
            ),
        ]

    def _run_artifact_smoke_tests(
        self,
        plans: list[ArtifactTransferPlan],
    ) -> list[dict[str, Any]]:
        expected_binary = {
            "github": "gh",
            "huggingface": "huggingface-cli",
            "zenodo": "curl",
        }
        results: list[dict[str, Any]] = []
        for plan in plans:
            head = plan.command[0] if len(plan.command) > 0 else ""
            passed = head == expected_binary[plan.provider] and len(plan.command) >= 2
            results.append(
                {
                    "provider": plan.provider,
                    "direction": plan.direction,
                    "passed": passed,
                    "command_head": head,
                    "shell_command": plan.shell_command,
                }
            )
        return results

    def _simulate_spot_resume(
        self,
        *,
        run_id: str,
        layout: ArtifactLayout,
        interrupted_after: PipelineStep,
        checkpoint_seeded: bool,
    ) -> SpotResumeSimulation:
        step_list = list(PIPELINE_STEPS)
        index = step_list.index(interrupted_after)
        completed = step_list[: index + 1]
        resume_from_step = step_list[index + 1] if index + 1 < len(step_list) else None
        checkpoint_path = layout.metadata_dir / "pipeline_checkpoint.json"
        checkpoint_recovered = (
            checkpoint_seeded
            or checkpoint_path.exists()
            or interrupted_after in ("ingest", "preprocess")
        )
        resume_command = (
            f"python -m tcpe pipeline --run-id {shlex.quote(run_id)} --resume"
            if resume_from_step is not None
            else f"python -m tcpe pipeline --run-id {shlex.quote(run_id)} --resume  # no-op"
        )
        return SpotResumeSimulation(
            interrupted_after_step=interrupted_after,
            resume_from_step=resume_from_step,
            completed_steps=completed,
            checkpoint_recovered=checkpoint_recovered,
            resume_command=resume_command,
        )


def build_artifact_transfer_plan(
    *,
    provider: ArtifactProvider,
    direction: ArtifactDirection,
    local_path: str | Path,
    remote_ref: str,
) -> ArtifactTransferPlan:
    """Create a provider-specific upload/download command plan."""
    path = Path(local_path)
    if str(remote_ref).strip() == "":
        raise ValueError("remote_ref must be non-empty.")

    if provider == "github":
        if direction == "push":
            command = ("gh", "release", "upload", remote_ref, str(path), "--clobber")
            notes = "GitHub release upload; remote_ref is the release tag."
        else:
            command = (
                "gh",
                "release",
                "download",
                remote_ref,
                "--pattern",
                path.name,
                "--dir",
                str(path.parent),
            )
            notes = "GitHub release download; remote_ref is the release tag."
    elif provider == "huggingface":
        repo_path = f"runs/{path.name}"
        if direction == "push":
            command = ("huggingface-cli", "upload", remote_ref, str(path), repo_path)
            notes = "Hugging Face Hub upload; remote_ref is the repo id."
        else:
            command = (
                "huggingface-cli",
                "download",
                remote_ref,
                repo_path,
                "--local-dir",
                str(path.parent),
            )
            notes = "Hugging Face Hub download; remote_ref is the repo id."
    else:
        if direction == "push":
            command = (
                "curl",
                "-X",
                "PUT",
                f"$ZENODO_BUCKET_URL/{remote_ref}/{path.name}",
                "--upload-file",
                str(path),
            )
            notes = "Zenodo upload; set ZENODO_BUCKET_URL to the deposition bucket URL."
        else:
            command = (
                "curl",
                "-L",
                f"$ZENODO_DOWNLOAD_URL/{remote_ref}/{path.name}",
                "-o",
                str(path),
            )
            notes = "Zenodo download; set ZENODO_DOWNLOAD_URL to the public API file prefix."

    return ArtifactTransferPlan(
        provider=provider,
        direction=direction,
        local_path=str(path),
        remote_ref=remote_ref,
        command=command,
        notes=notes,
    )


def artifact_sync_main(argv: list[str] | None = None) -> int:
    """Standalone CLI for provider-agnostic artifact sync command planning."""
    parser = argparse.ArgumentParser(
        prog="cloud_artifact_sync",
        description=(
            "Phase 17 artifact upload/download planner for GitHub, Hugging Face, and Zenodo."
        ),
    )
    parser.add_argument("--provider", choices=("github", "huggingface", "zenodo"), required=True)
    parser.add_argument("--direction", choices=("push", "pull"), required=True)
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--remote-ref", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Accepted for interface symmetry; command planning is always dry-run only.",
    )
    args = parser.parse_args(argv)

    plan = build_artifact_transfer_plan(
        provider=cast(ArtifactProvider, args.provider),
        direction=cast(ArtifactDirection, args.direction),
        local_path=Path(args.local_path),
        remote_ref=str(args.remote_ref),
    )
    payload = {
        "schema_version": ARTIFACT_SYNC_SCHEMA_VERSION,
        "dry_run": True,
        "plan": plan.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _render_runbook(
    *,
    run_id: str,
    config: TCPEConfig,
    selected: CloudHandoffConfig,
    run_presets: list[Phase17RunPreset],
    checkpoint_policy: dict[str, Any],
    budget_log: dict[str, Any],
    spot_resume: SpotResumeSimulation,
) -> str:
    lines: list[str] = []
    lines.append("# Phase 17 Cloud Handoff Runbook")
    lines.append("")
    lines.append(f"- Run id: `{run_id}`")
    lines.append(f"- Oracle VM: `{selected.oracle_vm_shape}` on `{selected.oracle_vm_os}`")
    lines.append(
        f"- Oracle Object Storage: `oci://{selected.oracle_namespace}/{selected.oracle_bucket}`"
    )
    lines.append(f"- Kaggle GPU preset: `{selected.kaggle_gpu_type} x{selected.kaggle_gpu_count}`")
    lines.append(f"- Kaggle working directory limit: `{selected.kaggle_working_dir_limit_gb}GB`")
    lines.append("")
    lines.append("## Dataset Contract")
    lines.append("")
    for dataset in selected.datasets:
        lines.append(
            f"- `{dataset.display_name}` -> `{dataset.processed_h5ad_name}` "
            f"({dataset.hvg_gene_target} HVGs, ~{dataset.expected_size_gb_min}-"
            f"{dataset.expected_size_gb_max}GB)"
        )
    lines.append("")
    lines.append("## tmux Sessions")
    lines.append("")
    lines.append("- Oracle preprocess seed session:")
    lines.append("  `tmux new-session -d -s tcpe-seed '<phase17 cloud seed command>'`")
    lines.append("- Oracle finalize session:")
    lines.append("  `tmux new-session -d -s tcpe-finalize '<phase15 pipeline resume command>'`")
    lines.append(
        "- Reattach with `tmux attach -t <session-name>` and inspect logs before terminating."
    )
    lines.append("")
    lines.append("## Execution Order")
    lines.append("")
    for preset in run_presets:
        lines.append(f"- `{preset.preset_name}` ({preset.executor}): `{preset.command}`")
    lines.append("")
    lines.append("## Recovery Rules")
    lines.append("")
    lines.append(
        "- Sync the entire run directory through the Oracle bucket after every completed step; "
        "the bucket is the source of truth across Oracle and Kaggle."
    )
    lines.append(
        f"- Checkpoint cadence: every `{selected.checkpoint_sync_interval_minutes}` minutes, and "
        "always immediately after `embed` and `train`."
    )
    lines.append(
        "- If Kaggle is interrupted, restore the run directory from Object Storage and resume "
        "with the same run id."
    )
    lines.append(
        f"- Spot simulation: interruption after `{spot_resume.interrupted_after_step}` resumes at "
        f"`{spot_resume.resume_from_step}` with `{spot_resume.resume_command}`."
    )
    lines.append("")
    lines.append("## Cost Guards")
    lines.append("")
    lines.append(
        f"- Oracle max runtime: "
        f"`{checkpoint_policy['oracle_policy']['max_runtime_minutes']}` minutes."
    )
    lines.append(
        f"- Oracle auto-stop command: `{checkpoint_policy['oracle_policy']['auto_stop_command']}`"
    )
    lines.append(
        f"- Kaggle max runtime: "
        f"`{checkpoint_policy['kaggle_policy']['max_runtime_minutes']}` minutes."
    )
    lines.append(f"- Budget status: `{budget_log['status']}`")
    lines.append("")
    lines.append("## Failure Recovery")
    lines.append("")
    lines.append(
        "- If `pipeline_checkpoint.json` is missing, regenerate the seed bundle from the last"
    )
    lines.append("  uploaded HVG-selected h5ad on Oracle, then re-upload the full run root.")
    lines.append(
        "- If `transport_model.ckpt` exists but the run stopped before Oracle finalization,"
    )
    lines.append(
        "  restore the Kaggle-produced run root and execute the Oracle resume command only."
    )
    lines.append("- If publication mirroring fails, keep Object Storage as authoritative and retry")
    lines.append("  GitHub/Hugging Face/Zenodo from the saved run manifest later.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Base artifact root in config: `{config.paths.artifact_root}`")
    lines.append(
        "- This phase does not start full training by itself; it prepares the handoff surface."
    )
    lines.append("")
    return "\n".join(lines)
