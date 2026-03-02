"""Phase 16 tests for local validation ladder and promotion gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from tcpe.config import DEFAULT_CONFIG_PATH, load_config
from tcpe.local_validation import (
    LocalValidationConfig,
    LocalValidationModule,
    ValidationStageSpec,
)
from tcpe.pipeline import PipelineRunResult


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def _write_temp_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Base config is not a mapping.")
    paths = payload.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise AssertionError("Config `paths` must be a mapping.")
    paths["artifact_root"] = (tmp_path / "artifacts").as_posix()
    paths["data_root"] = (tmp_path / "data").as_posix()
    paths["cache_root"] = (tmp_path / ".cache" / "tcpe").as_posix()

    model = payload.setdefault("model", {})
    if not isinstance(model, dict):
        raise AssertionError("Config `model` must be a mapping.")
    model["latent_dim"] = 20

    runtime = payload.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise AssertionError("Config `runtime` must be a mapping.")
    runtime["seed"] = 13

    config_path = tmp_path / "phase16_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _make_mock_pipeline_run(
    *,
    layout: Any,
    run_id: str,
    stage_name: str,
    model_beats: bool = True,
) -> PipelineRunResult:
    import anndata as ad

    checkpoints_dir = Path(layout.checkpoints_dir)
    reports_dir = Path(layout.reports_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = reports_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_card_dir = reports_dir / "model_card"
    model_card_dir.mkdir(parents=True, exist_ok=True)
    causal_dir = reports_dir / "causal"
    causal_dir.mkdir(parents=True, exist_ok=True)

    n_genes = 6
    genes = [f"g{idx}" for idx in range(n_genes)]
    adata = ad.AnnData(
        X=np.zeros((8, n_genes), dtype=np.float32),
        obs={"cell_id": [f"c{idx}" for idx in range(8)]},
        var={"gene_id": genes},
    )
    adata.var.index = genes
    truth = np.zeros((n_genes, n_genes), dtype=np.float64)
    truth[0, 1] = 0.8
    truth[2, 3] = -0.7
    truth[4, 5] = 0.6
    adata.uns["synthetic_ground_truth_adjacency"] = truth
    causal_adata_path = checkpoints_dir / "step_causal_adata.h5ad"
    adata.write_h5ad(causal_adata_path)

    eval_payload = {
        "schema_version": "evaluation_report_v1",
        "metrics": {
            "mae": 0.10,
            "pearson_top_1000_degs": 0.60,
            "calibration_error": 0.05,
        },
        "baseline_comparison": [
            {
                "baseline_name": "control_mean",
                "baseline_mae": 0.20,
                "model_mae": 0.10,
                "delta_mae_model_minus_baseline": -0.10,
                "model_beats_baseline_mae": model_beats,
                "baseline_pearson_mean_gene": 0.20,
                "model_pearson_top_1000_degs": 0.60,
                "delta_pearson_model_minus_baseline": 0.40,
            },
            {
                "baseline_name": "gene_level_mean",
                "baseline_mae": 0.18,
                "model_mae": 0.10,
                "delta_mae_model_minus_baseline": -0.08,
                "model_beats_baseline_mae": model_beats,
                "baseline_pearson_mean_gene": 0.30,
                "model_pearson_top_1000_degs": 0.60,
                "delta_pearson_model_minus_baseline": 0.30,
            },
        ],
        "failure_modes": [{"code": "none", "severity": "info", "message": "ok"}],
        "metadata": {"model_name": "ot_sinkhorn"},
    }
    eval_path = eval_dir / "evaluation_report.json"
    eval_path.write_text(json.dumps(eval_payload, indent=2, sort_keys=True), encoding="utf-8")

    adjacency = np.zeros_like(truth)
    adjacency[0, 1] = 0.95
    adjacency[2, 3] = 0.90
    adjacency[4, 5] = 0.92
    adjacency_path = causal_dir / "causal_graph_adjacency.npz"
    np.savez_compressed(
        adjacency_path,
        adjacency=adjacency,
        ci_lower=np.zeros_like(adjacency),
        ci_upper=np.zeros_like(adjacency),
        standard_error=np.zeros_like(adjacency),
    )
    causal_meta_path = causal_dir / "causal_graph_metadata.json"
    causal_meta_path.write_text(
        json.dumps(
            {"schema_version": "causal_graph_v1", "gene_ids": genes, "metadata": {}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    model_card_json = model_card_dir / "model_card.json"
    model_card_md = model_card_dir / "model_card.md"
    model_card_json.write_text(
        json.dumps({"schema_version": "model_card_v1"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model_card_md.write_text("# Card\n", encoding="utf-8")

    placeholder_adata_path = checkpoints_dir / f"{stage_name}_adata.h5ad"
    adata.write_h5ad(placeholder_adata_path)

    step_artifacts = {
        "ingest": {"adata_path": str(placeholder_adata_path)},
        "preprocess": {"adata_path": str(placeholder_adata_path)},
        "embed": {"adata_path": str(placeholder_adata_path)},
        "train": {
            "adata_path": str(placeholder_adata_path),
            "transport_checkpoint_path": str(checkpoints_dir / "transport.ckpt"),
            "transport_fit_summary_path": str(reports_dir / "transport_fit_summary.json"),
            "baseline_suite_json_path": str(reports_dir / "baseline_suite_results.json"),
            "transport_variant": "ot_sinkhorn",
            "baseline_schema_version": "baseline_suite_v1",
        },
        "causal": {
            "adata_path": str(causal_adata_path),
            "causal_adjacency_npz_path": str(adjacency_path),
            "causal_edge_table_csv_path": str(causal_dir / "causal_graph_edges.csv"),
            "causal_metadata_json_path": str(causal_meta_path),
            "causal_schema_version": "causal_graph_v1",
        },
        "evaluate": {
            "adata_path": str(placeholder_adata_path),
            "evaluation_report_json_path": str(eval_path),
            "prediction_npz_path": str(eval_dir / "transport_prediction.npz"),
        },
        "card": {
            "adata_path": str(placeholder_adata_path),
            "model_card_json_path": str(model_card_json),
            "model_card_markdown_path": str(model_card_md),
            "model_card_schema_version": "model_card_v1",
        },
    }
    return PipelineRunResult(
        run_id=run_id,
        status="completed",
        completed_steps=["ingest", "preprocess", "embed", "train", "causal", "evaluate", "card"],
        skipped_steps=[],
        resumed_from_checkpoint=False,
        checkpoint_path=Path(layout.metadata_dir) / "pipeline_checkpoint.json",
        run_report_path=Path(layout.metadata_dir) / "pipeline_run_report.json",
        artifact_manifest_path=Path(layout.metadata_dir) / "pipeline_artifact_manifest.json",
        artifact_bundle_dir=reports_dir / "cloud_handoff_bundle",
        step_artifacts=step_artifacts,
    )


def test_local_validation_ladder_passes_all_gates_with_mocked_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_anndata()
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")

    def fake_run(
        self: Any,
        *,
        config: Any,
        layout: Any,
        run_id: str,
        pipeline_config: Any,
        resume: bool,
    ) -> PipelineRunResult:
        stage_name = run_id.split("-")[-1]
        return _make_mock_pipeline_run(layout=layout, run_id=run_id, stage_name=stage_name)

    monkeypatch.setattr("tcpe.local_validation.PipelineModule.run", fake_run)

    module = LocalValidationModule()
    report = module.run(
        config=config,
        output_dir=tmp_path / "ladder",
        validation_config=LocalValidationConfig(
            stages=(
                ValidationStageSpec("synthetic", "synthetic", runtime_limit_seconds=999.0),
                ValidationStageSpec("adamson", "adamson", runtime_limit_seconds=999.0),
                ValidationStageSpec(
                    "replogle_sample",
                    "replogle_sample",
                    runtime_limit_seconds=999.0,
                ),
            ),
            run_pytest=False,
            run_coverage=False,
            pytest_passed_override=True,
            coverage_fraction_override=0.90,
            causal_min_auc=0.50,
            causal_min_f1=0.40,
        ),
    )
    assert module.status() == "phase16_local_validation_ready"
    assert report.all_gates_passed is True
    assert report.eligible_for_cloud is True
    assert report.report_json_path.exists()
    assert report.report_markdown_path.exists()
    assert len(report.stage_results) == 3


def test_local_validation_ladder_fails_when_coverage_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_anndata()
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")

    def fake_run(
        self: Any,
        *,
        config: Any,
        layout: Any,
        run_id: str,
        pipeline_config: Any,
        resume: bool,
    ) -> PipelineRunResult:
        stage_name = run_id.split("-")[-1]
        return _make_mock_pipeline_run(layout=layout, run_id=run_id, stage_name=stage_name)

    monkeypatch.setattr("tcpe.local_validation.PipelineModule.run", fake_run)

    report = LocalValidationModule().run(
        config=config,
        output_dir=tmp_path / "ladder_cov_fail",
        validation_config=LocalValidationConfig(
            stages=(
                ValidationStageSpec("synthetic", "synthetic", runtime_limit_seconds=999.0),
                ValidationStageSpec("adamson", "adamson", runtime_limit_seconds=999.0),
                ValidationStageSpec(
                    "replogle_sample",
                    "replogle_sample",
                    runtime_limit_seconds=999.0,
                ),
            ),
            run_pytest=False,
            run_coverage=False,
            pytest_passed_override=True,
            coverage_fraction_override=0.50,
            coverage_min_fraction=0.85,
            causal_min_auc=0.50,
            causal_min_f1=0.40,
        ),
    )
    assert report.all_gates_passed is False
    gate = next(item for item in report.gates if item.gate_name == "coverage_at_least_85_percent")
    assert gate.passed is False


def test_local_validation_ladder_baseline_gate_detects_non_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_anndata()
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")

    def fake_run(
        self: Any,
        *,
        config: Any,
        layout: Any,
        run_id: str,
        pipeline_config: Any,
        resume: bool,
    ) -> PipelineRunResult:
        stage_name = run_id.split("-")[-1]
        model_beats = stage_name != "adamson"
        return _make_mock_pipeline_run(
            layout=layout,
            run_id=run_id,
            stage_name=stage_name,
            model_beats=model_beats,
        )

    monkeypatch.setattr("tcpe.local_validation.PipelineModule.run", fake_run)

    report = LocalValidationModule().run(
        config=config,
        output_dir=tmp_path / "ladder_baseline_fail",
        validation_config=LocalValidationConfig(
            stages=(
                ValidationStageSpec("synthetic", "synthetic", runtime_limit_seconds=999.0),
                ValidationStageSpec("adamson", "adamson", runtime_limit_seconds=999.0),
                ValidationStageSpec(
                    "replogle_sample",
                    "replogle_sample",
                    runtime_limit_seconds=999.0,
                ),
            ),
            run_pytest=False,
            run_coverage=False,
            pytest_passed_override=True,
            coverage_fraction_override=0.90,
            causal_min_auc=0.50,
            causal_min_f1=0.40,
        ),
    )
    assert report.all_gates_passed is False
    baseline_gate = next(
        item for item in report.gates if item.gate_name == "baseline_dominance_on_real_datasets"
    )
    assert baseline_gate.passed is False
