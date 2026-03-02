"""Phase 15 tests for end-to-end pipeline orchestration and resume behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from tcpe.cli import main
from tcpe.config import DEFAULT_CONFIG_PATH, load_config
from tcpe.pipeline import PIPELINE_STEPS, PipelineConfig, PipelineModule
from tcpe.runtime.run_context import (
    build_artifact_layout,
    ensure_artifact_layout,
    generate_run_id,
    write_run_manifest,
)


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def _write_temp_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Base config is not a YAML mapping.")

    paths = payload.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise AssertionError("Base config `paths` is not a mapping.")
    paths["artifact_root"] = (tmp_path / "artifacts").as_posix()
    paths["data_root"] = (tmp_path / "data").as_posix()
    paths["cache_root"] = (tmp_path / ".cache" / "tcpe").as_posix()

    model = payload.setdefault("model", {})
    if not isinstance(model, dict):
        raise AssertionError("Base config `model` is not a mapping.")
    model["latent_dim"] = 24

    runtime = payload.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise AssertionError("Base config `runtime` is not a mapping.")
    runtime["seed"] = 19

    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_single_cli_pipeline_command_completes_synthetic_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _require_anndata()
    config_path = _write_temp_config(tmp_path)

    exit_code = main(
        [
            "pipeline",
            "--config",
            str(config_path),
            "--env",
            "local",
            "--synthetic-cells",
            "84",
            "--synthetic-genes",
            "40",
            "--synthetic-perturbations",
            "6",
            "--transport-epochs",
            "2",
            "--transport-latent-dim",
            "24",
            "--transport-hidden-dim",
            "64",
            "--causal-max-hvgs",
            "18",
            "--causal-bootstrap-iters",
            "0",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    result = payload["pipeline_result"]
    assert result["status"] == "completed"
    assert result["artifact_manifest_path"] is not None
    assert Path(result["artifact_manifest_path"]).exists()

    card_artifacts = result["step_artifacts"]["card"]
    assert Path(card_artifacts["model_card_json_path"]).exists()
    assert Path(card_artifacts["model_card_markdown_path"]).exists()


def test_partial_rerun_resumes_from_checkpoint(tmp_path: Path) -> None:
    _require_anndata()
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")

    run_id = generate_run_id(config=config, command_group="pipeline", provided="phase15-resume")
    layout = build_artifact_layout(config=config, command_group="pipeline", run_id=run_id)
    ensure_artifact_layout(layout)
    write_run_manifest(
        layout=layout,
        config=config,
        command_group="pipeline",
        run_id=run_id,
        config_path=config_path,
    )

    module = PipelineModule()
    partial_config = PipelineConfig(
        dataset_id="synthetic",
        synthetic_n_cells=84,
        synthetic_n_genes=40,
        synthetic_n_perturbations=6,
        seed=config.runtime.seed,
        transport_latent_dim=24,
        transport_hidden_dim=64,
        transport_epochs=2,
        causal_max_hvgs=18,
        causal_bootstrap_iterations=0,
        stop_after_step="train",
    )
    partial = module.run(
        config=config,
        layout=layout,
        run_id=run_id,
        pipeline_config=partial_config,
        resume=False,
    )
    assert partial.status == "partial"
    assert "train" in partial.completed_steps
    assert "causal" not in partial.completed_steps

    resumed = module.run(
        config=config,
        layout=layout,
        run_id=run_id,
        pipeline_config=replace(partial_config, stop_after_step=None),
        resume=True,
    )
    assert resumed.status == "completed"
    assert resumed.resumed_from_checkpoint is True
    assert set(["ingest", "preprocess", "embed", "train"]).issubset(set(resumed.skipped_steps))


def test_artifact_manifest_lists_required_outputs(tmp_path: Path) -> None:
    _require_anndata()
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")

    run_id = generate_run_id(config=config, command_group="pipeline", provided="phase15-manifest")
    layout = build_artifact_layout(config=config, command_group="pipeline", run_id=run_id)
    ensure_artifact_layout(layout)
    write_run_manifest(
        layout=layout,
        config=config,
        command_group="pipeline",
        run_id=run_id,
        config_path=config_path,
    )

    module = PipelineModule()
    result = module.run(
        config=config,
        layout=layout,
        run_id=run_id,
        pipeline_config=PipelineConfig(
            dataset_id="synthetic",
            synthetic_n_cells=84,
            synthetic_n_genes=40,
            synthetic_n_perturbations=6,
            seed=config.runtime.seed,
            transport_latent_dim=24,
            transport_hidden_dim=64,
            transport_epochs=2,
            causal_max_hvgs=18,
            causal_bootstrap_iterations=0,
        ),
        resume=False,
    )
    assert result.status == "completed"
    assert result.artifact_manifest_path is not None
    manifest_path = result.artifact_manifest_path
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "pipeline_artifact_manifest_v1"
    required = manifest["required_artifacts"]
    required_keys = {
        "ingest_adata",
        "preprocess_adata",
        "embed_adata",
        "transport_checkpoint",
        "transport_fit_summary",
        "baseline_suite_json",
        "causal_adjacency_npz",
        "causal_edge_table_csv",
        "causal_metadata_json",
        "evaluation_report_json",
        "prediction_npz",
        "model_card_json",
        "model_card_markdown",
    }
    assert required_keys.issubset(set(required.keys()))
    for key in required_keys:
        assert Path(required[key]).exists()
    assert set(manifest["step_artifacts"].keys()) == set(PIPELINE_STEPS)
