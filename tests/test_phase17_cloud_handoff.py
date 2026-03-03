"""Phase 17 tests for cloud handoff planning and resume seeding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from tcpe.cli import main
from tcpe.cloud_handoff import CloudHandoffModule, artifact_sync_main
from tcpe.config import DEFAULT_CONFIG_PATH, load_config
from tcpe.runtime.run_context import build_artifact_layout, generate_run_id


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

    config_path = tmp_path / "phase17_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_cloud_handoff_dry_run_plans_confirmed_oracle_and_kaggle_split(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")
    run_id = generate_run_id(config=config, command_group="cloud", provided="phase17-dry-run")
    layout = build_artifact_layout(config=config, command_group="cloud", run_id=run_id)

    module = CloudHandoffModule()
    result = module.run(
        config=config,
        layout=layout,
        run_id=run_id,
        simulate_interruption_after="train",
        dry_run=True,
    )

    assert module.status() == "phase17_cloud_handoff_ready"
    assert result.dry_run is True
    assert result.output_dir is None

    presets = {item.preset_name: item for item in result.resource_presets}
    oracle = presets["oracle_vm_cpu_jobs"]
    kaggle = presets["kaggle_t4_x2_training"]
    assert oracle.cpu_cores == 4
    assert oracle.ram_gb == 60
    assert kaggle.gpu_type == "T4"
    assert kaggle.gpu_count == 2
    assert kaggle.working_dir_limit_gb == 20

    assert result.spot_resume_simulation.resume_from_step == "causal"
    assert any("pipeline --env kaggle" in item.command for item in result.run_presets)
    assert all(item["passed"] for item in result.artifact_smoke_tests)


def test_cloud_handoff_non_dry_run_seeds_resume_checkpoint(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")
    run_id = generate_run_id(config=config, command_group="cloud", provided="phase17-seed")
    layout = build_artifact_layout(config=config, command_group="cloud", run_id=run_id)

    preprocessed_h5ad = tmp_path / "external_preprocessed.h5ad"
    preprocessed_h5ad.write_text("placeholder-h5ad", encoding="utf-8")

    result = CloudHandoffModule().run(
        config=config,
        layout=layout,
        run_id=run_id,
        preprocessed_h5ad_path=preprocessed_h5ad,
        dataset_label="replogle_k562_gwps",
        dry_run=False,
    )

    assert result.report_json_path is not None
    assert result.report_json_path.exists()
    assert result.runbook_markdown_path is not None
    assert result.runbook_markdown_path.exists()
    assert "oracle_vm_environment" in result.environment_spec_paths
    assert "kaggle_environment" in result.environment_spec_paths
    assert result.preprocessed_seed is not None

    checkpoint_path = Path(result.preprocessed_seed["checkpoint_path"])
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["completed_steps"] == ["ingest", "preprocess"]
    assert checkpoint_payload["run_id"] == run_id

    ingest_path = checkpoint_payload["step_artifacts"]["ingest"]["adata_path"]
    preprocess_path = checkpoint_payload["step_artifacts"]["preprocess"]["adata_path"]
    assert ingest_path == preprocess_path
    assert Path(ingest_path).exists()


def test_cloud_cli_dry_run_outputs_phase17_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_temp_config(tmp_path)

    exit_code = main(
        [
            "cloud",
            "--config",
            str(config_path),
            "--env",
            "local",
            "--run-id",
            "phase17-cli",
            "--simulate-interruption-after",
            "train",
            "--dry-run",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    result = payload["cloud_handoff_result"]
    assert result["dry_run"] is True
    assert result["spot_resume_simulation"]["resume_from_step"] == "causal"
    assert payload["message"] == "Phase 17 cloud handoff plan generated."


def test_artifact_sync_main_plans_provider_specific_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = artifact_sync_main(
        [
            "--provider",
            "github",
            "--direction",
            "push",
            "--local-path",
            "artifacts/sample.json",
            "--remote-ref",
            "tcpe-phase17",
            "--dry-run",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    plan = payload["plan"]
    assert plan["provider"] == "github"
    assert plan["direction"] == "push"
    assert plan["command"][0] == "gh"
