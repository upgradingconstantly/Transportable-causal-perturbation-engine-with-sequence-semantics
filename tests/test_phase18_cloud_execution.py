"""Phase 18 tests for cloud execution sequencing and launcher generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from tcpe.cli import main
from tcpe.cloud_execution import Phase18ExecutionConfig, Phase18ExecutionModule
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

    config_path = tmp_path / "phase18_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_phase18_dry_run_preserves_exact_execution_order(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")
    run_id = generate_run_id(config=config, command_group="cloud-exec", provided="phase18-dry")
    layout = build_artifact_layout(config=config, command_group="cloud-exec", run_id=run_id)

    result = Phase18ExecutionModule().run(
        config=config,
        layout=layout,
        run_id=run_id,
        execution_config=Phase18ExecutionConfig(),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.output_dir is None
    assert Phase18ExecutionModule().status() == "phase18_cloud_execution_ready"

    expected = [
        "01_oracle_stage_raw_upload",
        "02_kaggle_preprocess_embed",
        "03_kaggle_k562_essential_tuning",
        "04_kaggle_k562_gwps_transport",
        "05_kaggle_rpe1_transport_eval",
        "06_oracle_causal_iv_genomewide",
        "07_context_shift_and_model_card",
        "08_optional_schrodinger_bridge",
    ]
    assert [item.job_id for item in result.job_specs] == expected
    assert result.job_specs[0].executor == "oracle_vm"
    assert result.job_specs[1].executor == "kaggle"
    assert result.job_specs[6].executor == "oracle_vm"
    assert result.job_specs[7].optional is True
    assert result.job_specs[7].enabled is False


def test_phase18_non_dry_run_writes_launchers_and_runbook(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)
    config = load_config(config_path=config_path, environment="local")
    run_id = generate_run_id(config=config, command_group="cloud-exec", provided="phase18-files")
    layout = build_artifact_layout(config=config, command_group="cloud-exec", run_id=run_id)

    result = Phase18ExecutionModule().run(
        config=config,
        layout=layout,
        run_id=run_id,
        execution_config=Phase18ExecutionConfig(
            context_shift_executor="kaggle",
            include_optional_bridge=True,
        ),
        dry_run=False,
    )

    assert result.plan_json_path is not None
    assert result.plan_json_path.exists()
    assert result.runbook_markdown_path is not None
    assert result.runbook_markdown_path.exists()
    assert result.job_manifest_path is not None
    assert result.job_manifest_path.exists()
    assert result.sequence_checkpoint_path is not None
    assert result.sequence_checkpoint_path.exists()
    assert len(result.launcher_paths) == 8
    assert result.tmux_launcher_path is not None
    assert result.tmux_launcher_path.exists()

    first_launcher = result.launcher_paths["01_oracle_stage_raw_upload"]
    content = first_launcher.read_text(encoding="utf-8")
    assert "OCIObjectStorageClient" in content
    assert 'export WANDB_MODE="online"' in content
    assert "timeout 180m python - <<'PY'" in content

    tmux_text = result.tmux_launcher_path.read_text(encoding="utf-8")
    assert "tmux new-session -d -s tcpe18-01-raw" in tmux_text
    assert "tcpe18-06-causal" in tmux_text

    runbook = result.runbook_markdown_path.read_text(encoding="utf-8")
    assert "Phase 18 Cloud Execution Runbook" in runbook
    assert "W&B mode for cloud jobs: `online`" in runbook
    assert "08_optional_schrodinger_bridge" in runbook
    assert "disabled by default" not in runbook


def test_cloud_exec_cli_dry_run_outputs_phase18_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_temp_config(tmp_path)

    exit_code = main(
        [
            "cloud-exec",
            "--config",
            str(config_path),
            "--env",
            "local",
            "--run-id",
            "phase18-cli",
            "--context-shift-executor",
            "kaggle",
            "--dry-run",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    result = payload["cloud_execution_result"]
    assert result["dry_run"] is True
    assert result["job_specs"][6]["executor"] == "kaggle"
    assert payload["message"] == "Phase 18 cloud execution scripts generated."


def test_kaggle_overlay_sets_wandb_online() -> None:
    config = load_config(config_path=DEFAULT_CONFIG_PATH, environment="kaggle")
    assert config.runtime.wandb_mode == "online"
