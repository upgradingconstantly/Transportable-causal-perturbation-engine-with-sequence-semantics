"""Phase 2 tests for CLI groups, config validation, and deterministic run layout."""

from __future__ import annotations

from pathlib import Path

from tcpe.cli import COMMAND_GROUPS, build_parser, main
from tcpe.config import DEFAULT_CONFIG_PATH, load_config
from tcpe.runtime.run_context import build_artifact_layout, generate_run_id


def test_cli_help_contains_all_required_command_groups() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for group in COMMAND_GROUPS:
        assert group in help_text


def test_load_config_applies_environment_overlay() -> None:
    config = load_config(config_path=DEFAULT_CONFIG_PATH, environment="local")
    assert config.environment == "local"
    assert config.resources.max_ram_gb == 12
    assert config.resources.device == "cpu"


def test_invalid_config_fails_fast_in_cli(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "dataset:",
                "  primary_id: \"bad dataset id\"",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["data", "--config", str(invalid_config), "--env", "local", "--dry-run"])
    assert exit_code == 2


def test_run_id_and_artifact_layout_are_deterministic() -> None:
    config = load_config(config_path=DEFAULT_CONFIG_PATH, environment="local")
    run_id_a = generate_run_id(config=config, command_group="train")
    run_id_b = generate_run_id(config=config, command_group="train")
    assert run_id_a == run_id_b

    layout_a = build_artifact_layout(config=config, command_group="train", run_id=run_id_a)
    layout_b = build_artifact_layout(config=config, command_group="train", run_id=run_id_b)
    assert layout_a == layout_b
