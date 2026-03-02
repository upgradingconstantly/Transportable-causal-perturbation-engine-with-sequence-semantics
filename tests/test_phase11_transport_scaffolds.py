"""Phase 11 tests for flow/bridge transport scaffolds and variant dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tcpe.cli import main
from tcpe.config import DEFAULT_CONFIG_PATH
from tcpe.transport import (
    CLOUD_RECOMMENDED_TAG,
    EXPERIMENTAL_TAG,
    SchrodingerBridgeLightningModule,
    SchrodingerBridgeTransportStrategy,
    TransportModule,
    UnsupportedTransportPathError,
)


def _write_variant_config(tmp_path: Path, *, variant: str) -> Path:
    base_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    updated = base_text.replace("transport_variant: ot", f"transport_variant: {variant}")
    path = tmp_path / f"config_{variant}.yaml"
    path.write_text(updated, encoding="utf-8")
    return path


def test_flow_and_bridge_scaffolds_instantiate_with_expected_status_tags() -> None:
    module = TransportModule()
    assert module.status() == "phase11_transport_family_ready"

    flow_strategy = module.resolve_strategy(variant="flow")
    bridge_strategy = module.resolve_strategy(variant="bridge")

    flow_info = module.strategy_info("flow")
    bridge_info = module.strategy_info("bridge")

    assert flow_info.status_tags == (EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG)
    assert bridge_info.status_tags == (EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG)
    assert flow_strategy.variant_name == flow_info.strategy_name
    assert bridge_strategy.variant_name == bridge_info.strategy_name


def test_schrodinger_bridge_scaffold_has_manual_optimization_enabled() -> None:
    strategy = SchrodingerBridgeTransportStrategy()
    module = strategy.lightning_module
    assert isinstance(module, SchrodingerBridgeLightningModule)
    assert module.manual_optimization is True
    assert module.automatic_optimization is False


def test_scaffold_unsupported_paths_fail_gracefully_with_explicit_message() -> None:
    module = TransportModule()
    flow_strategy = module.resolve_strategy(variant="flow")
    bridge_strategy = module.resolve_strategy(variant="bridge")

    x = np.zeros((2, 3), dtype=np.float32)
    seq = np.zeros((2, 4), dtype=np.float32)
    cell = np.zeros((2, 5), dtype=np.float32)

    with pytest.raises(
        UnsupportedTransportPathError,
        match="Status tags: experimental, cloud-recommended",
    ):
        flow_strategy.fit(
            source_expression=x,
            target_expression=x,
            sequence_embedding=seq,
            cell_state_embedding=cell,
        )

    with pytest.raises(
        UnsupportedTransportPathError,
        match="Status tags: experimental, cloud-recommended",
    ):
        bridge_strategy.predict_distribution(
            source_expression=x,
            sequence_embedding=seq,
            cell_state_embedding=cell,
        )


def test_cli_train_dispatches_transport_variant_from_config_without_code_changes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_variant_config(tmp_path, variant="flow")
    exit_code = main(["train", "--config", str(config_path), "--env", "local", "--dry-run"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command_group"] == "train"
    dispatch = payload["transport_dispatch"]
    assert dispatch["variant"] == "flow"
    assert dispatch["status_tags"] == [EXPERIMENTAL_TAG, CLOUD_RECOMMENDED_TAG]
