"""Phase 10 tests for OT transport strategy and conditioning behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tcpe.baselines import REQUIRED_BASELINE_NAMES
from tcpe.synthetic_data import generate_synthetic_dataset
from tcpe.transport import (
    AdditiveConditioning,
    OTTransportConfig,
    OTTransportStrategy,
    TransportModule,
    TransportTrainingData,
)


def _require_anndata() -> None:
    pytest.importorskip("anndata")


@dataclass
class CapturingRun:
    """Minimal in-memory W&B-like run used for log synchronization checks."""

    records: list[dict[str, float]] = field(default_factory=list)
    finished: bool = False

    def log(self, data: dict[str, Any]) -> None:
        self.records.append({key: float(value) for key, value in data.items()})

    def finish(self) -> None:
        self.finished = True


def _build_strategy_and_data(seed: int = 7) -> tuple[OTTransportStrategy, TransportTrainingData]:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=180, n_genes=70, n_perturbations=10, seed=seed)
    training_data = TransportTrainingData.from_anndata(
        bundle.adata,
        expression_layer="normalized_log1p",
        source_policy="control_mean",
    )
    config = OTTransportConfig(
        input_dim=training_data.source_expression.shape[1],
        sequence_embedding_dim=training_data.sequence_embedding.shape[1],
        cell_state_embedding_dim=training_data.cell_state_embedding.shape[1],
        latent_dim=32,
        hidden_dim=96,
        learning_rate=2e-3,
        n_epochs=14,
        batch_size=48,
        sinkhorn_weight=0.05,
        sinkhorn_epsilon=0.2,
        sinkhorn_n_iters=25,
    )
    strategy = OTTransportStrategy(config=config, seed=seed)
    return strategy, training_data


def test_conditioning_heads_use_shared_latent_dim_and_addition_only() -> None:
    module = AdditiveConditioning(
        sequence_embedding_dim=7,
        cell_state_embedding_dim=5,
        latent_dim=11,
        combine_operator="add",
    )
    assert module.sequence_projection.out_features == 11
    assert module.cell_state_projection.out_features == 11

    sequence = torch.randn((4, 7), dtype=torch.float32)
    cell_state = torch.randn((4, 5), dtype=torch.float32)
    combined = module(sequence, cell_state)
    expected = module.project_sequence(sequence) + module.project_cell_state(cell_state)
    torch.testing.assert_close(combined, expected)

    with pytest.raises(ValueError, match="Raw concatenation is not allowed"):
        AdditiveConditioning(
            sequence_embedding_dim=7,
            cell_state_embedding_dim=5,
            latent_dim=11,
            combine_operator="concat",  # type: ignore[arg-type]
        )


def test_ot_training_converges_and_beats_control_mean_baseline() -> None:
    strategy, training_data = _build_strategy_and_data(seed=13)
    fit_result = strategy.fit(
        source_expression=training_data.source_expression,
        target_expression=training_data.target_expression,
        sequence_embedding=training_data.sequence_embedding,
        cell_state_embedding=training_data.cell_state_embedding,
    )

    start_loss = float(np.mean(fit_result.loss_history[:3]))
    end_loss = float(np.mean(fit_result.loss_history[-3:]))
    assert end_loss < start_loss

    prediction = strategy.predict_distribution(
        source_expression=training_data.source_expression,
        sequence_embedding=training_data.sequence_embedding,
        cell_state_embedding=training_data.cell_state_embedding,
    )
    control_mae = float(
        np.mean(np.abs(training_data.target_expression - training_data.source_expression))
    )
    model_mae = float(np.mean(np.abs(training_data.target_expression - prediction.mean)))
    assert model_mae < control_mae


def test_checkpoint_save_and_load_are_lossless(tmp_path: Path) -> None:
    strategy, training_data = _build_strategy_and_data(seed=22)
    strategy.fit(
        source_expression=training_data.source_expression,
        target_expression=training_data.target_expression,
        sequence_embedding=training_data.sequence_embedding,
        cell_state_embedding=training_data.cell_state_embedding,
    )

    subset = slice(0, 24)
    pre = strategy.predict_distribution(
        source_expression=training_data.source_expression[subset],
        sequence_embedding=training_data.sequence_embedding[subset],
        cell_state_embedding=training_data.cell_state_embedding[subset],
    )
    checkpoint_path = strategy.save(tmp_path / "ot_transport.ckpt")
    loaded = OTTransportStrategy.load(checkpoint_path)
    post = loaded.predict_distribution(
        source_expression=training_data.source_expression[subset],
        sequence_embedding=training_data.sequence_embedding[subset],
        cell_state_embedding=training_data.cell_state_embedding[subset],
    )

    np.testing.assert_allclose(pre.mean, post.mean, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(pre.variance, post.variance, atol=1e-6, rtol=1e-6)


def test_local_and_wandb_logs_are_synchronized() -> None:
    strategy, training_data = _build_strategy_and_data(seed=31)
    run = CapturingRun()
    fit_result = strategy.fit(
        source_expression=training_data.source_expression,
        target_expression=training_data.target_expression,
        sequence_embedding=training_data.sequence_embedding,
        cell_state_embedding=training_data.cell_state_embedding,
        wandb_run=run,
    )

    assert len(run.records) == fit_result.n_epochs
    assert len(strategy.local_log_history) == fit_result.n_epochs
    for local_row, logged_row in zip(strategy.local_log_history, run.records, strict=True):
        assert local_row == logged_row
    assert run.finished is False


def test_transport_module_blocks_training_until_baselines_are_ready() -> None:
    strategy, training_data = _build_strategy_and_data(seed=45)
    config = strategy.config
    module = TransportModule()
    assert module.status() == "phase11_transport_family_ready"

    incomplete = {"baselines": [{"baseline_name": "gene_level_mean"}]}
    with pytest.raises(RuntimeError, match="Missing baselines"):
        module.train_ot(
            baseline_suite=incomplete,
            training_data=training_data,
            config=config,
        )

    complete = {
        "baselines": [{"baseline_name": name} for name in REQUIRED_BASELINE_NAMES],
    }
    trained_strategy, fit_result = module.train_ot(
        baseline_suite=complete,
        training_data=training_data,
        config=config,
    )
    assert trained_strategy.variant_name == "ot_sinkhorn"
    assert fit_result.checkpoint_schema_version == "transport_ot_v1"
