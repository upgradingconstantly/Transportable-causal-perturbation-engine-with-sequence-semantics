"""Phase 14 tests for evaluator metrics and mandatory model-card outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tcpe.evaluation import (
    MANDATORY_MODEL_CARD_METRICS,
    EvaluationConfig,
    EvaluationError,
    EvaluationModule,
    ModelCardArtifacts,
    ModelCardValidationError,
)
from tcpe.synthetic_data import generate_synthetic_dataset


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def _resolve_expression(adata: Any, layer: str = "normalized_log1p") -> np.ndarray:
    matrix = adata.layers[layer]
    return np.asarray(matrix, dtype=np.float64)


def _build_eval_fixture(
    *,
    seed: int = 33,
) -> tuple[object, EvaluationModule, np.ndarray, np.ndarray]:
    bundle = generate_synthetic_dataset(n_cells=220, n_genes=120, n_perturbations=10, seed=seed)
    adata = bundle.adata
    evaluator = EvaluationModule()
    evaluator.run_baselines(adata, seed=seed, persist_to_anndata=True)

    y_true = _resolve_expression(adata)
    rng = np.random.default_rng(seed)
    y_pred = y_true + rng.normal(loc=0.0, scale=0.03, size=y_true.shape)
    y_var = np.full(y_true.shape, 0.04, dtype=np.float64)
    return adata, evaluator, y_pred, y_var


def test_model_card_files_are_generated_for_evaluation_runs(tmp_path: Path) -> None:
    _require_anndata()
    adata, evaluator, y_pred, y_var = _build_eval_fixture(seed=17)

    report, artifacts = evaluator.evaluate_and_generate_model_card(
        adata,
        prediction_mean=y_pred,
        prediction_variance=y_var,
        config=EvaluationConfig(model_name="ot_sinkhorn", model_version="phase10"),
        output_dir=tmp_path,
        file_prefix="phase14_model_card",
    )
    assert artifacts.json_path.exists()
    assert artifacts.markdown_path.exists()
    assert report.schema_version == "evaluation_report_v1"
    for key in MANDATORY_MODEL_CARD_METRICS:
        assert key in report.metrics
    assert "model_card_artifacts" in adata.uns


def test_missing_metric_sections_trigger_hard_failure() -> None:
    _require_anndata()
    adata, evaluator, y_pred, y_var = _build_eval_fixture(seed=22)
    report = evaluator.evaluate_predictions(
        adata,
        prediction_mean=y_pred,
        prediction_variance=y_var,
    )
    payload = evaluator._build_model_card_payload(report)
    metrics = dict(payload["metrics"])
    metrics.pop("calibration_error")
    payload["metrics"] = metrics

    with pytest.raises(ModelCardValidationError, match="calibration_error"):
        evaluator.validate_model_card_payload(payload, raise_on_error=True)


def test_model_card_json_schema_validation_passes(tmp_path: Path) -> None:
    _require_anndata()
    adata, evaluator, y_pred, y_var = _build_eval_fixture(seed=29)
    _, artifacts = evaluator.evaluate_and_generate_model_card(
        adata,
        prediction_mean=y_pred,
        prediction_variance=y_var,
        output_dir=tmp_path,
        file_prefix="phase14_json_validation",
    )

    payload = evaluator.validate_model_card_json(artifacts.json_path)
    assert payload["schema_version"] == "model_card_v1"
    assert set(MANDATORY_MODEL_CARD_METRICS).issubset(payload["metrics"].keys())


def test_markdown_card_renders_metrics_baselines_and_failure_modes(tmp_path: Path) -> None:
    _require_anndata()
    adata, evaluator, y_pred, y_var = _build_eval_fixture(seed=41)
    _, artifacts = evaluator.evaluate_and_generate_model_card(
        adata,
        prediction_mean=y_pred,
        prediction_variance=y_var,
        output_dir=tmp_path,
        file_prefix="phase14_markdown",
    )
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert "## Metrics" in markdown
    assert "## Baseline Comparisons" in markdown
    assert "## Failure Modes" in markdown
    assert "gene_level_mean" in markdown
    assert "control_mean" in markdown
    assert "linear_regression" in markdown


def test_run_completion_fails_without_model_card_artifacts() -> None:
    evaluator = EvaluationModule()
    with pytest.raises(EvaluationError, match="incomplete"):
        evaluator.assert_run_complete(None)

    fake = ModelCardArtifacts(
        json_path=Path("missing_model_card.json"),
        markdown_path=Path("missing_model_card.md"),
        schema_version="model_card_v1",
    )
    with pytest.raises(EvaluationError, match="Missing model card JSON artifact"):
        evaluator.assert_run_complete(fake)
