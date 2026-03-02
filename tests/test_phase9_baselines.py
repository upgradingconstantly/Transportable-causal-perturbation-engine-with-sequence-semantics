"""Phase 9 tests for mandatory baseline suite and training gate hooks."""

from __future__ import annotations

import pytest

from tcpe.baselines import (
    REQUIRED_BASELINE_NAMES,
    BaselineSuiteResult,
    run_baseline_suite,
)
from tcpe.evaluation import EvaluationModule
from tcpe.transport import TransportModule


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def test_random_edge_baseline_preserves_degree_distribution() -> None:
    _require_anndata()
    from tcpe.synthetic_data import generate_synthetic_dataset

    bundle = generate_synthetic_dataset(n_cells=180, n_genes=120, n_perturbations=10, seed=12)
    suite = run_baseline_suite(
        adata=bundle.adata,
        reference_adjacency=bundle.ground_truth.causal_adjacency,
        seed=99,
    )
    metrics = next(
        item.metrics for item in suite.baselines if item.baseline_name == "random_edge_grn"
    )
    assert int(metrics["edge_count_reference"]) == int(metrics["edge_count_random"])
    assert float(metrics["out_degree_l1_diff"]) == 0.0
    assert float(metrics["in_degree_l1_diff"]) == 0.0


def test_baseline_suite_contains_all_required_baselines_and_uniform_schema() -> None:
    _require_anndata()
    from tcpe.synthetic_data import generate_synthetic_dataset

    bundle = generate_synthetic_dataset(n_cells=200, n_genes=150, n_perturbations=10, seed=44)
    suite = run_baseline_suite(adata=bundle.adata, seed=44)
    names = set(suite.baseline_names())
    assert names == set(REQUIRED_BASELINE_NAMES)

    for result in suite.baselines:
        assert isinstance(result.baseline_name, str)
        assert isinstance(result.task, str)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.metadata, dict)


def test_evaluation_module_persists_baselines_for_later_evaluator_use() -> None:
    _require_anndata()
    from tcpe.baselines import BASELINE_RANDOM_GRN_UNS_KEY, BASELINE_RESULTS_UNS_KEY
    from tcpe.synthetic_data import generate_synthetic_dataset

    bundle = generate_synthetic_dataset(n_cells=160, n_genes=120, n_perturbations=8, seed=8)
    evaluator = EvaluationModule()
    suite = evaluator.run_baselines(adata=bundle.adata, seed=8, persist_to_anndata=True)

    assert evaluator.status() == "phase9_baselines_ready"
    assert isinstance(suite, BaselineSuiteResult)
    assert BASELINE_RESULTS_UNS_KEY in bundle.adata.uns
    assert BASELINE_RANDOM_GRN_UNS_KEY in bundle.adata.uns
    names = {
        item["baseline_name"]
        for item in bundle.adata.uns[BASELINE_RESULTS_UNS_KEY]["baselines"]
    }
    assert names == set(REQUIRED_BASELINE_NAMES)


def test_transport_training_gate_blocks_when_baselines_missing() -> None:
    module = TransportModule()
    assert module.status() == "phase11_transport_family_ready"
    incomplete = {"baselines": [{"baseline_name": "gene_level_mean"}]}
    assert module.can_train(incomplete) is False
    with pytest.raises(RuntimeError, match="Missing baselines"):
        module.assert_baselines_ready(incomplete)


def test_transport_training_gate_allows_when_baselines_complete() -> None:
    module = TransportModule()
    complete = {
        "baselines": [{"baseline_name": name} for name in REQUIRED_BASELINE_NAMES],
    }
    assert module.can_train(complete) is True
    module.assert_baselines_ready(complete)


def test_linear_baseline_beats_trivial_control_on_synthetic_mae() -> None:
    _require_anndata()
    from tcpe.synthetic_data import generate_synthetic_dataset

    bundle = generate_synthetic_dataset(n_cells=260, n_genes=140, n_perturbations=10, seed=55)
    suite = run_baseline_suite(adata=bundle.adata, seed=55)
    metrics_by_name = {item.baseline_name: item.metrics for item in suite.baselines}
    assert metrics_by_name["linear_regression"]["mae"] <= metrics_by_name["control_mean"]["mae"]
