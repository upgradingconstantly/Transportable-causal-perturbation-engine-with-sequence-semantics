"""Phase 12 tests for two-stage IV causal module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcpe.causal import (
    REQUIRED_IV_PROXY_COLUMNS,
    CausalConfig,
    CausalInputError,
    CausalModule,
)
from tcpe.synthetic_data import generate_synthetic_dataset


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def _make_confounding_adata(*, n_cells: int = 650, seed: int = 41) -> tuple[object, np.ndarray]:
    _require_anndata()
    import anndata as ad

    rng = np.random.default_rng(seed)
    z = rng.binomial(n=1, p=0.5, size=n_cells).astype(np.float64)
    confounder = rng.normal(loc=0.0, scale=1.0, size=n_cells)

    def noise(scale: float) -> np.ndarray:
        return rng.normal(loc=0.0, scale=scale, size=n_cells)

    gene_0 = (0.9 * z) + (1.8 * confounder) + noise(0.35)
    gene_1 = (1.6 * gene_0) + (1.8 * confounder) + noise(0.35)
    gene_2 = (-1.1 * gene_0) + (1.7 * confounder) + noise(0.35)
    gene_3 = (0.8 * confounder) + noise(0.40)
    gene_4 = noise(0.90)
    gene_5 = noise(0.90)
    normalized = np.column_stack(
        [gene_0, gene_1, gene_2, gene_3, gene_4, gene_5]
    ).astype(np.float32)

    min_shift = float(abs(np.min(normalized))) + 1.0
    shifted = normalized + min_shift
    counts = np.clip(np.round(np.expm1(shifted)), a_min=0.0, a_max=1e6).astype(np.int64)

    library_size = np.maximum(10_000.0 + (1_200.0 * confounder) + noise(100.0), 200.0)
    knockdown = np.clip(0.55 + (0.20 * _zscore(confounder)), 0.0, 1.0)
    perturbation_id = np.where(z > 0.5, "gene0_perturb", "ntc")
    condition = np.where(z > 0.5, "perturbed", "control")
    batch = np.where(confounder > 0, "batch_high", "batch_low")
    protocol = np.where(z > 0.5, "protocol_A", "protocol_B")

    obs = pd.DataFrame(
        {
            "cell_id": [f"conf_cell_{idx:05d}" for idx in range(n_cells)],
            "cell_type": ["k562_confounding"] * n_cells,
            "batch": batch,
            "condition": condition,
            "protocol": protocol,
            "library_size": library_size.astype(np.float64),
            "perturbation_id": perturbation_id,
            "knockdown_efficiency_proxy": knockdown.astype(np.float64),
        },
        index=[f"conf_cell_{idx:05d}" for idx in range(n_cells)],
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"gene_{idx}" for idx in range(6)],
            "gene_symbol": [f"G{idx}" for idx in range(6)],
            "chrom": ["chr1"] * 6,
            "strand": ["+"] * 6,
            "tss": [10_000 + idx for idx in range(6)],
            "highly_variable": [True] * 6,
        },
        index=[f"gene_{idx}" for idx in range(6)],
    )
    adata = ad.AnnData(X=counts, obs=obs, var=var)
    adata.layers["normalized_log1p"] = normalized

    truth = np.zeros((6, 6), dtype=np.float64)
    truth[0, 1] = 1.6
    truth[0, 2] = -1.1
    adata.uns["synthetic_ground_truth_adjacency"] = truth
    return adata, truth


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def test_fit_fails_when_required_iv_proxy_is_missing() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=180, n_genes=70, n_perturbations=10, seed=8)
    adata = bundle.adata.copy()
    adata.obs = adata.obs.drop(columns=["protocol"])

    module = CausalModule()
    with pytest.raises(CausalInputError, match="protocol"):
        module.fit(adata, config=CausalConfig(max_hvgs=30, bootstrap_iterations=0))


def test_iv_causal_graph_has_uncertainty_for_all_edges_and_cycle_permitting_metadata() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=260, n_genes=85, n_perturbations=10, seed=12)

    module = CausalModule()
    result = module.fit(
        bundle.adata,
        config=CausalConfig(max_hvgs=40, bootstrap_iterations=10, bootstrap_seed=9),
    )

    graph = result.graph
    n_genes = len(graph.gene_ids)
    assert graph.adjacency.shape == (n_genes, n_genes)
    assert graph.ci_lower.shape == (n_genes, n_genes)
    assert graph.ci_upper.shape == (n_genes, n_genes)
    assert graph.standard_error.shape == (n_genes, n_genes)
    assert np.allclose(np.diag(graph.adjacency), 0.0)
    assert np.all(graph.ci_upper >= graph.ci_lower)
    assert np.isfinite(graph.standard_error).all()
    assert graph.metadata["cyclic_allowed"] is True
    assert graph.metadata["dag_enforced"] is False
    assert graph.metadata["required_proxy_columns"] == list(REQUIRED_IV_PROXY_COLUMNS)


def test_synthetic_ground_truth_recovery_exceeds_minimum_threshold() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=300, n_genes=90, n_perturbations=10, seed=17)
    module = CausalModule()
    result = module.fit(
        bundle.adata,
        config=CausalConfig(max_hvgs=45, bootstrap_iterations=0),
    )

    selected = result.selected_gene_indices
    truth = bundle.ground_truth.causal_adjacency[np.ix_(selected, selected)]
    mask = ~np.eye(truth.shape[0], dtype=bool)
    true_flat = np.abs(truth[mask])
    pred_flat = np.abs(result.graph.adjacency[mask])

    true_edge_mask = true_flat > 0
    edge_count = int(np.sum(true_edge_mask))
    assert edge_count > 0
    top_idx = np.argsort(pred_flat)[::-1][:edge_count]
    overlap = int(np.sum(true_edge_mask[top_idx]))

    random_expected = float(edge_count * (edge_count / true_edge_mask.size))
    assert float(overlap) >= (1.20 * random_expected)


def test_known_confounding_case_iv_reduces_bias_vs_naive() -> None:
    adata, truth = _make_confounding_adata()
    module = CausalModule()
    result = module.fit(
        adata,
        config=CausalConfig(
            max_hvgs=6,
            bootstrap_iterations=0,
            stage1_ridge_alpha=1e-2,
            stage2_ridge_alpha=1e-2,
        ),
    )

    assert result.naive_adjacency is not None
    iv = result.graph.adjacency
    naive = result.naive_adjacency

    iv_error = abs(float(iv[0, 1] - truth[0, 1])) + abs(float(iv[0, 2] - truth[0, 2]))
    naive_error = abs(float(naive[0, 1] - truth[0, 1])) + abs(float(naive[0, 2] - truth[0, 2]))
    assert iv_error < naive_error


def test_export_graph_artifacts_writes_npz_csv_and_metadata_json(tmp_path: Path) -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=180, n_genes=70, n_perturbations=10, seed=19)
    module = CausalModule()
    result = module.fit(
        bundle.adata,
        config=CausalConfig(max_hvgs=32, bootstrap_iterations=5),
    )

    paths = module.export_graph_artifacts(
        result.graph,
        output_dir=tmp_path,
        file_prefix="phase12_causal",
    )
    assert paths.adjacency_npz_path.exists()
    assert paths.edge_table_csv_path.exists()
    assert paths.metadata_json_path.exists()

    metadata = json.loads(paths.metadata_json_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == result.graph.schema_version
    assert metadata["metadata"]["required_proxy_columns"] == list(REQUIRED_IV_PROXY_COLUMNS)
