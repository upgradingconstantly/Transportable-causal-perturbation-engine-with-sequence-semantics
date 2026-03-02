"""Phase 6 tests for preprocessing and QC module."""

from __future__ import annotations

import numpy as np
import pytest

from tcpe.anndata_schema import validate_anndata_schema
from tcpe.preprocessing import (
    CONFOUNDER_PROXY_COLUMNS_UNS_KEY,
    CONFOUNDER_PROXY_OBSM_KEY,
    NORMALIZED_LAYER_KEY,
    PREPROCESSING_UNS_KEY,
    PreprocessingConfig,
    PreprocessingModule,
)
from tcpe.synthetic_data import generate_synthetic_dataset


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def test_preprocessing_is_deterministic_with_same_seed_config() -> None:
    _require_anndata()
    bundle_a = generate_synthetic_dataset(n_cells=320, n_genes=2100, n_perturbations=10, seed=9)
    bundle_b = generate_synthetic_dataset(n_cells=320, n_genes=2100, n_perturbations=10, seed=9)

    module = PreprocessingModule()
    config = PreprocessingConfig(hvg_target=2000, seed=123)
    result_a = module.run(bundle_a.adata, config=config)
    result_b = module.run(bundle_b.adata, config=config)

    assert result_a.n_hvgs_selected == 2000
    assert result_b.n_hvgs_selected == 2000
    assert result_a.adata.var["gene_id"].tolist() == result_b.adata.var["gene_id"].tolist()
    np.testing.assert_allclose(
        result_a.adata.layers[NORMALIZED_LAYER_KEY],
        result_b.adata.layers[NORMALIZED_LAYER_KEY],
    )
    np.testing.assert_allclose(
        result_a.adata.obsm[CONFOUNDER_PROXY_OBSM_KEY],
        result_b.adata.obsm[CONFOUNDER_PROXY_OBSM_KEY],
    )


def test_hvg_count_matches_target_or_fallback() -> None:
    _require_anndata()
    module = PreprocessingModule()

    bundle_large = generate_synthetic_dataset(
        n_cells=250,
        n_genes=2300,
        n_perturbations=10,
        seed=13,
    )
    result_large = module.run(bundle_large.adata, config=PreprocessingConfig(hvg_target=2000))
    assert result_large.n_hvgs_selected == 2000
    assert result_large.hvg_fallback_used is False

    bundle_small = generate_synthetic_dataset(
        n_cells=250,
        n_genes=400,
        n_perturbations=10,
        seed=13,
    )
    result_small = module.run(bundle_small.adata, config=PreprocessingConfig(hvg_target=2000))
    assert result_small.n_hvgs_selected == 400
    assert result_small.hvg_fallback_used is True


def test_qc_filters_remove_known_bad_cells() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=220, n_genes=2100, n_perturbations=10, seed=21)
    adata = bundle.adata.copy()

    # Create low-quality cells that should fail cell-level QC.
    adata.X[:5, :] = 0
    adata.obs.iloc[:5, adata.obs.columns.get_loc("library_size")] = 0.0

    module = PreprocessingModule()
    result = module.run(
        adata,
        config=PreprocessingConfig(
            hvg_target=2000,
            min_counts_per_cell=10,
            min_genes_per_cell=5,
            min_cells_per_gene=3,
        ),
    )
    assert result.removed_cell_count >= 5
    assert result.n_cells_after_qc <= 215


def test_preprocessed_output_remains_schema_valid_and_records_metadata() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=300, n_genes=2100, n_perturbations=10, seed=33)
    module = PreprocessingModule()
    result = module.run(bundle.adata, config=PreprocessingConfig(hvg_target=2000))

    report = validate_anndata_schema(result.adata, mode="strict")
    assert report.is_valid
    assert PREPROCESSING_UNS_KEY in result.adata.uns
    assert CONFOUNDER_PROXY_COLUMNS_UNS_KEY in result.adata.uns
    assert CONFOUNDER_PROXY_OBSM_KEY in result.adata.obsm
    assert result.adata.obsm[CONFOUNDER_PROXY_OBSM_KEY].shape[0] == result.adata.n_obs
