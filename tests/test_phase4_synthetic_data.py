"""Phase 4 tests for synthetic dataset generation and export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tcpe.anndata_schema import validate_anndata_schema
from tcpe.synthetic_data import export_synthetic_dataset, generate_synthetic_dataset


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def test_synthetic_default_shape_and_ids() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(seed=17)
    adata = bundle.adata
    truth = bundle.ground_truth

    assert adata.shape == (500, 200)
    assert len(truth.perturbation_ids) == 10
    assert truth.causal_adjacency.shape == (200, 200)
    assert truth.direct_effects.shape == (10, 200)
    assert truth.transport_shift_by_perturbation.shape == (10, 200)
    assert sorted(adata.obs["perturbation_id"].unique().tolist()) == truth.perturbation_ids
    assert adata.var["gene_id"].tolist() == truth.gene_ids


def test_synthetic_is_deterministic_for_same_seed() -> None:
    _require_anndata()
    bundle_a = generate_synthetic_dataset(seed=101)
    bundle_b = generate_synthetic_dataset(seed=101)

    np.testing.assert_array_equal(bundle_a.adata.X, bundle_b.adata.X)
    np.testing.assert_allclose(
        bundle_a.ground_truth.causal_adjacency,
        bundle_b.ground_truth.causal_adjacency,
    )
    np.testing.assert_allclose(
        bundle_a.ground_truth.direct_effects,
        bundle_b.ground_truth.direct_effects,
    )
    np.testing.assert_allclose(
        bundle_a.ground_truth.transport_shift_by_perturbation,
        bundle_b.ground_truth.transport_shift_by_perturbation,
    )
    assert bundle_a.ground_truth.perturbation_ids == bundle_b.ground_truth.perturbation_ids
    assert bundle_a.ground_truth.gene_ids == bundle_b.ground_truth.gene_ids


def test_synthetic_schema_strict_passes() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(seed=5)
    report = validate_anndata_schema(bundle.adata, mode="strict")
    assert report.is_valid
    assert report.errors == []


def test_synthetic_export_aligns_truth_ids(tmp_path: Path) -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(seed=77)
    paths = export_synthetic_dataset(bundle=bundle, output_dir=tmp_path)

    assert paths.adata_path.exists()
    assert paths.arrays_path.exists()
    assert paths.metadata_path.exists()

    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    arrays = np.load(paths.arrays_path)

    assert metadata["gene_ids"] == bundle.ground_truth.gene_ids
    assert metadata["perturbation_ids"] == bundle.ground_truth.perturbation_ids
    assert metadata["cell_ids"] == bundle.adata.obs["cell_id"].tolist()
    assert arrays["causal_adjacency"].shape == (200, 200)
    assert arrays["direct_effects"].shape == (10, 200)
    assert arrays["transport_shift_by_perturbation"].shape == (10, 200)
