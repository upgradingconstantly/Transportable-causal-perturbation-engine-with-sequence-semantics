"""Phase 3 tests for canonical AnnData schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcpe.anndata_schema import (
    CELL_STATE_EMBEDDING_OBSM_KEY,
    NORMALIZED_LAYER_KEY,
    PERTURBATIONS_UNS_KEY,
    SCHEMA_VERSION,
    SCHEMA_VERSION_UNS_KEY,
    SEQUENCE_EMBEDDING_OBSM_KEY,
    AnnDataSchemaError,
    validate_anndata_schema,
)

anndata = pytest.importorskip("anndata")
AnnData = anndata.AnnData


def _make_valid_adata() -> AnnData:
    x = np.array([[2, 0, 1], [1, 1, 0], [0, 3, 1]], dtype=np.int64)
    obs = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3"],
            "cell_type": ["k562", "k562", "k562"],
            "batch": ["b1", "b1", "b2"],
            "condition": ["control", "perturbed", "perturbed"],
            "protocol": ["10x_v3", "10x_v3", "10x_v3"],
            "library_size": x.sum(axis=1).astype(float),
            "perturbation_id": ["p0", "p1", "p1"],
            "knockdown_efficiency_proxy": [0.0, 0.7, 0.6],
        },
        index=["c1", "c2", "c3"],
    )
    var = pd.DataFrame(
        {
            "gene_id": ["g1", "g2", "g3"],
            "gene_symbol": ["G1", "G2", "G3"],
            "chrom": ["chr1", "chr1", "chr2"],
            "strand": ["+", "-", "+"],
            "tss": [100, 200, 300],
        },
        index=["g1", "g2", "g3"],
    )
    adata = AnnData(X=x, obs=obs, var=var)
    adata.layers[NORMALIZED_LAYER_KEY] = np.log1p(
        (x / np.maximum(x.sum(axis=1, keepdims=True), 1)) * 1e4
    )
    adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY] = np.ones((adata.n_obs, 4), dtype=np.float32)
    adata.obsm[CELL_STATE_EMBEDDING_OBSM_KEY] = np.ones((adata.n_obs, 4), dtype=np.float32)
    adata.uns[PERTURBATIONS_UNS_KEY] = pd.DataFrame(
        [
            {
                "perturbation_id": "p0",
                "target_gene": "NTC",
                "chrom": "chr0",
                "start": -1,
                "end": -1,
                "strand": ".",
                "modality": "control",
                "dose": np.nan,
                "gRNA_id": "ntc_1",
            },
            {
                "perturbation_id": "p1",
                "target_gene": "TP53",
                "chrom": "chr17",
                "start": 7661779,
                "end": 7661799,
                "strand": "+",
                "modality": "crispri",
                "dose": 1.0,
                "gRNA_id": "tp53_1",
            },
        ]
    )
    adata.uns[SCHEMA_VERSION_UNS_KEY] = SCHEMA_VERSION
    return adata


def test_strict_validation_passes_for_valid_anndata() -> None:
    adata = _make_valid_adata()
    report = validate_anndata_schema(adata=adata, mode="strict")
    assert report.is_valid
    assert report.errors == []


def test_strict_validation_fails_with_precise_field_errors() -> None:
    adata = _make_valid_adata()
    del adata.obs["cell_type"]
    del adata.var["chrom"]
    adata.obsm.pop(SEQUENCE_EMBEDDING_OBSM_KEY)

    try:
        validate_anndata_schema(adata=adata, mode="strict")
    except AnnDataSchemaError as exc:
        message = str(exc)
        assert "Missing required `.obs` column: `cell_type`." in message
        assert "Missing required `.var` column: `chrom`." in message
        expected = (
            "Missing required embedding slot "
            f"`.obsm['{SEQUENCE_EMBEDDING_OBSM_KEY}']`."
        )
        assert expected in message
    else:
        raise AssertionError("Expected schema validation to fail in strict mode.")


def test_repair_mode_fills_safe_defaults_and_enables_strict_pass() -> None:
    adata = _make_valid_adata()
    del adata.obs["condition"]
    del adata.obs["protocol"]
    del adata.var["strand"]
    adata.layers.pop(NORMALIZED_LAYER_KEY)
    adata.obsm.pop(CELL_STATE_EMBEDDING_OBSM_KEY)
    adata.uns.pop(PERTURBATIONS_UNS_KEY)
    adata.uns.pop(SCHEMA_VERSION_UNS_KEY)

    repair_report = validate_anndata_schema(adata=adata, mode="repair")
    assert repair_report.is_valid
    assert any(item == "obs.condition" for item in repair_report.repaired_fields)
    assert any(item == "obs.protocol" for item in repair_report.repaired_fields)
    assert any(item == "var.strand" for item in repair_report.repaired_fields)
    assert any(item == f"layers.{NORMALIZED_LAYER_KEY}" for item in repair_report.repaired_fields)
    assert any(
        item == f"obsm.{CELL_STATE_EMBEDDING_OBSM_KEY}" for item in repair_report.repaired_fields
    )
    assert any(item == f"uns.{PERTURBATIONS_UNS_KEY}" for item in repair_report.repaired_fields)
    assert any(item == f"uns.{SCHEMA_VERSION_UNS_KEY}" for item in repair_report.repaired_fields)

    strict_report = validate_anndata_schema(adata=adata, mode="strict")
    assert strict_report.is_valid
