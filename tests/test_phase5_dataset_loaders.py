"""Phase 5 tests for dataset registry and ingestion loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcpe.anndata_schema import validate_anndata_schema
from tcpe.dataset_loaders import LOCAL_REPLOGLE_MAX_CELLS
from tcpe.ingestion import IngestionModule


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def _write_mock_tabular_csv(
    *,
    path: Path,
    n_cells: int,
    n_genes: int,
    include_optional_obs: bool = False,
) -> Path:
    rng = np.random.default_rng(123)
    perturbations = np.where(np.arange(n_cells) % 8 == 0, "ntc", "tp53_guidex")
    payload: dict[str, list[object]] = {
        "cell_id": [f"cell_{idx:05d}" for idx in range(n_cells)],
        "perturbation_id": perturbations.tolist(),
    }
    if include_optional_obs:
        payload["batch"] = [f"batch_{idx % 3}" for idx in range(n_cells)]
        payload["protocol"] = ["10x_v3"] * n_cells
        payload["condition"] = ["control" if val == "ntc" else "perturbed" for val in perturbations]
    for gene_idx in range(n_genes):
        payload[f"gene_{gene_idx:03d}"] = rng.poisson(lam=2.5, size=n_cells).astype(int).tolist()

    frame = pd.DataFrame(payload)
    frame.to_csv(path, index=False)
    return path


def test_registry_contains_required_phase5_loaders() -> None:
    module = IngestionModule()
    assert module.status() == "phase8_embedding_ready"
    assert module.available_datasets() == ["adamson", "lincs_stub", "replogle_sample"]


def test_lincs_stub_warns_and_raises_not_implemented(tmp_path: Path) -> None:
    module = IngestionModule()
    with pytest.warns(UserWarning, match="LINCS adapter is a Phase 5 stub"):
        with pytest.raises(NotImplementedError, match="LINCS adapter stub"):
            module.load_lincs_stub(cache_dir=str(tmp_path))


def test_adamson_ingest_pipeline_schema_valid(tmp_path: Path) -> None:
    _require_anndata()
    source_csv = _write_mock_tabular_csv(
        path=tmp_path / "adamson_mock.csv",
        n_cells=320,
        n_genes=24,
        include_optional_obs=False,
    )
    module = IngestionModule()
    result = module.load_adamson(cache_dir=str(tmp_path / "cache"), source_uri=str(source_csv))

    assert result.from_cache is False
    report = validate_anndata_schema(result.adata, mode="strict")
    assert report.is_valid
    assert result.adata.n_obs == 320
    assert result.adata.n_vars == 24


def test_replogle_sample_loader_enforces_local_5000_cap(tmp_path: Path) -> None:
    _require_anndata()
    source_csv = _write_mock_tabular_csv(
        path=tmp_path / "replogle_mock.csv",
        n_cells=6200,
        n_genes=15,
        include_optional_obs=True,
    )
    module = IngestionModule()
    with pytest.warns(UserWarning, match="exceeds local cap"):
        result = module.load_replogle_sample(
            cache_dir=str(tmp_path / "cache"),
            source_uri=str(source_csv),
            max_cells=8000,
        )
    assert result.adata.n_obs == LOCAL_REPLOGLE_MAX_CELLS


def test_loader_rerun_uses_cache_and_is_idempotent(tmp_path: Path) -> None:
    _require_anndata()
    source_csv = _write_mock_tabular_csv(
        path=tmp_path / "adamson_cache.csv",
        n_cells=240,
        n_genes=12,
        include_optional_obs=True,
    )
    module = IngestionModule()
    first = module.load_adamson(cache_dir=str(tmp_path / "cache"), source_uri=str(source_csv))
    second = module.load_adamson(cache_dir=str(tmp_path / "cache"), source_uri=str(source_csv))

    assert first.from_cache is False
    assert second.from_cache is True
    assert first.checksum_sha256 == second.checksum_sha256
    assert first.processed_path == second.processed_path
    assert first.adata.shape == second.adata.shape
