"""Phase 13 tests for context-shift split generation and leakage auditing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tcpe.context_shift import (
    ALL_SHIFT_TYPES,
    ContextShiftSplitModule,
    ShiftSplitConfig,
    ShiftSplitLeakageError,
    ShiftSplitManifest,
)
from tcpe.ingestion import IngestionModule
from tcpe.synthetic_data import generate_synthetic_dataset


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def _write_real_like_csv(path: Path, *, n_cells: int = 180, n_genes: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    perturbations = np.array(["ntc", "tp53_g1", "stat3_g1", "myc_g1"], dtype=object)
    cell_types = np.array(["K562", "RPE1"], dtype=object)
    protocols = np.array(["10x_v2", "10x_v3"], dtype=object)
    batch_ids = np.array(["batch_0", "batch_1", "batch_2"], dtype=object)

    indices = np.arange(n_cells)
    perturbation_id = perturbations[indices % perturbations.size]
    protocol = protocols[(indices // 2) % protocols.size]
    cell_type = cell_types[(indices // 3) % cell_types.size]
    batch = batch_ids[(indices // 5) % batch_ids.size]
    condition = np.where(perturbation_id == "ntc", "control", "perturbed")
    knockdown = np.where(perturbation_id == "ntc", 0.0, rng.uniform(0.2, 0.95, size=n_cells))

    payload: dict[str, list[object]] = {
        "cell_id": [f"real_cell_{idx:05d}" for idx in range(n_cells)],
        "perturbation_id": perturbation_id.tolist(),
        "cell_type": cell_type.tolist(),
        "protocol": protocol.tolist(),
        "batch": batch.tolist(),
        "condition": condition.tolist(),
        "knockdown_efficiency_proxy": knockdown.astype(float).tolist(),
    }
    for gene_idx in range(n_genes):
        payload[f"gene_{gene_idx:03d}"] = rng.poisson(lam=3.0, size=n_cells).astype(int).tolist()

    frame = pd.DataFrame(payload)
    frame.to_csv(path, index=False)
    return path


def _load_real_local_adata(tmp_path: Path) -> object:
    source_csv = _write_real_like_csv(tmp_path / "real_like.csv")
    module = IngestionModule()
    result = module.load_adamson(
        cache_dir=str(tmp_path / "cache"),
        source_uri=str(source_csv),
        force_refresh=True,
    )
    return result.adata


def test_split_manifest_is_reproducible_from_same_seed(tmp_path: Path) -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=260, n_genes=90, n_perturbations=10, seed=14)
    splitter = ContextShiftSplitModule()
    config = ShiftSplitConfig(seed=77, min_cells_per_split=12)

    manifest_a = splitter.generate_split(bundle.adata, shift_type="locus", config=config)
    manifest_b = splitter.generate_split(bundle.adata, shift_type="locus", config=config)
    assert manifest_a.to_dict() == manifest_b.to_dict()

    manifest_path = splitter.persist_manifest(manifest_a, tmp_path / "locus_manifest.json")
    loaded = splitter.load_manifest(manifest_path)
    assert loaded.to_dict() == manifest_a.to_dict()


def test_leakage_validator_fails_intentionally_corrupted_fixture() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=220, n_genes=80, n_perturbations=10, seed=21)
    splitter = ContextShiftSplitModule()
    manifest = splitter.generate_split(
        bundle.adata,
        shift_type="locus",
        config=ShiftSplitConfig(seed=99, min_cells_per_split=8),
    )
    assert len(manifest.train_indices) > 0

    corrupted = ShiftSplitManifest(
        **{
            **manifest.to_dict(),
            "test_indices": [manifest.train_indices[0], *manifest.test_indices],
        }
    )
    with pytest.raises(ShiftSplitLeakageError, match="Cell-index leakage"):
        splitter.validate_manifest(adata=bundle.adata, manifest=corrupted, raise_on_error=True)


@pytest.mark.parametrize("shift_type", ALL_SHIFT_TYPES)
def test_each_shift_type_generates_for_synthetic_and_real_local_dataset(
    shift_type: str,
    tmp_path: Path,
) -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=240, n_genes=80, n_perturbations=10, seed=31)
    real_adata = _load_real_local_adata(tmp_path / "real")
    splitter = ContextShiftSplitModule()
    config = ShiftSplitConfig(seed=123, min_cells_per_split=10)

    synthetic_manifest = splitter.generate_split(
        bundle.adata,
        shift_type=shift_type,  # type: ignore[arg-type]
        config=config,
    )
    real_manifest = splitter.generate_split(
        real_adata,
        shift_type=shift_type,  # type: ignore[arg-type]
        config=config,
    )

    for adata, manifest in ((bundle.adata, synthetic_manifest), (real_adata, real_manifest)):
        report = splitter.validate_manifest(adata=adata, manifest=manifest, raise_on_error=True)
        assert report["is_valid"] is True
        assert report["coverage"]["n_assigned_unique"] == int(adata.n_obs)
        assert report["coverage"]["n_missing"] == 0
        assert manifest.leakage_report["leakage_detected"] is False
        assert manifest.provenance["dataset_fingerprint_sha256"] != ""


def test_small_data_fallback_is_recorded_for_undersized_splits() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=90, n_genes=60, n_perturbations=6, seed=5)
    splitter = ContextShiftSplitModule()
    manifest = splitter.generate_split(
        bundle.adata,
        shift_type="protocol",
        config=ShiftSplitConfig(seed=8, min_cells_per_split=50),
    )
    assert manifest.fallback_applied is True
    assert manifest.fallback_reason is not None
