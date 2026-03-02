"""Synthetic dataset generation for TCPE Phase 4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tcpe.anndata_schema import (
    CELL_STATE_EMBEDDING_OBSM_KEY,
    NORMALIZED_LAYER_KEY,
    PERTURBATIONS_UNS_KEY,
    SCHEMA_VERSION,
    SCHEMA_VERSION_UNS_KEY,
    SEQUENCE_EMBEDDING_OBSM_KEY,
    validate_anndata_schema,
)

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any


class SyntheticDataError(RuntimeError):
    """Raised when synthetic dataset generation/export fails."""


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """Ground truth tensors and identifiers for synthetic benchmarking."""

    causal_adjacency: np.ndarray
    direct_effects: np.ndarray
    transport_shift_by_perturbation: np.ndarray
    confounder_matrix: np.ndarray
    perturbation_ids: list[str]
    gene_ids: list[str]
    target_gene_index: np.ndarray
    control_perturbation_id: str
    seed: int


@dataclass(frozen=True)
class SyntheticDatasetBundle:
    """Synthetic AnnData and associated ground-truth metadata."""

    adata: AnnData
    ground_truth: SyntheticGroundTruth


@dataclass(frozen=True)
class SyntheticExportPaths:
    """File outputs produced by synthetic dataset export."""

    adata_path: Path
    arrays_path: Path
    metadata_path: Path


def generate_synthetic_dataset(
    n_cells: int = 500,
    n_genes: int = 200,
    n_perturbations: int = 10,
    seed: int = 42,
) -> SyntheticDatasetBundle:
    """Generate deterministic synthetic Perturb-seq-style AnnData with known ground truth."""
    _validate_generation_inputs(
        n_cells=n_cells,
        n_genes=n_genes,
        n_perturbations=n_perturbations,
    )

    anndata_module = _require_anndata()
    rng = np.random.default_rng(seed)

    gene_ids = [f"gene_{index:04d}" for index in range(n_genes)]
    perturbation_ids = [f"p{index:03d}" for index in range(n_perturbations)]
    control_perturbation_id = perturbation_ids[0]

    adjacency = _build_causal_adjacency(n_genes=n_genes, rng=rng)
    direct_effects, target_gene_index = _build_perturbation_effects(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        adjacency=adjacency,
        rng=rng,
    )

    perturbation_index = _assign_perturbations(
        n_cells=n_cells,
        n_perturbations=n_perturbations,
        rng=rng,
    )
    batch_index = rng.integers(0, 3, size=n_cells)

    cell_cycle = rng.normal(0.0, 1.0, size=n_cells)
    lineage_state = rng.normal(0.0, 1.0, size=n_cells)
    confounder_matrix = np.column_stack([cell_cycle, lineage_state]).astype(np.float64)

    knockdown_efficiency = np.zeros(n_cells, dtype=np.float64)
    for cell_idx, perturb_idx in enumerate(perturbation_index):
        if perturb_idx == 0:
            continue
        efficiency = rng.beta(2.0, 2.0)
        efficiency *= 0.85 + (0.05 * batch_index[cell_idx])
        knockdown_efficiency[cell_idx] = float(np.clip(efficiency, 0.05, 1.0))

    base_log_expression = rng.normal(loc=-1.0, scale=0.5, size=n_genes)
    confounder_weights = rng.normal(loc=0.0, scale=0.15, size=(2, n_genes))
    batch_weights = rng.normal(loc=0.0, scale=0.08, size=(3, n_genes))

    direct_by_cell = direct_effects[perturbation_index]
    propagated_by_cell = direct_by_cell @ adjacency
    eta = (
        base_log_expression[None, :]
        + confounder_matrix @ confounder_weights
        + batch_weights[batch_index]
        + (knockdown_efficiency[:, None] * direct_by_cell)
        + (0.7 * knockdown_efficiency[:, None] * propagated_by_cell)
    )

    expected_rate = np.exp(np.clip(eta, -4.0, 3.5))
    library_size_factor = rng.lognormal(mean=0.3, sigma=0.45, size=n_cells)
    poisson_rate = expected_rate * library_size_factor[:, None]
    counts = rng.poisson(poisson_rate).astype(np.int64)
    _fix_zero_library_cells(counts=counts, rng=rng)

    cell_ids = [f"cell_{index:05d}" for index in range(n_cells)]
    var_frame = _build_var_metadata(gene_ids=gene_ids)
    obs_frame = _build_obs_metadata(
        cell_ids=cell_ids,
        perturbation_index=perturbation_index,
        perturbation_ids=perturbation_ids,
        batch_index=batch_index,
        knockdown_efficiency=knockdown_efficiency,
        counts=counts,
    )

    adata = cast(AnnData, anndata_module.AnnData(X=counts, obs=obs_frame, var=var_frame))
    adata.layers[NORMALIZED_LAYER_KEY] = _normalized_log1p(
        counts=counts,
        library_size=obs_frame["library_size"],
    )
    adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY] = _build_sequence_embeddings(
        perturbation_index=perturbation_index,
        n_perturbations=n_perturbations,
        rng=rng,
    )
    adata.obsm[CELL_STATE_EMBEDDING_OBSM_KEY] = _build_cell_state_embeddings(
        confounder_matrix=confounder_matrix,
        knockdown_efficiency=knockdown_efficiency,
        normalized=adata.layers[NORMALIZED_LAYER_KEY],
        rng=rng,
    )
    adata.uns[PERTURBATIONS_UNS_KEY] = _build_perturbation_metadata(
        perturbation_ids=perturbation_ids,
        target_gene_index=target_gene_index,
        var_frame=var_frame,
        control_perturbation_id=control_perturbation_id,
    )
    adata.uns[SCHEMA_VERSION_UNS_KEY] = SCHEMA_VERSION
    adata.uns["synthetic_generation"] = {
        "seed": seed,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_perturbations": n_perturbations,
    }

    transport_shift = _compute_transport_shift(
        normalized=adata.layers[NORMALIZED_LAYER_KEY],
        perturbation_index=perturbation_index,
        n_perturbations=n_perturbations,
        control_index=0,
    )
    ground_truth = SyntheticGroundTruth(
        causal_adjacency=adjacency.astype(np.float32),
        direct_effects=direct_effects.astype(np.float32),
        transport_shift_by_perturbation=transport_shift.astype(np.float32),
        confounder_matrix=confounder_matrix.astype(np.float32),
        perturbation_ids=perturbation_ids,
        gene_ids=gene_ids,
        target_gene_index=target_gene_index.astype(np.int64),
        control_perturbation_id=control_perturbation_id,
        seed=seed,
    )

    validate_anndata_schema(adata=adata, mode="strict")
    return SyntheticDatasetBundle(adata=adata, ground_truth=ground_truth)


def export_synthetic_dataset(
    bundle: SyntheticDatasetBundle,
    output_dir: str | Path,
    file_prefix: str = "synthetic_tcpe",
) -> SyntheticExportPaths:
    """Export synthetic AnnData plus ground-truth arrays/metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adata_path = output_path / f"{file_prefix}.h5ad"
    arrays_path = output_path / f"{file_prefix}_ground_truth_arrays.npz"
    metadata_path = output_path / f"{file_prefix}_ground_truth_metadata.json"

    bundle.adata.write_h5ad(adata_path)
    np.savez_compressed(
        arrays_path,
        causal_adjacency=bundle.ground_truth.causal_adjacency,
        direct_effects=bundle.ground_truth.direct_effects,
        transport_shift_by_perturbation=bundle.ground_truth.transport_shift_by_perturbation,
        confounder_matrix=bundle.ground_truth.confounder_matrix,
        target_gene_index=bundle.ground_truth.target_gene_index,
    )

    metadata = {
        "seed": bundle.ground_truth.seed,
        "control_perturbation_id": bundle.ground_truth.control_perturbation_id,
        "gene_ids": bundle.ground_truth.gene_ids,
        "perturbation_ids": bundle.ground_truth.perturbation_ids,
        "cell_ids": bundle.adata.obs["cell_id"].astype(str).tolist(),
        "shape": {"n_cells": int(bundle.adata.n_obs), "n_genes": int(bundle.adata.n_vars)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return SyntheticExportPaths(
        adata_path=adata_path,
        arrays_path=arrays_path,
        metadata_path=metadata_path,
    )


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise SyntheticDataError("`anndata` is required to generate synthetic datasets.") from exc
    return ad


def _validate_generation_inputs(n_cells: int, n_genes: int, n_perturbations: int) -> None:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    if n_genes <= 1:
        raise ValueError("n_genes must be greater than 1.")
    if n_perturbations < 2:
        raise ValueError("n_perturbations must be at least 2 (control + perturbations).")


def _build_causal_adjacency(n_genes: int, rng: np.random.Generator) -> np.ndarray:
    adjacency = np.zeros((n_genes, n_genes), dtype=np.float64)
    regulator_count = min(30, n_genes)
    targets_per_regulator = 3
    for regulator in range(regulator_count):
        targets = rng.choice(n_genes, size=targets_per_regulator, replace=False)
        weights = rng.normal(loc=0.0, scale=0.12, size=targets_per_regulator)
        adjacency[regulator, targets] = weights

    cycle_nodes = min(12, n_genes - (n_genes % 2))
    for node in range(0, cycle_nodes, 2):
        forward = float(rng.normal(loc=0.08, scale=0.02))
        backward = float(rng.normal(loc=-0.06, scale=0.02))
        adjacency[node, node + 1] = forward
        adjacency[node + 1, node] = backward

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _build_perturbation_effects(
    n_genes: int,
    n_perturbations: int,
    adjacency: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    effects = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    target_gene_index = np.full(n_perturbations, -1, dtype=np.int64)
    target_pool = np.linspace(0, n_genes - 1, num=(n_perturbations - 1), dtype=int)

    for perturb_idx in range(1, n_perturbations):
        target_idx = int(target_pool[perturb_idx - 1])
        target_gene_index[perturb_idx] = target_idx
        effects[perturb_idx, target_idx] = float(rng.normal(loc=-1.2, scale=0.15))

        downstream = np.where(np.abs(adjacency[target_idx]) > 0.0)[0]
        if downstream.size == 0:
            downstream = rng.choice(n_genes, size=3, replace=False)
        downstream_effect = adjacency[target_idx, downstream] * rng.normal(
            loc=1.0, scale=0.1, size=downstream.size
        )
        effects[perturb_idx, downstream] += downstream_effect

    return effects, target_gene_index


def _assign_perturbations(
    n_cells: int,
    n_perturbations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assignment = np.tile(
        np.arange(n_perturbations),
        int(np.ceil(n_cells / n_perturbations)),
    )[:n_cells]
    rng.shuffle(assignment)
    return assignment.astype(np.int64)


def _fix_zero_library_cells(counts: np.ndarray, rng: np.random.Generator) -> None:
    library = counts.sum(axis=1)
    zero_cells = np.where(library == 0)[0]
    for cell_idx in zero_cells:
        gene_idx = int(rng.integers(0, counts.shape[1]))
        counts[cell_idx, gene_idx] = 1


def _build_var_metadata(gene_ids: list[str]) -> pd.DataFrame:
    n_genes = len(gene_ids)
    chrom = [f"chr{(idx % 22) + 1}" for idx in range(n_genes)]
    strand = ["+" if idx % 2 == 0 else "-" for idx in range(n_genes)]
    tss = [100_000 + (idx * 173) for idx in range(n_genes)]
    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": [gene_id.upper() for gene_id in gene_ids],
            "chrom": chrom,
            "strand": strand,
            "tss": tss,
        },
        index=gene_ids,
    )


def _build_obs_metadata(
    cell_ids: list[str],
    perturbation_index: np.ndarray,
    perturbation_ids: list[str],
    batch_index: np.ndarray,
    knockdown_efficiency: np.ndarray,
    counts: np.ndarray,
) -> pd.DataFrame:
    assigned_perturbations = [perturbation_ids[idx] for idx in perturbation_index.tolist()]
    condition = [
        "control" if pert == perturbation_ids[0] else "perturbed"
        for pert in assigned_perturbations
    ]
    batch = [f"batch_{idx}" for idx in batch_index.tolist()]
    protocol = ["10x_v3_synthetic"] * len(cell_ids)
    library_size = counts.sum(axis=1).astype(np.float64)

    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_type": ["k562_synthetic"] * len(cell_ids),
            "batch": batch,
            "condition": condition,
            "protocol": protocol,
            "library_size": library_size,
            "perturbation_id": assigned_perturbations,
            "knockdown_efficiency_proxy": knockdown_efficiency.astype(np.float64),
        },
        index=cell_ids,
    )


def _normalized_log1p(counts: np.ndarray, library_size: pd.Series) -> np.ndarray:
    library = np.maximum(library_size.to_numpy(dtype=np.float64)[:, None], 1.0)
    normalized = (counts.astype(np.float64) / library) * 1e4
    return cast(np.ndarray, np.log1p(normalized).astype(np.float32))


def _build_sequence_embeddings(
    perturbation_index: np.ndarray,
    n_perturbations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbation_embedding = rng.normal(loc=0.0, scale=1.0, size=(n_perturbations, 16))
    return cast(np.ndarray, perturbation_embedding[perturbation_index].astype(np.float32))


def _build_cell_state_embeddings(
    confounder_matrix: np.ndarray,
    knockdown_efficiency: np.ndarray,
    normalized: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n_cells = confounder_matrix.shape[0]
    base_features = np.column_stack(
        [
            confounder_matrix,
            knockdown_efficiency.reshape(n_cells, 1),
            normalized[:, : min(5, normalized.shape[1])],
        ]
    )
    projection = rng.normal(loc=0.0, scale=0.5, size=(base_features.shape[1], 16))
    embeddings = base_features @ projection
    return embeddings.astype(np.float32)


def _build_perturbation_metadata(
    perturbation_ids: list[str],
    target_gene_index: np.ndarray,
    var_frame: pd.DataFrame,
    control_perturbation_id: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for perturbation_id, target_index in zip(perturbation_ids, target_gene_index, strict=True):
        if perturbation_id == control_perturbation_id:
            records.append(
                {
                    "perturbation_id": perturbation_id,
                    "target_gene": "NTC",
                    "chrom": "chr0",
                    "start": -1,
                    "end": -1,
                    "strand": ".",
                    "modality": "control",
                    "dose": 0.0,
                    "gRNA_id": f"{perturbation_id}_ntc",
                }
            )
            continue

        target_row = var_frame.iloc[int(target_index)]
        start = int(target_row["tss"]) - 25
        end = start + 20
        records.append(
            {
                "perturbation_id": perturbation_id,
                "target_gene": str(target_row["gene_symbol"]),
                "chrom": str(target_row["chrom"]),
                "start": start,
                "end": end,
                "strand": str(target_row["strand"]),
                "modality": "crispri",
                "dose": 1.0,
                "gRNA_id": f"{perturbation_id}_grna",
            }
        )
    return pd.DataFrame(records)


def _compute_transport_shift(
    normalized: np.ndarray,
    perturbation_index: np.ndarray,
    n_perturbations: int,
    control_index: int,
) -> np.ndarray:
    control_mask = perturbation_index == control_index
    if np.sum(control_mask) == 0:
        raise SyntheticDataError("Synthetic assignment must include at least one control cell.")

    control_mean = normalized[control_mask].mean(axis=0)
    shifts = np.zeros((n_perturbations, normalized.shape[1]), dtype=np.float64)
    for perturb_idx in range(n_perturbations):
        perturb_mask = perturbation_index == perturb_idx
        perturb_mean = normalized[perturb_mask].mean(axis=0)
        shifts[perturb_idx] = perturb_mean - control_mean
    return shifts
