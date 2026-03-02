"""Preprocessing and QC module for TCPE Phase 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from tcpe.anndata_schema import (
    NORMALIZED_LAYER_KEY,
    validate_anndata_schema,
)

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

PREPROCESSING_UNS_KEY = "preprocessing"
CONFOUNDER_PROXY_OBSM_KEY = "X_confounder_proxy"
CONFOUNDER_PROXY_COLUMNS_UNS_KEY = "confounder_proxy_columns"
PREPROCESSING_VERSION = "phase6_v1"


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot produce a valid model-ready AnnData."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for deterministic preprocessing and QC."""

    target_sum: float = 1e4
    hvg_target: int = 2000
    hvg_min_allowed: int = 2000
    hvg_max_allowed: int = 5000
    min_counts_per_cell: int = 50
    min_genes_per_cell: int = 10
    min_cells_per_gene: int = 3
    extract_confounder_proxies: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_sum <= 0:
            raise ValueError("target_sum must be positive.")
        if self.hvg_min_allowed <= 0 or self.hvg_max_allowed <= 0:
            raise ValueError("HVG allowed bounds must be positive.")
        if self.hvg_min_allowed > self.hvg_max_allowed:
            raise ValueError("hvg_min_allowed cannot be greater than hvg_max_allowed.")
        if not (self.hvg_min_allowed <= self.hvg_target <= self.hvg_max_allowed):
            raise ValueError(
                "hvg_target must be within the allowed range "
                f"[{self.hvg_min_allowed}, {self.hvg_max_allowed}]."
            )
        if self.min_counts_per_cell < 0:
            raise ValueError("min_counts_per_cell must be non-negative.")
        if self.min_genes_per_cell < 0:
            raise ValueError("min_genes_per_cell must be non-negative.")
        if self.min_cells_per_gene < 1:
            raise ValueError("min_cells_per_gene must be at least 1.")


@dataclass(frozen=True)
class PreprocessingResult:
    """Preprocessing output and summary statistics."""

    adata: AnnData
    n_cells_input: int
    n_genes_input: int
    n_cells_after_qc: int
    n_genes_after_qc: int
    n_hvgs_selected: int
    hvg_fallback_used: bool
    removed_cell_count: int
    removed_gene_count: int


class PreprocessingModule:
    """Reusable preprocessing pipeline for all datasets."""

    def run(
        self,
        adata: AnnData,
        config: PreprocessingConfig | None = None,
    ) -> PreprocessingResult:
        """Apply QC, normalization, log1p, HVG selection, and optional proxy extraction."""
        cfg = config if config is not None else PreprocessingConfig()
        validate_anndata_schema(adata=adata, mode="strict")

        working = adata.copy()
        n_cells_input = int(working.n_obs)
        n_genes_input = int(working.n_vars)

        counts_dense = _to_dense_counts(working.X)
        library_size = counts_dense.sum(axis=1)
        genes_detected = np.count_nonzero(counts_dense > 0, axis=1)
        keep_cells = (library_size >= cfg.min_counts_per_cell) & (
            genes_detected >= cfg.min_genes_per_cell
        )
        if int(np.sum(keep_cells)) == 0:
            raise PreprocessingError("QC removed all cells; adjust cell-level thresholds.")

        working = working[keep_cells, :].copy()
        counts_dense = counts_dense[keep_cells, :]

        expressing_cells_per_gene = np.count_nonzero(counts_dense > 0, axis=0)
        keep_genes = expressing_cells_per_gene >= cfg.min_cells_per_gene
        if int(np.sum(keep_genes)) == 0:
            raise PreprocessingError("QC removed all genes; adjust gene-level thresholds.")

        working = working[:, keep_genes].copy()
        counts_dense = counts_dense[:, keep_genes]

        # Ensure raw counts remain in .X and refresh library_size after QC masks.
        working.X = counts_dense.astype(np.int64)
        working.obs["library_size"] = counts_dense.sum(axis=1).astype(np.float64)

        normalized_log1p = _normalize_log1p(counts=counts_dense, target_sum=cfg.target_sum)
        working.layers[NORMALIZED_LAYER_KEY] = normalized_log1p.astype(np.float32)

        requested_hvg = cfg.hvg_target
        selected_hvg_count = min(requested_hvg, int(working.n_vars))
        hvg_fallback_used = selected_hvg_count != requested_hvg
        hvg_indices = _select_hvg_indices(
            normalized=normalized_log1p,
            gene_ids=working.var["gene_id"].astype(str).to_numpy(),
            n_keep=selected_hvg_count,
        )
        working = working[:, hvg_indices].copy()
        normalized_after_hvg = normalized_log1p[:, hvg_indices]
        working.layers[NORMALIZED_LAYER_KEY] = normalized_after_hvg.astype(np.float32)
        working.obs["library_size"] = np.asarray(working.X).sum(axis=1).astype(np.float64)

        if cfg.extract_confounder_proxies:
            proxy_matrix, proxy_columns = _extract_confounder_proxy_matrix(working.obs)
            working.obsm[CONFOUNDER_PROXY_OBSM_KEY] = proxy_matrix
            working.uns[CONFOUNDER_PROXY_COLUMNS_UNS_KEY] = proxy_columns

        working.uns[PREPROCESSING_UNS_KEY] = {
            "version": PREPROCESSING_VERSION,
            "config": asdict(cfg),
            "n_cells_input": n_cells_input,
            "n_genes_input": n_genes_input,
            "n_cells_after_qc": int(working.n_obs),
            "n_genes_after_qc": int(working.n_vars),
            "hvg": {
                "requested": requested_hvg,
                "selected": int(working.n_vars),
                "fallback_used": hvg_fallback_used,
            },
            "qc": {
                "removed_cells": int(n_cells_input - working.n_obs),
                "removed_genes": int(n_genes_input - working.n_vars),
                "min_counts_per_cell": cfg.min_counts_per_cell,
                "min_genes_per_cell": cfg.min_genes_per_cell,
                "min_cells_per_gene": cfg.min_cells_per_gene,
            },
            "target_sum": cfg.target_sum,
            "seed": cfg.seed,
            "extract_confounder_proxies": cfg.extract_confounder_proxies,
        }

        validate_anndata_schema(adata=working, mode="strict")
        return PreprocessingResult(
            adata=working,
            n_cells_input=n_cells_input,
            n_genes_input=n_genes_input,
            n_cells_after_qc=int(working.n_obs),
            n_genes_after_qc=int(working.n_vars),
            n_hvgs_selected=int(working.n_vars),
            hvg_fallback_used=hvg_fallback_used,
            removed_cell_count=int(n_cells_input - working.n_obs),
            removed_gene_count=int(n_genes_input - working.n_vars),
        )


def _to_dense_counts(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return cast(np.ndarray, x.toarray().astype(np.float64))
    return cast(np.ndarray, np.asarray(x, dtype=np.float64))


def _normalize_log1p(counts: np.ndarray, target_sum: float) -> np.ndarray:
    library_size = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    normalized = (counts / library_size) * target_sum
    return cast(np.ndarray, np.log1p(normalized))


def _select_hvg_indices(normalized: np.ndarray, gene_ids: np.ndarray, n_keep: int) -> np.ndarray:
    if n_keep <= 0:
        raise PreprocessingError("n_keep for HVG selection must be positive.")
    variance = np.var(normalized, axis=0)
    sort_indices = np.lexsort((gene_ids, -variance))
    return cast(np.ndarray, sort_indices[:n_keep])


def _extract_confounder_proxy_matrix(obs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns: list[str] = []
    proxy_parts: list[np.ndarray] = []

    library_size_source = (
        obs["library_size"]
        if "library_size" in obs.columns
        else pd.Series(np.zeros(obs.shape[0], dtype=np.float64), index=obs.index)
    )
    library_size = pd.to_numeric(library_size_source, errors="coerce").fillna(0.0)
    library_size_array = _zscore_column(library_size.to_numpy(dtype=np.float64))
    proxy_parts.append(library_size_array[:, None])
    columns.append("library_size_z")

    knockdown_source = (
        obs["knockdown_efficiency_proxy"]
        if "knockdown_efficiency_proxy" in obs.columns
        else pd.Series(np.zeros(obs.shape[0], dtype=np.float64), index=obs.index)
    )
    knockdown = pd.to_numeric(knockdown_source, errors="coerce").fillna(0.0)
    knockdown_array = _zscore_column(knockdown.to_numpy(dtype=np.float64))
    proxy_parts.append(knockdown_array[:, None])
    columns.append("knockdown_efficiency_proxy_z")

    for categorical_column in ("batch", "protocol"):
        if categorical_column not in obs.columns:
            continue
        categories = sorted(
            obs[categorical_column].astype(str).fillna("unknown").unique().tolist()
        )
        for category in categories:
            one_hot = (
                obs[categorical_column].astype(str).to_numpy() == category
            ).astype(np.float32)
            proxy_parts.append(one_hot[:, None])
            columns.append(f"{categorical_column}={category}")

    proxy_matrix = np.concatenate(proxy_parts, axis=1).astype(np.float32)
    return proxy_matrix, columns


def _zscore_column(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    zscored = (values - mean) / std
    return cast(np.ndarray, zscored.astype(np.float32))
