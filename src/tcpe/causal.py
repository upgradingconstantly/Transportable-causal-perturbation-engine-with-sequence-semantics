"""Causal module v1 with two-stage IV regression, proxy conditioning, and uncertainty."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd
from scipy import sparse

from tcpe.anndata_schema import NORMALIZED_LAYER_KEY

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

REQUIRED_IV_PROXY_COLUMNS: tuple[str, ...] = (
    "batch",
    "library_size",
    "knockdown_efficiency_proxy",
    "protocol",
)
CAUSAL_GRAPH_SCHEMA_VERSION = "causal_graph_v1"
CAUSAL_ESTIMATOR_VERSION = "phase12_two_stage_iv_v1"
CAUSAL_GRAPH_UNS_KEY = "causal_graph"
CAUSAL_DIAGNOSTICS_UNS_KEY = "causal_diagnostics"

CONTROL_PERTURBATION_IDS = {"ntc", "ctrl", "control", "p000"}


class CausalError(RuntimeError):
    """Base error for causal-module operations."""


class CausalInputError(CausalError):
    """Raised when required causal inputs are missing or malformed."""


@dataclass(frozen=True)
class CausalConfig:
    """Configuration for two-stage IV causal graph estimation."""

    expression_layer: str = NORMALIZED_LAYER_KEY
    max_hvgs: int = 128
    stage1_ridge_alpha: float = 1e-3
    stage2_ridge_alpha: float = 1e-3
    bootstrap_iterations: int = 24
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    compute_naive_graph: bool = True

    def __post_init__(self) -> None:
        if self.max_hvgs <= 1:
            raise ValueError("max_hvgs must be greater than 1.")
        if self.stage1_ridge_alpha < 0:
            raise ValueError("stage1_ridge_alpha must be non-negative.")
        if self.stage2_ridge_alpha < 0:
            raise ValueError("stage2_ridge_alpha must be non-negative.")
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be non-negative.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1).")


@dataclass(frozen=True)
class CausalGraphArtifact:
    """Estimated directional causal graph with uncertainty intervals."""

    schema_version: str
    estimator_version: str
    gene_ids: list[str]
    adjacency: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    standard_error: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "estimator_version": self.estimator_version,
            "gene_ids": list(self.gene_ids),
            "adjacency": self.adjacency.tolist(),
            "ci_lower": self.ci_lower.tolist(),
            "ci_upper": self.ci_upper.tolist(),
            "standard_error": self.standard_error.tolist(),
            "metadata": dict(self.metadata),
        }

    def to_edge_table(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        n_genes = len(self.gene_ids)
        for source_idx in range(n_genes):
            for target_idx in range(n_genes):
                if source_idx == target_idx:
                    continue
                records.append(
                    {
                        "source_gene_id": self.gene_ids[source_idx],
                        "target_gene_id": self.gene_ids[target_idx],
                        "weight": float(self.adjacency[source_idx, target_idx]),
                        "standard_error": float(self.standard_error[source_idx, target_idx]),
                        "ci_lower": float(self.ci_lower[source_idx, target_idx]),
                        "ci_upper": float(self.ci_upper[source_idx, target_idx]),
                    }
                )
        return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class CausalFitResult:
    """Output of two-stage IV fitting with graph and diagnostics."""

    graph: CausalGraphArtifact
    selected_gene_indices: np.ndarray
    selected_gene_ids: list[str]
    stage1_design_columns: list[str]
    stage2_proxy_columns: list[str]
    diagnostics: dict[str, float]
    naive_adjacency: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_gene_indices"] = self.selected_gene_indices.tolist()
        if self.naive_adjacency is not None:
            payload["naive_adjacency"] = self.naive_adjacency.tolist()
        return payload


@dataclass(frozen=True)
class CausalExportPaths:
    """Artifact paths exported by causal graph exporter."""

    adjacency_npz_path: Path
    edge_table_csv_path: Path
    metadata_json_path: Path


class CausalEstimator(Protocol):
    """Protocol for causal estimators."""

    def fit(
        self,
        adata: AnnData,
        *,
        config: CausalConfig | None = None,
        persist_to_anndata: bool = True,
    ) -> CausalFitResult:
        """Estimate causal graph from AnnData and return fit artifacts."""

    def infer_graph(
        self,
        adata: AnnData,
        *,
        config: CausalConfig | None = None,
        persist_to_anndata: bool = True,
    ) -> CausalGraphArtifact:
        """Infer causal graph only."""

    def edge_uncertainty(self, graph: CausalGraphArtifact) -> pd.DataFrame:
        """Return edge-level uncertainty table."""


class CausalModule:
    """Phase 12 causal module using two-stage IV with required proxies."""

    def status(self) -> str:
        return "phase12_iv_ready"

    def fit(
        self,
        adata: AnnData,
        *,
        config: CausalConfig | None = None,
        persist_to_anndata: bool = True,
    ) -> CausalFitResult:
        selected_config = config if config is not None else CausalConfig()
        result = _fit_two_stage_iv(adata=adata, config=selected_config)

        if persist_to_anndata:
            adata.uns[CAUSAL_GRAPH_UNS_KEY] = result.graph.to_dict()
            adata.uns[CAUSAL_DIAGNOSTICS_UNS_KEY] = dict(result.diagnostics)

        return result

    def infer_graph(
        self,
        adata: AnnData,
        *,
        config: CausalConfig | None = None,
        persist_to_anndata: bool = True,
    ) -> CausalGraphArtifact:
        result = self.fit(
            adata,
            config=config,
            persist_to_anndata=persist_to_anndata,
        )
        return result.graph

    def edge_uncertainty(self, graph: CausalGraphArtifact) -> pd.DataFrame:
        return graph.to_edge_table()

    def export_graph_artifacts(
        self,
        graph: CausalGraphArtifact,
        *,
        output_dir: str | Path,
        file_prefix: str = "causal_graph",
    ) -> CausalExportPaths:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        adjacency_npz_path = output_path / f"{file_prefix}_adjacency.npz"
        edge_table_csv_path = output_path / f"{file_prefix}_edges.csv"
        metadata_json_path = output_path / f"{file_prefix}_metadata.json"

        np.savez_compressed(
            adjacency_npz_path,
            adjacency=graph.adjacency,
            ci_lower=graph.ci_lower,
            ci_upper=graph.ci_upper,
            standard_error=graph.standard_error,
        )
        graph.to_edge_table().to_csv(edge_table_csv_path, index=False)
        metadata_json_path.write_text(
            json.dumps(
                {
                    "schema_version": graph.schema_version,
                    "estimator_version": graph.estimator_version,
                    "gene_ids": graph.gene_ids,
                    "metadata": graph.metadata,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return CausalExportPaths(
            adjacency_npz_path=adjacency_npz_path,
            edge_table_csv_path=edge_table_csv_path,
            metadata_json_path=metadata_json_path,
        )


def _fit_two_stage_iv(
    *,
    adata: AnnData,
    config: CausalConfig,
) -> CausalFitResult:
    expression = _resolve_expression_matrix(adata=adata, expression_layer=config.expression_layer)
    selected_gene_indices, selected_gene_ids = _select_hvg_subset(
        adata=adata,
        expression=expression,
        max_hvgs=config.max_hvgs,
    )
    response = expression[:, selected_gene_indices]

    proxy_design, proxy_columns = _build_proxy_design_matrix(adata.obs)
    instrument_design, instrument_columns = _build_instrument_design_matrix(adata.obs)

    stage1_design = np.concatenate(
        [np.ones((response.shape[0], 1), dtype=np.float64), instrument_design, proxy_design],
        axis=1,
    )
    stage1_columns = (
        ["intercept"]
        + [f"instrument::{name}" for name in instrument_columns]
        + [f"proxy::{name}" for name in proxy_columns]
    )
    stage1_coef = _ridge_solution(
        design=stage1_design,
        response=response,
        alpha=config.stage1_ridge_alpha,
    )
    purified_expression = stage1_design @ stage1_coef

    adjacency = _fit_stage2_graph(
        response=response,
        endogenous_predictors=purified_expression,
        proxy_design=proxy_design,
        ridge_alpha=config.stage2_ridge_alpha,
    )
    naive_adjacency = None
    if config.compute_naive_graph:
        naive_adjacency = _fit_stage2_graph(
            response=response,
            endogenous_predictors=response,
            proxy_design=None,
            ridge_alpha=config.stage2_ridge_alpha,
        )

    standard_error, ci_lower, ci_upper = _estimate_bootstrap_uncertainty(
        response=response,
        proxy_design=proxy_design,
        instrument_design=instrument_design,
        stage1_ridge_alpha=config.stage1_ridge_alpha,
        stage2_ridge_alpha=config.stage2_ridge_alpha,
        bootstrap_iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    )

    np.fill_diagonal(adjacency, 0.0)
    np.fill_diagonal(ci_lower, 0.0)
    np.fill_diagonal(ci_upper, 0.0)
    np.fill_diagonal(standard_error, 0.0)
    if naive_adjacency is not None:
        np.fill_diagonal(naive_adjacency, 0.0)

    diagnostics = _build_diagnostics(
        response=response,
        purified=purified_expression,
        adjacency=adjacency,
        naive_adjacency=naive_adjacency,
        adata=adata,
        selected_gene_indices=selected_gene_indices,
    )
    metadata: dict[str, Any] = {
        "required_proxy_columns": list(REQUIRED_IV_PROXY_COLUMNS),
        "proxy_design_columns": proxy_columns,
        "instrument_columns": instrument_columns,
        "n_cells": int(response.shape[0]),
        "n_genes_selected": int(response.shape[1]),
        "expression_layer": config.expression_layer,
        "bootstrap_iterations": int(config.bootstrap_iterations),
        "confidence_level": float(config.confidence_level),
        "cyclic_allowed": True,
        "dag_enforced": False,
    }

    graph = CausalGraphArtifact(
        schema_version=CAUSAL_GRAPH_SCHEMA_VERSION,
        estimator_version=CAUSAL_ESTIMATOR_VERSION,
        gene_ids=selected_gene_ids,
        adjacency=adjacency.astype(np.float32),
        ci_lower=ci_lower.astype(np.float32),
        ci_upper=ci_upper.astype(np.float32),
        standard_error=standard_error.astype(np.float32),
        metadata=metadata,
    )

    return CausalFitResult(
        graph=graph,
        selected_gene_indices=selected_gene_indices.astype(np.int64),
        selected_gene_ids=selected_gene_ids,
        stage1_design_columns=stage1_columns,
        stage2_proxy_columns=proxy_columns,
        diagnostics=diagnostics,
        naive_adjacency=None if naive_adjacency is None else naive_adjacency.astype(np.float32),
    )


def _resolve_expression_matrix(adata: AnnData, expression_layer: str) -> np.ndarray:
    if expression_layer in adata.layers:
        matrix = adata.layers[expression_layer]
    elif expression_layer == "X":
        matrix = adata.X
    else:
        raise CausalInputError(
            f"Expression layer '{expression_layer}' not found in AnnData layers and is not 'X'."
        )

    if sparse.issparse(matrix):
        return cast(np.ndarray, matrix.toarray().astype(np.float64))
    return cast(np.ndarray, np.asarray(matrix, dtype=np.float64))


def _select_hvg_subset(
    *,
    adata: AnnData,
    expression: np.ndarray,
    max_hvgs: int,
) -> tuple[np.ndarray, list[str]]:
    if expression.shape[1] <= 1:
        raise CausalInputError("Causal inference requires at least 2 genes.")

    if "highly_variable" in adata.var.columns:
        hv_flags = adata.var["highly_variable"].to_numpy(dtype=bool)
        hv_indices = np.flatnonzero(hv_flags)
        if hv_indices.size > 0:
            selected = hv_indices[:max_hvgs]
        else:
            selected = np.array([], dtype=np.int64)
    else:
        selected = np.array([], dtype=np.int64)

    if selected.size == 0:
        gene_variance = np.var(expression, axis=0)
        selected = np.argsort(gene_variance)[::-1][:max_hvgs]

    selected = np.asarray(selected, dtype=np.int64)
    if selected.size <= 1:
        raise CausalInputError("Selected gene subset must contain at least 2 genes.")

    if "gene_id" in adata.var.columns:
        gene_ids_source = adata.var["gene_id"].astype(str).to_numpy()
    else:
        gene_ids_source = np.asarray(adata.var_names.astype(str))
    gene_ids = [str(gene_ids_source[idx]) for idx in selected.tolist()]
    return selected, gene_ids


def _build_proxy_design_matrix(obs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    missing = [column for column in REQUIRED_IV_PROXY_COLUMNS if column not in obs.columns]
    if missing:
        missing_joined = ", ".join(missing)
        raise CausalInputError(
            "Two-stage IV requires proxy columns in `.obs`: "
            f"{', '.join(REQUIRED_IV_PROXY_COLUMNS)}. Missing: {missing_joined}."
        )

    library_size = pd.to_numeric(obs["library_size"], errors="coerce").fillna(0.0)
    knockdown = pd.to_numeric(obs["knockdown_efficiency_proxy"], errors="coerce").fillna(0.0)
    numeric_matrix = np.column_stack(
        [
            _zscore(library_size.to_numpy(dtype=np.float64)),
            _zscore(knockdown.to_numpy(dtype=np.float64)),
        ]
    )
    numeric_columns = ["library_size_z", "knockdown_efficiency_proxy_z"]

    batch = obs["batch"].astype(str).str.strip().replace("", "unknown")
    protocol = obs["protocol"].astype(str).str.strip().replace("", "unknown")
    batch_dummies = pd.get_dummies(batch, prefix="batch", dtype=np.float64)
    protocol_dummies = pd.get_dummies(protocol, prefix="protocol", dtype=np.float64)
    categorical_frame = pd.concat([batch_dummies, protocol_dummies], axis=1)
    categorical_matrix = categorical_frame.to_numpy(dtype=np.float64)
    categorical_columns = categorical_frame.columns.astype(str).tolist()

    proxy_design = np.concatenate([numeric_matrix, categorical_matrix], axis=1)
    if proxy_design.shape[1] == 0:
        raise CausalInputError("Proxy design matrix is empty after encoding.")
    return proxy_design, numeric_columns + categorical_columns


def _build_instrument_design_matrix(obs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if "perturbation_id" not in obs.columns:
        raise CausalInputError("Two-stage IV requires `.obs['perturbation_id']` as instruments.")

    perturbation = obs["perturbation_id"].astype(str).str.strip().replace("", "unassigned")
    categories = sorted(set(perturbation.tolist()))
    if len(categories) < 2:
        raise CausalInputError(
            "Two-stage IV requires at least two perturbation categories in "
            "`.obs['perturbation_id']`."
        )

    reference = _select_instrument_reference(perturbation=perturbation, obs=obs)
    instrument_frame = pd.get_dummies(perturbation, prefix="perturb", dtype=np.float64)
    reference_column = f"perturb_{reference}"
    if reference_column in instrument_frame.columns:
        instrument_frame = instrument_frame.drop(columns=[reference_column])

    instrument_columns = instrument_frame.columns.astype(str).tolist()
    if len(instrument_columns) == 0:
        raise CausalInputError(
            "Two-stage IV instrumentation is empty after reference-category removal."
        )
    return instrument_frame.to_numpy(dtype=np.float64), instrument_columns


def _select_instrument_reference(perturbation: pd.Series, obs: pd.DataFrame) -> str:
    lower = perturbation.str.lower()
    control_mask = lower.isin(CONTROL_PERTURBATION_IDS)
    if "condition" in obs.columns:
        control_mask = control_mask | obs["condition"].astype(str).str.lower().eq("control")

    if bool(control_mask.any()):
        return str(perturbation[control_mask].iloc[0])
    return str(perturbation.iloc[0])


def _fit_stage2_graph(
    *,
    response: np.ndarray,
    endogenous_predictors: np.ndarray,
    proxy_design: np.ndarray | None,
    ridge_alpha: float,
) -> np.ndarray:
    n_cells, n_genes = response.shape
    adjacency = np.zeros((n_genes, n_genes), dtype=np.float64)
    intercept = np.ones((n_cells, 1), dtype=np.float64)
    proxy_matrix = (
        proxy_design
        if proxy_design is not None
        else np.zeros((n_cells, 0), dtype=np.float64)
    )

    for target_idx in range(n_genes):
        source_indices = [idx for idx in range(n_genes) if idx != target_idx]
        source_matrix = endogenous_predictors[:, source_indices]
        stage2_design = np.concatenate([intercept, source_matrix, proxy_matrix], axis=1)
        coef = _ridge_solution(
            design=stage2_design,
            response=response[:, [target_idx]],
            alpha=ridge_alpha,
        )[:, 0]
        edge_weights = coef[1 : 1 + len(source_indices)]
        adjacency[source_indices, target_idx] = edge_weights

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _estimate_bootstrap_uncertainty(
    *,
    response: np.ndarray,
    proxy_design: np.ndarray,
    instrument_design: np.ndarray,
    stage1_ridge_alpha: float,
    stage2_ridge_alpha: float,
    bootstrap_iterations: int,
    seed: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_genes = response.shape[1]
    if bootstrap_iterations <= 0:
        zero = np.zeros((n_genes, n_genes), dtype=np.float64)
        return zero, zero, zero

    rng = np.random.default_rng(seed)
    estimates = np.zeros((bootstrap_iterations, n_genes, n_genes), dtype=np.float64)

    for iteration in range(bootstrap_iterations):
        sampled_idx = rng.integers(0, response.shape[0], size=response.shape[0])
        response_sample = response[sampled_idx]
        proxy_sample = proxy_design[sampled_idx]
        instrument_sample = instrument_design[sampled_idx]
        stage1_design = np.concatenate(
            [
                np.ones((response_sample.shape[0], 1), dtype=np.float64),
                instrument_sample,
                proxy_sample,
            ],
            axis=1,
        )
        stage1_coef = _ridge_solution(
            design=stage1_design,
            response=response_sample,
            alpha=stage1_ridge_alpha,
        )
        purified_sample = stage1_design @ stage1_coef
        estimates[iteration] = _fit_stage2_graph(
            response=response_sample,
            endogenous_predictors=purified_sample,
            proxy_design=proxy_sample,
            ridge_alpha=stage2_ridge_alpha,
        )

    ddof = 1 if bootstrap_iterations > 1 else 0
    standard_error = np.std(estimates, axis=0, ddof=ddof)
    alpha = 1.0 - confidence_level
    ci_lower = np.quantile(estimates, alpha / 2.0, axis=0)
    ci_upper = np.quantile(estimates, 1.0 - (alpha / 2.0), axis=0)
    return standard_error, ci_lower, ci_upper


def _ridge_solution(
    *,
    design: np.ndarray,
    response: np.ndarray,
    alpha: float,
) -> np.ndarray:
    xtx = design.T @ design
    ridge = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    ridge[0, 0] = 0.0
    lhs = xtx + ridge
    rhs = design.T @ response
    try:
        return cast(np.ndarray, np.linalg.solve(lhs, rhs))
    except np.linalg.LinAlgError:
        return cast(np.ndarray, np.linalg.pinv(lhs) @ rhs)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def _build_diagnostics(
    *,
    response: np.ndarray,
    purified: np.ndarray,
    adjacency: np.ndarray,
    naive_adjacency: np.ndarray | None,
    adata: AnnData,
    selected_gene_indices: np.ndarray,
) -> dict[str, float]:
    stage1_r2 = _mean_r2(y_true=response, y_pred=purified)
    offdiag_l1 = float(np.mean(np.abs(adjacency[~np.eye(adjacency.shape[0], dtype=bool)])))
    diagnostics: dict[str, float] = {
        "stage1_r2_mean": stage1_r2,
        "iv_offdiag_l1_mean": offdiag_l1,
    }

    if naive_adjacency is not None:
        naive_l1 = float(
            np.mean(np.abs(naive_adjacency[~np.eye(naive_adjacency.shape[0], dtype=bool)]))
        )
        diagnostics["naive_offdiag_l1_mean"] = naive_l1
        diagnostics["iv_minus_naive_offdiag_l1_mean"] = offdiag_l1 - naive_l1

    truth = _extract_truth_subset(adata=adata, selected_gene_indices=selected_gene_indices)
    if truth is not None:
        diagnostics["iv_truth_abs_corr"] = _offdiag_abs_corr(y_true=truth, y_pred=adjacency)
        if naive_adjacency is not None:
            diagnostics["naive_truth_abs_corr"] = _offdiag_abs_corr(
                y_true=truth,
                y_pred=naive_adjacency,
            )
    return diagnostics


def _extract_truth_subset(adata: AnnData, selected_gene_indices: np.ndarray) -> np.ndarray | None:
    raw = adata.uns.get("synthetic_ground_truth_adjacency")
    if raw is None:
        return None

    truth = np.asarray(raw, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        return None
    if np.max(selected_gene_indices) >= truth.shape[0]:
        return None
    subset = truth[np.ix_(selected_gene_indices, selected_gene_indices)]
    return cast(np.ndarray, subset)


def _offdiag_abs_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.eye(y_true.shape[0], dtype=bool)
    true_flat = np.abs(y_true[mask])
    pred_flat = np.abs(y_pred[mask])
    if np.std(true_flat) <= 1e-12 or np.std(pred_flat) <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(true_flat, pred_flat)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def _mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape for R2 calculation.")
    numerator = np.sum(np.square(y_true - y_pred), axis=0)
    denominator = np.sum(np.square(y_true - y_true.mean(axis=0, keepdims=True)), axis=0)
    safe_denominator = np.where(denominator <= 1e-12, 1.0, denominator)
    r2 = 1.0 - (numerator / safe_denominator)
    return float(np.mean(r2))
