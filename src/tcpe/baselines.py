"""Mandatory baseline suite for TCPE Phase 9."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy import sparse

from tcpe.anndata_schema import NORMALIZED_LAYER_KEY

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

BASELINE_SUITE_SCHEMA_VERSION = "baseline_suite_v1"
BASELINE_RESULTS_UNS_KEY = "baseline_suite_results"
BASELINE_RANDOM_GRN_UNS_KEY = "baseline_random_grn_adjacency"
REQUIRED_BASELINE_NAMES: tuple[str, ...] = (
    "gene_level_mean",
    "control_mean",
    "linear_regression",
    "random_edge_grn",
)


class BaselineError(RuntimeError):
    """Raised when baseline generation fails."""


@dataclass(frozen=True)
class BaselineResult:
    """Uniform schema for an individual baseline output."""

    baseline_name: str
    task: Literal["expression", "grn"]
    metrics: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineSuiteResult:
    """Uniform schema for full baseline-suite outputs."""

    schema_version: str
    expression_layer: str
    n_train_cells: int
    n_eval_cells: int
    n_genes: int
    baselines: list[BaselineResult]
    metadata: dict[str, Any]
    random_grn_adjacency: np.ndarray

    def baseline_names(self) -> list[str]:
        return [item.baseline_name for item in self.baselines]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["baselines"] = [baseline.to_dict() for baseline in self.baselines]
        payload["random_grn_adjacency"] = self.random_grn_adjacency.tolist()
        return payload


def run_baseline_suite(
    adata: AnnData,
    *,
    expression_layer: str = NORMALIZED_LAYER_KEY,
    train_mask: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    reference_adjacency: np.ndarray | None = None,
    seed: int = 42,
) -> BaselineSuiteResult:
    """Run all required baselines and return a uniform results object."""
    matrix = _resolve_expression_matrix(adata=adata, expression_layer=expression_layer)
    perturbation_ids = _resolve_perturbation_ids(adata=adata)
    condition = (
        adata.obs["condition"].astype(str).to_numpy()
        if "condition" in adata.obs.columns
        else np.array(["unknown"] * adata.n_obs)
    )

    train_idx = _resolve_mask(mask=train_mask, n_obs=adata.n_obs, default=True)
    eval_idx = _resolve_mask(mask=eval_mask, n_obs=adata.n_obs, default=True)

    y_train = matrix[train_idx]
    y_eval = matrix[eval_idx]
    perturb_train = perturbation_ids[train_idx]
    perturb_eval = perturbation_ids[eval_idx]
    condition_train = condition[train_idx]

    gene_mean_pred, gene_mean_meta = _gene_level_mean_baseline(
        y_train=y_train,
        perturb_train=perturb_train,
        perturb_eval=perturb_eval,
    )
    control_pred, control_meta = _control_mean_baseline(
        y_train=y_train,
        perturb_train=perturb_train,
        condition_train=condition_train,
        n_eval=y_eval.shape[0],
    )
    linear_pred, linear_meta = _linear_regression_baseline(
        y_train=y_train,
        perturb_train=perturb_train,
        perturb_eval=perturb_eval,
    )

    expression_results = [
        BaselineResult(
            baseline_name="gene_level_mean",
            task="expression",
            metrics=_regression_metrics(y_true=y_eval, y_pred=gene_mean_pred),
            metadata=gene_mean_meta,
        ),
        BaselineResult(
            baseline_name="control_mean",
            task="expression",
            metrics=_regression_metrics(y_true=y_eval, y_pred=control_pred),
            metadata=control_meta,
        ),
        BaselineResult(
            baseline_name="linear_regression",
            task="expression",
            metrics=_regression_metrics(y_true=y_eval, y_pred=linear_pred),
            metadata=linear_meta,
        ),
    ]

    ref_adj, ref_source = _resolve_reference_adjacency(
        adata=adata,
        y_train=y_train,
        reference_adjacency=reference_adjacency,
    )
    random_adj = _degree_preserving_random_graph(reference_adjacency=ref_adj, seed=seed)
    random_result = BaselineResult(
        baseline_name="random_edge_grn",
        task="grn",
        metrics=_grn_baseline_metrics(reference_adjacency=ref_adj, random_adjacency=random_adj),
        metadata={"reference_source": ref_source},
    )

    baselines = expression_results + [random_result]
    suite = BaselineSuiteResult(
        schema_version=BASELINE_SUITE_SCHEMA_VERSION,
        expression_layer=expression_layer,
        n_train_cells=int(np.sum(train_idx)),
        n_eval_cells=int(np.sum(eval_idx)),
        n_genes=int(matrix.shape[1]),
        baselines=baselines,
        metadata={"seed": seed},
        random_grn_adjacency=random_adj.astype(np.float32),
    )
    return suite


def _resolve_expression_matrix(adata: AnnData, expression_layer: str) -> np.ndarray:
    if expression_layer in adata.layers:
        matrix = adata.layers[expression_layer]
    elif expression_layer == "X":
        matrix = adata.X
    else:
        raise BaselineError(
            f"Expression layer '{expression_layer}' not found in AnnData layers and is not 'X'."
        )

    if sparse.issparse(matrix):
        return cast(np.ndarray, matrix.toarray().astype(np.float64))
    return cast(np.ndarray, np.asarray(matrix, dtype=np.float64))


def _resolve_perturbation_ids(adata: AnnData) -> np.ndarray:
    if "perturbation_id" not in adata.obs.columns:
        raise BaselineError("AnnData `.obs` must include `perturbation_id` for baseline suite.")
    values = adata.obs["perturbation_id"].astype(str).to_numpy()
    return cast(np.ndarray, values)


def _resolve_mask(mask: np.ndarray | None, n_obs: int, default: bool) -> np.ndarray:
    if mask is None:
        return np.full(n_obs, default, dtype=bool)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != (n_obs,):
        raise BaselineError(f"Mask shape {mask_array.shape} does not match expected ({n_obs},).")
    if int(np.sum(mask_array)) == 0:
        raise BaselineError("Provided mask selects zero cells.")
    return mask_array


def _gene_level_mean_baseline(
    *,
    y_train: np.ndarray,
    perturb_train: np.ndarray,
    perturb_eval: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    means_by_perturbation: dict[str, np.ndarray] = {}
    for perturbation_id in sorted(set(perturb_train.tolist())):
        mask = perturb_train == perturbation_id
        means_by_perturbation[perturbation_id] = y_train[mask].mean(axis=0)

    global_mean = y_train.mean(axis=0)
    predictions = np.zeros((len(perturb_eval), y_train.shape[1]), dtype=np.float64)
    unseen = 0
    for idx, perturbation_id in enumerate(perturb_eval.tolist()):
        if perturbation_id in means_by_perturbation:
            predictions[idx] = means_by_perturbation[perturbation_id]
        else:
            predictions[idx] = global_mean
            unseen += 1

    metadata = {
        "n_train_perturbations": len(means_by_perturbation),
        "n_eval_unseen_perturbations": unseen,
        "fallback": "global_train_mean",
    }
    return predictions, metadata


def _control_mean_baseline(
    *,
    y_train: np.ndarray,
    perturb_train: np.ndarray,
    condition_train: np.ndarray,
    n_eval: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    control_mask = _detect_controls(perturbation_ids=perturb_train, condition=condition_train)
    if int(np.sum(control_mask)) == 0:
        control_vector = y_train.mean(axis=0)
        fallback_used = True
    else:
        control_vector = y_train[control_mask].mean(axis=0)
        fallback_used = False
    predictions = np.repeat(control_vector[None, :], repeats=n_eval, axis=0)
    metadata = {
        "n_train_control_cells": int(np.sum(control_mask)),
        "fallback_used": fallback_used,
    }
    return predictions, metadata


def _linear_regression_baseline(
    *,
    y_train: np.ndarray,
    perturb_train: np.ndarray,
    perturb_eval: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    categories = sorted(set(perturb_train.tolist()))
    cat_to_col = {name: idx for idx, name in enumerate(categories)}
    x_train = np.zeros((y_train.shape[0], len(categories) + 1), dtype=np.float64)
    x_train[:, 0] = 1.0
    for row_idx, perturbation_id in enumerate(perturb_train.tolist()):
        x_train[row_idx, cat_to_col[perturbation_id] + 1] = 1.0

    coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

    x_eval = np.zeros((len(perturb_eval), len(categories) + 1), dtype=np.float64)
    x_eval[:, 0] = 1.0
    unseen = 0
    for row_idx, perturbation_id in enumerate(perturb_eval.tolist()):
        if perturbation_id in cat_to_col:
            x_eval[row_idx, cat_to_col[perturbation_id] + 1] = 1.0
        else:
            unseen += 1
    predictions = x_eval @ coef
    metadata = {
        "n_train_perturbations": len(categories),
        "n_eval_unseen_perturbations": unseen,
        "fallback": "intercept_only_for_unseen",
    }
    return predictions, metadata


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise BaselineError(
            f"Prediction shape {y_pred.shape} does not match true shape {y_true.shape}."
        )
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    pearson = _mean_gene_pearson(y_true=y_true, y_pred=y_pred)
    return {"mae": mae, "rmse": rmse, "pearson_mean_gene": pearson}


def _mean_gene_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correlations: list[float] = []
    for gene_idx in range(y_true.shape[1]):
        true_col = y_true[:, gene_idx]
        pred_col = y_pred[:, gene_idx]
        if np.std(true_col) <= 1e-8 or np.std(pred_col) <= 1e-8:
            continue
        corr = float(np.corrcoef(true_col, pred_col)[0, 1])
        if np.isnan(corr):
            continue
        correlations.append(corr)
    if len(correlations) == 0:
        return 0.0
    return float(np.mean(correlations))


def _detect_controls(perturbation_ids: np.ndarray, condition: np.ndarray) -> np.ndarray:
    control_conditions = np.array([value.lower() == "control" for value in condition], dtype=bool)
    control_ids = np.array(
        [
            value.lower() in {"ntc", "ctrl", "control", "p000"}
            for value in perturbation_ids
        ],
        dtype=bool,
    )
    return cast(np.ndarray, control_conditions | control_ids)


def _resolve_reference_adjacency(
    *,
    adata: AnnData,
    y_train: np.ndarray,
    reference_adjacency: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if reference_adjacency is not None:
        return _validate_square_adjacency(reference_adjacency), "argument"

    if "inferred_grn_adjacency" in adata.uns:
        adjacency = np.asarray(adata.uns["inferred_grn_adjacency"], dtype=np.float64)
        return _validate_square_adjacency(adjacency), "adata.uns.inferred_grn_adjacency"

    if "synthetic_ground_truth_adjacency" in adata.uns:
        adjacency = np.asarray(adata.uns["synthetic_ground_truth_adjacency"], dtype=np.float64)
        return _validate_square_adjacency(adjacency), "adata.uns.synthetic_ground_truth_adjacency"

    proxy = _derive_proxy_adjacency(y_train=y_train)
    return proxy, "derived_proxy_expression_graph"


def _validate_square_adjacency(adjacency: np.ndarray) -> np.ndarray:
    if len(adjacency.shape) != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise BaselineError("Reference adjacency must be a square 2D matrix.")
    if adjacency.shape[0] == 0:
        raise BaselineError("Reference adjacency must be non-empty.")
    result = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(result, 0.0)
    return result


def _derive_proxy_adjacency(y_train: np.ndarray) -> np.ndarray:
    n_genes = y_train.shape[1]
    if n_genes < 2:
        return np.zeros((n_genes, n_genes), dtype=np.float64)

    corr = np.corrcoef(y_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    adjacency = np.zeros_like(corr)
    top_k = min(2, max(1, n_genes - 1))
    for src in range(n_genes):
        target_idx = np.argsort(np.abs(corr[src]))[::-1][:top_k]
        adjacency[src, target_idx] = corr[src, target_idx]
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _degree_preserving_random_graph(
    *,
    reference_adjacency: np.ndarray,
    seed: int,
    n_swaps_factor: int = 10,
) -> np.ndarray:
    adjacency = np.asarray(reference_adjacency, dtype=np.float64)
    edges = [(i, j, float(adjacency[i, j])) for i, j in zip(*np.nonzero(adjacency), strict=False)]
    edge_pairs = {(i, j) for i, j, _ in edges}
    if len(edges) < 2:
        randomized = np.zeros_like(adjacency)
        for i, j, weight in edges:
            randomized[i, j] = weight
        return randomized

    rng = np.random.default_rng(seed)
    edges_mutable = list(edges)
    swap_attempts = len(edges_mutable) * n_swaps_factor
    for _ in range(swap_attempts):
        idx_a, idx_b = rng.choice(len(edges_mutable), size=2, replace=False).tolist()
        a, b, w_ab = edges_mutable[idx_a]
        c, d, w_cd = edges_mutable[idx_b]
        if len({a, b, c, d}) < 4:
            continue
        new_edge_1 = (a, d)
        new_edge_2 = (c, b)
        if new_edge_1[0] == new_edge_1[1] or new_edge_2[0] == new_edge_2[1]:
            continue
        if new_edge_1 in edge_pairs or new_edge_2 in edge_pairs:
            continue

        edge_pairs.remove((a, b))
        edge_pairs.remove((c, d))
        edge_pairs.add(new_edge_1)
        edge_pairs.add(new_edge_2)

        edges_mutable[idx_a] = (new_edge_1[0], new_edge_1[1], w_ab)
        edges_mutable[idx_b] = (new_edge_2[0], new_edge_2[1], w_cd)

    randomized = np.zeros_like(adjacency)
    for src, dst, weight in edges_mutable:
        randomized[src, dst] = weight
    np.fill_diagonal(randomized, 0.0)
    return randomized


def _grn_baseline_metrics(
    *,
    reference_adjacency: np.ndarray,
    random_adjacency: np.ndarray,
) -> dict[str, float]:
    ref_binary = (reference_adjacency != 0).astype(np.int64)
    rnd_binary = (random_adjacency != 0).astype(np.int64)
    ref_out = ref_binary.sum(axis=1)
    ref_in = ref_binary.sum(axis=0)
    rnd_out = rnd_binary.sum(axis=1)
    rnd_in = rnd_binary.sum(axis=0)
    overlap = int(np.sum((ref_binary == 1) & (rnd_binary == 1)))
    edge_count = int(np.sum(ref_binary))
    return {
        "edge_count_reference": float(edge_count),
        "edge_count_random": float(np.sum(rnd_binary)),
        "out_degree_l1_diff": float(np.sum(np.abs(ref_out - rnd_out))),
        "in_degree_l1_diff": float(np.sum(np.abs(ref_in - rnd_in))),
        "edge_overlap_fraction": float(overlap / edge_count) if edge_count > 0 else 0.0,
    }
