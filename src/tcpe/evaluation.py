"""Evaluation, baseline harness, and model-card generation for TCPE Phase 14."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import sparse

from tcpe.anndata_schema import NORMALIZED_LAYER_KEY
from tcpe.baselines import (
    BASELINE_RANDOM_GRN_UNS_KEY,
    BASELINE_RESULTS_UNS_KEY,
    BaselineSuiteResult,
    run_baseline_suite,
)

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

EVALUATION_REPORT_SCHEMA_VERSION = "evaluation_report_v1"
MODEL_CARD_SCHEMA_VERSION = "model_card_v1"
EVALUATION_REPORT_UNS_KEY = "evaluation_report"
MODEL_CARD_ARTIFACTS_UNS_KEY = "model_card_artifacts"
MANDATORY_MODEL_CARD_METRICS: tuple[str, ...] = (
    "mae",
    "pearson_top_1000_degs",
    "calibration_error",
)


class EvaluationError(RuntimeError):
    """Raised when evaluation or report generation fails."""


class ModelCardValidationError(EvaluationError):
    """Raised when model-card payloads fail schema checks."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for Phase 14 evaluator computations."""

    expression_layer: str = NORMALIZED_LAYER_KEY
    top_k_degs: int = 1000
    calibration_bins: int = 10
    failure_mae_threshold: float = 1.00
    failure_pearson_threshold: float = 0.10
    failure_calibration_threshold: float = 0.75
    seed: int = 42
    model_name: str = "transport_model"
    model_version: str = "unknown"
    persist_to_anndata: bool = True

    def __post_init__(self) -> None:
        if self.top_k_degs <= 0:
            raise ValueError("top_k_degs must be positive.")
        if self.calibration_bins <= 0:
            raise ValueError("calibration_bins must be positive.")
        if self.failure_mae_threshold < 0:
            raise ValueError("failure_mae_threshold must be non-negative.")
        if self.failure_calibration_threshold < 0:
            raise ValueError("failure_calibration_threshold must be non-negative.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


@dataclass(frozen=True)
class EvaluationReport:
    """Structured report for model-vs-baseline evaluation."""

    schema_version: str
    metrics: dict[str, float]
    baseline_comparison: list[dict[str, Any]]
    failure_modes: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCardArtifacts:
    """Filesystem artifacts emitted for a model card."""

    json_path: Path
    markdown_path: Path
    schema_version: str
    validation_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["json_path"] = str(self.json_path)
        payload["markdown_path"] = str(self.markdown_path)
        return payload


class EvaluationModule:
    """Phase 14 evaluation module with mandatory model-card generation."""

    def status(self) -> str:
        return "phase9_baselines_ready"

    def run_baselines(
        self,
        adata: AnnData,
        *,
        expression_layer: str = NORMALIZED_LAYER_KEY,
        train_mask: np.ndarray | None = None,
        eval_mask: np.ndarray | None = None,
        reference_adjacency: np.ndarray | None = None,
        seed: int = 42,
        persist_to_anndata: bool = True,
    ) -> BaselineSuiteResult:
        """Run all required baselines and optionally persist results into AnnData."""
        suite = run_baseline_suite(
            adata=adata,
            expression_layer=expression_layer,
            train_mask=train_mask,
            eval_mask=eval_mask,
            reference_adjacency=reference_adjacency,
            seed=seed,
        )
        if persist_to_anndata:
            adata.uns[BASELINE_RESULTS_UNS_KEY] = suite.to_dict()
            adata.uns[BASELINE_RANDOM_GRN_UNS_KEY] = suite.random_grn_adjacency
        return suite

    def evaluate_predictions(
        self,
        adata: AnnData,
        *,
        prediction_mean: np.ndarray,
        prediction_variance: np.ndarray | None = None,
        baseline_suite: BaselineSuiteResult | dict[str, Any] | None = None,
        config: EvaluationConfig | None = None,
        eval_mask: np.ndarray | None = None,
        persist_to_anndata: bool | None = None,
    ) -> EvaluationReport:
        """Compute required metrics, baseline comparisons, and failure modes."""
        selected = config if config is not None else EvaluationConfig()
        persist = selected.persist_to_anndata if persist_to_anndata is None else persist_to_anndata

        expression = _resolve_expression_matrix(
            adata=adata,
            expression_layer=selected.expression_layer,
        )
        eval_idx = _resolve_mask(mask=eval_mask, n_obs=int(adata.n_obs), default=True)
        y_true = expression[eval_idx]
        y_pred = np.asarray(prediction_mean, dtype=np.float64)
        if y_pred.shape != y_true.shape:
            raise EvaluationError(
                f"prediction_mean shape {y_pred.shape} does not match evaluated target shape "
                f"{y_true.shape}."
            )

        y_var: np.ndarray | None = None
        if prediction_variance is not None:
            y_var = np.asarray(prediction_variance, dtype=np.float64)
            if y_var.shape != y_true.shape:
                raise EvaluationError(
                    f"prediction_variance shape {y_var.shape} does not match target shape "
                    f"{y_true.shape}."
                )

        control_profile = _resolve_control_profile(adata=adata, expression=expression)
        metrics = self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_var=y_var,
            control_profile=control_profile,
            top_k_degs=selected.top_k_degs,
            calibration_bins=selected.calibration_bins,
        )

        baseline_payload = self._resolve_or_run_baselines(
            adata=adata,
            baseline_suite=baseline_suite,
            expression_layer=selected.expression_layer,
            eval_mask=eval_idx,
            seed=selected.seed,
        )
        baseline_comparison = _build_baseline_comparison_table(
            baseline_payload=baseline_payload,
            model_metrics=metrics,
        )
        failure_modes = _build_failure_modes(
            model_metrics=metrics,
            baseline_comparison=baseline_comparison,
            config=selected,
        )

        report = EvaluationReport(
            schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
            metrics=metrics,
            baseline_comparison=baseline_comparison,
            failure_modes=failure_modes,
            metadata={
                "model_name": selected.model_name,
                "model_version": selected.model_version,
                "expression_layer": selected.expression_layer,
                "n_eval_cells": int(y_true.shape[0]),
                "n_genes": int(y_true.shape[1]),
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "baseline_suite_schema_version": str(
                    baseline_payload.get("schema_version", "unknown")
                ),
            },
        )
        if persist:
            adata.uns[EVALUATION_REPORT_UNS_KEY] = report.to_dict()
        return report

    def evaluate_and_generate_model_card(
        self,
        adata: AnnData,
        *,
        prediction_mean: np.ndarray,
        prediction_variance: np.ndarray | None = None,
        baseline_suite: BaselineSuiteResult | dict[str, Any] | None = None,
        config: EvaluationConfig | None = None,
        eval_mask: np.ndarray | None = None,
        output_dir: str | Path,
        file_prefix: str = "model_card",
    ) -> tuple[EvaluationReport, ModelCardArtifacts]:
        """Run full Phase 14 evaluation and enforce model-card completeness."""
        report = self.evaluate_predictions(
            adata,
            prediction_mean=prediction_mean,
            prediction_variance=prediction_variance,
            baseline_suite=baseline_suite,
            config=config,
            eval_mask=eval_mask,
            persist_to_anndata=True,
        )
        artifacts = self.generate_model_card(
            report,
            output_dir=output_dir,
            file_prefix=file_prefix,
        )
        self.assert_run_complete(artifacts)
        adata.uns[MODEL_CARD_ARTIFACTS_UNS_KEY] = artifacts.to_dict()
        return report, artifacts

    def generate_model_card(
        self,
        report: EvaluationReport,
        *,
        output_dir: str | Path,
        file_prefix: str = "model_card",
    ) -> ModelCardArtifacts:
        """Write model card JSON and Markdown artifacts with schema validation."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / f"{file_prefix}.json"
        markdown_path = root / f"{file_prefix}.md"

        payload = self._build_model_card_payload(report)
        validation = self.validate_model_card_payload(payload, raise_on_error=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        markdown = _render_model_card_markdown(payload)
        markdown_path.write_text(markdown, encoding="utf-8")

        artifacts = ModelCardArtifacts(
            json_path=json_path,
            markdown_path=markdown_path,
            schema_version=MODEL_CARD_SCHEMA_VERSION,
            validation_report=validation,
        )
        return artifacts

    def validate_model_card_json(self, path: str | Path) -> dict[str, Any]:
        """Load and validate model-card JSON payload."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ModelCardValidationError("Model card JSON root must be an object.")
        self.validate_model_card_payload(cast(dict[str, Any], payload), raise_on_error=True)
        return cast(dict[str, Any], payload)

    def validate_model_card_payload(
        self,
        payload: dict[str, Any],
        *,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        """Validate required sections and mandatory metrics in model-card payload."""
        errors: list[str] = []
        schema_version = payload.get("schema_version")
        if str(schema_version) != MODEL_CARD_SCHEMA_VERSION:
            errors.append(
                f"Unsupported model-card schema version '{schema_version}'. "
                f"Expected '{MODEL_CARD_SCHEMA_VERSION}'."
            )

        for key in ("metrics", "baseline_comparison", "failure_modes", "metadata"):
            if key not in payload:
                errors.append(f"Missing required model-card section `{key}`.")

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            errors.append("`metrics` section must be an object.")
        else:
            for metric_key in MANDATORY_MODEL_CARD_METRICS:
                if metric_key not in metrics:
                    errors.append(f"Missing mandatory metric `{metric_key}` in `metrics` section.")
                    continue
                try:
                    float(metrics[metric_key])
                except (TypeError, ValueError):
                    errors.append(f"Metric `{metric_key}` must be numeric.")

        baseline_comparison = payload.get("baseline_comparison")
        if not isinstance(baseline_comparison, list):
            errors.append("`baseline_comparison` section must be a list.")
        elif len(baseline_comparison) == 0:
            errors.append("`baseline_comparison` must include at least one row.")

        failure_modes = payload.get("failure_modes")
        if not isinstance(failure_modes, list):
            errors.append("`failure_modes` section must be a list.")
        elif len(failure_modes) == 0:
            errors.append("`failure_modes` must include at least one entry.")

        validation = {"is_valid": len(errors) == 0, "errors": errors}
        if raise_on_error and not validation["is_valid"]:
            raise ModelCardValidationError(
                "Model card validation failed:\n- " + "\n- ".join(errors)
            )
        return validation

    def assert_run_complete(self, artifacts: ModelCardArtifacts | None) -> None:
        """Enforce that a run is incomplete unless model-card artifacts exist and validate."""
        if artifacts is None:
            raise EvaluationError(
                "Evaluation run is incomplete unless model card artifacts exist and validate."
            )
        if not artifacts.json_path.exists():
            raise EvaluationError(f"Missing model card JSON artifact: {artifacts.json_path}")
        if not artifacts.markdown_path.exists():
            raise EvaluationError(
                f"Missing model card Markdown artifact: {artifacts.markdown_path}"
            )

        self.validate_model_card_json(artifacts.json_path)
        markdown_text = artifacts.markdown_path.read_text(encoding="utf-8")
        required_markdown_sections = ("## Metrics", "## Baseline Comparisons", "## Failure Modes")
        for section in required_markdown_sections:
            if section not in markdown_text:
                raise EvaluationError(
                    f"Model card markdown missing required section `{section}`."
                )

    def _build_model_card_payload(self, report: EvaluationReport) -> dict[str, Any]:
        return {
            "schema_version": MODEL_CARD_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "evaluation_schema_version": report.schema_version,
            "metrics": report.metrics,
            "baseline_comparison": report.baseline_comparison,
            "failure_modes": report.failure_modes,
            "metadata": report.metadata,
        }

    def _resolve_or_run_baselines(
        self,
        *,
        adata: AnnData,
        baseline_suite: BaselineSuiteResult | dict[str, Any] | None,
        expression_layer: str,
        eval_mask: np.ndarray,
        seed: int,
    ) -> dict[str, Any]:
        if isinstance(baseline_suite, BaselineSuiteResult):
            return baseline_suite.to_dict()
        if isinstance(baseline_suite, dict):
            return baseline_suite
        if BASELINE_RESULTS_UNS_KEY in adata.uns:
            existing = adata.uns[BASELINE_RESULTS_UNS_KEY]
            if isinstance(existing, dict):
                return cast(dict[str, Any], existing)

        generated = self.run_baselines(
            adata,
            expression_layer=expression_layer,
            eval_mask=eval_mask,
            seed=seed,
            persist_to_anndata=True,
        )
        return generated.to_dict()

    def _compute_metrics(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_var: np.ndarray | None,
        control_profile: np.ndarray,
        top_k_degs: int,
        calibration_bins: int,
    ) -> dict[str, float]:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        pearson = _pearson_on_top_degs(
            y_true=y_true,
            y_pred=y_pred,
            control_profile=control_profile,
            top_k=top_k_degs,
        )
        calibration = _calibration_error(
            y_true=y_true,
            y_pred=y_pred,
            y_var=y_var,
            n_bins=calibration_bins,
        )
        return {
            "mae": mae,
            "pearson_top_1000_degs": pearson,
            "calibration_error": calibration,
        }


def _resolve_expression_matrix(adata: AnnData, expression_layer: str) -> np.ndarray:
    if expression_layer in adata.layers:
        matrix = adata.layers[expression_layer]
    elif expression_layer == "X":
        matrix = adata.X
    else:
        raise EvaluationError(
            f"Expression layer '{expression_layer}' not found in AnnData layers and is not 'X'."
        )
    if sparse.issparse(matrix):
        return cast(np.ndarray, matrix.toarray().astype(np.float64))
    return cast(np.ndarray, np.asarray(matrix, dtype=np.float64))


def _resolve_mask(mask: np.ndarray | None, n_obs: int, default: bool) -> np.ndarray:
    if mask is None:
        return np.full(n_obs, default, dtype=bool)
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != (n_obs,):
        raise EvaluationError(f"Mask shape {mask_array.shape} does not match expected ({n_obs},).")
    if int(np.sum(mask_array)) == 0:
        raise EvaluationError("Provided mask selects zero cells.")
    return mask_array


def _resolve_control_profile(adata: AnnData, expression: np.ndarray) -> np.ndarray:
    if "condition" in adata.obs.columns:
        condition = adata.obs["condition"].astype(str).to_numpy()
    else:
        condition = np.array(["unknown"] * int(adata.n_obs), dtype=object)

    if "perturbation_id" in adata.obs.columns:
        perturbation_ids = adata.obs["perturbation_id"].astype(str).to_numpy()
    else:
        perturbation_ids = np.array(["unknown"] * int(adata.n_obs), dtype=object)

    control_mask = _detect_controls(
        perturbation_ids=cast(np.ndarray, perturbation_ids),
        condition=cast(np.ndarray, condition),
    )
    if int(np.sum(control_mask)) == 0:
        return cast(np.ndarray, np.mean(expression, axis=0))
    return cast(np.ndarray, np.mean(expression[control_mask], axis=0))


def _detect_controls(perturbation_ids: np.ndarray, condition: np.ndarray) -> np.ndarray:
    control_conditions = np.array([value.lower() == "control" for value in condition], dtype=bool)
    control_ids = np.array(
        [value.lower() in {"ntc", "ctrl", "control", "p000"} for value in perturbation_ids],
        dtype=bool,
    )
    return cast(np.ndarray, control_conditions | control_ids)


def _pearson_on_top_degs(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    control_profile: np.ndarray,
    top_k: int,
) -> float:
    if y_true.shape != y_pred.shape:
        raise EvaluationError(
            f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}."
        )

    mean_true = np.mean(y_true, axis=0)
    deg_scores = np.abs(mean_true - control_profile.reshape(-1))
    if deg_scores.size == 0:
        return 0.0

    k = min(top_k, int(y_true.shape[1]))
    top_idx = np.argsort(deg_scores)[::-1][:k]

    correlations: list[float] = []
    for gene_idx in top_idx.tolist():
        true_col = y_true[:, gene_idx]
        pred_col = y_pred[:, gene_idx]
        if float(np.std(true_col)) <= 1e-8 or float(np.std(pred_col)) <= 1e-8:
            continue
        corr = float(np.corrcoef(true_col, pred_col)[0, 1])
        if np.isnan(corr):
            continue
        correlations.append(corr)

    if len(correlations) == 0:
        return 0.0
    return float(np.mean(correlations))


def _calibration_error(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_var: np.ndarray | None,
    n_bins: int,
) -> float:
    abs_error = np.abs(y_true - y_pred).reshape(-1)
    if y_var is None:
        pred_std = np.zeros_like(abs_error, dtype=np.float64)
    else:
        pred_std = np.sqrt(np.clip(y_var.reshape(-1), a_min=1e-12, a_max=None))

    if abs_error.size == 0:
        return 0.0

    if n_bins <= 1:
        return float(np.mean(np.abs(pred_std - abs_error)))

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(pred_std, quantiles)
    if float(np.max(edges) - np.min(edges)) <= 1e-12:
        return float(np.mean(np.abs(pred_std - abs_error)))

    weighted_error = 0.0
    total = float(abs_error.size)
    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (pred_std >= left) & (pred_std <= right)
        else:
            mask = (pred_std >= left) & (pred_std < right)
        count = int(np.sum(mask))
        if count == 0:
            continue
        bin_pred_std = float(np.mean(pred_std[mask]))
        bin_abs_error = float(np.mean(abs_error[mask]))
        weighted_error += (count / total) * abs(bin_pred_std - bin_abs_error)
    return float(weighted_error)


def _build_baseline_comparison_table(
    *,
    baseline_payload: dict[str, Any],
    model_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_mae = float(model_metrics["mae"])
    model_pearson = float(model_metrics["pearson_top_1000_degs"])

    baselines_raw = baseline_payload.get("baselines", [])
    if not isinstance(baselines_raw, list):
        return rows

    for raw in baselines_raw:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("task", "")) != "expression":
            continue
        metrics = raw.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        baseline_name = str(raw.get("baseline_name", "unknown"))
        baseline_mae = _safe_float(metrics.get("mae"), default=np.nan)
        baseline_pearson = _safe_float(metrics.get("pearson_mean_gene"), default=np.nan)
        delta_mae = (
            float(model_mae - baseline_mae) if np.isfinite(baseline_mae) else float("nan")
        )
        delta_pearson = (
            float(model_pearson - baseline_pearson)
            if np.isfinite(baseline_pearson)
            else float("nan")
        )
        model_beats_mae = bool(np.isfinite(baseline_mae) and model_mae < baseline_mae)
        rows.append(
            {
                "baseline_name": baseline_name,
                "baseline_mae": float(baseline_mae),
                "model_mae": model_mae,
                "delta_mae_model_minus_baseline": delta_mae,
                "model_beats_baseline_mae": model_beats_mae,
                "baseline_pearson_mean_gene": float(baseline_pearson),
                "model_pearson_top_1000_degs": model_pearson,
                "delta_pearson_model_minus_baseline": delta_pearson,
            }
        )

    rows.sort(key=lambda item: float(item["baseline_mae"]))
    return rows


def _build_failure_modes(
    *,
    model_metrics: dict[str, float],
    baseline_comparison: list[dict[str, Any]],
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    mae = float(model_metrics["mae"])
    pearson = float(model_metrics["pearson_top_1000_degs"])
    calibration = float(model_metrics["calibration_error"])
    if mae > config.failure_mae_threshold:
        failures.append(
            {
                "code": "high_mae",
                "severity": "critical",
                "metric": "mae",
                "value": mae,
                "threshold": config.failure_mae_threshold,
                "message": "Model MAE exceeds configured failure threshold.",
            }
        )
    if pearson < config.failure_pearson_threshold:
        failures.append(
            {
                "code": "low_deg_pearson",
                "severity": "warning",
                "metric": "pearson_top_1000_degs",
                "value": pearson,
                "threshold": config.failure_pearson_threshold,
                "message": "DEG-focused Pearson is below configured threshold.",
            }
        )
    if calibration > config.failure_calibration_threshold:
        failures.append(
            {
                "code": "poor_calibration",
                "severity": "warning",
                "metric": "calibration_error",
                "value": calibration,
                "threshold": config.failure_calibration_threshold,
                "message": "Calibration error exceeds configured threshold.",
            }
        )

    non_improved = [row for row in baseline_comparison if not bool(row["model_beats_baseline_mae"])]
    if non_improved:
        failures.append(
            {
                "code": "baseline_underperformance",
                "severity": "warning",
                "metric": "delta_mae_model_minus_baseline",
                "value": int(len(non_improved)),
                "threshold": 0,
                "message": (
                    "Model does not beat one or more baseline MAE values. "
                    f"Non-improved baselines: {len(non_improved)}."
                ),
            }
        )

    if len(failures) == 0:
        failures.append(
            {
                "code": "no_failure_modes_detected",
                "severity": "info",
                "metric": "n_failures",
                "value": 0,
                "threshold": 0,
                "message": "No configured failure mode was triggered.",
            }
        )
    return failures


def _safe_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
        if np.isnan(numeric):
            return default
        return numeric
    except (TypeError, ValueError):
        return default


def _render_model_card_markdown(payload: dict[str, Any]) -> str:
    metrics = cast(dict[str, Any], payload["metrics"])
    baseline_rows = cast(list[dict[str, Any]], payload["baseline_comparison"])
    failure_modes = cast(list[dict[str, Any]], payload["failure_modes"])
    metadata = cast(dict[str, Any], payload["metadata"])

    lines: list[str] = []
    lines.append("# TCPE Model Card")
    lines.append("")
    lines.append(f"- Schema version: `{payload['schema_version']}`")
    lines.append(f"- Generated at (UTC): `{payload['generated_at_utc']}`")
    lines.append(f"- Evaluation schema version: `{payload['evaluation_schema_version']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- MAE: `{float(metrics['mae']):.6f}`")
    lines.append(
        "- Pearson on top-1000 DEGs: "
        f"`{float(metrics['pearson_top_1000_degs']):.6f}`"
    )
    lines.append(f"- Calibration error: `{float(metrics['calibration_error']):.6f}`")
    lines.append("")
    lines.append("## Baseline Comparisons")
    lines.append("")
    lines.append(
        "| baseline | baseline_mae | model_mae | delta(model-baseline) | "
        "model_beats_baseline |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for row in baseline_rows:
        lines.append(
            f"| {row['baseline_name']} | {float(row['baseline_mae']):.6f} | "
            f"{float(row['model_mae']):.6f} | "
            f"{float(row['delta_mae_model_minus_baseline']):.6f} | "
            f"{bool(row['model_beats_baseline_mae'])} |"
        )
    lines.append("")
    lines.append("## Failure Modes")
    lines.append("")
    for mode in failure_modes:
        lines.append(
            f"- `{mode['severity']}` `{mode['code']}`: {mode['message']} "
            f"(value={mode['value']}, threshold={mode['threshold']})"
        )
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for key in sorted(metadata.keys()):
        lines.append(f"- {key}: `{metadata[key]}`")
    lines.append("")
    return "\n".join(lines)
