"""Phase 16 local validation ladder and promotion-gate enforcement."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from scipy.stats import rankdata

from tcpe.config import TCPEConfig
from tcpe.pipeline import PipelineConfig, PipelineModule, PipelineRunResult
from tcpe.runtime.run_context import (
    build_artifact_layout,
    ensure_artifact_layout,
    generate_run_id,
    write_run_manifest,
)

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

LOCAL_VALIDATION_SCHEMA_VERSION = "local_validation_ladder_v1"
ValidationStageName = Literal["synthetic", "adamson", "replogle_sample"]


class LocalValidationError(RuntimeError):
    """Raised when phase-16 local validation cannot complete."""


@dataclass(frozen=True)
class ValidationStageSpec:
    """Single stage definition for the phase-16 local ladder."""

    stage_name: ValidationStageName
    dataset_id: str
    runtime_limit_seconds: float
    source_uri: str | None = None
    synthetic_n_cells: int = 120
    synthetic_n_genes: int = 60
    synthetic_n_perturbations: int = 8

    def __post_init__(self) -> None:
        if self.stage_name not in ("synthetic", "adamson", "replogle_sample"):
            raise ValueError(f"Unsupported stage_name '{self.stage_name}'.")
        if str(self.dataset_id).strip() == "":
            raise ValueError("dataset_id must be non-empty.")
        if self.runtime_limit_seconds <= 0:
            raise ValueError("runtime_limit_seconds must be positive.")
        if self.synthetic_n_cells <= 0:
            raise ValueError("synthetic_n_cells must be positive.")
        if self.synthetic_n_genes <= 1:
            raise ValueError("synthetic_n_genes must be greater than 1.")
        if self.synthetic_n_perturbations < 2:
            raise ValueError("synthetic_n_perturbations must be at least 2.")


def _default_stage_specs() -> tuple[ValidationStageSpec, ...]:
    return (
        ValidationStageSpec(
            stage_name="synthetic",
            dataset_id="synthetic",
            runtime_limit_seconds=10.0 * 60.0,
            synthetic_n_cells=140,
            synthetic_n_genes=70,
            synthetic_n_perturbations=8,
        ),
        ValidationStageSpec(
            stage_name="adamson",
            dataset_id="adamson",
            runtime_limit_seconds=45.0 * 60.0,
        ),
        ValidationStageSpec(
            stage_name="replogle_sample",
            dataset_id="replogle_sample",
            runtime_limit_seconds=90.0 * 60.0,
        ),
    )


@dataclass(frozen=True)
class LocalValidationConfig:
    """Configuration for phase-16 validation ladder execution."""

    stages: tuple[ValidationStageSpec, ...] = field(default_factory=_default_stage_specs)
    seed: int = 42
    transport_latent_dim: int = 32
    transport_hidden_dim: int = 96
    transport_epochs: int = 5
    transport_batch_size: int = 48
    transport_learning_rate: float = 2e-3
    transport_sinkhorn_weight: float = 0.05
    causal_max_hvgs: int = 40
    causal_bootstrap_iterations: int = 0
    coverage_min_fraction: float = 0.85
    peak_ram_max_gb: float = 12.0
    required_baseline_wins: tuple[str, ...] = ("control_mean", "gene_level_mean")
    causal_min_auc: float = 0.55
    causal_min_f1: float = 0.05
    run_pytest: bool = True
    run_coverage: bool = True
    pytest_command: tuple[str, ...] | None = None
    coverage_command: tuple[str, ...] | None = None
    pytest_passed_override: bool | None = None
    coverage_fraction_override: float | None = None

    def __post_init__(self) -> None:
        if len(self.stages) == 0:
            raise ValueError("At least one validation stage is required.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if self.transport_latent_dim <= 0:
            raise ValueError("transport_latent_dim must be positive.")
        if self.transport_hidden_dim <= 0:
            raise ValueError("transport_hidden_dim must be positive.")
        if self.transport_epochs <= 0:
            raise ValueError("transport_epochs must be positive.")
        if self.transport_batch_size <= 0:
            raise ValueError("transport_batch_size must be positive.")
        if self.transport_learning_rate <= 0:
            raise ValueError("transport_learning_rate must be positive.")
        if self.transport_sinkhorn_weight < 0:
            raise ValueError("transport_sinkhorn_weight must be non-negative.")
        if self.causal_max_hvgs <= 1:
            raise ValueError("causal_max_hvgs must be greater than 1.")
        if self.causal_bootstrap_iterations < 0:
            raise ValueError("causal_bootstrap_iterations must be non-negative.")
        if not 0.0 <= self.coverage_min_fraction <= 1.0:
            raise ValueError("coverage_min_fraction must be in [0, 1].")
        if self.peak_ram_max_gb <= 0:
            raise ValueError("peak_ram_max_gb must be positive.")
        if not 0.0 <= self.causal_min_auc <= 1.0:
            raise ValueError("causal_min_auc must be in [0, 1].")
        if not 0.0 <= self.causal_min_f1 <= 1.0:
            raise ValueError("causal_min_f1 must be in [0, 1].")
        if self.coverage_fraction_override is not None and not (
            0.0 <= self.coverage_fraction_override <= 1.0
        ):
            raise ValueError("coverage_fraction_override must be in [0, 1] when provided.")


@dataclass(frozen=True)
class ValidationStageResult:
    """Runtime, memory, and metric outputs for one ladder stage."""

    stage_name: ValidationStageName
    dataset_id: str
    run_id: str
    pipeline_status: str
    runtime_seconds: float
    runtime_limit_seconds: float
    peak_memory_bytes: int
    peak_memory_gb: float
    memory_measurement_method: str
    metrics: dict[str, float]
    metric_deltas: dict[str, float]
    baseline_win_flags: dict[str, bool]
    causal_recovery: dict[str, float] | None
    step_artifacts: dict[str, dict[str, Any]]
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationGateResult:
    """Pass/fail result for one promotion gate."""

    gate_name: str
    passed: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalValidationReport:
    """Complete phase-16 local validation report."""

    schema_version: str
    generated_at_utc: str
    run_id: str
    pytest_passed: bool
    coverage_fraction: float
    config: dict[str, Any]
    stage_results: list[ValidationStageResult]
    gates: list[ValidationGateResult]
    all_gates_passed: bool
    eligible_for_cloud: bool
    issues: list[str]
    report_json_path: Path
    report_markdown_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage_results"] = [item.to_dict() for item in self.stage_results]
        payload["gates"] = [item.to_dict() for item in self.gates]
        payload["report_json_path"] = str(self.report_json_path)
        payload["report_markdown_path"] = str(self.report_markdown_path)
        return payload


class LocalValidationModule:
    """Phase-16 module for strict local promotion-gate validation."""

    def status(self) -> str:
        return "phase16_local_validation_ready"

    def run(
        self,
        *,
        config: TCPEConfig,
        validation_config: LocalValidationConfig | None = None,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
    ) -> LocalValidationReport:
        selected = validation_config if validation_config is not None else LocalValidationConfig()
        resolved_run_id = run_id or (
            f"{generate_run_id(config=config, command_group='pipeline')}-phase16"
        )
        root = (
            Path(output_dir)
            if output_dir is not None
            else config.paths.artifact_root / config.environment / "validation" / resolved_run_id
        )
        root.mkdir(parents=True, exist_ok=True)

        pytest_passed = self._resolve_pytest_gate(selected=selected, output_dir=root)
        coverage_fraction = self._resolve_coverage_fraction(selected=selected, output_dir=root)

        stage_results: list[ValidationStageResult] = []
        for stage in selected.stages:
            stage_results.append(
                self._run_stage(
                    config=config,
                    validation_config=selected,
                    stage=stage,
                    parent_run_id=resolved_run_id,
                )
            )

        gates = self._evaluate_gates(
            selected=selected,
            pytest_passed=pytest_passed,
            coverage_fraction=coverage_fraction,
            stage_results=stage_results,
        )
        issues = [gate.details for gate in gates if not gate.passed]
        all_gates_passed = all(gate.passed for gate in gates)

        report_json_path = root / "local_validation_report.json"
        report_markdown_path = root / "local_validation_report.md"
        report = LocalValidationReport(
            schema_version=LOCAL_VALIDATION_SCHEMA_VERSION,
            generated_at_utc=datetime.now(UTC).isoformat(),
            run_id=resolved_run_id,
            pytest_passed=pytest_passed,
            coverage_fraction=coverage_fraction,
            config=asdict(selected),
            stage_results=stage_results,
            gates=gates,
            all_gates_passed=all_gates_passed,
            eligible_for_cloud=all_gates_passed,
            issues=issues,
            report_json_path=report_json_path,
            report_markdown_path=report_markdown_path,
        )

        report_json_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_markdown_path.write_text(
            _render_markdown(report),
            encoding="utf-8",
        )
        return report

    def _run_stage(
        self,
        *,
        config: TCPEConfig,
        validation_config: LocalValidationConfig,
        stage: ValidationStageSpec,
        parent_run_id: str,
    ) -> ValidationStageResult:
        stage_run_id = f"{parent_run_id}-{stage.stage_name}"
        layout = build_artifact_layout(config=config, command_group="pipeline", run_id=stage_run_id)
        ensure_artifact_layout(layout)
        write_run_manifest(
            layout=layout,
            config=config,
            command_group="pipeline",
            run_id=stage_run_id,
            config_path=Path("<phase16-local-validation>"),
        )

        dataset_source_uris = (
            {stage.dataset_id: stage.source_uri}
            if stage.source_uri is not None
            else {}
        )
        pipeline_config = PipelineConfig(
            dataset_id=stage.dataset_id,
            synthetic_n_cells=stage.synthetic_n_cells,
            synthetic_n_genes=stage.synthetic_n_genes,
            synthetic_n_perturbations=stage.synthetic_n_perturbations,
            dataset_source_uris=dataset_source_uris,
            seed=validation_config.seed,
            transport_latent_dim=validation_config.transport_latent_dim,
            transport_hidden_dim=validation_config.transport_hidden_dim,
            transport_epochs=validation_config.transport_epochs,
            transport_batch_size=validation_config.transport_batch_size,
            transport_learning_rate=validation_config.transport_learning_rate,
            transport_sinkhorn_weight=validation_config.transport_sinkhorn_weight,
            causal_max_hvgs=validation_config.causal_max_hvgs,
            causal_bootstrap_iterations=validation_config.causal_bootstrap_iterations,
        )

        module = PipelineModule()
        pipeline_result, runtime_seconds, peak_memory_bytes, memory_method = _run_with_memory_probe(
            lambda: module.run(
                config=config,
                layout=layout,
                run_id=stage_run_id,
                pipeline_config=pipeline_config,
                resume=False,
            )
        )
        if not isinstance(pipeline_result, PipelineRunResult):
            raise LocalValidationError(
                f"Unexpected pipeline result type for stage '{stage.stage_name}'."
            )

        metrics, metric_deltas, baseline_win_flags = _extract_evaluation_metrics(
            pipeline_result.step_artifacts
        )
        causal_recovery = None
        if stage.stage_name == "synthetic":
            causal_recovery = _extract_causal_recovery_metrics(pipeline_result.step_artifacts)

        issues: list[str] = []
        if pipeline_result.status != "completed":
            issues.append(f"Pipeline stage did not complete: status={pipeline_result.status}.")
        if runtime_seconds > stage.runtime_limit_seconds:
            issues.append(
                f"Runtime {runtime_seconds:.2f}s exceeds "
                f"stage limit {stage.runtime_limit_seconds:.2f}s."
            )

        return ValidationStageResult(
            stage_name=stage.stage_name,
            dataset_id=stage.dataset_id,
            run_id=stage_run_id,
            pipeline_status=pipeline_result.status,
            runtime_seconds=runtime_seconds,
            runtime_limit_seconds=stage.runtime_limit_seconds,
            peak_memory_bytes=peak_memory_bytes,
            peak_memory_gb=peak_memory_bytes / float(1024**3),
            memory_measurement_method=memory_method,
            metrics=metrics,
            metric_deltas=metric_deltas,
            baseline_win_flags=baseline_win_flags,
            causal_recovery=causal_recovery,
            step_artifacts=cast(dict[str, dict[str, Any]], pipeline_result.step_artifacts),
            issues=issues,
        )

    def _resolve_pytest_gate(
        self,
        *,
        selected: LocalValidationConfig,
        output_dir: Path,
    ) -> bool:
        if selected.pytest_passed_override is not None:
            return bool(selected.pytest_passed_override)
        if not selected.run_pytest:
            return False

        command = (
            selected.pytest_command
            if selected.pytest_command is not None
            else (sys.executable, "-m", "pytest", "-q", "-ra")
        )
        result = _run_command(command=command, cwd=Path.cwd())
        (output_dir / "pytest_stdout.log").write_text(result.stdout, encoding="utf-8")
        (output_dir / "pytest_stderr.log").write_text(result.stderr, encoding="utf-8")
        return result.returncode == 0

    def _resolve_coverage_fraction(
        self,
        *,
        selected: LocalValidationConfig,
        output_dir: Path,
    ) -> float:
        if selected.coverage_fraction_override is not None:
            return float(selected.coverage_fraction_override)
        if not selected.run_coverage:
            return 0.0

        coverage_json = output_dir / "coverage.json"
        default_command = (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=src",
            f"--cov-report=json:{coverage_json}",
        )
        command = (
            selected.coverage_command
            if selected.coverage_command is not None
            else default_command
        )
        result = _run_command(command=command, cwd=Path.cwd())
        (output_dir / "coverage_stdout.log").write_text(result.stdout, encoding="utf-8")
        (output_dir / "coverage_stderr.log").write_text(result.stderr, encoding="utf-8")

        if coverage_json.exists():
            payload = json.loads(coverage_json.read_text(encoding="utf-8"))
            totals = payload.get("totals", {})
            percent = float(totals.get("percent_covered", 0.0))
            return percent / 100.0
        return _parse_coverage_from_stdout(result.stdout)

    def _evaluate_gates(
        self,
        *,
        selected: LocalValidationConfig,
        pytest_passed: bool,
        coverage_fraction: float,
        stage_results: list[ValidationStageResult],
    ) -> list[ValidationGateResult]:
        stage_by_name = {stage.stage_name: stage for stage in stage_results}
        gates: list[ValidationGateResult] = []
        gates.append(
            ValidationGateResult(
                gate_name="tests_pass_100",
                passed=pytest_passed,
                details=(
                    "Unit + integration tests passed."
                    if pytest_passed
                    else "Unit + integration tests did not pass."
                ),
            )
        )
        coverage_passed = coverage_fraction >= selected.coverage_min_fraction
        gates.append(
            ValidationGateResult(
                gate_name="coverage_at_least_85_percent",
                passed=coverage_passed,
                details=(
                    f"Coverage={coverage_fraction:.3f}, "
                    f"required>={selected.coverage_min_fraction:.3f}."
                ),
            )
        )

        for stage in stage_results:
            gates.append(
                ValidationGateResult(
                    gate_name=f"{stage.stage_name}_runtime_limit",
                    passed=stage.runtime_seconds <= stage.runtime_limit_seconds,
                    details=(
                        f"{stage.stage_name} runtime={stage.runtime_seconds:.2f}s, "
                        f"limit={stage.runtime_limit_seconds:.2f}s."
                    ),
                )
            )

        memory_limit_bytes = int(selected.peak_ram_max_gb * float(1024**3))
        max_stage_memory = max((stage.peak_memory_bytes for stage in stage_results), default=0)
        gates.append(
            ValidationGateResult(
                gate_name="peak_ram_under_limit",
                passed=max_stage_memory <= memory_limit_bytes,
                details=(
                    f"Peak observed={max_stage_memory / float(1024**3):.3f}GB, "
                    f"limit={selected.peak_ram_max_gb:.3f}GB."
                ),
            )
        )

        baseline_required = set(selected.required_baseline_wins)
        baseline_gate_passed = True
        baseline_notes: list[str] = []
        for stage_name in ("adamson", "replogle_sample"):
            stage_result = stage_by_name.get(cast(ValidationStageName, stage_name))
            if stage_result is None:
                baseline_gate_passed = False
                baseline_notes.append(f"Stage `{stage_name}` missing.")
                continue
            stage_flags = stage_result.baseline_win_flags
            missing = sorted(
                name for name in baseline_required if not bool(stage_flags.get(name, False))
            )
            if missing:
                baseline_gate_passed = False
                baseline_notes.append(f"{stage_name} missing required wins: {', '.join(missing)}.")
        if len(baseline_notes) == 0:
            baseline_notes.append("Model beats required baselines on Adamson and Replogle stages.")
        gates.append(
            ValidationGateResult(
                gate_name="baseline_dominance_on_real_datasets",
                passed=baseline_gate_passed,
                details=" ".join(baseline_notes),
            )
        )

        synthetic = stage_by_name.get("synthetic")
        causal_gate_passed = False
        causal_details = "Synthetic stage missing."
        if synthetic is not None and synthetic.causal_recovery is not None:
            auc = float(synthetic.causal_recovery.get("auc", 0.0))
            f1 = float(synthetic.causal_recovery.get("f1", 0.0))
            causal_gate_passed = (auc >= selected.causal_min_auc) and (f1 >= selected.causal_min_f1)
            causal_details = (
                f"Synthetic causal recovery auc={auc:.4f} (min={selected.causal_min_auc:.4f}), "
                f"f1={f1:.4f} (min={selected.causal_min_f1:.4f})."
            )
        gates.append(
            ValidationGateResult(
                gate_name="synthetic_causal_graph_recovery",
                passed=causal_gate_passed,
                details=causal_details,
            )
        )
        return gates


@dataclass(frozen=True)
class _CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _run_command(command: tuple[str, ...], cwd: Path) -> _CommandResult:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return _CommandResult(
        returncode=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def _parse_coverage_from_stdout(stdout: str) -> float:
    lines = stdout.splitlines()
    for line in reversed(lines):
        if "TOTAL" not in line:
            continue
        tokens = line.split()
        for token in reversed(tokens):
            if token.endswith("%"):
                try:
                    return float(token.rstrip("%")) / 100.0
                except ValueError:
                    continue
    return 0.0


def _run_with_memory_probe(
    runner: Any,
) -> tuple[Any, float, int, str]:
    start = time.perf_counter()
    peak_memory_bytes = 0
    measurement_method = "tracemalloc"

    try:
        import psutil
    except ImportError:
        tracemalloc.start()
        result = runner()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        runtime_seconds = float(time.perf_counter() - start)
        return result, runtime_seconds, int(peak), measurement_method

    process = psutil.Process()
    stop_event = Event()

    def monitor() -> None:
        nonlocal peak_memory_bytes
        while not stop_event.is_set():
            try:
                rss = int(process.memory_info().rss)
                peak_memory_bytes = max(peak_memory_bytes, rss)
            except Exception:
                pass
            stop_event.wait(0.05)

    monitor_thread = Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        result = runner()
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)

    try:
        peak_memory_bytes = max(peak_memory_bytes, int(process.memory_info().rss))
    except Exception:
        pass
    runtime_seconds = float(time.perf_counter() - start)
    return result, runtime_seconds, int(peak_memory_bytes), "psutil_rss"


def _extract_evaluation_metrics(
    step_artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, bool]]:
    evaluate_payload = step_artifacts.get("evaluate")
    if evaluate_payload is None:
        raise LocalValidationError("Missing `evaluate` step artifacts.")
    report_path = evaluate_payload.get("evaluation_report_json_path")
    if not isinstance(report_path, str):
        raise LocalValidationError("Evaluate step missing `evaluation_report_json_path`.")
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LocalValidationError("Evaluation report payload root must be an object.")

    metrics_raw = payload.get("metrics", {})
    if not isinstance(metrics_raw, dict):
        raise LocalValidationError("Evaluation report `metrics` must be an object.")
    metrics = {
        "mae": float(metrics_raw.get("mae", 0.0)),
        "pearson_top_1000_degs": float(metrics_raw.get("pearson_top_1000_degs", 0.0)),
        "calibration_error": float(metrics_raw.get("calibration_error", 0.0)),
    }

    rows = payload.get("baseline_comparison", [])
    if not isinstance(rows, list):
        raise LocalValidationError("Evaluation report `baseline_comparison` must be a list.")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_name[str(row.get("baseline_name", "unknown"))] = row

    metric_deltas: dict[str, float] = {}
    baseline_win_flags: dict[str, bool] = {}
    for baseline_name in ("control_mean", "gene_level_mean", "linear_regression"):
        row = by_name.get(baseline_name)
        if row is None:
            baseline_win_flags[baseline_name] = False
            continue
        baseline_win_flags[baseline_name] = bool(row.get("model_beats_baseline_mae", False))
        delta = row.get("delta_mae_model_minus_baseline")
        if delta is not None:
            metric_deltas[f"delta_mae_vs_{baseline_name}"] = float(delta)
    return metrics, metric_deltas, baseline_win_flags


def _extract_causal_recovery_metrics(
    step_artifacts: dict[str, dict[str, Any]],
) -> dict[str, float] | None:
    causal_payload = step_artifacts.get("causal")
    if causal_payload is None:
        return None
    adata_path_raw = causal_payload.get("adata_path")
    adjacency_path_raw = causal_payload.get("causal_adjacency_npz_path")
    metadata_path_raw = causal_payload.get("causal_metadata_json_path")
    path_candidates = (adata_path_raw, adjacency_path_raw, metadata_path_raw)
    if not all(isinstance(item, str) for item in path_candidates):
        return None

    adata_path = Path(cast(str, adata_path_raw))
    adjacency_path = Path(cast(str, adjacency_path_raw))
    metadata_path = Path(cast(str, metadata_path_raw))
    if not (adata_path.exists() and adjacency_path.exists() and metadata_path.exists()):
        return None

    adata = _require_anndata().read_h5ad(adata_path)
    truth_raw = adata.uns.get("synthetic_ground_truth_adjacency")
    if truth_raw is None:
        return None
    truth = np.asarray(truth_raw, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1]:
        return None

    predicted = np.load(adjacency_path)["adjacency"].astype(np.float64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        return None
    gene_ids = metadata.get("gene_ids", [])
    if not isinstance(gene_ids, list):
        return None
    if "gene_id" in adata.var.columns:
        var_gene_ids = adata.var["gene_id"].astype(str).tolist()
    else:
        var_gene_ids = adata.var_names.astype(str).tolist()
    index_by_id = {gene_id: idx for idx, gene_id in enumerate(var_gene_ids)}
    selected_indices: list[int] = []
    for gene_id in gene_ids:
        key = str(gene_id)
        if key not in index_by_id:
            return None
        selected_indices.append(int(index_by_id[key]))

    truth_subset = truth[np.ix_(selected_indices, selected_indices)]
    if truth_subset.shape != predicted.shape:
        return None

    mask = ~np.eye(predicted.shape[0], dtype=bool)
    y_true = np.abs(truth_subset[mask]) > 1e-12
    scores = np.abs(predicted[mask])
    auc = _binary_auroc(y_true=y_true, scores=scores)
    f1 = _best_f1(y_true=y_true, scores=scores)
    return {"auc": auc, "f1": f1}


def _binary_auroc(*, y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_true))
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(np.sum(ranks[y_true]))
    auc = (rank_sum_pos - (positives * (positives + 1) / 2.0)) / (positives * negatives)
    return float(max(0.0, min(1.0, auc)))


def _best_f1(*, y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_true))
    if positives == 0:
        return 0.0
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return 0.0
    best = 0.0
    for threshold in thresholds.tolist():
        pred = scores >= float(threshold)
        tp = int(np.sum(pred & y_true))
        fp = int(np.sum(pred & ~y_true))
        fn = int(np.sum(~pred & y_true))
        if tp == 0:
            continue
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall <= 1e-12:
            continue
        f1 = (2.0 * precision * recall) / (precision + recall)
        best = max(best, float(f1))
    return best


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - dependency behavior.
        raise LocalValidationError("`anndata` is required for local validation.") from exc
    return ad


def _render_markdown(report: LocalValidationReport) -> str:
    lines: list[str] = []
    lines.append("# Phase 16 Local Validation Report")
    lines.append("")
    lines.append(f"- Schema: `{report.schema_version}`")
    lines.append(f"- Generated at (UTC): `{report.generated_at_utc}`")
    lines.append(f"- Run id: `{report.run_id}`")
    lines.append(f"- Pytest gate passed: `{report.pytest_passed}`")
    lines.append(f"- Coverage fraction: `{report.coverage_fraction:.4f}`")
    lines.append(f"- Eligible for cloud: `{report.eligible_for_cloud}`")
    lines.append("")
    lines.append("## Stage Results")
    lines.append("")
    lines.append("| stage | dataset | runtime_s | limit_s | peak_ram_gb | status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for stage in report.stage_results:
        lines.append(
            f"| {stage.stage_name} | {stage.dataset_id} | {stage.runtime_seconds:.2f} | "
            f"{stage.runtime_limit_seconds:.2f} | {stage.peak_memory_gb:.3f} | "
            f"{stage.pipeline_status} |"
        )
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for gate in report.gates:
        lines.append(f"- `{gate.gate_name}` -> `{gate.passed}`: {gate.details}")
    lines.append("")
    if report.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in report.issues:
            lines.append(f"- {issue}")
    else:
        lines.append("## Issues")
        lines.append("")
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)
