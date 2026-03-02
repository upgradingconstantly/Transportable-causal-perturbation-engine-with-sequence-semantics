"""Context-shift split engine for TCPE Phase 13."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from tcpe.anndata_schema import PERTURBATIONS_UNS_KEY, SCHEMA_VERSION_UNS_KEY

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

ShiftType = Literal["cell_type", "locus", "dose_strength", "protocol"]
ALL_SHIFT_TYPES: tuple[ShiftType, ...] = ("cell_type", "locus", "dose_strength", "protocol")

SHIFT_SPLIT_SCHEMA_VERSION = "shift_split_manifest_v1"
SHIFT_SPLITS_UNS_KEY = "context_shift_splits"


class ShiftSplitError(RuntimeError):
    """Raised when context-shift split generation fails."""


class ShiftSplitLeakageError(ShiftSplitError):
    """Raised when a split manifest fails leakage or integrity validation."""


@dataclass(frozen=True)
class ShiftSplitConfig:
    """Configuration for deterministic context-shift split generation."""

    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    min_cells_per_split: int = 32

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1).")
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0, 1).")
        if (self.val_fraction + self.test_fraction) >= 1.0:
            raise ValueError("val_fraction + test_fraction must be less than 1.0.")
        if self.min_cells_per_split <= 0:
            raise ValueError("min_cells_per_split must be positive.")


@dataclass(frozen=True)
class ShiftSplitManifest:
    """Serializable split manifest with provenance and leakage audit."""

    schema_version: str
    shift_type: ShiftType
    base_seed: int
    effective_seed: int
    config: dict[str, Any]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    train_groups: list[str]
    val_groups: list[str]
    test_groups: list[str]
    fallback_applied: bool
    fallback_reason: str | None
    leakage_report: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ShiftSplitManifest:
        shift_type = str(payload.get("shift_type", "")).strip()
        if shift_type not in ALL_SHIFT_TYPES:
            raise ShiftSplitError(
                f"Unsupported shift_type '{shift_type}' in split manifest payload."
            )
        typed_shift = cast(ShiftType, shift_type)
        return cls(
            schema_version=str(payload.get("schema_version", SHIFT_SPLIT_SCHEMA_VERSION)),
            shift_type=typed_shift,
            base_seed=int(payload.get("base_seed", 0)),
            effective_seed=int(payload.get("effective_seed", 0)),
            config=dict(payload.get("config", {})),
            train_indices=[int(item) for item in payload.get("train_indices", [])],
            val_indices=[int(item) for item in payload.get("val_indices", [])],
            test_indices=[int(item) for item in payload.get("test_indices", [])],
            train_groups=[str(item) for item in payload.get("train_groups", [])],
            val_groups=[str(item) for item in payload.get("val_groups", [])],
            test_groups=[str(item) for item in payload.get("test_groups", [])],
            fallback_applied=bool(payload.get("fallback_applied", False)),
            fallback_reason=(
                None
                if payload.get("fallback_reason") is None
                else str(payload.get("fallback_reason"))
            ),
            leakage_report=dict(payload.get("leakage_report", {})),
            provenance=dict(payload.get("provenance", {})),
        )


class ShiftSplitGenerator(Protocol):
    """Public interface for deterministic context-shift split generators."""

    def generate_split(
        self,
        adata: AnnData,
        *,
        shift_type: ShiftType,
        config: ShiftSplitConfig | None = None,
    ) -> ShiftSplitManifest:
        """Generate one deterministic split manifest."""


@dataclass(frozen=True)
class _GroupAssignment:
    train_groups: tuple[str, ...]
    val_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    fallback_applied: bool
    fallback_reason: str | None


class ContextShiftSplitModule:
    """Phase 13 context-shift split module with leakage checks and manifests."""

    def status(self) -> str:
        return "phase13_context_shift_ready"

    def generate_split(
        self,
        adata: AnnData,
        *,
        shift_type: ShiftType,
        config: ShiftSplitConfig | None = None,
    ) -> ShiftSplitManifest:
        selected = config if config is not None else ShiftSplitConfig()
        groups = _resolve_shift_groups(adata=adata, shift_type=shift_type)
        effective_seed = _derive_effective_seed(base_seed=selected.seed, shift_type=shift_type)
        assignment = _assign_groups(
            groups=groups,
            config=selected,
            effective_seed=effective_seed,
        )

        train_idx = _indices_for_groups(groups=groups, selected_groups=assignment.train_groups)
        val_idx = _indices_for_groups(groups=groups, selected_groups=assignment.val_groups)
        test_idx = _indices_for_groups(groups=groups, selected_groups=assignment.test_groups)

        provenance = _build_provenance(
            adata=adata,
            shift_type=shift_type,
            groups=groups,
            assignment=assignment,
            selected=selected,
            effective_seed=effective_seed,
        )
        preliminary = ShiftSplitManifest(
            schema_version=SHIFT_SPLIT_SCHEMA_VERSION,
            shift_type=shift_type,
            base_seed=selected.seed,
            effective_seed=effective_seed,
            config={
                "seed": selected.seed,
                "val_fraction": selected.val_fraction,
                "test_fraction": selected.test_fraction,
                "min_cells_per_split": selected.min_cells_per_split,
            },
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_groups=list(assignment.train_groups),
            val_groups=list(assignment.val_groups),
            test_groups=list(assignment.test_groups),
            fallback_applied=assignment.fallback_applied,
            fallback_reason=assignment.fallback_reason,
            leakage_report={},
            provenance=provenance,
        )
        leakage_report = self.validate_manifest(
            adata=adata,
            manifest=preliminary,
            raise_on_error=False,
        )
        manifest = ShiftSplitManifest(
            **{
                **preliminary.to_dict(),
                "leakage_report": leakage_report,
            }
        )
        self.validate_manifest(adata=adata, manifest=manifest, raise_on_error=True)
        return manifest

    def generate_all_splits(
        self,
        adata: AnnData,
        *,
        config: ShiftSplitConfig | None = None,
        shift_types: tuple[ShiftType, ...] = ALL_SHIFT_TYPES,
    ) -> dict[ShiftType, ShiftSplitManifest]:
        manifests: dict[ShiftType, ShiftSplitManifest] = {}
        for shift_type in shift_types:
            manifests[shift_type] = self.generate_split(adata, shift_type=shift_type, config=config)
        return manifests

    def annotate_anndata(
        self,
        adata: AnnData,
        manifests: dict[ShiftType, ShiftSplitManifest],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            shift_type: manifest.to_dict() for shift_type, manifest in manifests.items()
        }
        adata.uns[SHIFT_SPLITS_UNS_KEY] = payload
        return payload

    def persist_manifest(self, manifest: ShiftSplitManifest, output_path: str | Path) -> Path:
        path = Path(output_path)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{manifest.shift_type}_split_manifest.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def persist_manifests(
        self,
        manifests: dict[ShiftType, ShiftSplitManifest],
        *,
        output_dir: str | Path,
        file_prefix: str = "context_shift",
    ) -> dict[ShiftType, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[ShiftType, Path] = {}
        for shift_type, manifest in manifests.items():
            filename = f"{file_prefix}_{shift_type}_manifest.json"
            paths[shift_type] = self.persist_manifest(manifest, root / filename)
        return paths

    def load_manifest(self, path: str | Path) -> ShiftSplitManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ShiftSplitError(f"Split manifest payload is not a JSON object: {path}")
        return ShiftSplitManifest.from_dict(cast(dict[str, Any], payload))

    def validate_manifest(
        self,
        *,
        adata: AnnData,
        manifest: ShiftSplitManifest,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        groups = _resolve_shift_groups(adata=adata, shift_type=manifest.shift_type)
        n_obs = int(adata.n_obs)

        train = np.asarray(manifest.train_indices, dtype=np.int64)
        val = np.asarray(manifest.val_indices, dtype=np.int64)
        test = np.asarray(manifest.test_indices, dtype=np.int64)
        partitions = {"train": train, "val": val, "test": test}

        errors: list[str] = []
        out_of_range_counts: dict[str, int] = {}
        duplicate_counts: dict[str, int] = {}
        partition_cell_counts: dict[str, int] = {}
        partition_group_counts: dict[str, int] = {}

        for name, indices in partitions.items():
            unique_indices = np.unique(indices)
            duplicate_counts[name] = int(indices.size - unique_indices.size)
            if duplicate_counts[name] > 0:
                errors.append(f"Partition `{name}` contains duplicate cell indices.")

            out_of_range_mask = (indices < 0) | (indices >= n_obs)
            out_of_range_counts[name] = int(np.sum(out_of_range_mask))
            if out_of_range_counts[name] > 0:
                errors.append(f"Partition `{name}` contains out-of-range cell indices.")

            valid_indices = indices[~out_of_range_mask]
            partition_cell_counts[name] = int(valid_indices.size)
            if valid_indices.size > 0:
                part_groups = groups[valid_indices]
                partition_group_counts[name] = int(np.unique(part_groups).size)
            else:
                partition_group_counts[name] = 0

        train_val_overlap = sorted(set(train.tolist()) & set(val.tolist()))
        train_test_overlap = sorted(set(train.tolist()) & set(test.tolist()))
        val_test_overlap = sorted(set(val.tolist()) & set(test.tolist()))
        if train_val_overlap:
            errors.append("Cell-index leakage between train and val partitions.")
        if train_test_overlap:
            errors.append("Cell-index leakage between train and test partitions.")
        if val_test_overlap:
            errors.append("Cell-index leakage between val and test partitions.")

        assigned = set(train.tolist()) | set(val.tolist()) | set(test.tolist())
        missing = sorted(set(range(n_obs)) - assigned)
        if missing:
            errors.append("Split manifest does not cover all cells exactly once.")

        train_group_set = set(groups[train].tolist()) if train.size > 0 else set()
        val_group_set = set(groups[val].tolist()) if val.size > 0 else set()
        test_group_set = set(groups[test].tolist()) if test.size > 0 else set()

        overlap_train_val_groups = sorted(train_group_set & val_group_set)
        overlap_train_test_groups = sorted(train_group_set & test_group_set)
        overlap_val_test_groups = sorted(val_group_set & test_group_set)
        if overlap_train_val_groups:
            errors.append("Shift-group leakage between train and val partitions.")
        if overlap_train_test_groups:
            errors.append("Shift-group leakage between train and test partitions.")
        if overlap_val_test_groups:
            errors.append("Shift-group leakage between val and test partitions.")

        leakage_detected = bool(
            train_val_overlap
            or train_test_overlap
            or val_test_overlap
            or overlap_train_val_groups
            or overlap_train_test_groups
            or overlap_val_test_groups
        )

        report: dict[str, Any] = {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "leakage_detected": leakage_detected,
            "cell_index_overlap": {
                "train_val": train_val_overlap,
                "train_test": train_test_overlap,
                "val_test": val_test_overlap,
            },
            "group_overlap": {
                "train_val": overlap_train_val_groups,
                "train_test": overlap_train_test_groups,
                "val_test": overlap_val_test_groups,
            },
            "partition_cell_counts": partition_cell_counts,
            "partition_group_counts": partition_group_counts,
            "duplicate_index_counts": duplicate_counts,
            "out_of_range_counts": out_of_range_counts,
            "coverage": {
                "n_obs": n_obs,
                "n_assigned_unique": int(len(assigned)),
                "n_missing": int(len(missing)),
            },
        }
        if raise_on_error and not report["is_valid"]:
            raise ShiftSplitLeakageError(
                "Split manifest validation failed:\n- " + "\n- ".join(errors)
            )
        return report


def _assign_groups(
    *,
    groups: np.ndarray,
    config: ShiftSplitConfig,
    effective_seed: int,
) -> _GroupAssignment:
    unique_groups = sorted({str(value) for value in groups.tolist()})
    if len(unique_groups) == 0:
        raise ShiftSplitError("Cannot generate split: no shift groups are available.")

    rng = np.random.default_rng(effective_seed)
    permuted = np.asarray(unique_groups, dtype=object)
    rng.shuffle(permuted)
    ordered = [str(item) for item in permuted.tolist()]
    counts = _count_cells_by_group(groups=groups)

    fallback_applied = False
    fallback_reason: str | None = None

    if len(ordered) >= 3:
        n_groups = len(ordered)
        n_val = max(1, int(round(config.val_fraction * n_groups)))
        n_test = max(1, int(round(config.test_fraction * n_groups)))
        while (n_val + n_test) >= n_groups:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        split_point_train = n_groups - n_val - n_test
        split_point_val = n_groups - n_test
        train_groups = tuple(ordered[:split_point_train])
        val_groups = tuple(ordered[split_point_train:split_point_val])
        test_groups = tuple(ordered[split_point_val:])
    elif len(ordered) == 2:
        train_groups = (ordered[0],)
        val_groups = ()
        test_groups = (ordered[1],)
        fallback_applied = True
        fallback_reason = "insufficient_shift_groups_for_three_way_split"
    else:
        train_groups = (ordered[0],)
        val_groups = ()
        test_groups = ()
        fallback_applied = True
        fallback_reason = "single_shift_group_only_train_split_possible"

    train_count = _count_for_selected_groups(counts=counts, selected=train_groups)
    val_count = _count_for_selected_groups(counts=counts, selected=val_groups)
    test_count = _count_for_selected_groups(counts=counts, selected=test_groups)
    if (
        (val_count > 0 and val_count < config.min_cells_per_split)
        or (test_count > 0 and test_count < config.min_cells_per_split)
        or (train_count > 0 and train_count < config.min_cells_per_split)
    ):
        fallback_applied = True
        fallback_reason = "undersized_partition_fallback_applied"
        if val_groups:
            train_groups = tuple(sorted(set(train_groups) | set(val_groups)))
            val_groups = ()

        train_count = _count_for_selected_groups(counts=counts, selected=train_groups)
        test_count = _count_for_selected_groups(counts=counts, selected=test_groups)

        if test_count > 0 and test_count < config.min_cells_per_split:
            groups_by_size = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if len(groups_by_size) >= 2:
                selected_test = groups_by_size[1][0]
                test_groups = (selected_test,)
                train_groups = tuple(
                    item[0] for item in groups_by_size if item[0] != selected_test
                )
            else:
                test_groups = ()
                train_groups = tuple(item[0] for item in groups_by_size)

        train_count = _count_for_selected_groups(counts=counts, selected=train_groups)
        if train_count < config.min_cells_per_split:
            train_groups = tuple(group for group in unique_groups)
            val_groups = ()
            test_groups = ()
            fallback_reason = "dataset_too_small_for_holdout_all_assigned_to_train"

    return _GroupAssignment(
        train_groups=train_groups,
        val_groups=val_groups,
        test_groups=test_groups,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
    )


def _indices_for_groups(*, groups: np.ndarray, selected_groups: tuple[str, ...]) -> list[int]:
    if len(selected_groups) == 0:
        return []
    mask = np.isin(groups, np.asarray(selected_groups, dtype=object))
    return [int(idx) for idx in np.flatnonzero(mask).tolist()]


def _count_cells_by_group(*, groups: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(groups, return_counts=True)
    return {
        str(group): int(count)
        for group, count in zip(values.tolist(), counts.tolist(), strict=True)
    }


def _count_for_selected_groups(*, counts: dict[str, int], selected: tuple[str, ...]) -> int:
    return int(sum(counts.get(group, 0) for group in selected))


def _build_provenance(
    *,
    adata: AnnData,
    shift_type: ShiftType,
    groups: np.ndarray,
    assignment: _GroupAssignment,
    selected: ShiftSplitConfig,
    effective_seed: int,
) -> dict[str, Any]:
    counts = _count_cells_by_group(groups=groups)
    dataset_name = None
    if "dataset_name" in adata.uns:
        dataset_name = str(adata.uns["dataset_name"])

    fingerprint = _dataset_fingerprint(adata=adata, groups=groups)
    return {
        "generator": "ContextShiftSplitModule",
        "schema_version": SHIFT_SPLIT_SCHEMA_VERSION,
        "shift_type": shift_type,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "dataset_name": dataset_name,
        "anndata_schema_version": str(adata.uns.get(SCHEMA_VERSION_UNS_KEY, "unknown")),
        "group_count": int(len(counts)),
        "group_cell_counts": counts,
        "train_groups": list(assignment.train_groups),
        "val_groups": list(assignment.val_groups),
        "test_groups": list(assignment.test_groups),
        "base_seed": int(selected.seed),
        "effective_seed": int(effective_seed),
        "dataset_fingerprint_sha256": fingerprint,
    }


def _dataset_fingerprint(*, adata: AnnData, groups: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(adata.n_obs)).encode("utf-8"))
    digest.update(str(int(adata.n_vars)).encode("utf-8"))

    obs_columns = ("cell_id", "perturbation_id", "cell_type", "protocol")
    for column in obs_columns:
        if column not in adata.obs.columns:
            continue
        values = adata.obs[column].astype(str).tolist()
        digest.update(column.encode("utf-8"))
        digest.update("||".join(values).encode("utf-8"))

    digest.update("||".join(groups.tolist()).encode("utf-8"))
    return digest.hexdigest()


def _derive_effective_seed(*, base_seed: int, shift_type: ShiftType) -> int:
    payload = f"{base_seed}|{shift_type}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    derived = int(digest, 16)
    return int((base_seed + derived) % (2**32 - 1))


def _resolve_shift_groups(*, adata: AnnData, shift_type: ShiftType) -> np.ndarray:
    if shift_type == "cell_type":
        return _require_obs_text_column(adata=adata, column="cell_type")
    if shift_type == "protocol":
        return _require_obs_text_column(adata=adata, column="protocol")
    if shift_type == "locus":
        return _resolve_locus_groups(adata=adata)
    if shift_type == "dose_strength":
        return _resolve_dose_strength_groups(adata=adata)
    raise ShiftSplitError(f"Unsupported shift type: {shift_type}")


def _require_obs_text_column(*, adata: AnnData, column: str) -> np.ndarray:
    if column not in adata.obs.columns:
        raise ShiftSplitError(f"AnnData `.obs` is missing required column `{column}`.")
    values = adata.obs[column].astype(str).str.strip()
    if (values == "").any():
        raise ShiftSplitError(f"AnnData `.obs['{column}']` contains empty values.")
    return cast(np.ndarray, values.to_numpy(dtype=object))


def _resolve_locus_groups(*, adata: AnnData) -> np.ndarray:
    perturbation_ids = _require_obs_text_column(adata=adata, column="perturbation_id")
    perturbation_table = _coerce_perturbation_table(adata.uns.get(PERTURBATIONS_UNS_KEY))
    if perturbation_table is None:
        raise ShiftSplitError(
            "AnnData `.uns['perturbations']` is missing or malformed for locus shift splitting."
        )
    by_id: dict[str, dict[str, Any]] = {}
    for _, row in perturbation_table.iterrows():
        perturbation_id = str(row.get("perturbation_id", "")).strip()
        if perturbation_id == "":
            continue
        by_id[perturbation_id] = cast(dict[str, Any], row.to_dict())

    labels: list[str] = []
    for perturbation_id in perturbation_ids.tolist():
        record = by_id.get(str(perturbation_id))
        if record is None:
            labels.append(f"perturbation:{perturbation_id}")
            continue
        chrom = str(record.get("chrom", "unknown")).strip()
        strand = str(record.get("strand", ".")).strip() or "."
        start = _coerce_int(record.get("start"))
        end = _coerce_int(record.get("end"))
        if chrom == "" or chrom.lower() == "unknown" or start is None or end is None:
            labels.append(f"perturbation:{perturbation_id}")
            continue
        if start < 1 or end < 1:
            labels.append(f"perturbation:{perturbation_id}")
            continue
        labels.append(f"{_normalize_chrom(chrom)}:{start}-{end}:{strand}")
    return np.asarray(labels, dtype=object)


def _resolve_dose_strength_groups(*, adata: AnnData) -> np.ndarray:
    perturbation_ids = _require_obs_text_column(adata=adata, column="perturbation_id")
    perturbation_table = _coerce_perturbation_table(adata.uns.get(PERTURBATIONS_UNS_KEY))
    if perturbation_table is None:
        raise ShiftSplitError(
            "AnnData `.uns['perturbations']` is missing or malformed for dose-strength splitting."
        )

    dose_by_perturbation: dict[str, float] = {}
    for _, row in perturbation_table.iterrows():
        perturbation_id = str(row.get("perturbation_id", "")).strip()
        if perturbation_id == "":
            continue
        dose = _coerce_float(row.get("dose"))
        dose_by_perturbation[perturbation_id] = float(0.0 if dose is None else max(dose, 0.0))

    if "knockdown_efficiency_proxy" not in adata.obs.columns:
        raise ShiftSplitError(
            "AnnData `.obs` is missing required `knockdown_efficiency_proxy` "
            "for dose_strength split."
        )
    knockdown = pd.to_numeric(adata.obs["knockdown_efficiency_proxy"], errors="coerce").fillna(0.0)
    knockdown_values = np.asarray(knockdown.to_numpy(dtype=np.float64), dtype=np.float64)

    dose_values = np.asarray(
        [
            dose_by_perturbation.get(str(perturbation_id), 0.0)
            for perturbation_id in perturbation_ids
        ],
        dtype=np.float64,
    )
    effective_strength = np.clip(dose_values * np.clip(knockdown_values, 0.0, None), 0.0, None)
    return _rank_bin_labels(values=effective_strength, n_bins=4)


def _rank_bin_labels(*, values: np.ndarray, n_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=object)
    if n_bins <= 1 or np.allclose(values, values[0]):
        return np.asarray(["strength_bin_0"] * values.size, dtype=object)

    order = np.argsort(values, kind="mergesort")
    bin_ids = np.zeros(values.size, dtype=np.int64)
    for rank, idx in enumerate(order.tolist()):
        bin_id = int((rank * n_bins) / max(values.size, 1))
        bin_ids[int(idx)] = min(bin_id, n_bins - 1)
    labels = [f"strength_bin_{int(bin_id)}" for bin_id in bin_ids.tolist()]
    return np.asarray(labels, dtype=object)


def _coerce_perturbation_table(raw_object: Any) -> pd.DataFrame | None:
    if isinstance(raw_object, pd.DataFrame):
        return raw_object.copy()
    if isinstance(raw_object, list):
        if all(isinstance(item, dict) for item in raw_object):
            return pd.DataFrame(raw_object)
        return None
    if isinstance(raw_object, dict):
        records: list[dict[str, Any]] = []
        for key, value in raw_object.items():
            if not isinstance(value, dict):
                return None
            record = dict(value)
            record.setdefault("perturbation_id", str(key))
            records.append(record)
        return pd.DataFrame(records)
    return None


def _normalize_chrom(chrom: str) -> str:
    cleaned = chrom.strip()
    if cleaned.lower().startswith("chr"):
        cleaned = cleaned[3:]
    return f"chr{cleaned}"


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if np.isnan(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None
