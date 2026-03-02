"""Canonical AnnData schema contract and validators for TCPE Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

SCHEMA_VERSION = "anndata_v1"
SCHEMA_VERSION_UNS_KEY = "tcpe_schema_version"
NORMALIZED_LAYER_KEY = "normalized_log1p"
SEQUENCE_EMBEDDING_OBSM_KEY = "X_sequence_embedding"
CELL_STATE_EMBEDDING_OBSM_KEY = "X_cell_state_embedding"
PERTURBATIONS_UNS_KEY = "perturbations"

REQUIRED_OBS_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "cell_type",
    "batch",
    "condition",
    "protocol",
    "library_size",
    "perturbation_id",
    "knockdown_efficiency_proxy",
)
REQUIRED_VAR_COLUMNS: tuple[str, ...] = ("gene_id", "gene_symbol", "chrom", "strand", "tss")
REQUIRED_PERTURBATION_FIELDS: tuple[str, ...] = (
    "target_gene",
    "chrom",
    "start",
    "end",
    "strand",
    "modality",
    "dose",
    "gRNA_id",
)

ValidationMode = Literal["strict", "repair"]


class AnnDataSchemaError(ValueError):
    """Raised when an AnnData object does not satisfy schema requirements."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "AnnData schema validation failed:\n- " + "\n- ".join(errors)
        super().__init__(message)


@dataclass
class AnnDataValidationReport:
    """Validation output for schema checks."""

    is_valid: bool
    mode: ValidationMode
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repaired_fields: list[str] = field(default_factory=list)


def validate_anndata_schema(
    adata: AnnData, mode: ValidationMode = "strict", raise_on_error: bool = True
) -> AnnDataValidationReport:
    """Validate AnnData against the TCPE canonical schema."""
    if mode not in ("strict", "repair"):
        raise ValueError(f"Unsupported validation mode: {mode}")

    errors: list[str] = []
    warnings: list[str] = []
    repaired_fields: list[str] = []

    _validate_raw_counts_x(adata=adata, errors=errors)
    _validate_or_repair_obs(adata=adata, mode=mode, errors=errors, repaired_fields=repaired_fields)
    _validate_or_repair_var(adata=adata, mode=mode, errors=errors, repaired_fields=repaired_fields)
    _validate_or_repair_normalized_layer(
        adata=adata, mode=mode, errors=errors, repaired_fields=repaired_fields
    )
    _validate_or_repair_embeddings(
        adata=adata, mode=mode, errors=errors, repaired_fields=repaired_fields
    )
    _validate_or_repair_perturbations(
        adata=adata, mode=mode, errors=errors, repaired_fields=repaired_fields
    )
    _validate_or_repair_schema_version(
        adata=adata, mode=mode, errors=errors, repaired_fields=repaired_fields
    )

    report = AnnDataValidationReport(
        is_valid=(len(errors) == 0),
        mode=mode,
        errors=errors,
        warnings=warnings,
        repaired_fields=repaired_fields,
    )
    if errors and raise_on_error:
        raise AnnDataSchemaError(errors)
    return report


def _validate_raw_counts_x(adata: AnnData, errors: list[str]) -> None:
    if adata.n_obs == 0:
        errors.append("AnnData must contain at least one observation (cell).")
    if adata.n_vars == 0:
        errors.append("AnnData must contain at least one variable (gene).")

    if sparse.issparse(adata.X):
        x_data = adata.X.data
        if np.any(x_data < 0):
            errors.append("`.X` must contain non-negative raw counts.")
        if not np.allclose(x_data, np.round(x_data)):
            errors.append("`.X` must store integer-like raw counts.")
        return

    x_dense = np.asarray(adata.X)
    if np.any(x_dense < 0):
        errors.append("`.X` must contain non-negative raw counts.")
    if not np.allclose(x_dense, np.round(x_dense)):
        errors.append("`.X` must store integer-like raw counts.")


def _validate_or_repair_obs(
    adata: AnnData,
    mode: ValidationMode,
    errors: list[str],
    repaired_fields: list[str],
) -> None:
    repair_defaults: dict[str, Any] = {
        "cell_id": adata.obs_names.astype(str).tolist(),
        "cell_type": "unknown",
        "batch": "unknown",
        "condition": "unknown",
        "protocol": "unknown",
        "library_size": _library_size_from_x(adata),
        "perturbation_id": "unassigned",
        "knockdown_efficiency_proxy": 0.0,
    }

    for column in REQUIRED_OBS_COLUMNS:
        if column not in adata.obs.columns:
            if mode == "repair":
                adata.obs[column] = repair_defaults[column]
                repaired_fields.append(f"obs.{column}")
            else:
                errors.append(f"Missing required `.obs` column: `{column}`.")

    if "library_size" in adata.obs.columns:
        coerced = pd.to_numeric(adata.obs["library_size"], errors="coerce")
        if coerced.isna().any():
            if mode == "repair":
                fallback = _library_size_from_x(adata)
                adata.obs["library_size"] = coerced.fillna(
                    pd.Series(fallback, index=adata.obs.index)
                )
                repaired_fields.append("obs.library_size")
            else:
                errors.append("`.obs['library_size']` must be numeric for all cells.")
        if (pd.to_numeric(adata.obs["library_size"], errors="coerce") < 0).any():
            errors.append("`.obs['library_size']` must be non-negative.")

    if "knockdown_efficiency_proxy" in adata.obs.columns:
        coerced_knockdown = pd.to_numeric(adata.obs["knockdown_efficiency_proxy"], errors="coerce")
        if coerced_knockdown.isna().all() and mode == "strict":
            errors.append("`.obs['knockdown_efficiency_proxy']` must contain numeric values.")
        if mode == "repair":
            adata.obs["knockdown_efficiency_proxy"] = coerced_knockdown.fillna(0.0)
        else:
            adata.obs["knockdown_efficiency_proxy"] = coerced_knockdown

    for text_column in (
        "cell_id",
        "cell_type",
        "batch",
        "condition",
        "protocol",
        "perturbation_id",
    ):
        if text_column in adata.obs.columns:
            converted = adata.obs[text_column].astype(str).str.strip()
            if (converted == "").any():
                errors.append(f"`.obs['{text_column}']` contains empty values.")
            adata.obs[text_column] = converted


def _validate_or_repair_var(
    adata: AnnData,
    mode: ValidationMode,
    errors: list[str],
    repaired_fields: list[str],
) -> None:
    defaults: dict[str, Any] = {
        "gene_id": adata.var_names.astype(str).tolist(),
        "gene_symbol": adata.var_names.astype(str).tolist(),
        "chrom": "unknown",
        "strand": ".",
        "tss": -1,
    }
    for column in REQUIRED_VAR_COLUMNS:
        if column not in adata.var.columns:
            if mode == "repair":
                adata.var[column] = defaults[column]
                repaired_fields.append(f"var.{column}")
            else:
                errors.append(f"Missing required `.var` column: `{column}`.")

    if "tss" in adata.var.columns:
        tss_series = pd.to_numeric(adata.var["tss"], errors="coerce")
        if tss_series.isna().any():
            if mode == "repair":
                adata.var["tss"] = tss_series.fillna(-1).astype(int)
                repaired_fields.append("var.tss")
            else:
                errors.append("`.var['tss']` must be numeric.")

    for text_column in ("gene_id", "gene_symbol", "chrom", "strand"):
        if text_column in adata.var.columns:
            converted = adata.var[text_column].astype(str).str.strip()
            if (converted == "").any():
                errors.append(f"`.var['{text_column}']` contains empty values.")
            adata.var[text_column] = converted


def _validate_or_repair_normalized_layer(
    adata: AnnData,
    mode: ValidationMode,
    errors: list[str],
    repaired_fields: list[str],
) -> None:
    if NORMALIZED_LAYER_KEY not in adata.layers:
        if mode == "repair":
            adata.layers[NORMALIZED_LAYER_KEY] = _compute_normalized_log1p(adata.X)
            repaired_fields.append(f"layers.{NORMALIZED_LAYER_KEY}")
        else:
            errors.append(
                f"Missing required normalized layer `.layers['{NORMALIZED_LAYER_KEY}']`."
            )
        return

    layer = adata.layers[NORMALIZED_LAYER_KEY]
    if layer.shape != adata.X.shape:
        errors.append(
            f"`.layers['{NORMALIZED_LAYER_KEY}']` shape {layer.shape} does not match `.X` shape "
            f"{adata.X.shape}."
        )
        return
    if sparse.issparse(layer):
        if np.any(layer.data < 0):
            errors.append(f"`.layers['{NORMALIZED_LAYER_KEY}']` must contain non-negative values.")
    else:
        if np.any(np.asarray(layer) < 0):
            errors.append(f"`.layers['{NORMALIZED_LAYER_KEY}']` must contain non-negative values.")


def _validate_or_repair_embeddings(
    adata: AnnData,
    mode: ValidationMode,
    errors: list[str],
    repaired_fields: list[str],
) -> None:
    required_keys = (SEQUENCE_EMBEDDING_OBSM_KEY, CELL_STATE_EMBEDDING_OBSM_KEY)
    for key in required_keys:
        if key not in adata.obsm:
            if mode == "repair":
                adata.obsm[key] = np.zeros((adata.n_obs, 1), dtype=np.float32)
                repaired_fields.append(f"obsm.{key}")
            else:
                errors.append(f"Missing required embedding slot `.obsm['{key}']`.")
            continue

        value = adata.obsm[key]
        if value.shape[0] != adata.n_obs:
            errors.append(
                f"`.obsm['{key}']` has {value.shape[0]} rows but expected {adata.n_obs} rows."
            )
        if len(value.shape) != 2 or value.shape[1] <= 0:
            errors.append(
                f"`.obsm['{key}']` must be a 2D matrix with positive embedding dimension."
            )


def _validate_or_repair_perturbations(
    adata: AnnData,
    mode: ValidationMode,
    errors: list[str],
    repaired_fields: list[str],
) -> None:
    perturbation_ids = (
        adata.obs["perturbation_id"].astype(str).tolist()
        if "perturbation_id" in adata.obs.columns
        else ["unassigned"] * adata.n_obs
    )
    unique_ids = sorted(set(perturbation_ids))

    if PERTURBATIONS_UNS_KEY not in adata.uns:
        if mode == "repair":
            adata.uns[PERTURBATIONS_UNS_KEY] = _default_perturbation_records(unique_ids)
            repaired_fields.append(f"uns.{PERTURBATIONS_UNS_KEY}")
        else:
            errors.append(f"Missing required `.uns['{PERTURBATIONS_UNS_KEY}']` metadata object.")
            return

    raw_object = adata.uns.get(PERTURBATIONS_UNS_KEY)
    records = _coerce_perturbation_records(raw_object)
    if records is None:
        errors.append(
            f"`.uns['{PERTURBATIONS_UNS_KEY}']` must be a dataframe, mapping, or list of mappings."
        )
        return

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("perturbation_id", "")).strip()
        if record_id == "":
            errors.append(
                f"`.uns['{PERTURBATIONS_UNS_KEY}']` records must include non-empty "
                "`perturbation_id`."
            )
            continue

        missing_fields = [field for field in REQUIRED_PERTURBATION_FIELDS if field not in record]
        if missing_fields:
            if mode == "repair":
                _fill_perturbation_defaults(record=record, perturbation_id=record_id)
                repaired_fields.append(
                    f"uns.{PERTURBATIONS_UNS_KEY}.{record_id}:" + ",".join(missing_fields)
                )
            else:
                errors.append(
                    f"Perturbation `{record_id}` is missing fields: {', '.join(missing_fields)}."
                )

        record["perturbation_id"] = record_id
        records_by_id[record_id] = record

    for perturbation_id in unique_ids:
        if perturbation_id not in records_by_id:
            if mode == "repair":
                records_by_id[perturbation_id] = _default_perturbation_record(perturbation_id)
                repaired_fields.append(f"uns.{PERTURBATIONS_UNS_KEY}.{perturbation_id}")
            else:
                errors.append(
                    f"`.obs['perturbation_id']` contains `{perturbation_id}` which is missing in "
                    f"`.uns['{PERTURBATIONS_UNS_KEY}']`."
                )

    for perturbation_id, record in records_by_id.items():
        _validate_perturbation_record(perturbation_id=perturbation_id, record=record, errors=errors)

    adata.uns[PERTURBATIONS_UNS_KEY] = pd.DataFrame(
        [records_by_id[key] for key in sorted(records_by_id.keys())]
    )


def _validate_or_repair_schema_version(
    adata: AnnData,
    mode: ValidationMode,
    errors: list[str],
    repaired_fields: list[str],
) -> None:
    current = adata.uns.get(SCHEMA_VERSION_UNS_KEY)
    if current is None:
        if mode == "repair":
            adata.uns[SCHEMA_VERSION_UNS_KEY] = SCHEMA_VERSION
            repaired_fields.append(f"uns.{SCHEMA_VERSION_UNS_KEY}")
        else:
            errors.append(f"Missing `.uns['{SCHEMA_VERSION_UNS_KEY}']` schema version marker.")
        return
    if str(current) != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema version `{current}` in `.uns['{SCHEMA_VERSION_UNS_KEY}']`; "
            f"expected `{SCHEMA_VERSION}`."
        )


def _library_size_from_x(adata: AnnData) -> np.ndarray:
    if sparse.issparse(adata.X):
        summed = adata.X.sum(axis=1)
        return cast(np.ndarray, np.asarray(summed, dtype=np.float64).reshape(-1))
    dense = np.asarray(adata.X, dtype=np.float64)
    summed_dense = np.sum(dense, axis=1, dtype=np.float64)
    return cast(np.ndarray, np.asarray(summed_dense, dtype=np.float64).reshape(-1))


def _compute_normalized_log1p(x: Any) -> Any:
    if sparse.issparse(x):
        x_csr = x.tocsr().astype(np.float64)
        library_size = np.asarray(x_csr.sum(axis=1)).ravel()
        scale = np.divide(
            1e4,
            library_size,
            out=np.zeros_like(library_size),
            where=library_size > 0,
        )
        scaled = sparse.diags(scale) @ x_csr
        scaled.data = np.log1p(scaled.data)
        return scaled

    x_dense = np.asarray(x, dtype=np.float64)
    library_size_dense = x_dense.sum(axis=1, keepdims=True)
    scale_dense = np.divide(
        1e4, library_size_dense, out=np.zeros_like(library_size_dense), where=library_size_dense > 0
    )
    return np.log1p(x_dense * scale_dense)


def _coerce_perturbation_records(raw_object: Any) -> list[dict[str, Any]] | None:
    if isinstance(raw_object, pd.DataFrame):
        return cast(list[dict[str, Any]], raw_object.to_dict(orient="records"))
    if isinstance(raw_object, list):
        if all(isinstance(item, dict) for item in raw_object):
            return [dict(item) for item in raw_object]
        return None
    if isinstance(raw_object, dict):
        records: list[dict[str, Any]] = []
        for key, value in raw_object.items():
            if not isinstance(value, dict):
                return None
            record = dict(value)
            record.setdefault("perturbation_id", str(key))
            records.append(record)
        return records
    return None


def _default_perturbation_record(perturbation_id: str) -> dict[str, Any]:
    return {
        "perturbation_id": perturbation_id,
        "target_gene": "unknown",
        "chrom": "unknown",
        "start": -1,
        "end": -1,
        "strand": ".",
        "modality": "unknown",
        "dose": np.nan,
        "gRNA_id": perturbation_id,
    }


def _default_perturbation_records(perturbation_ids: list[str]) -> list[dict[str, Any]]:
    return [_default_perturbation_record(perturbation_id=item) for item in perturbation_ids]


def _fill_perturbation_defaults(record: dict[str, Any], perturbation_id: str) -> None:
    defaults = _default_perturbation_record(perturbation_id)
    for key, value in defaults.items():
        record.setdefault(key, value)


def _validate_perturbation_record(
    perturbation_id: str, record: dict[str, Any], errors: list[str]
) -> None:
    start = _coerce_numeric(record.get("start"))
    end = _coerce_numeric(record.get("end"))
    if start is None:
        errors.append(f"Perturbation `{perturbation_id}` has non-numeric `start`.")
    if end is None:
        errors.append(f"Perturbation `{perturbation_id}` has non-numeric `end`.")
    if start is not None and end is not None and start >= 0 and end >= 0 and start > end:
        errors.append(
            f"Perturbation `{perturbation_id}` has invalid interval with `start` > `end`."
        )

    dose = _coerce_numeric(record.get("dose"))
    if dose is None and not pd.isna(record.get("dose")):
        errors.append(f"Perturbation `{perturbation_id}` has non-numeric `dose`.")

    for field_name in ("target_gene", "chrom", "strand", "modality", "gRNA_id"):
        value = str(record.get(field_name, "")).strip()
        if value == "":
            errors.append(f"Perturbation `{perturbation_id}` has empty `{field_name}`.")


def _coerce_numeric(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if np.isnan(numeric):
            return numeric
        return numeric
    except (TypeError, ValueError):
        return None
