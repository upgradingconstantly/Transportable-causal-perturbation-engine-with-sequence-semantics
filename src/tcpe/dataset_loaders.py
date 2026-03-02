"""Dataset ingestion registry and loaders for TCPE Phase 5."""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse
from urllib.request import urlretrieve

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


LOCAL_REPLOGLE_MAX_CELLS = 5000
TCPE_HARMONIZATION_VERSION = "phase5_v1"


class DatasetLoaderError(RuntimeError):
    """Raised when dataset loading cannot be completed."""


@dataclass(frozen=True)
class DatasetLoadResult:
    """Output bundle from dataset loader execution."""

    dataset_id: str
    adata: AnnData
    raw_path: Path
    processed_path: Path
    checksum_sha256: str
    from_cache: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetLoader(Protocol):
    """Protocol for pluggable dataset loaders."""

    dataset_id: str

    def load(
        self,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> DatasetLoadResult:
        """Load and harmonize a dataset into canonical AnnData."""


class DatasetRegistry:
    """Dataset loader registry with pluggable loader dispatch."""

    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}

    def register(self, loader: DatasetLoader) -> None:
        if loader.dataset_id in self._loaders:
            raise ValueError(f"Loader already registered: {loader.dataset_id}")
        self._loaders[loader.dataset_id] = loader

    def get(self, dataset_id: str) -> DatasetLoader:
        if dataset_id not in self._loaders:
            available = ", ".join(sorted(self._loaders.keys()))
            raise KeyError(f"Unknown dataset '{dataset_id}'. Available: {available}")
        return self._loaders[dataset_id]

    def available(self) -> list[str]:
        return sorted(self._loaders.keys())

    def load(
        self,
        dataset_id: str,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> DatasetLoadResult:
        loader = self.get(dataset_id)
        return loader.load(
            cache_dir=cache_dir,
            source_uri=source_uri,
            force_refresh=force_refresh,
            **kwargs,
        )


class TabularPerturbSeqLoader:
    """Generic loader for tabular cell-by-gene Perturb-seq data."""

    def __init__(
        self,
        *,
        dataset_id: str,
        default_source_uri: str | None,
        default_cell_type: str,
        default_protocol: str,
        default_batch_prefix: str,
        hard_max_cells: int | None = None,
        sample_seed: int = 42,
    ) -> None:
        self.dataset_id = dataset_id
        self.default_source_uri = default_source_uri
        self.default_cell_type = default_cell_type
        self.default_protocol = default_protocol
        self.default_batch_prefix = default_batch_prefix
        self.hard_max_cells = hard_max_cells
        self.sample_seed = sample_seed

    def load(
        self,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
        max_cells: int | None = None,
        **kwargs: Any,
    ) -> DatasetLoadResult:
        _ = kwargs
        source = source_uri if source_uri is not None else self.default_source_uri
        if source is None:
            raise DatasetLoaderError(
                f"No source URI was provided for dataset '{self.dataset_id}'."
            )

        cache_root = Path(cache_dir) / self.dataset_id
        raw_dir = cache_root / "raw"
        processed_dir = cache_root / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_path = _cache_source_file(
            source=source,
            dataset_id=self.dataset_id,
            raw_dir=raw_dir,
            force_refresh=force_refresh,
        )
        checksum = _sha256_file(raw_path)
        resolved_cap = _resolve_requested_cap(
            requested=max_cells,
            hard_max=self.hard_max_cells,
            dataset_id=self.dataset_id,
        )
        cap_tag = f"cap{resolved_cap}" if resolved_cap is not None else "capall"
        cache_key = f"{checksum[:16]}_{cap_tag}"
        processed_path = processed_dir / f"{self.dataset_id}_{cache_key}.h5ad"
        manifest_path = processed_dir / f"{self.dataset_id}_{cache_key}.manifest.json"

        if not force_refresh and processed_path.exists() and manifest_path.exists():
            manifest = _read_json(manifest_path)
            if _is_cache_manifest_valid(
                manifest=manifest,
                expected_checksum=checksum,
                expected_cap=resolved_cap,
            ):
                adata = _require_anndata().read_h5ad(processed_path)
                return DatasetLoadResult(
                    dataset_id=self.dataset_id,
                    adata=adata,
                    raw_path=raw_path,
                    processed_path=processed_path,
                    checksum_sha256=checksum,
                    from_cache=True,
                    metadata={"cache_manifest": manifest},
                )

        adata, parse_meta = self._parse_tabular_raw(
            raw_path=raw_path,
            max_cells=resolved_cap,
        )
        repair_report = validate_anndata_schema(adata=adata, mode="repair")
        validate_anndata_schema(adata=adata, mode="strict")

        adata.write_h5ad(processed_path)
        manifest_payload = {
            "dataset_id": self.dataset_id,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "checksum_sha256": checksum,
            "max_cells": resolved_cap,
            "harmonization_version": TCPE_HARMONIZATION_VERSION,
            "repaired_fields": repair_report.repaired_fields,
            "parse_metadata": parse_meta,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return DatasetLoadResult(
            dataset_id=self.dataset_id,
            adata=adata,
            raw_path=raw_path,
            processed_path=processed_path,
            checksum_sha256=checksum,
            from_cache=False,
            metadata={"cache_manifest": manifest_payload},
        )

    def _parse_tabular_raw(
        self,
        *,
        raw_path: Path,
        max_cells: int | None,
    ) -> tuple[AnnData, dict[str, Any]]:
        table = pd.read_csv(raw_path)
        if table.empty:
            raise DatasetLoaderError(f"Raw table is empty for dataset '{self.dataset_id}'.")

        reserved_obs_columns = {
            "cell_id",
            "cell_type",
            "batch",
            "condition",
            "protocol",
            "library_size",
            "perturbation_id",
            "knockdown_efficiency_proxy",
        }
        gene_columns = [col for col in table.columns if col not in reserved_obs_columns]
        if len(gene_columns) == 0:
            raise DatasetLoaderError(
                f"Raw table for '{self.dataset_id}' must include one or more gene count columns."
            )

        if "cell_id" not in table.columns:
            table["cell_id"] = [f"{self.dataset_id}_cell_{idx:06d}" for idx in range(len(table))]
        if "perturbation_id" not in table.columns:
            table["perturbation_id"] = "unassigned"

        if max_cells is not None and len(table) > max_cells:
            rng = np.random.default_rng(self.sample_seed)
            selected = np.sort(rng.choice(len(table), size=max_cells, replace=False))
            table = table.iloc[selected].reset_index(drop=True)

        counts = (
            table[gene_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .round()
            .astype(np.int64)
            .to_numpy()
        )
        library_size = counts.sum(axis=1).astype(np.float64)

        obs = _harmonize_obs_metadata(
            table=table,
            counts_library_size=library_size,
            default_cell_type=self.default_cell_type,
            default_protocol=self.default_protocol,
            default_batch_prefix=self.default_batch_prefix,
        )
        var = _harmonize_var_metadata(gene_columns=gene_columns)
        adata = cast(AnnData, _require_anndata().AnnData(X=counts, obs=obs, var=var))

        adata.layers[NORMALIZED_LAYER_KEY] = _compute_normalized_log1p(
            counts=counts,
            library_size=library_size,
        )
        adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY] = np.zeros((adata.n_obs, 1), dtype=np.float32)
        adata.obsm[CELL_STATE_EMBEDDING_OBSM_KEY] = np.zeros((adata.n_obs, 1), dtype=np.float32)
        adata.uns[PERTURBATIONS_UNS_KEY] = _build_default_perturbation_metadata(
            perturbation_ids=obs["perturbation_id"].astype(str).tolist()
        )
        adata.uns[SCHEMA_VERSION_UNS_KEY] = SCHEMA_VERSION
        adata.uns["dataset_name"] = self.dataset_id
        adata.uns["harmonization_version"] = TCPE_HARMONIZATION_VERSION
        adata.uns["raw_source_path"] = str(raw_path)

        parse_meta = {
            "n_rows_raw": int(len(table)),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "max_cells_applied": max_cells,
        }
        return adata, parse_meta


class AdamsonLoader(TabularPerturbSeqLoader):
    """Adamson et al. 2016 dataset loader."""

    def __init__(self) -> None:
        super().__init__(
            dataset_id="adamson",
            default_source_uri=None,
            default_cell_type="K562",
            default_protocol="perturbseq_adamson_2016",
            default_batch_prefix="adamson_batch",
            hard_max_cells=None,
            sample_seed=42,
        )


class ReplogleSampleLoader(TabularPerturbSeqLoader):
    """Replogle sample loader with strict local max-cell cap."""

    def __init__(self) -> None:
        super().__init__(
            dataset_id="replogle_sample",
            default_source_uri=None,
            default_cell_type="K562",
            default_protocol="perturbseq_replogle_2022",
            default_batch_prefix="replogle_batch",
            hard_max_cells=LOCAL_REPLOGLE_MAX_CELLS,
            sample_seed=42,
        )


class LINCSAdapterStubLoader:
    """LINCS L1000 adapter stub for Phase 5."""

    dataset_id = "lincs_stub"

    def load(
        self,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> DatasetLoadResult:
        _ = (cache_dir, source_uri, force_refresh, kwargs)
        warnings.warn(
            (
                "LINCS adapter is a Phase 5 stub. Full LINCS preprocessing "
                "is deferred to a later phase."
            ),
            UserWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "LINCS adapter stub: full LINCS ingestion/preprocessing is not implemented in Phase 5."
        )


def build_default_dataset_registry() -> DatasetRegistry:
    """Construct the default Phase 5 dataset registry."""
    registry = DatasetRegistry()
    registry.register(AdamsonLoader())
    registry.register(ReplogleSampleLoader())
    registry.register(LINCSAdapterStubLoader())
    return registry


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise DatasetLoaderError("`anndata` is required for dataset ingestion.") from exc
    return ad


def _cache_source_file(
    *,
    source: str | Path,
    dataset_id: str,
    raw_dir: Path,
    force_refresh: bool,
) -> Path:
    parsed = urlparse(str(source))
    if parsed.scheme in ("http", "https", "ftp"):
        filename = Path(parsed.path).name or f"{dataset_id}_source.csv"
        destination = raw_dir / filename
        if force_refresh or not destination.exists():
            urlretrieve(str(source), destination)
        return destination

    source_path = Path(source)
    if not source_path.exists():
        raise DatasetLoaderError(f"Source file does not exist: {source_path}")
    destination = raw_dir / source_path.name
    if force_refresh or not destination.exists():
        shutil.copy2(source_path, destination)
    return destination


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DatasetLoaderError(f"Cache manifest is malformed: {path}")
    return cast(dict[str, Any], payload)


def _is_cache_manifest_valid(
    *,
    manifest: dict[str, Any],
    expected_checksum: str,
    expected_cap: int | None,
) -> bool:
    if str(manifest.get("checksum_sha256")) != expected_checksum:
        return False
    if manifest.get("harmonization_version") != TCPE_HARMONIZATION_VERSION:
        return False
    if manifest.get("max_cells") != expected_cap:
        return False
    return True


def _resolve_requested_cap(
    *,
    requested: int | None,
    hard_max: int | None,
    dataset_id: str,
) -> int | None:
    if requested is not None and requested <= 0:
        raise DatasetLoaderError("max_cells must be positive when provided.")
    if hard_max is None:
        return requested
    if requested is None:
        return hard_max
    if requested > hard_max:
        warnings.warn(
            (
                f"Requested max_cells={requested} exceeds local cap for '{dataset_id}'. "
                f"Using capped value {hard_max}."
            ),
            UserWarning,
            stacklevel=2,
        )
        return hard_max
    return requested


def _harmonize_obs_metadata(
    *,
    table: pd.DataFrame,
    counts_library_size: np.ndarray,
    default_cell_type: str,
    default_protocol: str,
    default_batch_prefix: str,
) -> pd.DataFrame:
    harmonized = pd.DataFrame(index=table.index.copy())
    harmonized["cell_id"] = table["cell_id"].astype(str)
    harmonized["cell_type"] = (
        table["cell_type"].astype(str)
        if "cell_type" in table.columns
        else default_cell_type
    )

    if "batch" in table.columns:
        harmonized["batch"] = table["batch"].astype(str)
    else:
        harmonized["batch"] = [f"{default_batch_prefix}_0"] * len(table)

    if "condition" in table.columns:
        harmonized["condition"] = table["condition"].astype(str)
    else:
        perturb_series = table["perturbation_id"].astype(str).str.lower()
        harmonized["condition"] = np.where(
            perturb_series.isin({"ntc", "ctrl", "control", "p000"}),
            "control",
            "perturbed",
        )

    harmonized["protocol"] = (
        table["protocol"].astype(str)
        if "protocol" in table.columns
        else default_protocol
    )

    if "library_size" in table.columns:
        lib_size = pd.to_numeric(table["library_size"], errors="coerce").fillna(
            pd.Series(counts_library_size)
        )
        harmonized["library_size"] = lib_size.astype(np.float64)
    else:
        harmonized["library_size"] = counts_library_size.astype(np.float64)

    harmonized["perturbation_id"] = table["perturbation_id"].astype(str)
    perturbation_series = harmonized["perturbation_id"].astype(str).str.lower()

    if "knockdown_efficiency_proxy" in table.columns:
        knockdown = pd.to_numeric(table["knockdown_efficiency_proxy"], errors="coerce").fillna(0.0)
        harmonized["knockdown_efficiency_proxy"] = knockdown.astype(np.float64)
    else:
        harmonized["knockdown_efficiency_proxy"] = np.where(
            perturbation_series.isin({"ntc", "ctrl", "control", "p000"}),
            0.0,
            1.0,
        ).astype(np.float64)

    harmonized.index = harmonized["cell_id"].astype(str)
    return harmonized


def _harmonize_var_metadata(gene_columns: list[str]) -> pd.DataFrame:
    var = pd.DataFrame(index=gene_columns)
    var["gene_id"] = gene_columns
    var["gene_symbol"] = [item.upper() for item in gene_columns]
    var["chrom"] = "unknown"
    var["strand"] = "."
    var["tss"] = -1
    return var


def _compute_normalized_log1p(counts: np.ndarray, library_size: np.ndarray) -> np.ndarray:
    denom = np.maximum(library_size.reshape(-1, 1), 1.0)
    normalized = (counts.astype(np.float64) / denom) * 1e4
    return cast(np.ndarray, np.log1p(normalized).astype(np.float32))


def _build_default_perturbation_metadata(perturbation_ids: list[str]) -> pd.DataFrame:
    unique_ids = sorted(set(perturbation_ids))
    records: list[dict[str, Any]] = []
    for perturbation_id in unique_ids:
        lower = perturbation_id.lower()
        is_control = lower in {"ntc", "ctrl", "control", "p000"}
        records.append(
            {
                "perturbation_id": perturbation_id,
                "target_gene": "NTC" if is_control else perturbation_id.split("_")[0].upper(),
                "chrom": "unknown",
                "start": -1,
                "end": -1,
                "strand": ".",
                "modality": "control" if is_control else "crispri",
                "dose": 0.0 if is_control else 1.0,
                "gRNA_id": perturbation_id,
            }
        )
    return pd.DataFrame(records)
