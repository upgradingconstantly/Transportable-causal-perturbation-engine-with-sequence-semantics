"""Genomic reference access and sequence-window extraction for TCPE Phase 7."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from tcpe.anndata_schema import PERTURBATIONS_UNS_KEY

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

ENSEMBL_GRCH38_CHR22_URL = (
    "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz"
)
SEQUENCE_WINDOWS_UNS_KEY = "sequence_windows"
SEQUENCE_WINDOW_AUDIT_UNS_KEY = "sequence_window_audit"

ExtractionStatus = Literal["ok", "fallback"]


class SequenceWindowError(RuntimeError):
    """Raised when sequence-window extraction cannot proceed."""


@dataclass(frozen=True)
class ReferenceDownloadResult:
    """Result metadata for local reference downloads."""

    fasta_path: Path
    source_url: str
    checksum_sha256: str
    from_cache: bool


@dataclass(frozen=True)
class SequenceWindowConfig:
    """Configuration for locus-centered sequence window extraction."""

    window_size: int = 2000
    min_window_size: int = 1000
    max_window_size: int = 2000
    include_sequence: bool = True

    def __post_init__(self) -> None:
        if self.min_window_size <= 0 or self.max_window_size <= 0:
            raise ValueError("Window-size bounds must be positive.")
        if self.min_window_size > self.max_window_size:
            raise ValueError("min_window_size cannot exceed max_window_size.")
        if not (self.min_window_size <= self.window_size <= self.max_window_size):
            raise ValueError(
                "window_size must be within the configured bounds "
                f"[{self.min_window_size}, {self.max_window_size}]."
            )


@dataclass(frozen=True)
class SequenceWindowRecord:
    """Single perturbation sequence-window extraction result."""

    perturbation_id: str
    status: ExtractionStatus
    reason: str
    chrom: str
    strand: str
    locus_start: int
    locus_end: int
    center: int
    window_start: int
    window_end: int
    window_size_requested: int
    window_size_observed: int
    sequence: str
    cache_key: str
    from_cache: bool

    def to_row(self, include_sequence: bool) -> dict[str, Any]:
        payload = asdict(self)
        if not include_sequence:
            payload["sequence"] = ""
        return payload


@dataclass(frozen=True)
class SequenceWindowSummary:
    """Summary of sequence-window extraction over a perturbation set."""

    n_total: int
    n_ok: int
    n_fallback: int
    cache_hits: int
    cache_misses: int


class GenomeReference:
    """In-memory genomic reference abstraction."""

    def __init__(
        self,
        *,
        sequences_by_chrom: dict[str, str],
        source_path: Path | None,
        source_name: str,
    ) -> None:
        if len(sequences_by_chrom) == 0:
            raise ValueError("Reference must contain at least one chromosome sequence.")
        normalized: dict[str, str] = {}
        for chrom_name, sequence in sequences_by_chrom.items():
            norm = normalize_chrom_name(chrom_name)
            normalized[norm] = sequence.upper()
        self._sequences = normalized
        self.source_path = source_path
        self.source_name = source_name

    @classmethod
    def from_fasta(cls, fasta_path: str | Path, source_name: str | None = None) -> GenomeReference:
        path = Path(fasta_path)
        if not path.exists():
            raise SequenceWindowError(f"FASTA file does not exist: {path}")
        sequences = _parse_fasta(path)
        return cls(
            sequences_by_chrom=sequences,
            source_path=path,
            source_name=source_name or path.name,
        )

    @staticmethod
    def download_local_chr22_reference(
        *,
        cache_dir: str | Path,
        force_refresh: bool = False,
    ) -> ReferenceDownloadResult:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        destination = cache_root / Path(ENSEMBL_GRCH38_CHR22_URL).name
        from_cache = destination.exists() and not force_refresh
        if force_refresh or not destination.exists():
            urlretrieve(ENSEMBL_GRCH38_CHR22_URL, destination)
            from_cache = False
        checksum = _sha256_file(destination)
        return ReferenceDownloadResult(
            fasta_path=destination,
            source_url=ENSEMBL_GRCH38_CHR22_URL,
            checksum_sha256=checksum,
            from_cache=from_cache,
        )

    @classmethod
    def from_local_chr22_reference(
        cls,
        *,
        cache_dir: str | Path,
        force_refresh: bool = False,
    ) -> GenomeReference:
        download_result = cls.download_local_chr22_reference(
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        return cls.from_fasta(
            fasta_path=download_result.fasta_path,
            source_name="ensembl_grch38_release110_chr22",
        )

    def available_chromosomes(self) -> list[str]:
        return sorted(self._sequences.keys())

    def get_chrom_sequence(self, chrom: str) -> str:
        norm = normalize_chrom_name(chrom)
        if norm not in self._sequences:
            raise SequenceWindowError(
                f"Chromosome '{chrom}' (normalized '{norm}') not found in reference."
            )
        return self._sequences[norm]

    def get_chrom_length(self, chrom: str) -> int:
        return len(self.get_chrom_sequence(chrom))


class SequenceWindowExtractor:
    """Extract locus-centered windows with cache and audit metadata."""

    def __init__(
        self,
        *,
        reference: GenomeReference,
        config: SequenceWindowConfig | None = None,
    ) -> None:
        self.reference = reference
        self.config = config if config is not None else SequenceWindowConfig()
        self._cache: dict[str, SequenceWindowRecord] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def extract_from_perturbation_table(
        self,
        table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, SequenceWindowSummary]:
        if "perturbation_id" not in table.columns:
            raise SequenceWindowError("Perturbation table must include `perturbation_id`.")

        records: list[SequenceWindowRecord] = []
        for _, row in table.sort_values("perturbation_id").iterrows():
            records.append(self.extract_single(row.to_dict()))

        result_rows = [record.to_row(self.config.include_sequence) for record in records]
        result_df = pd.DataFrame(result_rows)
        summary = SequenceWindowSummary(
            n_total=len(records),
            n_ok=int(np.sum(result_df["status"] == "ok")) if len(records) else 0,
            n_fallback=int(np.sum(result_df["status"] == "fallback")) if len(records) else 0,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )
        return result_df, summary

    def annotate_anndata(self, adata: AnnData) -> SequenceWindowSummary:
        perturb_table = _coerce_perturbation_table(adata.uns.get(PERTURBATIONS_UNS_KEY))
        if perturb_table is None:
            raise SequenceWindowError(
                "AnnData is missing parseable "
                f"`.uns['{PERTURBATIONS_UNS_KEY}']` perturbation metadata."
            )

        windows_df, summary = self.extract_from_perturbation_table(perturb_table)
        adata.uns[SEQUENCE_WINDOWS_UNS_KEY] = windows_df
        adata.uns[SEQUENCE_WINDOW_AUDIT_UNS_KEY] = {
            "reference_source_name": self.reference.source_name,
            "reference_source_path": (
                str(self.reference.source_path) if self.reference.source_path else None
            ),
            "config": asdict(self.config),
            "summary": asdict(summary),
        }
        return summary

    def extract_single(self, perturbation_row: dict[str, Any]) -> SequenceWindowRecord:
        perturbation_id = str(perturbation_row.get("perturbation_id", "")).strip() or "unknown"
        chrom_raw = str(perturbation_row.get("chrom", "")).strip()
        strand = str(perturbation_row.get("strand", "+")).strip() or "+"
        start = _coerce_int(perturbation_row.get("start"))
        end = _coerce_int(perturbation_row.get("end"))

        if chrom_raw == "" or chrom_raw.lower() == "unknown":
            return _fallback_record(
                perturbation_id=perturbation_id,
                reason="missing_or_unknown_chromosome",
                chrom=chrom_raw or "unknown",
                strand=strand,
                start=start,
                end=end,
                window_size=self.config.window_size,
            )
        if start is None or end is None or start < 1 or end < 1:
            return _fallback_record(
                perturbation_id=perturbation_id,
                reason="missing_or_invalid_coordinates",
                chrom=chrom_raw,
                strand=strand,
                start=start,
                end=end,
                window_size=self.config.window_size,
            )
        if end < start:
            return _fallback_record(
                perturbation_id=perturbation_id,
                reason="invalid_interval_end_before_start",
                chrom=chrom_raw,
                strand=strand,
                start=start,
                end=end,
                window_size=self.config.window_size,
            )

        cache_key = _build_cache_key(
            chrom=chrom_raw,
            start=start,
            end=end,
            strand=strand,
            window_size=self.config.window_size,
        )
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self._cache_hits += 1
            return SequenceWindowRecord(
                **{
                    **asdict(cached),
                    "perturbation_id": perturbation_id,
                    "from_cache": True,
                }
            )

        try:
            chrom_sequence = self.reference.get_chrom_sequence(chrom_raw)
        except SequenceWindowError:
            return _fallback_record(
                perturbation_id=perturbation_id,
                reason="chromosome_not_found_in_reference",
                chrom=chrom_raw,
                strand=strand,
                start=start,
                end=end,
                window_size=self.config.window_size,
            )

        chrom_length = len(chrom_sequence)
        center = int((start + end) / 2)
        window_start, window_end = _compute_window_bounds(
            center=center,
            chrom_length=chrom_length,
            window_size=self.config.window_size,
        )
        window_seq = chrom_sequence[window_start - 1 : window_end]
        if window_seq == "":
            return _fallback_record(
                perturbation_id=perturbation_id,
                reason="empty_window_after_bounds",
                chrom=chrom_raw,
                strand=strand,
                start=start,
                end=end,
                window_size=self.config.window_size,
            )

        oriented_seq = window_seq
        if strand == "-":
            oriented_seq = reverse_complement(window_seq)

        record = SequenceWindowRecord(
            perturbation_id=perturbation_id,
            status="ok",
            reason="ok",
            chrom=normalize_chrom_name(chrom_raw),
            strand=strand,
            locus_start=start,
            locus_end=end,
            center=center,
            window_start=window_start,
            window_end=window_end,
            window_size_requested=self.config.window_size,
            window_size_observed=len(oriented_seq),
            sequence=oriented_seq,
            cache_key=cache_key,
            from_cache=False,
        )
        self._cache[cache_key] = record
        self._cache_misses += 1
        return record


def normalize_chrom_name(chrom: str) -> str:
    normalized = chrom.strip()
    if normalized.lower().startswith("chr"):
        normalized = normalized[3:]
    return normalized.upper()


def reverse_complement(sequence: str) -> str:
    translation = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(translation)[::-1]


def _build_cache_key(*, chrom: str, start: int, end: int, strand: str, window_size: int) -> str:
    norm_chrom = normalize_chrom_name(chrom)
    payload = f"{norm_chrom}:{start}:{end}:{strand}:{window_size}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        numeric = int(float(value))
        return numeric
    except (TypeError, ValueError):
        return None


def _compute_window_bounds(*, center: int, chrom_length: int, window_size: int) -> tuple[int, int]:
    if chrom_length <= 0:
        raise SequenceWindowError("Chromosome length must be positive.")
    if chrom_length <= window_size:
        return 1, chrom_length

    left_half = (window_size - 1) // 2
    right_half = window_size - left_half - 1
    start = center - left_half
    end = center + right_half

    if start < 1:
        shift = 1 - start
        start += shift
        end += shift
    if end > chrom_length:
        shift = end - chrom_length
        end -= shift
        start -= shift

    start = max(1, start)
    end = min(chrom_length, end)
    return start, end


def _fallback_record(
    *,
    perturbation_id: str,
    reason: str,
    chrom: str,
    strand: str,
    start: int | None,
    end: int | None,
    window_size: int,
) -> SequenceWindowRecord:
    return SequenceWindowRecord(
        perturbation_id=perturbation_id,
        status="fallback",
        reason=reason,
        chrom=normalize_chrom_name(chrom) if chrom else "UNKNOWN",
        strand=strand or "+",
        locus_start=start or -1,
        locus_end=end or -1,
        center=-1,
        window_start=-1,
        window_end=-1,
        window_size_requested=window_size,
        window_size_observed=0,
        sequence="",
        cache_key="",
        from_cache=False,
    )


def _parse_fasta(path: Path) -> dict[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    sequences: dict[str, list[str]] = {}
    current_name: str | None = None

    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for raw_line in handle:
            line = raw_line.strip()
            if line == "":
                continue
            if line.startswith(">"):
                header = line[1:].split()[0]
                current_name = header
                sequences.setdefault(current_name, [])
                continue
            if current_name is None:
                raise SequenceWindowError(f"Malformed FASTA without header in {path}.")
            sequences[current_name].append(line)

    if len(sequences) == 0:
        raise SequenceWindowError(f"No sequences found in FASTA: {path}")
    return {name: "".join(parts) for name, parts in sequences.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _coerce_perturbation_table(raw_object: Any) -> pd.DataFrame | None:
    if isinstance(raw_object, pd.DataFrame):
        return raw_object.copy()
    if isinstance(raw_object, list) and all(isinstance(item, dict) for item in raw_object):
        return pd.DataFrame(cast(list[dict[str, Any]], raw_object))
    if isinstance(raw_object, dict):
        rows: list[dict[str, Any]] = []
        for key, value in raw_object.items():
            if not isinstance(value, dict):
                return None
            row = dict(value)
            row.setdefault("perturbation_id", str(key))
            rows.append(row)
        return pd.DataFrame(rows)
    return None


def save_sequence_window_cache(
    *,
    output_path: str | Path,
    table: pd.DataFrame,
    summary: SequenceWindowSummary,
) -> Path:
    path = Path(output_path)
    payload = {
        "summary": asdict(summary),
        "records": table.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
