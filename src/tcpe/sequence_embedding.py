"""Sequence embedding providers and AnnData annotation utilities for TCPE Phase 8."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

from tcpe.anndata_schema import PERTURBATIONS_UNS_KEY, SEQUENCE_EMBEDDING_OBSM_KEY
from tcpe.sequence_windows import SEQUENCE_WINDOWS_UNS_KEY

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

SEQUENCE_EMBEDDING_METADATA_UNS_KEY = "sequence_embedding_metadata"


class EmbeddingProviderError(RuntimeError):
    """Base error for embedding-provider failures."""


class EmbeddingProviderDisabledError(EmbeddingProviderError):
    """Raised when a real-model provider is disabled in local development mode."""


class EmbeddingDimensionError(EmbeddingProviderError):
    """Raised when embedding dimensions are inconsistent with provider metadata."""


class SequenceEmbeddingProvider(Protocol):
    """Interface contract for perturbation-sequence embedding providers."""

    provider_name: str
    provider_version: str
    embedding_dim: int

    def embed_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        """Embed ordered sequences into a 2D float matrix [n_sequences, embedding_dim]."""


@dataclass(frozen=True)
class SequenceEmbeddingMetadata:
    """Metadata persisted with sequence embeddings in AnnData."""

    provider_name: str
    provider_version: str
    embedding_dim: int
    source: str
    n_cells: int
    n_unique_perturbations: int
    n_sequences_embedded: int
    n_missing_sequences_fallback: int


class DeterministicMockSequenceEmbeddingProvider:
    """Deterministic hash-based sequence embedder for fast local development."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        seed: int = 42,
        provider_version: str = "mock_sequence_v1",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.provider_name = "deterministic_mock_sequence"
        self.provider_version = provider_version
        self.embedding_dim = embedding_dim
        self.seed = seed

    def embed_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        for idx, sequence in enumerate(sequences):
            vectors[idx] = self._embed_single(sequence)
        return vectors

    def _embed_single(self, sequence: str) -> np.ndarray:
        canonical = sequence.upper().strip()
        payload = f"{self.seed}|{canonical}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(rng_seed)
        return cast(np.ndarray, rng.normal(0.0, 1.0, size=self.embedding_dim).astype(np.float32))


class HuggingFaceSequenceEmbeddingProvider:
    """Skeleton adapter for real DNA foundation models (disabled by default locally)."""

    def __init__(
        self,
        *,
        model_id: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        embedding_dim: int = 256,
        enabled: bool = False,
        provider_version: str = "hf_sequence_skeleton_v1",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.model_id = model_id
        self.provider_name = f"hf_sequence::{model_id}"
        self.provider_version = provider_version
        self.embedding_dim = embedding_dim
        self.enabled = enabled

    def embed_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        if not self.enabled:
            raise EmbeddingProviderDisabledError(
                "HuggingFaceSequenceEmbeddingProvider is disabled in local mode. "
                "Use deterministic mock provider or enable explicitly in cloud runs."
            )
        raise NotImplementedError(
            "Phase 8 skeleton: real Hugging Face sequence embedding inference "
            "is not implemented yet."
        )


def annotate_sequence_embeddings(
    adata: AnnData,
    *,
    provider: SequenceEmbeddingProvider,
    sequences_by_perturbation: dict[str, str] | None = None,
) -> SequenceEmbeddingMetadata:
    """Generate sequence embeddings per cell and store them in canonical AnnData locations."""
    perturbation_ids = _get_perturbation_ids(adata)
    sequence_map, source_name = _resolve_sequence_map(
        adata=adata,
        perturbation_ids=perturbation_ids,
        override=sequences_by_perturbation,
    )

    unique_ids = sorted(set(perturbation_ids))
    available_ids = [
        pid
        for pid in unique_ids
        if pid in sequence_map and sequence_map[pid].strip() != ""
    ]
    missing_ids = sorted(set(unique_ids) - set(available_ids))

    sequence_payload = [sequence_map[pid] for pid in available_ids]
    if len(sequence_payload) > 0:
        embedded = provider.embed_sequences(sequence_payload)
        _validate_provider_output(
            matrix=embedded,
            expected_rows=len(sequence_payload),
            expected_dim=provider.embedding_dim,
            provider_name=provider.provider_name,
        )
        lookup = {pid: embedded[idx] for idx, pid in enumerate(available_ids)}
    else:
        lookup = {}

    fallback_vector = np.zeros((provider.embedding_dim,), dtype=np.float32)
    cell_embeddings = np.zeros((len(perturbation_ids), provider.embedding_dim), dtype=np.float32)
    for idx, perturbation_id in enumerate(perturbation_ids):
        cell_embeddings[idx] = lookup.get(perturbation_id, fallback_vector)

    adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY] = cell_embeddings.astype(np.float32)
    metadata = SequenceEmbeddingMetadata(
        provider_name=provider.provider_name,
        provider_version=provider.provider_version,
        embedding_dim=provider.embedding_dim,
        source=source_name,
        n_cells=int(adata.n_obs),
        n_unique_perturbations=len(unique_ids),
        n_sequences_embedded=len(available_ids),
        n_missing_sequences_fallback=len(missing_ids),
    )
    adata.uns[SEQUENCE_EMBEDDING_METADATA_UNS_KEY] = asdict(metadata)
    return metadata


def _get_perturbation_ids(adata: AnnData) -> list[str]:
    if "perturbation_id" not in adata.obs.columns:
        raise EmbeddingProviderError(
            "AnnData `.obs` must contain `perturbation_id` for sequence embedding."
        )
    values = adata.obs["perturbation_id"].astype(str).tolist()
    return cast(list[str], values)


def _resolve_sequence_map(
    *,
    adata: AnnData,
    perturbation_ids: list[str],
    override: dict[str, str] | None,
) -> tuple[dict[str, str], str]:
    if override is not None:
        normalized = {str(key): str(value) for key, value in override.items()}
        return normalized, "override_map"

    if SEQUENCE_WINDOWS_UNS_KEY in adata.uns:
        table = adata.uns[SEQUENCE_WINDOWS_UNS_KEY]
        if isinstance(table, pd.DataFrame):
            if "perturbation_id" in table.columns and "sequence" in table.columns:
                mapping = {
                    str(row["perturbation_id"]): str(row["sequence"])
                    for _, row in table.iterrows()
                }
                return mapping, "sequence_windows_uns"

    if PERTURBATIONS_UNS_KEY in adata.uns:
        perturb_table = _coerce_perturbation_table(adata.uns[PERTURBATIONS_UNS_KEY])
        if perturb_table is not None:
            mapping = {
                str(row["perturbation_id"]): _deterministic_placeholder_sequence(
                    str(row["perturbation_id"])
                )
                for _, row in perturb_table.iterrows()
                if "perturbation_id" in perturb_table.columns
            }
            return mapping, "perturbations_placeholder"

    unique_ids = sorted(set(perturbation_ids))
    fallback = {pid: _deterministic_placeholder_sequence(pid) for pid in unique_ids}
    return fallback, "fallback_placeholder"


def _deterministic_placeholder_sequence(perturbation_id: str) -> str:
    digest = hashlib.sha256(perturbation_id.encode("utf-8")).hexdigest()
    bases = np.array(list("ACGT"))
    seq_chars = [bases[int(char, 16) % 4] for char in digest[:120]]
    return "".join(seq_chars)


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
            record = dict(value)
            record.setdefault("perturbation_id", str(key))
            rows.append(record)
        return pd.DataFrame(rows)
    return None


def _validate_provider_output(
    *,
    matrix: np.ndarray,
    expected_rows: int,
    expected_dim: int,
    provider_name: str,
) -> None:
    if len(matrix.shape) != 2:
        raise EmbeddingDimensionError(
            f"{provider_name} returned shape {matrix.shape}; expected 2D matrix."
        )
    if matrix.shape[0] != expected_rows:
        raise EmbeddingDimensionError(
            f"{provider_name} returned {matrix.shape[0]} rows; expected {expected_rows}."
        )
    if matrix.shape[1] != expected_dim:
        raise EmbeddingDimensionError(
            f"{provider_name} returned embedding dim {matrix.shape[1]}; expected {expected_dim}."
        )
