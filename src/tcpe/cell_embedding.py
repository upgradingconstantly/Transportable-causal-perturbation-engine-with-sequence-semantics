"""Cell-state embedding providers and AnnData annotation utilities for TCPE Phase 8."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from scipy import sparse

from tcpe.anndata_schema import CELL_STATE_EMBEDDING_OBSM_KEY, NORMALIZED_LAYER_KEY
from tcpe.sequence_embedding import (
    EmbeddingDimensionError,
    EmbeddingProviderDisabledError,
    EmbeddingProviderError,
)

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any

CELL_STATE_EMBEDDING_METADATA_UNS_KEY = "cell_state_embedding_metadata"


class CellStateEmbeddingProvider(Protocol):
    """Interface contract for cell-state embedding providers."""

    provider_name: str
    provider_version: str
    embedding_dim: int

    def embed_expression(self, expression_matrix: np.ndarray) -> np.ndarray:
        """Embed per-cell expression matrix [n_cells, n_genes] into [n_cells, embedding_dim]."""


@dataclass(frozen=True)
class CellStateEmbeddingMetadata:
    """Metadata persisted with cell-state embeddings in AnnData."""

    provider_name: str
    provider_version: str
    embedding_dim: int
    source_layer: str
    n_cells: int
    n_genes_input: int


class DeterministicMockCellStateEmbeddingProvider:
    """Deterministic projection-based mock embedder for local development."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        seed: int = 42,
        provider_version: str = "mock_cell_state_v1",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.provider_name = "deterministic_mock_cell_state"
        self.provider_version = provider_version
        self.embedding_dim = embedding_dim
        self.seed = seed

    def embed_expression(self, expression_matrix: np.ndarray) -> np.ndarray:
        if len(expression_matrix.shape) != 2:
            raise EmbeddingProviderError("expression_matrix must be 2D [n_cells, n_genes].")
        n_cells, n_genes = expression_matrix.shape
        if n_cells == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if n_genes == 0:
            return np.zeros((n_cells, self.embedding_dim), dtype=np.float32)

        standardized = _row_standardize(expression_matrix.astype(np.float64))
        projection = self._projection_matrix(n_genes=n_genes)
        embeddings = standardized @ projection
        return cast(np.ndarray, embeddings.astype(np.float32))

    def _projection_matrix(self, *, n_genes: int) -> np.ndarray:
        payload = f"{self.seed}|{n_genes}|{self.embedding_dim}"
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(rng_seed)
        matrix = rng.normal(0.0, 1.0, size=(n_genes, self.embedding_dim))
        matrix /= np.sqrt(max(n_genes, 1))
        return matrix


class HuggingFaceCellStateEmbeddingProvider:
    """Skeleton adapter for real single-cell foundation models (disabled by default locally)."""

    def __init__(
        self,
        *,
        model_id: str = "bowang-lab/scGPT",
        embedding_dim: int = 256,
        enabled: bool = False,
        provider_version: str = "hf_cell_state_skeleton_v1",
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.model_id = model_id
        self.provider_name = f"hf_cell_state::{model_id}"
        self.provider_version = provider_version
        self.embedding_dim = embedding_dim
        self.enabled = enabled

    def embed_expression(self, expression_matrix: np.ndarray) -> np.ndarray:
        if not self.enabled:
            raise EmbeddingProviderDisabledError(
                "HuggingFaceCellStateEmbeddingProvider is disabled in local mode. "
                "Use deterministic mock provider or enable explicitly in cloud runs."
            )
        _ = expression_matrix
        raise NotImplementedError(
            "Phase 8 skeleton: real Hugging Face cell-state embedding inference "
            "is not implemented yet."
        )


def annotate_cell_state_embeddings(
    adata: AnnData,
    *,
    provider: CellStateEmbeddingProvider,
    source_layer: str = NORMALIZED_LAYER_KEY,
) -> CellStateEmbeddingMetadata:
    """Generate cell-state embeddings and store in canonical AnnData locations."""
    matrix = _resolve_expression_matrix(adata=adata, source_layer=source_layer)
    embedded = provider.embed_expression(matrix)
    _validate_provider_output(
        matrix=embedded,
        expected_rows=adata.n_obs,
        expected_dim=provider.embedding_dim,
        provider_name=provider.provider_name,
    )

    adata.obsm[CELL_STATE_EMBEDDING_OBSM_KEY] = embedded.astype(np.float32)
    metadata = CellStateEmbeddingMetadata(
        provider_name=provider.provider_name,
        provider_version=provider.provider_version,
        embedding_dim=provider.embedding_dim,
        source_layer=source_layer,
        n_cells=int(adata.n_obs),
        n_genes_input=int(matrix.shape[1]),
    )
    adata.uns[CELL_STATE_EMBEDDING_METADATA_UNS_KEY] = asdict(metadata)
    return metadata


def _resolve_expression_matrix(adata: AnnData, source_layer: str) -> np.ndarray:
    if source_layer in adata.layers:
        matrix = adata.layers[source_layer]
    elif source_layer == "X":
        matrix = adata.X
    else:
        raise EmbeddingProviderError(
            f"Requested source_layer '{source_layer}' not found in AnnData layers."
        )

    if sparse.issparse(matrix):
        return cast(np.ndarray, matrix.toarray().astype(np.float64))
    return cast(np.ndarray, np.asarray(matrix, dtype=np.float64))


def _row_standardize(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, keepdims=True)
    safe_stds = np.where(stds <= 1e-8, 1.0, stds)
    return cast(np.ndarray, (matrix - means) / safe_stds)


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
