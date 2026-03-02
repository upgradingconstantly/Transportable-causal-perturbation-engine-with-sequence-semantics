"""Phase 8 tests for embedding provider interfaces and mock implementations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from tcpe.anndata_schema import (
    CELL_STATE_EMBEDDING_OBSM_KEY,
    SEQUENCE_EMBEDDING_OBSM_KEY,
    validate_anndata_schema,
)
from tcpe.cell_embedding import (
    CELL_STATE_EMBEDDING_METADATA_UNS_KEY,
    DeterministicMockCellStateEmbeddingProvider,
    annotate_cell_state_embeddings,
)
from tcpe.ingestion import IngestionModule
from tcpe.sequence_embedding import (
    SEQUENCE_EMBEDDING_METADATA_UNS_KEY,
    DeterministicMockSequenceEmbeddingProvider,
    EmbeddingDimensionError,
    annotate_sequence_embeddings,
)
from tcpe.synthetic_data import generate_synthetic_dataset


def _require_anndata() -> None:
    pytest.importorskip("anndata")


def test_mock_sequence_provider_is_deterministic() -> None:
    provider_a = DeterministicMockSequenceEmbeddingProvider(embedding_dim=32, seed=11)
    provider_b = DeterministicMockSequenceEmbeddingProvider(embedding_dim=32, seed=11)
    sequences = ["ACGTACGT", "TTTTCCCC", "GGGGAAAA"]

    emb_a = provider_a.embed_sequences(sequences)
    emb_b = provider_b.embed_sequences(sequences)
    assert emb_a.shape == (3, 32)
    assert emb_b.shape == (3, 32)
    np.testing.assert_allclose(emb_a, emb_b)


def test_mock_cell_provider_is_deterministic() -> None:
    provider_a = DeterministicMockCellStateEmbeddingProvider(embedding_dim=24, seed=7)
    provider_b = DeterministicMockCellStateEmbeddingProvider(embedding_dim=24, seed=7)
    expression = np.array([[1.0, 0.0, 3.0], [2.0, 5.0, 1.0]], dtype=np.float64)

    emb_a = provider_a.embed_expression(expression)
    emb_b = provider_b.embed_expression(expression)
    assert emb_a.shape == (2, 24)
    np.testing.assert_allclose(emb_a, emb_b)


@dataclass
class AlternateMockSequenceProvider:
    """Alternative provider used to test interface stability under provider switching."""

    provider_name: str = "alternate_sequence_mock"
    provider_version: str = "alt_v1"
    embedding_dim: int = 16

    def embed_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        for idx, seq in enumerate(sequences):
            matrix[idx, :] = float(len(seq) % 5)
        return matrix


def test_provider_switch_changes_metadata_not_slot_interface() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=140, n_genes=120, n_perturbations=8, seed=5)
    adata = bundle.adata

    seq_provider_1 = DeterministicMockSequenceEmbeddingProvider(embedding_dim=16, seed=101)
    seq_provider_2 = AlternateMockSequenceProvider(embedding_dim=16)
    cell_provider = DeterministicMockCellStateEmbeddingProvider(embedding_dim=16, seed=101)

    metadata_1 = annotate_sequence_embeddings(adata=adata, provider=seq_provider_1)
    annotate_cell_state_embeddings(adata=adata, provider=cell_provider)
    shape_before = adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY].shape

    metadata_2 = annotate_sequence_embeddings(adata=adata, provider=seq_provider_2)
    shape_after = adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY].shape

    assert metadata_1.provider_name != metadata_2.provider_name
    provider_name = adata.uns[SEQUENCE_EMBEDDING_METADATA_UNS_KEY]["provider_name"]
    assert provider_name == metadata_2.provider_name
    assert shape_before == shape_after
    assert SEQUENCE_EMBEDDING_OBSM_KEY in adata.obsm
    assert CELL_STATE_EMBEDDING_OBSM_KEY in adata.obsm


def test_embedding_dimension_mismatch_fails_early() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=60, n_genes=80, n_perturbations=6, seed=2)
    adata = bundle.adata

    @dataclass
    class BadSequenceProvider:
        provider_name: str = "bad_sequence_provider"
        provider_version: str = "v0"
        embedding_dim: int = 10

        def embed_sequences(self, sequences: Sequence[str]) -> np.ndarray:
            return np.zeros((len(sequences), 9), dtype=np.float32)

    with pytest.raises(EmbeddingDimensionError, match="embedding dim"):
        annotate_sequence_embeddings(adata=adata, provider=BadSequenceProvider())


def test_mock_embedding_pipeline_end_to_end_on_synthetic() -> None:
    _require_anndata()
    bundle = generate_synthetic_dataset(n_cells=180, n_genes=200, n_perturbations=10, seed=17)
    module = IngestionModule()
    adata = bundle.adata

    seq_meta = module.annotate_sequence_embeddings(adata)
    cell_meta = module.annotate_cell_state_embeddings(adata)

    assert SEQUENCE_EMBEDDING_OBSM_KEY in adata.obsm
    assert CELL_STATE_EMBEDDING_OBSM_KEY in adata.obsm
    assert adata.obsm[SEQUENCE_EMBEDDING_OBSM_KEY].shape[0] == adata.n_obs
    assert adata.obsm[CELL_STATE_EMBEDDING_OBSM_KEY].shape[0] == adata.n_obs
    assert SEQUENCE_EMBEDDING_METADATA_UNS_KEY in adata.uns
    assert CELL_STATE_EMBEDDING_METADATA_UNS_KEY in adata.uns
    assert int(seq_meta["embedding_dim"]) > 0
    assert int(cell_meta["embedding_dim"]) > 0

    report = validate_anndata_schema(adata=adata, mode="strict")
    assert report.is_valid
