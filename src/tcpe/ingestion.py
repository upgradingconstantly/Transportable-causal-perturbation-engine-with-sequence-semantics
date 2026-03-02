"""Ingestion module placeholder."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcpe.anndata_schema import AnnDataValidationReport, ValidationMode, validate_anndata_schema
from tcpe.cell_embedding import (
    CellStateEmbeddingProvider,
    DeterministicMockCellStateEmbeddingProvider,
    annotate_cell_state_embeddings,
)
from tcpe.dataset_loaders import (
    DatasetLoadResult,
    DatasetRegistry,
    build_default_dataset_registry,
)
from tcpe.preprocessing import (
    PreprocessingConfig,
    PreprocessingModule,
    PreprocessingResult,
)
from tcpe.sequence_embedding import (
    DeterministicMockSequenceEmbeddingProvider,
    SequenceEmbeddingProvider,
    annotate_sequence_embeddings,
)
from tcpe.sequence_windows import (
    GenomeReference,
    SequenceWindowConfig,
    SequenceWindowExtractor,
    SequenceWindowSummary,
)
from tcpe.synthetic_data import SyntheticDatasetBundle, generate_synthetic_dataset

if TYPE_CHECKING:
    from anndata import AnnData
else:
    AnnData = Any


class IngestionModule:
    """Phase 8 ingestion scaffold with loading, preprocessing, windows, and embeddings."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self._registry = registry if registry is not None else build_default_dataset_registry()
        self._preprocessor = PreprocessingModule()

    def status(self) -> str:
        return "phase8_embedding_ready"

    def available_datasets(self) -> list[str]:
        """List all registered dataset loaders."""
        return self._registry.available()

    def validate_schema(
        self, adata: AnnData, mode: ValidationMode = "strict"
    ) -> AnnDataValidationReport:
        """Validate input AnnData against the canonical TCPE schema."""
        return validate_anndata_schema(adata=adata, mode=mode)

    def build_synthetic_dataset(
        self,
        n_cells: int = 500,
        n_genes: int = 200,
        n_perturbations: int = 10,
        seed: int = 42,
    ) -> SyntheticDatasetBundle:
        """Create the canonical Phase 4 synthetic dataset bundle."""
        return generate_synthetic_dataset(
            n_cells=n_cells,
            n_genes=n_genes,
            n_perturbations=n_perturbations,
            seed=seed,
        )

    def load_dataset(
        self,
        dataset_id: str,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> DatasetLoadResult:
        """Load a dataset via the registry using harmonized schema-compliant output."""
        return self._registry.load(
            dataset_id=dataset_id,
            cache_dir=cache_dir,
            source_uri=source_uri,
            force_refresh=force_refresh,
            **kwargs,
        )

    def load_adamson(
        self,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
    ) -> DatasetLoadResult:
        """Load and harmonize Adamson data."""
        return self.load_dataset(
            "adamson",
            cache_dir=cache_dir,
            source_uri=source_uri,
            force_refresh=force_refresh,
        )

    def load_replogle_sample(
        self,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
        max_cells: int | None = None,
    ) -> DatasetLoadResult:
        """Load and harmonize Replogle sample data with local max-cell cap."""
        return self.load_dataset(
            "replogle_sample",
            cache_dir=cache_dir,
            source_uri=source_uri,
            force_refresh=force_refresh,
            max_cells=max_cells,
        )

    def load_lincs_stub(
        self,
        *,
        cache_dir: str | Path,
        source_uri: str | Path | None = None,
        force_refresh: bool = False,
    ) -> DatasetLoadResult:
        """Invoke the LINCS adapter stub (raises NotImplementedError)."""
        return self.load_dataset(
            "lincs_stub",
            cache_dir=cache_dir,
            source_uri=source_uri,
            force_refresh=force_refresh,
        )

    def preprocess_adata(
        self,
        adata: AnnData,
        config: PreprocessingConfig | None = None,
    ) -> PreprocessingResult:
        """Run standardized Phase 6 preprocessing on an AnnData object."""
        return self._preprocessor.run(adata=adata, config=config)

    def build_local_chr22_reference(
        self,
        *,
        cache_dir: str | Path,
        force_refresh: bool = False,
    ) -> GenomeReference:
        """Create local development reference from Ensembl GRCh38 chr22 FASTA."""
        return GenomeReference.from_local_chr22_reference(
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )

    def extract_sequence_windows(
        self,
        adata: AnnData,
        *,
        reference: GenomeReference,
        config: SequenceWindowConfig | None = None,
    ) -> SequenceWindowSummary:
        """Extract and store sequence windows for all perturbations in AnnData."""
        extractor = SequenceWindowExtractor(reference=reference, config=config)
        return extractor.annotate_anndata(adata)

    def annotate_sequence_embeddings(
        self,
        adata: AnnData,
        *,
        provider: SequenceEmbeddingProvider | None = None,
        sequences_by_perturbation: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Annotate sequence embeddings using mock provider by default."""
        selected = (
            provider
            if provider is not None
            else DeterministicMockSequenceEmbeddingProvider()
        )
        metadata = annotate_sequence_embeddings(
            adata=adata,
            provider=selected,
            sequences_by_perturbation=sequences_by_perturbation,
        )
        return asdict(metadata)

    def annotate_cell_state_embeddings(
        self,
        adata: AnnData,
        *,
        provider: CellStateEmbeddingProvider | None = None,
        source_layer: str = "normalized_log1p",
    ) -> dict[str, Any]:
        """Annotate cell-state embeddings using mock provider by default."""
        selected = (
            provider
            if provider is not None
            else DeterministicMockCellStateEmbeddingProvider()
        )
        metadata = annotate_cell_state_embeddings(
            adata=adata,
            provider=selected,
            source_layer=source_layer,
        )
        return asdict(metadata)
