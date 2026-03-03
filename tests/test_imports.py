"""Phase 1 import smoke tests."""

import importlib


def test_package_imports() -> None:
    modules = [
        "tcpe",
        "tcpe.anndata_schema",
        "tcpe.cli",
        "tcpe.baselines",
        "tcpe.config",
        "tcpe.dataset_loaders",
        "tcpe.preprocessing",
        "tcpe.types",
        "tcpe.ingestion",
        "tcpe.sequence_embedding",
        "tcpe.sequence_windows",
        "tcpe.cell_embedding",
        "tcpe.transport",
        "tcpe.causal",
        "tcpe.context_shift",
        "tcpe.cloud_handoff",
        "tcpe.pipeline",
        "tcpe.local_validation",
        "tcpe.evaluation",
        "tcpe.runtime.seed",
        "tcpe.runtime.run_context",
        "tcpe.runtime.wandb_scaffold",
        "tcpe.synthetic_data",
    ]

    for module_name in modules:
        imported = importlib.import_module(module_name)
        assert imported is not None
