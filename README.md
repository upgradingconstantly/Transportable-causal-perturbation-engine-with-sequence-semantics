# Transportable Causal Perturbation Engine (TCPE)

Phase 1-18 bootstrap for the TCPE project.

## Phase 1-18 Contents
- Python package scaffold with six module stubs:
  - ingestion
  - sequence embedding
  - cell-state embedding
  - transport
  - causal
  - evaluation
- Shared config/types/runtime utilities
- Deterministic seed helpers
- W&B offline-first scaffold
- CLI scaffold with command groups:
  - data
  - embed
  - train
  - causal
  - eval
  - card
  - pipeline
  - cloud
  - cloud-exec
- Hierarchical config overlays:
  - `configs/base.yaml`
  - `configs/env/local.yaml`
  - `configs/env/kaggle.yaml`
  - `configs/env/azure.yaml`
- Lint/type/test tooling and CI
- Canonical AnnData schema validator (`strict` and `repair` modes):
  - required `.obs`, `.var`, `.layers`, `.obsm`, and `.uns` contract checks
  - schema version marker support
  - repair-mode safe default filling for repairable fields
- Synthetic data generator (Phase 4):
  - deterministic in-memory Poisson counts (`500` cells x `200` genes, `10` perturbations)
  - simulated confounders, knockdown efficiency variation, and causal graph effects
  - exports `.h5ad` + ground-truth arrays/metadata for testing
- Dataset ingestion layer (Phase 5):
  - pluggable dataset registry with `adamson`, `replogle_sample`, and `lincs_stub` loaders
  - loader-level raw-source caching + checksum-validated processed cache reuse
  - Replogle local hard cap enforcement at `5,000` cells
  - metadata harmonization into canonical AnnData schema v1 fields
  - LINCS adapter intentionally stubbed with explicit `NotImplementedError`
- Preprocessing and QC module (Phase 6):
  - library-size normalization + `log1p` transform in a standard preprocessing path
  - HVG selection with configurable target (default `2,000`, allowed range `2,000-5,000`)
  - QC filters for low-quality cells and genes
  - deterministic preprocessing metadata recorded in `.uns['preprocessing']`
  - optional confounder proxy extraction to `.obsm['X_confounder_proxy']`
- Sequence-window extraction module (Phase 7):
  - genomic reference abstraction for FASTA loading and chromosome lookup
  - local reference mode for Ensembl GRCh38 release 110 chromosome 22 FASTA
  - locus-centered window extraction (`1000-2000bp`) with edge/boundary handling
  - strand-aware sequence orientation via reverse complement for `-` strand
  - cache keyed by perturbation locus and extraction settings
  - audit metadata persisted in `.uns['sequence_window_audit']`
- Embedding provider layer (Phase 8):
  - `SequenceEmbeddingProvider` and `CellStateEmbeddingProvider` interface contracts
  - deterministic mock sequence and cell-state embedders for local fast runs
  - Hugging Face adapter skeletons for DNA/cell foundation models (disabled locally by default)
  - canonical embedding storage in:
    - `.obsm['X_sequence_embedding']`
    - `.obsm['X_cell_state_embedding']`
  - provider metadata persisted in `.uns` with provider name/version/source/dimension
- Baseline suite + transport pre-training gate (Phase 9):
  - required baselines:
    - `gene_level_mean`
    - `control_mean`
    - `linear_regression`
    - `random_edge_grn` (degree-matching randomization)
  - uniform baseline result schema (`baseline_suite_v1`)
  - evaluator hook to persist baseline outputs for downstream reporting
  - transport readiness gate that blocks training until all mandatory baselines are present
- Transport module core (Phase 10):
  - `TransportStrategy` interface with `fit`, `predict_distribution`, `save`, and `load`
  - OT Sinkhorn strategy with entropic regularization and checkpoint schema versioning
  - latent encoder + uncertainty head for predictive variance outputs
  - additive conditioning with separate sequence/cell-state projection heads to shared latent dim
  - optional Lightning-compatible wrapper + synchronized local/W&B metric logging
- Flow/bridge pluggability scaffolds (Phase 11):
  - flow-matching transport scaffold wired to the shared strategy interface
  - Schrodinger-bridge scaffold with manual-optimization Lightning-style module placeholder
  - config/CLI variant dispatch for `ot`, `flow`, and `bridge` without code changes
  - explicit scaffold status tags: `experimental`, `cloud-recommended`
- Causal module v1 (Phase 12):
  - two-stage IV estimator with perturbation instruments and proxy-conditioned regressions
  - required proxy extraction from AnnData `.obs`: `batch`, `library_size`, `knockdown_efficiency_proxy`, `protocol`
  - same proxy design matrix passed to both Stage 1 and Stage 2
  - cyclic directed adjacency estimation (no DAG constraint)
  - bootstrap uncertainty intervals for all directed edges + artifact export (`npz`, `csv`, `json`)
- Context-shift split engine (Phase 13):
  - deterministic split generation across four shift types:
    - `cell_type`
    - `locus`
    - `dose_strength`
    - `protocol`
  - leakage checks for cell-index overlap, group overlap, and full-cell coverage
  - undersized split fallback logic with explicit manifest-level fallback metadata
  - auditable split manifests with provenance and deterministic seed derivation
- Evaluator + model card generator (Phase 14):
  - required metrics:
    - `mae`
    - `pearson_top_1000_degs`
    - `calibration_error`
  - baseline-vs-model comparison table generation for expression baselines
  - failure-mode section generation from threshold checks and baseline underperformance
  - model card output to JSON + Markdown with strict schema validation
  - evaluation run completion gate that fails unless model card artifacts exist and validate
- End-to-end pipeline orchestrator (Phase 15):
  - single pipeline run chain:
    - ingest
    - preprocess
    - embed
    - train
    - causal
    - evaluate
    - model card
  - step-level checkpointing and resume support for interrupted/partial runs
  - standardized artifact manifest for cloud handoff packaging
  - cloud handoff bundle export with required output files
  - classified run failures (`data`, `config`, `model`, `infra`, `unknown`)
- Local validation ladder (Phase 16):
  - strict promotion-gate runner for:
    - synthetic
    - adamson
    - replogle_sample
  - stage runtime and peak-memory collection with per-stage limits
  - baseline-dominance gate checks for `control_mean` and `gene_level_mean`
  - synthetic causal recovery gate checks (AUC/F1 thresholds)
  - local eligibility report output:
    - `local_validation_report.json`
    - `local_validation_report.md`
- Cloud handoff preparation (Phase 17):
  - Oracle VM + Kaggle split presets aligned to the confirmed infra contract
  - resume-ready checkpoint seeding from externally preprocessed HVG-selected `.h5ad`
  - GitHub/Hugging Face/Zenodo artifact transfer command planning
  - budget log + interruption recovery simulation
  - tmux-oriented cloud runbook generation
- Cloud execution sequence (Phase 18):
  - exact eight-step Oracle/Kaggle launcher generation in project-plan order
  - OCI Python SDK object-storage transfer helper for bucket I/O
  - tmux launcher script for Oracle jobs and execution runbook output
  - runtime timeout guards, shutdown instructions, and initialized sequence checkpoint
  - Kaggle cloud mode defaults updated to `WANDB_MODE=online`

## Runtime
- Python `3.11`

## Local Setup (Windows PowerShell)
```powershell
.\scripts\bootstrap.ps1
.\scripts\check.ps1
```

## Local Setup (Linux/macOS)
```bash
bash scripts/bootstrap.sh
bash scripts/check.sh
```

## Developer Workflow
1. Install git hooks:
   ```bash
   .venv/bin/pre-commit install
   ```
2. Run checks before commit:
   ```bash
   bash scripts/check.sh
   ```
3. CI runs `ruff`, `mypy`, and `pytest` on push/pull request.

## CLI Usage
```bash
python -m tcpe --help
python -m tcpe data --help
python -m tcpe train --env local --dry-run
python -m tcpe cloud --env local --dry-run
python -m tcpe cloud-exec --env local --dry-run
```

Example output includes validated config environment, deterministic run id, and resolved artifact layout.

## W&B Local Defaults
- `WANDB_MODE=offline` by default via `.env.example`.
- Artifacts are written under `artifacts/wandb/`.
