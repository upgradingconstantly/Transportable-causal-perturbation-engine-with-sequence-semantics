# Transportable Causal Perturbation Engine (TCPE) Execution Plan
## Local-First Coding + Small-Dataset Validation + Cloud-Ready Handoff

## Summary
This plan assumes the repository currently contains only the two specification documents and no implementation code.  
The implementation sequence is fixed to match your requirement: code locally, validate with progressively larger small datasets, and only then move heavy runs to cloud compute (Kaggle/Azure).  
The plan is designed to make the system scientifically credible and engineering-complete for the six required components: ingestion/standardization, sequence embedding, cell-state embedding, transport model, causal module, and mandatory context-shift evaluator/model card generation.  
Baselines are now implemented before any transport model training.

## Locked Decisions (from your answers)
1. Workflow is `Package + CLI`.
2. Experiment tracking is `W&B + local logs`.
3. Local validation data ladder is fixed: `in-memory synthetic -> Adamson (small real dataset) -> sampled Replogle K562 (5,000 cells)`.
4. LINCS L1000 is in scope as `adapter stub now`, full implementation later.
5. Foundation-model integration strategy is `deterministic mock embedders first`.
6. Training framework is `PyTorch Lightning wrappers` with `plain PyTorch nn.Module models`; Schrödinger bridge uses Lightning `manual_optimization`.

## Phase-by-Phase Implementation Plan

### Phase 1 - Repository and Engineering Bootstrap
Objective: establish reproducible engineering foundation before domain logic.

Build steps:
1. Create Python package layout for six components plus shared config/types/utilities.
2. Standardize runtime on Python 3.11.
3. Add dependency management with lockfile and environment bootstrap scripts for Windows (local) and Linux (cloud).
4. Add code quality tooling: `ruff`, `mypy`, `pytest`, `pre-commit`.
5. Add CI pipeline to run lint/type/unit tests on every push.
6. Set deterministic defaults for random seeds and numeric backend behavior.
7. Add W&B integration scaffold with `offline` mode default locally.

Validation:
1. `pip/conda install` succeeds on laptop.
2. CI runs green on a trivial test.
3. Package imports cleanly.
4. Offline W&B run writes local artifacts.

Exit gate:
All developer setup commands run from clean checkout in under 15 minutes.

---

### Phase 2 - CLI and Configuration System
Objective: make all workflows scriptable and cloud-portable from day one.

Build steps:
1. Implement CLI entrypoint with command groups: `data`, `embed`, `train`, `causal`, `eval`, `card`, `pipeline`.
2. Implement hierarchical config system with environment overlays: `local`, `kaggle`, `azure`.
3. Add typed config validation at startup (paths, memory limits, dataset identifiers, model options).
4. Add run ID generation and artifact directory conventions.

Validation:
1. `--help` works for all commands.
2. Invalid configs fail fast with actionable errors.
3. Local run IDs produce deterministic directory structure.

Exit gate:
Every future phase can be executed via CLI with config only, no notebook dependence.

---

### Phase 3 - Canonical AnnData Contract (Schema v1)
Objective: lock data contracts so all modules interoperate without ad hoc assumptions.

Build steps:
1. Define canonical AnnData requirements for `.X`, `.obs`, `.var`, `.layers`, `.obsm`, `.uns`.
2. Enforce raw counts in `.X` and normalized values in dedicated layers.
3. Define required `.obs` fields: `cell_id`, `cell_type`, `batch`, `condition`, `protocol`, `library_size`, `perturbation_id`, `knockdown_efficiency_proxy`.
4. Define required `.var` fields: `gene_id`, `gene_symbol`, `chrom`, `strand`, `tss`.
5. Define required perturbation metadata object with locus support: `target_gene`, `chrom`, `start`, `end`, `strand`, `modality`, `dose`, `gRNA_id`.
6. Define required embedding slots: sequence embeddings and cell-state embeddings.
7. Implement schema validator with strict mode and repair mode.

Validation:
1. Synthetic AnnData passes strict validation.
2. Corrupt fixtures fail with precise error messages.
3. Repair mode can fill safe defaults for optional fields.

Exit gate:
No downstream module accepts AnnData unless schema validation passes.

---

### Phase 4 - Synthetic Dataset Generator (Primary Local Test Bed)
Objective: create deterministic, fast, no-download data to validate interfaces and algorithms.

Build steps:
1. Generate in-memory Poisson count matrices with target shape `500 cells x 200 genes`.
2. Simulate `10 perturbations` with known causal graph and known confounders.
3. Simulate variable knockdown efficiency and batch effects.
4. Generate ground-truth transport shift and ground-truth causal adjacency for benchmark assertions.
5. Export synthetic AnnData plus truth metadata for testing.

Validation:
1. Generator outputs identical data with same seed.
2. Ground-truth files align with generated IDs.
3. Synthetic data passes AnnData schema v1.

Exit gate:
Synthetic fixture becomes default dataset for unit/integration tests.

---

### Phase 5 - Data Ingestion Layer (Real Data Ready)
Objective: implement real-data entrypoints without overloading laptop resources.

Build steps:
1. Build dataset registry with pluggable loaders.
2. Implement Adamson loader (download, parse, map into schema v1).
3. Implement sampled Replogle loader with hard cap to 5,000 cells locally.
4. Add loader-level caching and checksum validation.
5. Implement LINCS adapter stub with full interface contract and explicit `NotImplemented` for heavy preprocessing.
6. Add metadata harmonization rules across datasets.

Validation:
1. Adamson ingest pipeline produces schema-valid AnnData.
2. Replogle sample loader enforces 5,000-cell cap.
3. LINCS stub passes interface tests and emits clear warning.
4. Loader re-run uses cache and is idempotent.

Exit gate:
Local machine can ingest Adamson and sampled Replogle without exceeding memory budget.

---

### Phase 6 - Preprocessing and QC Module
Objective: standardize preprocessing exactly once and reuse everywhere.

Build steps:
1. Implement library-size normalization.
2. Implement `log1p` transform.
3. Implement HVG selection with configurable target (`2,000` default, range `2,000-5,000`).
4. Implement QC filters for low-quality cells/genes.
5. Record preprocessing parameters in `.uns` for reproducibility.
6. Add optional confounder proxy extraction from technical metadata.

Validation:
1. Preprocessing outputs deterministic given same seed/config.
2. HVG count matches configured target or documented fallback.
3. QC removes known bad synthetic cells.
4. Output AnnData still passes schema validator.

Exit gate:
Any raw ingested dataset can be transformed into standardized model-ready AnnData with one command.

---

### Phase 7 - Genomic Reference and Sequence Window Extraction
Objective: implement locus-aware perturbation representation foundation.

Build steps:
1. Implement GRCh38 reference access abstraction.
2. Implement local development reference mode using the Ensembl GRCh38 chromosome 22 FASTA from `https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/`.
3. Implement window extraction centered on perturbation site with configurable width (`1000-2000 bp`).
4. Handle edge cases: chromosome boundaries, missing coordinates, strand orientation.
5. Add sequence cache keyed by perturbation locus.
6. Persist sequence-window metadata for auditability.

Validation:
1. Boundary tests for start/end clipping pass.
2. Reverse-strand extraction consistency tests pass.
3. Missing-coordinate cases route to explicit fallback policy.
4. Extracted windows are deterministic and cacheable.

Exit gate:
Each perturbation in schema v1 has either valid sequence window or explicit fallback reason.

---

### Phase 8 - Embedding Provider Interfaces and Deterministic Mocks
Objective: make foundation-model integration swappable while keeping local tests fast.

Build steps:
1. Define `SequenceEmbeddingProvider` interface.
2. Define `CellStateEmbeddingProvider` interface.
3. Implement deterministic mock sequence embedder (hash/seed based vectors).
4. Implement deterministic mock cell embedder.
5. Implement adapter skeletons for real HF models (Nucleotide Transformer/HyenaDNA/DNABERT-2 and scGPT/Geneformer), disabled by default locally.
6. Store embeddings in standardized AnnData slots and persist metadata about provider/version/dim.

Validation:
1. Mock embedders return stable vectors across runs.
2. Provider switching changes source metadata but not interface.
3. Embedding dimension validation catches mismatch early.
4. End-to-end pipeline works with mock providers only.

Exit gate:
All downstream training/evaluation runs without real foundation-model downloads.

---

### Phase 9 - Baseline Suite (Mandatory Honesty Gate)
Objective: enforce required baseline comparisons before transport training begins.

Build steps:
1. Implement gene-level mean baseline.
2. Implement control mean baseline.
3. Implement linear regression perturbation baseline.
4. Implement random-edge GRN baseline with degree-matching.
5. Add uniform baseline result schema.
6. Integrate baseline execution hooks so baseline outputs are available for pre-training validation and later evaluator use.

Validation:
1. Baselines execute on synthetic/Adamson/Replogle-sample without manual intervention.
2. Random GRN baseline preserves requested degree distribution.
3. Baseline outputs are directly comparable to main model outputs.
4. Baseline commands run successfully before any transport model training command is allowed.

Exit gate:
No transport training is permitted until baseline suite is runnable and validated.

---

### Phase 10 - Transport Module Core (OT First, Pluggable by Design)
Objective: implement first real transport model with architecture ready for additional variants.

Build steps:
1. Define transport strategy interface with standardized `fit`, `predict_distribution`, and artifact serialization.
2. Implement latent encoder `nn.Module` (framework-agnostic).
3. Implement OT coupling variant using entropic regularization (POT/Sinkhorn).
4. Wrap training in Lightning `LightningModule` for logging/checkpointing.
5. Implement conditioning with two separate learned linear projection heads, one for sequence embeddings and one for cell-state embeddings.
6. Project both embedding types into the same latent conditioning dimension, then combine by element-wise addition.
7. Explicitly disallow raw concatenation as the conditioning combine operator.
8. Implement predictive uncertainty output (sampling-based or residual variance head).
9. Add checkpoint schema versioning.

Validation:
1. Training loop converges on synthetic data.
2. Predicted perturbed distribution improves over trivial baseline on synthetic data.
3. Model checkpoint save/load is lossless.
4. Lightning logs and local W&B logs are synchronized.
5. Unit tests verify both projection heads map to same dimension and the combiner path is element-wise addition only.

Exit gate:
OT variant can train/infer locally on synthetic and Adamson-scale data.

---

### Phase 11 - Flow and Schrödinger Bridge Scaffolds
Objective: keep pluggability promise while avoiding premature heavy implementation locally.

Build steps:
1. Implement flow-matching module skeleton with shared interface compliance.
2. Implement Schrödinger-bridge module skeleton with Lightning `manual_optimization`.
3. Wire both into CLI/config dispatch.
4. Add explicit status tags: `experimental`, `cloud-recommended`.

Validation:
1. Both variants instantiate and pass smoke tests.
2. Config selects variant without code changes.
3. Unsupported paths fail gracefully with explicit message.

Exit gate:
Transport family is pluggable now; full heavy training deferred to cloud phases.

---

### Phase 12 - Causal Module v1 (IV + Confounder Proxies + Uncertainty)
Objective: implement confounder-robust causal GRN estimation compatible with Perturb-seq realities.

Build steps:
1. Define causal estimator interface with standardized input/output contracts.
2. Implement two-stage IV pipeline inspired by ADAPRE logic.
3. Extract proxy variables from AnnData schema v1 `.obs` using exactly: `batch`, `library_size`, `knockdown_efficiency_proxy`, `protocol`.
4. Build a proxy design matrix with consistent encoding and pass it to Stage 1 and Stage 2.
5. Stage 1 uses perturbation indicators plus the proxy matrix to obtain purified endogenous signals.
6. Stage 2 estimates directional causal effects among HVGs using purified signals while also conditioning on the same proxy matrix.
7. Allow cyclic adjacency estimation (no DAG enforcement).
8. Add uncertainty estimation per edge (bootstrap or robust standard errors).
9. Export graph artifacts with weights, confidence intervals, and metadata.

Validation:
1. Synthetic ground-truth causal graph recovery exceeds minimum threshold.
2. Known confounding synthetic scenarios show reduced bias vs naive regression.
3. Cyclic synthetic graph can be recovered without crashes.
4. Uncertainty intervals are emitted for all edges.
5. Missing any of the required proxy fields triggers explicit schema/fit-time failure.

Exit gate:
Causal module produces weighted, uncertain, cycle-permitting GRN artifact on local datasets.

---

### Phase 13 - Context-Shift Split Engine
Objective: formalize and automate shift-aware evaluation.

Build steps:
1. Implement split generator for four shift types: cell type, locus, dose/strength, protocol.
2. Add leakage checks and provenance report for each split.
3. Add small-data compatible fallback logic when a split is undersized.
4. Persist split manifests for reproducibility and cloud reruns.
5. Add deterministic split seed management.

Validation:
1. Split manifests are reproducible from same seed.
2. Leakage tests fail intentionally corrupted fixtures.
3. Each split type can be generated from synthetic and at least one real local dataset.

Exit gate:
Shift-based train/val/test partitions are generated automatically and auditable.

---

### Phase 14 - Evaluator and Model Card Generator (Mandatory Output)
Objective: produce standardized machine/human reports on every run.

Build steps:
1. Implement metric computation: MAE, Pearson on top-1000 DEGs, calibration error.
2. Implement baseline-vs-model comparison tables.
3. Implement failure-mode section generation.
4. Implement model card outputs in JSON and Markdown with schema versioning.
5. Enforce mandatory model card creation in pipeline completion criteria.

Validation:
1. Model card files are generated for every evaluation run.
2. Missing metric sections trigger hard failure.
3. JSON schema validation passes.
4. Markdown card renders complete summaries and baseline comparisons.

Exit gate:
A run is considered incomplete unless model card artifacts exist and validate.

---

### Phase 15 - End-to-End Pipeline Composition
Objective: connect all six components into one deterministic pipeline.

Build steps:
1. Implement pipeline orchestrator command that chains ingest -> preprocess -> embed -> train -> causal -> evaluate -> model card.
2. Add run-state checkpoints to resume after interruption.
3. Add standardized artifact bundle export for cloud handoff.
4. Add robust error classification (data, config, model, infra).

Validation:
1. Full synthetic end-to-end run completes from single CLI command.
2. Partial reruns can resume from checkpoints.
3. Artifact manifest lists all required outputs.

Exit gate:
Single command can run full local pipeline without manual patching.

---

### Phase 16 - Local Validation Ladder (Strict Promotion Gates)
Objective: verify correctness at increasing realism while staying within laptop constraints.

Build steps:
1. Run full test ladder on synthetic data.
2. Run full pipeline on Adamson real dataset.
3. Run full pipeline on Replogle K562 sampled 5,000 cells.
4. Collect runtime, memory, and metric deltas for each stage.
5. Record issues and patch until all gates pass.

Validation gates:
1. Unit + integration test suite pass at 100%.
2. Test coverage target is at least 85%.
3. Synthetic end-to-end runtime under 10 minutes on local machine.
4. Adamson end-to-end runtime under 45 minutes on local machine.
5. Replogle-sample end-to-end runtime under 90 minutes on local machine.
6. Peak RAM under 12GB in all local runs.
7. Main model beats at least control mean and gene-level mean baselines on Adamson and Replogle-sample under selected metrics.
8. Causal module passes synthetic graph recovery threshold (AUC/F1 threshold predeclared in config).

Exit gate:
Only after all gates pass does the project become eligible for cloud-scale execution.

---

### Phase 17 - Cloud Handoff Preparation (No Full Training Yet)
Objective: make cloud execution operationally safe and reproducible.

Build steps:
1. Create environment specs for Kaggle and Azure (Linux).
2. Add cloud run configs and resource presets.
3. Add scripts for artifact upload/download (GitHub/HF/Zenodo).
4. Add checkpoint and resume policy suitable for spot interruptions.
5. Add cost guards: max runtime, auto-stop reminders, budget logging.
6. Add tmux-based remote execution runbook and failure recovery guide.

Validation:
1. Dry-run command planning works on cloud config without heavy computation.
2. Artifact push/pull smoke tests pass.
3. Spot interruption simulation recovers from last checkpoint.

Exit gate:
Cloud runbooks are execution-ready with zero ambiguity.

---

### Phase 18 - Planned Cloud Execution Sequence (After Local Completion)
Objective: execute heavy workloads in the right order with minimal Azure credit burn.

Execution order:
1. Kaggle/SageMaker: full Replogle preprocessing and frozen-embedder generation first.
2. Kaggle/Colab: iterative subset tuning for transport and causal hyperparameters.
3. Azure NC16as_T4_v3 spot: full K562 transport training.
4. Azure NC16as_T4_v3 spot: full RPE1 transport training and cross-cell-type evaluation.
5. Azure CPU-heavy session: full causal IV genome-wide run.
6. Azure/Kaggle: full context-shift sweep and model card generation.
7. Optional Azure on-demand: Schrödinger bridge full run if OT/flow baselines are stable.

Operational rules:
1. Enable auto-shutdown on Azure before first run.
2. Use spot instances by default; switch to on-demand only for critical long jobs.
3. Deallocate VM immediately after each job.
4. Store weights/results externally before teardown.

Exit gate:
Cloud phase considered complete only when final model cards and baseline comparisons are published artifacts.

## Public APIs / Interfaces / Types (Implementation Contract)

1. `CanonicalAnnDataV1` contract with strict validator and schema version in `.uns`.
2. `PerturbationSite` typed object with locus-aware fields and modality metadata.
3. `SequenceEmbeddingProvider` interface with `embed(sequences) -> ndarray`.
4. `CellStateEmbeddingProvider` interface with `embed(expression) -> ndarray`.
5. `TransportStrategy` interface with `fit`, `predict_distribution`, `save`, `load`.
6. `CausalEstimator` interface with `fit`, `infer_graph`, `edge_uncertainty`.
7. `ShiftSplitGenerator` interface returning deterministic split manifests.
8. `BaselineRunner` interface returning standardized metrics payload.
9. `ModelCardBuilder` interface emitting JSON + Markdown from evaluator outputs.
10. CLI command contract where each stage accepts config path, run ID, artifact root, and seed.

## Test Cases and Scenarios

### Unit tests
1. Schema validation positive/negative fixtures.
2. Sequence window extraction boundary and strand tests.
3. Mock embedding determinism and dimension checks.
4. OT coupling numerical stability tests.
5. IV estimator behavior on controlled synthetic confounding with required proxies.
6. Split generator leakage detection.
7. Conditioning-head tests verifying separate projection heads, equal latent dimension, and element-wise addition combiner.

### Integration tests
1. Synthetic full pipeline in one command.
2. Adamson ingest -> preprocess -> embed(mocks) -> train(OT) -> causal -> evaluate -> model card.
3. Replogle 5,000-cell sample same pipeline.
4. Resume-from-checkpoint pipeline continuation.
5. Baseline suite executes successfully before first transport training run.

### Robustness tests
1. Missing perturbation coordinates.
2. Undersized split partitions.
3. High class imbalance across perturbations.
4. Interrupted training with checkpoint resume.
5. Invalid config and missing file handling.
6. Missing required IV proxy columns (`batch`, `library_size`, `knockdown_efficiency_proxy`, `protocol`).

### Reproducibility tests
1. Same seed yields same split manifests and near-identical metrics.
2. Cross-platform path handling (Windows local, Linux cloud).
3. Artifact hashes stable across reruns where expected.

## Acceptance Criteria (Project-Level)
1. Six modules run end-to-end through CLI.
2. Mandatory model card generation enforced.
3. Baselines always executed and reported, and baseline suite is operational before transport training.
4. Small-data local ladder passes all runtime/memory constraints.
5. Causal module outputs uncertain cyclic GRN artifact.
6. Transport module supports OT fully and flow/bridge via pluggable interface.
7. Transport conditioning uses separate projection heads to a shared latent space with element-wise addition.
8. Causal IV pipeline uses schema-defined proxies in both regression stages.
9. Cloud handoff package is complete and runnable without architectural changes.

## Assumptions and Defaults
1. Current repository has no implementation code; full build is greenfield.
2. Python 3.11 is standard runtime for local and cloud.
3. Local machine constraints are strict: 16GB RAM, no dedicated GPU.
4. Foundation models are not retrained; used as frozen providers later.
5. Local development uses deterministic mock embedders by default.
6. LINCS implementation is stub-only in this phase set; full pipeline deferred.
7. OT variant is first fully operational transport model.
8. W&B runs offline locally and online in cloud.
9. Azure credits are preserved for full-scale runs only after local promotion gates pass.
10. Local genomic reference slice uses Ensembl GRCh38 chromosome 22 FASTA from release 110 FTP path.
11. Exact external accession IDs beyond Replogle are treated as discoverable implementation details, not design blockers.

