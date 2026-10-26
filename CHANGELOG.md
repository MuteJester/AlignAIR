# Changelog

All notable changes to AlignAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [Unreleased]

### Added
- QA execution layers, per-test RNG/cache isolation, outbound-network rejection, bounded test execution,
  shuffled component checks, critical-file branch/statement coverage gates, and persisted CI reports.
- Checksum-pinned public-model acceptance fixtures and required release-claim execution gates. Model
  checks no longer depend on private model filenames; release builds require their validation job.
- Installed-wheel API contracts and independent, hand-calculated convergence-evaluator fixtures.
- Standard-library-only `alignair.errors` package: shared exception categories, stable codes,
  versioned JSON diagnostics, scoped operation context, and opt-in reporting/logging helpers.
- CLI `--debug` and `--error-format text|json` flags, available at every command level.
- Contributor guidance for raising, chaining, reporting, and extending domain errors.

### Changed
- Fully removed simulator segments are treated as absent; visible segments still require valid germline
  annotations. Training losses and metrics exclude absent/inverted gene supervision instead of using
  placeholder coordinates. Crop handling accepts the same absent-segment contract.
- Tests are grouped by package responsibility, with separate acceptance, robustness, and packaging
  suites. AA guard and benchmark checks have focused subdirectories; obsolete fixtures and the
  standalone test runner are removed. Optional local-artifact checks have a `local_data` marker.
  The `test-research` extra supplies HMM/scorer test dependencies for full-suite CI; contributor
  instructions and the website testing guide describe the layout and commands.
- Release smoke validation lives in `tests/packaging/release_smoke.sh`, shared by the wheel and Docker jobs
  in CI and release workflows. AA research command documentation lives in the website maintainer guide;
  the top-level `scripts/` directory is retired.
- Profile handling is organized under `alignair.aa.guard.profiles` into source metadata,
  HMM banks, allele alignment, and assembly. Existing imports remain available. The package
  command replaces `build_profiles.py`, taking explicitly pinned local inputs and output paths;
  translation reuses the shared codon table and HMM construction reuses the existing builder.
- AA guard compute accounting and production-seed selection now live in
  `alignair.aa.guard.training`; study runners supply their accounting paths explicitly.
- AA guard sweep and LOCO training are package APIs and module commands with explicit
  inputs. Removed their standalone scripts and script-path imports; AUROC uses NumPy.
- Sweep manifest preparation lives in `alignair.aa.guard.training.manifests` and the
  existing training command (`freeze-sweep`). Experiment settings and all paths are explicit;
  the historical partition and manifest hashing format are preserved.
- LOCO manifest preparation shares the training manifest module and writer (`freeze-loco`),
  reusing development-only seed selection and source-vocabulary helpers. Removed its standalone
  script and import-time artifact reads; audit paths describe the supplied inputs.
- Consolidated OAS collectors and catalog probing under `alignair.aa.guard.data_sources`.
  Shared bounded downloads, CSV parsing, cohort exclusions, and optional domain verification
  replace four standalone scripts while retaining the distinct sampling policies and output filenames.
- Moved the GenAIRR simulated-positive collector into the source-adapter package with explicit
  output paths, preserving its study settings, filtering order, record identities, and census.
- Moved IgSF collection into the source-adapter package, reusing shared HTTP handling and preserving
  its UniProt query, InterPro selection, exclusions, lineage ordering, and output filenames.
- Moved SAbDab VHH/murine collection into the source-adapter package with explicit inputs,
  preserving sampling limits, ID exclusions, sequence deduplication, and output filenames.
- Moved active 10x TCR collection into the source-adapter package, replacing standalone download
  and cache code with shared HTTP handling while preserving study identities and sampling rules.
- Moved current cohort assembly/gating into `alignair.aa.guard.cohort`, reusing the existing
  allocation, structural-generation, and audit helpers with explicit input/output paths.
- Removed the standalone disjointness audit wrapper; `alignair.aa.guard.disjoint` provides
  the command alongside its existing audit helpers, with explicit input paths and shared diagnostics.
- Consolidated donor-genotype examples, corrected the sample-manifest loop, and shortened example
  instructions to link to the website guides. Sample data and the tutorial merger are preserved.
- Public error-handling documentation lives in the website API reference. Local AI notes, plans,
  research reports, and supporting artifacts are consolidated under the ignored `Notes and Plans/`
  directory; research script and test paths follow the new location.
- All existing custom exceptions inherit from `AlignAIRError`, retaining their original import
  paths and built-in catchability. Public validation and model-loading paths adopt domain errors.
- CLI operational errors use stderr; unexpected defects return status 70 and interruption returns
  130. Normal operational failures retain status 1, and argparse usage errors retain status 2.
- Model container failures distinguish malformed headers, missing sections, and integrity errors.
- Registry source failures retain their cause instead of becoming misleading unknown-model errors.
- Invalid training annotations report the stage, record position, and missing gene fields without
  inferring replacement coordinates.

### Removed
- Historical `calibrate_stage2.py` simulator-specific calibration runner. APS/RAPS and group-based
  fitting remain in `aa.conformal`; embedded calibration loading and existing model artifacts are unchanged.
- Redundant `train_stage1_trunc.py` wrapper. The shared truncation trainer, warm-start/checkpoint
  support, and convergence-driver integration remain unchanged.
- Redundant `train_strategic_igh.py` wrapper and historical training defaults.
  The strategic model setup and trainer remain in the existing package module.
- Historical `train_eval_aa_igh.py` allele-head experiment runner and held-out sampling recipe.
  Existing training and evaluation APIs remain in `aa.train_gym` and `aa.eval_gym`.
- Redundant `benchmark_aa_igh.py` wrapper for the historical allele-head checkpoint.
  The package benchmark implementation and shared allele-recovery metrics remain unchanged.
- Historical `eval_baseline_complete.py` report runner and study-specific junction-masking analysis.
  Shared inference, metrics, coordinate projection, and alignment code remain unchanged.
- Historical `eval_strategic.py` comparison runner and its duplicate batching/cohort setup.
  Shared inference and checkpoint evaluation remain; evaluator documentation describes its own policy.
- Historical `error_forensics.py` allele-head diagnostic runner. Shared nearest-germline
  calculations and their regression tests remain in the existing forensic module.
- Historical `derisk_local_aligner.py` comparison runner, its duplicate augmentation functions,
  and study-specific decision rule. Shared local alignment and augmentation remain in the package.
- Unused `mask_recovery.py` simulator-truth diagnostic for the historical allele-head model.
  Translation, forensic, and local-alignment implementations remain unchanged.
- Unused `diagnose_ambiguity.py` diagnostic for the historical allele-head checkpoint,
  including its import-time model loading. Protein-equivalence and allele-expansion APIs remain.
- Redundant `phase1_proof.py` experiment wrapper and console summary. The package learning-proof
  implementation, held-out manifest support, and historical proof artifacts remain unchanged.
- Redundant `phase0_report.py` wrapper. Phase-0 report calculation and writing remain
  in the existing `aa.phase0` and `aa.report` modules.
- Historical convergence `confirm_promotion.py` runner with fixed incumbent/winner checkpoints
  and study-specific promotion criteria. Reusable paired statistics and metrics remain in the package.
- Redundant convergence `run_study.py` CLI wrapper and its historical experiment defaults.
  The existing package driver and checkpoint-selection implementation remain unchanged.
- Historical Phase-3 `truncation_stress.py` diagnostic runner. Shared cropping and conformal
  code remains; product tests verify each full/trunc/short group uses its own frozen quantile.
- Historical Phase-3 `evaluate.py` runner tied to the strategic research checkpoint.
  Cohort validation, family scoring, statistics, and taxonomy mapping remain in their package modules.
- Standalone `validate_product.py` experiment/report harness. Existing product tests cover its
  runtime contracts; research/product class-set and top-call parity now covers V/D/J across
  full-length and cropped inputs without importing the benchmark harness into the product package.
- Historical Gate-3 `split_freeze.py` writer. Read-only exclusion replay retains the original
  split policy, checked against captured synthetic assignments and a fixed canonical digest.
  Existing corpus partitions, fingerprints, and access audits remain unchanged.
- Retired Gate-3 `fit_select.py` experiment runner and its unused report output path.
  Package scorer helpers and the newer guard's selection implementation remain unchanged.
- Historical `validate_prefilter.py` report runner. Structural prefilter rules and
  their behavior tests remain; no runtime rule-demotion mechanism depended on this report.
- Historical Gate-3 `cleanup.py` runner tied to the retired emitter format.
  Versioned domain-preparation helpers and the frozen corpus used for exclusion replay remain unchanged.
- Preliminary `assemble_feasibility.py` Gate-3 report runner, superseded by the
  post-cleanup report. The reusable feasibility calculator and its tests remain.
- Local `mock_oneshot.py` experiment runner and its unused historical summary writer.
  Core bundle/access/evaluation APIs remain; selection tests retain separate real/simulator
  rejection checks and demonstrate why pooled success cannot override a failing group.
- Historical `capture_license_evidence.py` snapshot runner. Archived evidence references
  and profile-source attribution metadata remain unchanged.
- One-off `c0_diagnostic.py` report runner for the historical Gate-3 FIT cohort.
  C0 decision rules and their use in dual evaluation remain in the package.
- Redundant `train_candidate.py` smoke runner and its duplicate domain-label mapping.
  Training uses the package engine, with smoke coverage in the training tests.
- Superseded fresh-cohort orchestration (`hydrate_fresh.py`) and its legacy wrapper tests.
  The current Stage-1 gate and its allocation, deduplication, and disjointness helpers remain.
- Retired VDJServer TCR and legacy OAS-light cache collectors. Neither is used by the active
  cohort workflow; 10x TCR and the maintained OAS collectors provide the current sources.
- Archived pre-v3 benchmark inputs and comparison report, their dedicated validation workflow,
  and the unused website widget. Documentation no longer cites the retired comparison.

### Fixed
- Benchmark process pools use an explicit spawn context, avoiding fork-after-thread initialization
  while preserving deterministic case and coverage-acceptance ordering.
- Offline Hugging Face loading uses a cache-only lookup, avoiding SDK metadata requests even when
  the SDK's ordinary download path builds telemetry headers before checking local files.
- Profile-bank assembly verifies expected input checksums and stops on missing required profiles
  instead of publishing an incomplete bank. Invalid annotated frames are rejected;
  existing output directories are refused, and runtime metadata uses the installed pyhmmer version.
- LOCO preparation verifies the frozen sweep hash, partition, profile bank, class map, and feature
  order; it refuses invalid or out-of-program seed results, fresh metrics, and existing output files.
- Sweep manifest preparation refuses unreadable or malformed fresh-exclusion sources,
  ambiguous duplicate development IDs, sequence overlap, and existing output files.
- The disjointness audit command checks Phase-3 therapeutic names and sequence hashes using
  the supplied labels, rather than passing empty exclusions. Invalid fresh JSONL reports its line number.
- Cohort freezing binds the exact source snapshots audited, detects inputs changed during the audit,
  and refuses to overwrite an existing frozen manifest. A stopped cohort audit returns failure status.
- 10x downloads respect their byte limit, and cached raw inputs are filtered against the current
  denylist. Removed the normalized-unit shortcut that could reuse records without rechecking exclusions.
- IgSF collection validates cached UniProt checksums and refuses to silently shorten out-of-bounds
  domain annotations.
- OAS probes retry an inaccessible chain even when the other chain already makes a study eligible.
  Download caches validate request identity and checksums; incomplete CSV rows are excluded.
- Guard feature caches bind profile-bank, feature-order, and fold-mask identities.
  Training resumes restore batch-sampling RNG state and reject mismatched or unbound checkpoints.
- Genotype-constrained CLI prediction passes the validated allele mapping to inference instead
  of the loader's `(genotype, unknown)` result tuple.

## [3.0.0] - 2026-07-18

A complete PyTorch rewrite of AlignAIR (`alignair` package) with a first-class CLI, a self-contained
model format, pretrained models, and a modern documentation site. This is a major, breaking release —
the legacy TensorFlow lineage is gone.

### Added
- **`alignair` CLI**: `predict`, `train`, `demo`, `models`, `reference`, `convert`, `validate-airr`,
  `compare`, `analyze`, `benchmark`, `genotype`, `doctor`, plus `--version`.
- **Pretrained models** published to a public hub (`AlignAIR/AlignAIR-pretrained`): human IGH,
  IGK+IGL, and TRB. `--model <id>` downloads, hash-verifies, and caches automatically — no login.
  `alignair models list/get/info/...` manages the catalog, which updates live from the registry.
- **Self-contained `.alignair` model format**: weights + the embedded, fingerprinted germline
  reference + model card in one file that loads **without executing any pickle**.
- **Donor-genotype constraint**: `predict --genotype` restricts calls to a subset of the model's
  reference (YAML or FASTA) with no retraining. A model is a **fixed-reference classifier** — novel
  alleles require training (see the model contract).
- **`alignair train`**: train a model for your own reference/species from a built-in GenAIRR
  dataconfig or custom V/D/J FASTAs; exports a self-contained, pickle-free bundle + `model_card.md`,
  `reference_manifest.json`, and `validation_report.json`.
- **AIRR-C output**: schema-valid rearrangement TSV (`sequence_alignment`, `*_cigar`, `*_identity`),
  a per-gene candidate-set column (`*_call_set`), and a `<output>.run.json` provenance sidecar.
- **Bounded-memory streaming** prediction for repertoire-scale inputs; per-read metadata join
  (`--metadata`, e.g. 10x annotations).
- **In-model orientation** detection/correction (forward / reverse-complement / complement / reverse).
- CPU **Docker** image, cross-platform CI, and a compiled (optional) Cython germline-CIGAR kernel.

### Changed
- Output is a schema-valid AIRR rearrangement TSV; ambiguity is surfaced as a candidate set.
- Documentation rebuilt as a React/TypeScript site (reference, guides, and interactive lessons); license clarified as GPL-3.0-or-later.

### Removed
- The entire legacy TensorFlow lineage (`app.py` entrypoint, SavedModel workflow, `src/AlignAIR`).
