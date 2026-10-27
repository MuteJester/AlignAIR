# Changelog

All notable changes to AlignAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [Unreleased]

### Added
- Explicit protein substitution policies in the shared training workflow, with unaugmented
  validation, versioned checkpoint provenance and augmentation-policy checks on resume.
- One public training request/workflow for nucleotide allele, protein allele and protein-class models,
  with shared validation, resume, checkpoints, update callbacks and raw prediction after reload.
- Portable input/label metadata and ordered output-label schemas, preserved through safe export and
  conversion. Reload and resume check label identity; the website documents the full training workflow.
- Shared sequence-content results, explicit unknown/ambiguity handling and crop metadata, with
  nucleotide file and prediction policy presets in the data input module.
- Source-aware nucleotide read batches and a unique output-ID allocator that reserves generated
  names across chunks. Metadata lookup now has a dedicated module with eager and disk-backed APIs.
- Shared raw sequence-file parsing for FASTA, FASTQ, CSV/TSV, text, gzip and stdin, with
  structured source-location errors and explicit file ownership, independent of sequence alphabets.
- Shared sequence normalization with configurable alphabets, stable source indices, structured failures,
  and checked prediction/error interleaving; protein product interfaces retain their existing IDs and JSON fields.
- Shared sequence-input adaptation, mode-safe prediction batches, multi-head probabilities, and named
  component-feature capture, with explicit tokenizer/window policies and inference API documentation.
- Shared allele-cohort evaluation with optional sparse predicted sets, scalar-head observations,
  explicit equivalence-map coverage, and metadata-head-aware reporting for either input alphabet.
- Shared supervision-aware cohort observation collection for nucleotide/protein and categorical/multilabel
  models, with original record indices, per-field support, shape checks, and mode-safe inference.
- Shared evaluation statistics and exact-class/boundary reports, validated observation contracts,
  bounded NumPy bootstrap resampling, and a website API guide.
- Shared task-driven one-batch capacity diagnostics for nucleotide and protein models, with weighted
  evaluation loss and final task metrics; class-diversity sampling policies with explicit quota checks.
- Shared training-engine factory for task selection, weight/device placement, stable parameter binding,
  replaceable optimizers, and update-based scheduler construction, with API documentation.
- Shared curriculum record sources with explicit automatic/amplicon policies, provider injection,
  deterministic component/panel seeds, translated-read eligibility, and a website API guide.
- Shared reference-bound model preparation in `alignair.core`, with explicit nucleotide/protein input
  and allele/protein-class label policies, configurable components, and delayed model allocation.
- Model setup API guide covering label identity, head defaults, adapters, and preflight validation.
- Shared `alignair.train.training_iterations` API for explicit optimizer-step windows, lazy batch
  consumption, completed-step results, and scoped diagnostics across nucleotide and protein runners.
- Shared protein augmentation and validated CIGAR mapping, with conservative fragment-compatible
  crop labels, classification-only masks, and a website API guide.
- Shared protein-class sample builder with explicit productivity/stop policies, plus source-independent
  budgeted collection and preparation/selection ledgers under `alignair.data.sampling`.
- Sample preparation API guide covering rejection reasons, hard failures, selection counts, and seed budgets.
- Shared protein reference views derived from the ordered nucleotide `ReferenceSet`, with explicit
  anchor and hash-scope policies, deterministic equivalence labels, and reference API documentation.
- Shared `alignair.train.TrainingEngine` for optimizer steps, full optimized-parameter gradient
  checks/clipping, constraints, scheduler ordering, and mode-safe validation. Website API guide added.
- Atomic training checkpoint publication, shared snapshot/research/self-contained checkpoint helpers,
  and one process-RNG state contract with malformed-state rollback.
- Shared training samples, batches, and source adapters in `alignair.data`, with explicit window
  policies, positional-label padding, device transfer, and preserved supervision/compatible-set masks.
- Website training-batch API guide covering nucleotide records, translated allele targets, and
  categorical samples.
- Shared `alignair.tasks` contracts for multilabel/categorical objectives, optional-head losses,
  supervised metric totals, and validation aggregation. Trainers accept explicit task implementations.
- Shared configurable nucleotide/protein model assembly, named architecture component factories, and
  a common layer-inspection interface. Construction choices round-trip through model configuration.
- Website model-composition API guide with component, checkpoint, and score-semantics contracts.
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
- Removed the independent strategic protein training module. Protein-class substitution experiments
  use `TrainingConfig` and `run_training`; study and product callers use reference-bound model setup.
- Public training types and orchestration now live in `alignair.train`; root and historical object-API
  imports point to the same owners. The independent `aa.train_gym` loop is removed and its tests migrated.
- The common trainer selects existing record/batch preparation from a reference-bound setup.
  New runs seed model initialization and sources, and report actual completed steps.
- `Aligner.predict_raw` handles either input space. The AIRR pipeline explicitly requires nucleotide
  allele models; uncalibrated protein checkpoints cannot be mistaken for calibrated product predictions.
- Nucleotide file validation, prediction, raw forward, genotype alignment and calibration share
  content preparation. Protein validation keeps its strict policy through the same normalization owner.
- In-memory nucleotide prediction rejects non-string values and bare non-batch inputs, strips outer
  whitespace, and maps unknown characters to N before tokenization, alignment and reporting.
  Literal P input no longer becomes padding. Eager empty batches raise a typed input error.
- Genotype repertoire alignment consumes generators once and uses prepared read lengths while retaining
  crop flags in prediction records. Assay replay uses shared adaptation/forward execution with verified pins.
- Nucleotide CLI metadata joins use original source IDs, preventing generated suffixes from matching
  another read's metadata. Unnamed reads receive no metadata through invented fallback join keys.
- Both metadata loaders reject duplicate keys. Inserts are transactional; temporary indexes support
  context management and idempotent closure. Header/rename collisions fail before rows are lost.
- 10x metadata normalization preserves nonempty canonical AIRR values and fills gaps from aliases.
- Protein CLI and nucleotide readers share file parsing. The protein CLI no longer reads a second
  whole-file string, supports the shared formats, and reports empty FASTA entries as record failures.
- Unnamed FASTA entries reach downstream ID assignment and validation. Malformed FASTA/table
  structure now fails explicitly; nucleotide ambiguity cleaning and chunking remain caller policies.
- Protein product prediction normalizes each batch once. Duplicate input dispatch, ID assignment,
  per-item validation calls, and placeholder-slot reconstruction have been removed.
- Non-string sequence values produce structured invalid_type failures in skip mode. Normalized
  sequence lengths are checked after case conversion; invalid normalization windows fail preflight.
- Research, product, feature extraction, and nucleotide prediction share padding/forward mechanics;
  duplicate implementations are removed. The evaluation context now lives in core runtime.
- Research probability inference enters evaluation mode and restores caller flags. Product inference
  keeps its verified tokenizer and device-local softmax; research keeps CPU softmax arithmetic.
- Checkpoint evaluation reuses clean-cohort V/J/D probabilities and conformal test probabilities,
  reducing redundant model passes and supporting configurations without a D head.
- Protein allele evaluation and stratified benchmarking use shared observation collection and allele counters;
  duplicate inference/counting loops and dummy-tensor counter initialization have been removed.
- Unmapped alleles no longer form a false equivalence class. Class accuracy and within-class error
  fractions use explicit mapped-comparison support. Boundary/scalar errors use their observed row counts.
- Empty allele cohorts report zero records; disabled metadata heads report null metrics. Stratified
  protein benchmarking now restores caller module modes after inference.
- Frozen protein-cohort evaluation delegates prediction collection to the shared owner, excludes
  unobserved classes without dropping observed boundaries, and reports explicit per-field support.
- Categorical batch adaptation preserves gene-level supervision masks alongside class-only masks.
- Frozen protein-cohort reports and convergence macro accuracy use shared evaluation owners. Gym
  competence statistics retain their seeded Python resampling convention under the same statistics module.
- Statistical imports defer benchmark loading; existing evaluation and historical estimator imports remain available.
- Capacity-proof training and diversity selection now live in `train.diagnostics` and `data.sampling`;
  the protein proof study imports shared owners, and `aa.train` retains compatibility imports only.
- Alignment runners and capacity/frozen-cohort proofs use one training-state initializer. Public
  nucleotide training creates task weights there when omitted; supplied weights retain their identity.
- Optimizer assembly deduplicates shared parameters, checks custom parameter groups and scheduler
  ownership, and rejects device transfers that split parameter ties. Root task-container parameters
  are included in ownership checks and gradient diagnostics. Clipping validation has one owner.
- Nucleotide and protein runners share `train.gym.streams`; removed stream construction from the
  nucleotide trainer and the private protein source bridge. Receptor detection is shared with the
  experiment builder. Source ordering, per-locus mutation caps, and resume seed schedules are preserved.
- Locus tagging leaves provider-owned records unchanged, repeated component dictionaries are
  independent, and colliding custom provenance names retain every reference panel.
- Protein setup helpers, public nucleotide training entries, and training plans use one model setup
  owner and construct the shared `AlignAIR` class. `aa_igh_config` delegates to the shared count-based
  factory. Reference allele-order validation
  lives on `ReferenceSet` and is reused by protein views and model setup.
- Nucleotide, protein allele, categorical, and cropped-protein runners use one batch/step iteration
  owner. Removed the duplicated loops and the AA-specific next-batch helper. Runner-specific history,
  validation, checkpoint timing, scientific policies, and resume step semantics are preserved.
- Protein crop targets now use aligned reference codons instead of slicing prototypes with read
  coordinates. Missing reference evidence excludes classification while preserving known boundaries.
  Trimmed V reads can now be cropped. Full-state Stage-1 resume requires the matching versioned
  sample policy; older checkpoints can initialize weights through `init_path`.
- Replaced duplicate strict/curriculum protein sample assembly with one builder. Historical `aa.data`
  imports delegate to shared owners; migrated the implementation out of that module and removed
  `_class_sample`. Shared translation eligibility also serves protein allele-batch source filtering.
- Collection distinguishes scientific rejections, selection skips, callback failures, and untracked
  historical selection outcomes. Malformed frames and length-changing tokenizers raise typed errors.
- Germline translation, protein label maps, and class-map assembly live under `alignair.reference`.
  AA training, audit, and product callers use the shared owners; historical imports reexport them.
  Source allele order, class IDs, V/J versus V/D/J hash scopes, and the existing wire schema are preserved.
- Protein views quarantine invalid anchors and reject inconsistent source indices or empty required
  label spaces. Serialized equivalence maps validate class IDs and exact allele/member agreement.
- Renamed shared neural-network components to `alignair.neural` for clearer package navigation.
  Internal callers and tests use the new name; `alignair.nn` imports remain compatibility reexports.
- NT and AA alignment runners use one training engine; migrated optimizer loops, snapshot writers,
  and RNG implementations are removed. Historical trainer and AA resume imports delegate to it.
- Gradient checks and clipping include learned task weights. Non-finite norms abort before optimizer
  updates; validation restores individual module modes and cannot leave later steps in evaluation mode.
- New research checkpoints include Python RNG state. RNG restoration failures now raise a typed error
  instead of continuing with partially restored generators. Resume documentation explicitly describes
  re-seeded simulator/augmentation streams rather than claiming exact uninterrupted replay.
- Tokenization, codon projection, query-coordinate validation, target construction, and allele-set
  encoding have shared owners under `alignair.data`. Historical imports delegate to these owners;
  the separate stage-one categorical collator is removed.
- Record prefix windows clip boundary targets and exclude genes lost from the retained read.
  Protein allele batches exclude absent/inverted genes from supervision. Translated-record eligibility
  now excludes lengths inconsistent with the declared junction frame; it does not infer another frame.
- Loss implementations and batch metrics have a shared owner; historical NT/AA imports delegate to it.
- Validation weights metrics by supervised rows or positions. Categorical metrics respect invalid genes
  and compatible class sets. Excluded empty sets and fully ignored state labels have finite zero loss.
- The AA model constructor delegates to the shared AlignAIR implementation; categorical scores,
  separate boundary masks, and optional heads are explicit configuration choices.
- ModelXRay, AA representation extraction, and the assay profiler use shared model inspection points.
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
