# Changelog

All notable changes to AlignAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [Unreleased]

These changes make up AlignAIR 4 (the release number is not set yet). AlignAIR 4 is a new network and a new model
contract: AlignAIR 3 model files do not run on it.

### Changed
- **One network, the chip (`chip-v1`).** A dilated-convolution trunk gives every base its features once; a region
  parser, V/J anchor heads, a mutation reader and per-segment scans of the bound reference's alleles read them, and
  deterministic steps refine the calls and boundaries against the germline. The `conv-v1` network is removed.
- **AlignAIR 3 model files are refused** with the version that runs them: a `.pt`, a `conv-v1` `.alignair`, or a
  container without `reference_json` ("this is an AlignAIR 3 model ...; use alignair<4 to run it"). The hub catalog's
  current entries (`alignair-igh-human`, `alignair-igkl-human`, `alignair-tcrb-human` at 1.0.0) are AlignAIR 3
  models; AlignAIR 4 pretrained models are to be published one per locus.
- **The reference is an input, not weights.** A model scores the alleles of the reference it is bound to: its
  embedded reference by default, or another one (see Added). The fixed-reference contract and its reference-identity
  check are gone; novel alleles are callable without retraining.
- **`--genotype` binds the genotype's subset of the reference as the bank.** Genotype files are YAML or JSON keyed by
  segment (`V`, `D`, `J`); a name the bound reference does not hold is refused, as is a segment left with no allele.
  `--genotype-method` and the post-hoc genotype masks are removed. A read whose V gene is outside the genotype gets no
  V call (`airr_assembly_status` `partial`, reason `no_v`).
- **Calls.** `v_call` lists, after the call, the V alleles that match the read as well (IgBLAST's convention for
  equally scoring alleles); an allele the read matches at more bases than the model's V call becomes the call; on a
  multi-locus model each read keeps to one locus and the record carries its V call's locus; a D-presence head gates
  the D call, and D is aligned without gaps.
- **Junction and productivity follow by rule from the V and J anchors** the model finds in the read;
  `productive_prediction` repeats the derived call. The learned productivity and chain-type heads and allele
  temperature calibration are removed.
- **Reads are batched with reads of their 32-position width bucket**, so other reads' lengths never change a read's
  row; the default batch size follows the device (256 on a GPU, 64 on the CPU, 16 per worker with `--threads`).
- **`alignair genotype`** now infers a donor's V genotype from an AIRR table: `alignair genotype --airr TSV --out
  JSON [--locus] [--explain] [--min-reads]`.
- Model loading takes no dataconfigs or `trust_pickle`: `alignair predict` loses `--dataconfig` and `--trust-pickle`,
  `alignair benchmark` loses `--trust-pickle`.
- A model file written for sharing records neither the host nor the user it was written by.
- **A reference of another locus** (every locus-showing allele name of one other locus, e.g. TRB on an IGH model) binds
  with one `LocusMismatchWarning` ("the bound reference is TRB, but the model was trained on IGH") and keeps its own
  locus, so the records' `locus` comes from the alleles called.
- **`<out>.run.json`** records `--reference` and `--genotype` (paths and SHA-256), the fingerprints of the reference the
  calls came from (the bound one, with the embedded one's beside them) and the genotype bank's allele order.
- **Genotype files** that are not YAML/JSON keyed by V, D, J (a FASTA, a broken file, a list, a segment given one
  string) or that are keyed by gene names are refused with a ConfigurationError whose hint gives the format, instead
  of an internal error or a misleading "alleles the reference does not hold".
- `alignair reference list --chain` matches a chain type, a locus or a whole token of either or of the name (`IGH`
  no longer lists light chains).
- `alignair predict`, `demo` and `benchmark` without `--device` resolve `auto` (CUDA, then Apple MPS, then CPU), as the
  Python API does.
- The missing-segment refusal has a hint; the refusal to share a checkpoint names `alignair export`.
- Training reads get library flanks by default (a 5' leader, a primer masked to N, a 3' constant region), in
  `alignair train`, `run_training` and `train_model`.
- **Error classes end in `Error`.** `DuplicateMetadataId` is `DuplicateMetadataIdError`, `PredictionCountMismatch` is
  `PredictionCountMismatchError`, `alignair.registry.cache.IntegrityError` is
  `alignair.registry.DownloadIntegrityError`, and `NonFiniteLossError` is `NonFiniteTrainingError`; the old names are
  gone. `OperationIOError` is an `OSError`.
- **Modules moved.** `alignair.compare` is `alignair.analyze.compare`; the germline reader
  (`alignair.predict.heuristic_matcher`, `alignair.predict.germline`) is the `alignair.predict.germline` package
  (`HeuristicGermlineMatcher` is `locate.GermlineLocator`, `match_one` is `locate`); `alignair.train.gym` is
  `alignair.train.records`; `alignair.train.build` is `alignair.train.bundle` (its dataconfig builder is
  `alignair.genairr.custom`); `build_experiment` is `alignair.genairr.experiment.build_experiment`.
- **Python API signatures.** `build_airr(reads, reference, has_d=True)` replaces `chain="heavy"/"light"`;
  `PredictConfig(loci=...)` replaces `chain_types=`; `select_alleles(probs, names, threshold, cap)` is the one
  allele-set rule; `load_genotype(path)` returns the genotype dict and `to_genotype_set(genotype, locus)` builds the
  AIRR GenotypeSet; `save_model(path, model, reference, *, dataconfigs=(), ...)` embeds the reference it is given, and
  `save_checkpoint`/`save_rotating` take the reference too; `build_experiment` refuses a parameter it does not read
  (a typo no longer simulates the default) and needs `invert_d_prob`; `NucleotideBatchAdapter(augment_orientation=)`
  takes True or False and refuses any other value.
- **Model files** no longer store allele counts, input/label spaces or a label schema; the reference digests identify
  the model. An earlier AlignAIR 4 development build cannot read a file this one writes. A stored configuration that
  is invalid is a `ModelIntegrityError` (the file's fault). A training checkpoint resumes only onto the reference it
  was trained on, its sequences included (`init_from` warm-starts from it otherwise).
- **Metadata joins** (`load_metadata`, `build_metadata_index`) rename a column that collides with an AIRR field to
  `meta_<column>` by default, as `alignair predict --metadata` does.
- **`alignair info`** and **`alignair models info`** print one model card: the id and version (or "(local)"), species
  and locus when set, the model's allele counts, allele order and sections.
- **`alignair analyze`** reads CSV as well as TSV; its QC counts the `orientation` column's four transforms
  (`reverse_complement`, `complement`, `reverse`, `forward`; was `reoriented`); the always-empty V confidence
  statistics are gone.
- **`alignair benchmark --dataconfig`** takes one name; unknown `--strata` are refused, naming the strata.
- **`alignair doctor`** lists the core dependencies (torch, numpy, GenAIRR, safetensors, PyYAML, platformdirs) and the
  optional ones (zstandard, huggingface-hub), reports each compiled kernel, and its hint for a missing dependency is
  `pip install alignair`.
- `--locus` is spelled as the model spells it; a bare model spec naming a `.alignair` file, or a path below
  `org/repo`, is a path ("model file does not exist"), not a Hugging Face repo; a failed `alignair predict` publishes
  no rejects file; `<out>.run.json` records the predicting AlignAIR as `alignair_version` and the model's as
  `model_created_by_alignair`; `ReferenceSet.from_fasta` refuses, naming them, alleles whose name does not tell their
  segment; a model without a D never gets a D in its AIRR rows, whatever the bound reference holds.
- The CLI options that had no help text have it (`benchmark`, `analyze --format`, `models prune --keep`, `info`,
  `train --lr`/`--seed`).
- The `test` extra adds ruff (`>=0.17.1,<0.18`): `tests/quality/test_lint.py` fails on any finding in `src/alignair`
  under `pyproject.toml`'s `[tool.ruff]` settings.

### Added
- `alignair predict --reference FASTA` and, in Python, `Aligner.from_pretrained(..., reference=...)` and
  `Aligner.bound_to(reference)`: bind another germline reference, novel alleles and new genes included. Missing IMGT
  gaps and anchors are taken from the model's own alleles (the same sequence, or a same-length variant differing at
  no more than 10% of bases) or GenAIRR's resolver (`alignair.reference.annotate`); alleles left without an anchor
  are named on stderr (`alignair.reference.anchor_report`). A reference missing one of the model's segments is
  refused.
- `alignair predict --infer-genotype` (with `--genotype-out`, `--genotype-explain`, `--genotype-min-reads`): two
  passes over the input infer the donor's V genotype by TIgGER's frequency rule, re-call the reads it settles, and
  write it as an AIRR GenotypeSet.
- `alignair predict --threads N`: on the CPU, N single-threaded worker processes.
- `alignair export CHECKPOINT OUT_DIR [--model-id --model-version --description --no-validate --overwrite]`: a training
  checkpoint becomes a pickle-free bundle (model, model card, reference manifest, validation report) whose card the
  catalog accepts as it is. The model card names the model, its reference and version (e.g. OGRDB V8), allele counts,
  training steps, intended use and the genotype and reference-swap usage.
- Publishing to the shared catalog: `alignair.registry.publish_bundle` lays a bundle out beside its card and manifest
  (`<id>/<version>.alignair`, `.model_card.md`, `.reference_manifest.json`, recorded with their SHA-256 and checked by
  the validator), and `upload_version` sends one version's files and `registry.json` in one commit. `push_to_hub`
  uploads the card and manifest beside a model file with it.
- Training: `--data-workers` generates curriculum records in worker processes with batches identical to a serial
  run; an init seed separates the initial weights from the data seed; opt-in novel-allele training banks
  (`TrainingConfig.novel_alleles`); `train_args` record the recipe.
- A V allele listed by two loci belongs to both (TRAV/DV alleles in TRA and TRD).

### Removed
- The amino-acid (protein) track: `predict-aa`, `predict_aa`, `load_aa_model`, protein references and class labels,
  conformal and allele calibration, the probability and feature-extraction APIs, augmentation, class sampling and
  partitioning, and the cohort, coverage and report evaluation (kept at tag `aa-track-v3`).
- The WFA, parasail and Biopython germline readers (`alignair.align`) and those dependencies; the heuristic matcher
  is the one germline reader.
- The `alignair_benchmark` package and other research code; the head-to-head evaluation is the source-only
  `alignair_assay` package.
- `PredictConfig.selector` and `pad_mode` (with `SELECTORS`, `largest_gap`, `max_likelihood_percentage`), and
  `correct_segments(max_len=, pad_mode=)`.
- `AlignAIR.from_dataconfigs`, `AlignAIRConfig.from_dataconfigs`/`from_reference` and the allele-count and
  label-space settings: build a model for a reference with `alignair.core.prepare_model`
  (`ReferenceSet.from_dataconfigs` stays).
- `ReferenceSet.from_yaml`, `to_yaml` and `infer_locus`; a saved reference reloads with the locus schema it was saved
  with.
- `alignair.data.tokenizer` and `alignair.data.encoders` (the tokens are `alignair.core.tokens`; `tokenize_batch`).
- The ten AIRR extension columns that were never filled: `*_resolved_call`, `*_call_level`, `*_set_confidence` and
  `is_contaminant`.
- `alignair reference export --dataconfig/--index` and the `export-reference` verb: `alignair reference export MODEL
  --fasta OUT` writes a model file's germline FASTA.
- `alignair convert`: AlignAIR 3 `.pt` checkpoints are refused (run them with `alignair<4`).

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
