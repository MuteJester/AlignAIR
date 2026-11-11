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
- **`alignair convert`** converts lab chip model files (`lab-chip-v1`, `lab-chipw64-v1`) to `chip-v1`; it no longer
  converts `.pt` checkpoints.
- Model loading takes no dataconfigs or `trust_pickle`: `alignair predict` loses `--dataconfig` and `--trust-pickle`,
  `alignair benchmark` loses `--trust-pickle`.
- A model file written for sharing records neither the host nor the user it was written by.
- Training reads get library flanks by default (a 5' leader, a primer masked to N, a 3' constant region), in
  `alignair train`, `run_training` and `train_model`.

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
