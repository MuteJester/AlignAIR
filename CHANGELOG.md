# Changelog

All notable changes to AlignAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

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
