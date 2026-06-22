# Changelog

All notable changes to AlignAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic versioning.

## [Unreleased]

The PyTorch rewrite (`alignair` package) and its CLI.

### Added
- `alignair` CLI: `predict`, `train`, `model` (list/download/inspect), `validate-airr`,
  `doctor`, `bundle`, plus `--version`.
- **Dynamic genotype**: supply a reference at predict time as YAML or FASTA (`--genotype`) —
  an allele subset and/or novel alleles; the model conditions on exactly what you provide.
- **`alignair train`**: train a model for your own reference/species from a built-in GenAIRR
  dataconfig or custom V/D/J FASTAs; writes a self-contained bundle (custom references are
  embedded) + `model_card.md` + `validation_report.json`. Fine-tune via `--base-model`.
- **Model zoo**: `alignair model` + id/HF-repo resolution for `--model`; `from_pretrained`.
- **AIRR-C compliance**: schema-valid rearrangement output (`sequence_alignment`, `*_cigar`),
  `validate-airr`, and a `<output>.run.json` provenance sidecar.
- Classical **parasail** V reader (`--v-reader parasail`): faster + sharper heavy-SHM V calling.
- Working CPU **Docker** image (`alignair` entrypoint, `doctor` healthcheck), CI + release workflows.

### Changed
- Output is now schema-valid AIRR rearrangement TSV with calibrated uncertainty columns.
- Documentation rewritten around the PyTorch CLI; license clarified as GPL-3.0-or-later.

### Removed
- The legacy TensorFlow `app.py` entrypoint and SavedModel workflow.
