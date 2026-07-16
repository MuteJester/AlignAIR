<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="220" alt="AlignAIR logo" />
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  A neural aligner for immunoglobulin (IG) and T-cell receptor (TCR) repertoires. A single end-to-end
  model predicts V/D/J allele calls, segment coordinates, and the junction, and writes standard AIRR
  output. Constrain calls to a donor genotype (a subset of the model's reference), or train a model for
  your own reference / species.<br>
  <a href="https://hub.docker.com/r/thomask90/alignair"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair"></a>
  <a href="https://doi.org/10.1093/nar/gkaf651"><img src="https://img.shields.io/badge/DOI-10.1093%2Fnar%2Fgkaf651-blue" alt="DOI"></a>
  <a href="LICENSE"><img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
</p>

---

## Overview

- **End-to-end neural model.** A single network detects read orientation, localizes the V/D/J segments, and calls alleles from a shared representation, producing V/D/J calls, segment coordinates, the junction, productivity, and mutation rate in one pass, with no multi-stage heuristic pipeline.
- **Self-contained models.** Each model embeds a fingerprinted germline reference and loads without executing any pickle. Pretrained human IGH, IGK+IGL, and TRB models are a command away (`--model <id>`); the germline catalog travels with the model.
- **Donor genotype constraint.** At inference you can restrict calls to a subset of the model's reference (a donor genotype, as YAML or FASTA) with no retraining. Adding alleles, a new species, or a new locus requires training a new compatible model.
- **Uncertainty-aware.** When a read cannot distinguish alleles (e.g. short fragments), AlignAIR reports an **equivalence set** (`*_call_set`) rather than forcing a single call. Optional per-allele confidence calibration is available as a separate step.
- **AIRR output.** Standard AIRR rearrangement TSV (V/D/J calls, coordinates, junction, productivity) that reads directly into Change-O / Scirpy / Immcantation.
- **Evaluated against IgBLAST.** On a 4,400-case / 22-stratum benchmark (bootstrap CIs, Bonferroni-corrected), AlignAIR reports higher accuracy on 23 of 24 metrics, with the largest improvements on short fragments, reverse-complement / arbitrary orientation, and D/J calling. See [benchmarks](https://mutejester.github.io/AlignAIR/#/docs/benchmarks).

## Install

```bash
pip install "AlignAIR[cli]"            # core + CLI (recommended)
alignair doctor                        # verify Python / PyTorch+CUDA / GenAIRR
```

Install extras: `[cli]` (CLI + model download + AIRR validation + parasail), `[train]` (training
extras), `[all]`. PyTorch is auto-detected for GPU; for a CPU-only install, `pip install torch
--index-url https://download.pytorch.org/whl/cpu` first.

Or Docker (no local install needed):

```bash
docker pull thomask90/alignair:latest
docker run --rm thomask90/alignair:latest doctor

# align reads: mount an input dir + an output dir, and persist the model cache across runs
docker run --rm \
  -v "$PWD:/data" -v alignair-cache:/home/appuser/.cache/alignair \
  thomask90/alignair:latest \
  predict --input /data/reads.fasta --out /data/out.tsv --model alignair-igh-human
```

The default image is CPU-only; pin a version tag (`thomask90/alignair:3.0.0`) for reproducibility. The
container runs as a non-root user, so mount a writable output dir (add `--user $(id -u):$(id -g)` if
your host uid differs). Models are not baked in - the `alignair-cache` volume above keeps a downloaded
`--model <id>` from being re-fetched on every run. GPU is auto-detected when you run in a CUDA base image.

## Quick start

See it work end-to-end in one command - offline, no model download needed (it trains a tiny demo
model, aligns simulated reads, validates the AIRR output, and runs the donor-genotype path):

```bash
alignair demo
```

Or use a **pretrained model** - downloaded automatically from the public
[model hub](https://huggingface.co/AlignAIR/AlignAIR-pretrained) on first use, no login:

```bash
alignair models list                          # human IGH, IGK+IGL, TRB (fetched live from the hub)
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human

# restrict calls to a donor's genotype (a subset of the model's reference) - YAML or FASTA
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human --genotype donor.yaml
```

`--model <id>` downloads + hash-verifies + caches the model on first use (pin a version with
`--model <id>@<version>`). `--genotype` constrains the run to a subset of the model's reference - no
retraining.

Prefer your own reference or species? Train a model, then align with it:

```bash
alignair train --dataconfig HUMAN_IGH_OGRDB --out my_model --preset desktop   # ~minutes on a GPU
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle/model.alignair
```

See [`examples/`](examples/) for runnable data.

## Reference: donor subsets now, new references by training

Each model is tied to the germline reference it was trained on (embedded and fingerprinted in the
model file). What you can do with it:

- **Constrain to a donor's genotype** - a subset of the model's reference - with `--genotype donor.yaml`
  or `donor.fasta` at predict time. No retraining; calls are restricted to that donor's alleles.
- **Add alleles / a new species / a new locus** - this changes the model's allele universe, so **train
  a new compatible model** (novel alleles are not callable by a model that was not trained on them):

```bash
# train for any of GenAIRR's ~90 built-in references (human, mouse, rat, rabbit, dog, ...)
alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh --preset desktop

# or train from your OWN germline FASTAs (custom reference or species)
alignair train --v-fasta v.fasta --d-fasta d.fasta --j-fasta j.fasta \
  --chain-type BCR_HEAVY --out runs/my_ref --preset desktop
```

This writes checkpoints to `runs/.../` plus a self-contained, pickle-free `runs/.../bundle/model.alignair`
(the reference is **embedded**), a `model_card.md`, a `reference_manifest.json`, and a
`validation_report.json`. Presets: `quick` (smoke), `desktop`, `full` (paper-grade). Preview the
reference/config/model size without training with `--plan`. Then just
`alignair predict ... --model runs/.../bundle/model.alignair`.

## Output

`alignair predict` writes a **schema-valid AIRR rearrangement TSV** (validates against the official
`airr` library; reads back with Change-O / Immcantation): `sequence_id`, `sequence`, `rev_comp`,
`productive`, `v_call`/`d_call`/`j_call`, `junction`/`junction_aa`, gapped `sequence_alignment` /
`germline_alignment`, per-gene `*_cigar`, `*_identity`, and sequence/germline coordinates, plus a
per-gene equivalence-set column (`*_call_set`) for alleles a read cannot distinguish. The gapped
alignment fields are produced by AlignAIR's own IMGT-gap reconstruction (no external aligner required;
`parasail` is bundled in `[cli]` for exact CIGARs and the fast reader).

Every run also writes a **`<output>.run.json` provenance sidecar** (model + fingerprint, reference,
command, device, seed, and package versions). Validate any TSV explicitly:

```bash
alignair validate-airr out.tsv      # -> "VALID AIRR-C rearrangement"
```

## Commands

| Command | Purpose |
| --- | --- |
| `alignair demo` | offline end-to-end trial (tiny train → predict → validate → genotype) |
| `alignair predict` | align reads → AIRR rearrangement TSV |
| `alignair train` | train a model for your own reference / species (built-in dataconfig or custom FASTA) |
| `alignair models` | list / download / manage pretrained models |
| `alignair reference` | list built-in references, or export a model's reference |
| `alignair compare` | agreement report between two AIRR TSVs (e.g. AlignAIR vs IgBLAST) on your data |
| `alignair validate-airr` | validate a rearrangement TSV against the AIRR-C schema |
| `alignair doctor` | check the environment (Python, PyTorch+CUDA, GenAIRR, parasail) |
| `alignair convert` | convert a legacy checkpoint into a versioned, fingerprinted `.alignair` |

Run `alignair <command> --help` for options.

## Documentation

Full docs, reference, and interactive lessons: **<https://mutejester.github.io/AlignAIR/>**

- [Getting started](https://mutejester.github.io/AlignAIR/#/docs/getting-started)
- [Pretrained models](https://mutejester.github.io/AlignAIR/#/docs/models)
- [Command-line reference](https://mutejester.github.io/AlignAIR/#/docs/cli)
- [Benchmarks](https://mutejester.github.io/AlignAIR/#/docs/benchmarks)
- [Design & internals](https://mutejester.github.io/AlignAIR/#/docs/design)
- [Troubleshooting](https://mutejester.github.io/AlignAIR/#/docs/troubleshooting)

## Development

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use AlignAIR, please cite:

> Thomas Konstantinovsky, Ayelet Peres, Ran Eisenberg, Pazit Polak, Ofir Lindenbaum, Gur Yaari.
> Enhancing sequence alignment of adaptive immune receptors through multi-task deep learning.
> *Nucleic Acids Research*, Volume 53, Issue 13, 22 July 2025, gkaf651.
> <https://doi.org/10.1093/nar/gkaf651>

## License

GPL-3.0-or-later (see [`LICENSE`](LICENSE)).

## Contact

Issues: [GitHub issues](https://github.com/MuteJester/AlignAIR/issues) · Email: thomaskon90@gmail.com · Site: https://alignair.ai
