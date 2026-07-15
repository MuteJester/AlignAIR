<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="220" alt="AlignAIR logo" />
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  A neural aligner for immunoglobulin (IG) and T‑cell receptor (TCR) repertoires — more accurate than
  IgBLAST, with uncertainty‑aware calls and standard AIRR output. Constrain calls to a donor genotype (a
  subset of the model's reference), or train a model for your own reference / species.<br>
  <a href="https://hub.docker.com/r/thomask90/alignair"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair"></a>
  <a href="https://doi.org/10.5281/zenodo.15687939"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15687939.svg" alt="DOI"></a>
  <a href="LICENSE"><img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
</p>

---

## Why AlignAIR

- **More accurate than IgBLAST across the board.** On a 4,400‑case / 22‑stratum benchmark (bootstrap CIs, Bonferroni‑corrected) AlignAIR wins **23 of 24 metrics** — biggest on short fragments, reverse‑complement / arbitrary orientation, and D/J calling, where classical seed‑and‑extend degrades. See [benchmarks](docs/benchmarks.md).
- **Donor genotype constraint.** Each model is trained against an embedded, fingerprinted germline reference. At inference you can restrict calls to a **compatible subset or donor genotype** drawn from that reference (as YAML or FASTA) — no retraining — which sharpens accuracy on ambiguous reads. Adding alleles the model was not trained on, changing species, or changing the allele universe requires **training or fine‑tuning** a compatible model.
- **Uncertainty‑aware.** When a read can't distinguish alleles (e.g. short fragments), AlignAIR reports an **equivalence set** of the alleles it cannot tell apart (`*_call_set`) instead of forcing a single guess. Optional per‑allele confidence calibration is available as a separate step.
- **AIRR output.** Standard AIRR rearrangement TSV (V/D/J calls, coordinates, junction, productivity) plus uncertainty columns.

## Install

```bash
pip install "AlignAIR[cli]"            # core + CLI (recommended)
alignair doctor                        # verify Python / PyTorch+CUDA / GenAIRR
```

Install extras: `[cli]` (CLI + model download + AIRR validation), `[reader]` (parasail — faster,
sharper V calling), `[train]` (training extras), `[all]`. PyTorch is auto‑detected for GPU; for a
CPU‑only install, `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

Or Docker:

```bash
docker pull thomask90/alignair:latest
docker run --rm thomask90/alignair:latest doctor
```

> GPU is auto‑detected for inference; the default Docker image is CPU‑only (small + portable).

## Quick start

See it work end-to-end in one command — offline, no model download needed (it trains a tiny demo
model, aligns simulated reads, validates the AIRR output, and runs the donor-genotype path):

```bash
alignair demo
```

> Pretrained models are not published yet. Today you train your own (below); `alignair model list`
> shows the catalog and its status. `--model` accepts a local bundle / `.pt`, and (once published)
> a catalog id or `org/name` Hugging Face repo id.

Train a model for your reference, then align with it:

```bash
alignair train --dataconfig HUMAN_IGH_OGRDB --out my_model --preset desktop   # ~minutes on a GPU
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle/model.alignair

# restrict calls to a donor's genotype (a subset of the model's reference) — YAML or FASTA
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle/model.alignair --genotype donor.yaml
```

`--genotype` constrains the run to a subset of the model's reference — no retraining. See
[`examples/`](examples/) for runnable data.

## Reference: donor subsets now, new references by training

Each model is tied to the germline reference it was trained on (embedded and fingerprinted in the
model file). What you can do with it:

- **Constrain to a donor's genotype** — a subset of the model's reference — with `--genotype donor.yaml`
  or `donor.fasta` at predict time. No retraining; calls are restricted to that donor's alleles.
- **Add alleles / a new species / a new locus** — this changes the model's allele universe, so **train
  or fine‑tune** a compatible model (novel alleles are not callable by a model that was not trained on
  them):

```bash
# train for any of GenAIRR's ~90 built-in references (human, mouse, rat, rabbit, dog, …)
alignair train --dataconfig MOUSE_IGH_IMGT --out runs/mouse_igh --preset desktop

# or train from your OWN germline FASTAs (custom reference or species)
alignair train --v-fasta v.fasta --d-fasta d.fasta --j-fasta j.fasta \
  --chain-type BCR_HEAVY --out runs/my_ref --preset desktop
```

This writes checkpoints to `runs/.../` plus a self-contained, pickle-free `runs/.../bundle/model.alignair`
(the reference is **embedded**), a `model_card.md`, a `reference_manifest.json`, and a
`validation_report.json`. Presets: `quick` (smoke), `desktop`, `full` (paper‑grade). Preview the
reference/config/model size without training with `--plan`; resume an interrupted run with
`--resume runs/.../model.alignair`. Then just `alignair predict … --model runs/.../bundle/model.alignair`.

## Output

`alignair predict` writes a **schema‑valid AIRR rearrangement TSV** (validates against the official
`airr` library; reads back with Change‑O / Immcantation): `sequence_id`, `sequence`, `rev_comp`,
`productive`, `v_call`/`d_call`/`j_call`, `junction`/`junction_aa`, gapped `sequence_alignment` /
`germline_alignment`, per‑gene `*_cigar`, `*_identity`, and sequence/germline coordinates, plus a
per‑gene equivalence‑set column (`*_call_set`) for alleles a read cannot distinguish. The alignment
fields come from a real gapped alignment (parasail).

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
| `alignair model` | list / download / inspect pretrained models |
| `alignair reference` | validate / convert a germline reference (YAML ↔ FASTA) |
| `alignair compare` | agreement report between two AIRR TSVs (e.g. AlignAIR vs IgBLAST) on your data |
| `alignair validate-airr` | validate a rearrangement TSV against the AIRR-C schema |
| `alignair doctor` | check the environment (Python, PyTorch+CUDA, GenAIRR, parasail) |
| `alignair convert` | convert a legacy checkpoint into a versioned, fingerprinted `.alignair` |

Run `alignair <command> --help` for options.

## Documentation

- [Getting started](docs/getting_started.md)
- [Benchmarks — AlignAIR vs IgBLAST](docs/benchmarks.md)
- [DNAlignAIR design & internals](docs/dnalignair.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Adoption roadmap](docs/architecture/adoption_roadmap.md)

## Development

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and the
[adoption roadmap](docs/architecture/adoption_roadmap.md).

## Citation

```
doi:10.5281/zenodo.15687939
```

## License

GPL‑3.0‑or‑later (see [`LICENSE`](LICENSE)).

## Contact

Issues: [GitHub issues](https://github.com/MuteJester/AlignAIR/issues) · Email: thomaskon90@gmail.com · Site: https://alignair.ai
