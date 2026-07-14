<p align="center">
  <img src="https://alignair.ai/_next/static/media/logo_alignair11bw.17e5d8d6.svg" width="220" alt="AlignAIR logo" />
</p>

<h1 align="center">AlignAIR</h1>
<p align="center">
  A neural aligner for immunoglobulin (IG) and T‑cell receptor (TCR) repertoires that
  <b>conditions on your reference at runtime</b> — use a pretrained human IG/TCR model, or bring
  your own reference (any subset, novel alleles, or species).<br>
  <a href="https://hub.docker.com/r/thomask90/alignair"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/thomask90/alignair"></a>
  <a href="https://doi.org/10.5281/zenodo.15687939"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15687939.svg" alt="DOI"></a>
  <a href="LICENSE"><img alt="GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
</p>

---

## Why AlignAIR

- **More accurate than IgBLAST across the board.** On a 4,400‑case / 22‑stratum benchmark (bootstrap CIs, Bonferroni‑corrected) AlignAIR wins **23 of 24 metrics** — biggest on short fragments, reverse‑complement / arbitrary orientation, and D/J calling, where classical seed‑and‑extend degrades. See [benchmarks](docs/benchmarks.md).
- **Dynamic genotype.** The allele reference is an **input, not baked into the weights.** Supply a genotype as YAML or FASTA — a subset of the trained reference and/or **novel alleles never seen in training** — and the model conditions on exactly what you provide. (Validated: novel alleles are called as accurately as trained ones.)
- **Calibrated uncertainty.** When a read can't distinguish alleles (e.g. short fragments), AlignAIR reports a calibrated **equivalence set** and degrades gracefully to gene/family level instead of guessing.
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
model, aligns simulated reads, validates the AIRR output, and runs the dynamic-genotype path):

```bash
alignair demo
```

> Pretrained models are not published yet. Today you train your own (below); `alignair model list`
> shows the catalog and its status. `--model` accepts a local bundle / `.pt`, and (once published)
> a catalog id or `org/name` Hugging Face repo id.

Train a model for your reference, then align with it:

```bash
alignair train --reference HUMAN_IGH_OGRDB -o my_model --preset desktop   # ~minutes on a GPU
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle

# align against a donor genotype (fewer alleles and/or NOVEL alleles) — YAML or FASTA
alignair predict --input reads.fasta --out out.tsv --model my_model/bundle --genotype donor.yaml
```

`--genotype` simply *becomes* the reference for the run — no retraining, no extra flags. See
[`examples/`](examples/) for runnable data.

## Bring your own reference

Because the reference is an input, AlignAIR works on any reference you hand it:

- **A donor's genotype** (a subset of known alleles) — `--genotype donor.yaml` or `donor.fasta` at predict time.
- **Novel alleles** — list them in the genotype file with their sequence; they're embedded and aligned at predict time, no retraining.
- **A new species / locus** — **train your own model**:

```bash
# train for any of GenAIRR's ~90 built-in references (human, mouse, rat, rabbit, dog, …)
alignair train --reference MOUSE_IGH_IMGT -o runs/mouse_igh --preset desktop

# or train from your OWN germline FASTAs (custom/novel reference or species)
alignair train --v-fasta v.fasta --d-fasta d.fasta --j-fasta j.fasta \
  --chain-type BCR_HEAVY -o runs/my_ref --preset desktop

# fine-tune from an existing model instead of training from scratch
alignair train --reference HUMAN_IGK_OGRDB --base-model human_igh_bundle/ -o runs/igk
```

This writes a self-contained `runs/.../bundle/` (custom references are **embedded** in the bundle),
plus `model_card.md` and `validation_report.json`. Presets: `smoke` (quick check), `desktop`,
`standard` (paper‑grade). Training fits the equivalence‑set calibration automatically and
checkpoints each interval — resume an interrupted run with `--resume` (re‑run the same command and
add the flag). Then just `alignair predict … --model runs/.../bundle`.

## Output

`alignair predict` writes a **schema‑valid AIRR rearrangement TSV** (validates against the official
`airr` library; reads back with Change‑O / Immcantation): `sequence_id`, `sequence`, `rev_comp`,
`productive`, `v_call`/`d_call`/`j_call`, `junction`/`junction_aa`, gapped `sequence_alignment` /
`germline_alignment`, per‑gene `*_cigar`, `*_identity`, and sequence/germline coordinates, plus
calibrated‑uncertainty extension columns (`*_call_set`, `*_call_level`, `*_set_confidence`). The
alignment fields come from a real gapped alignment (parasail).

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
