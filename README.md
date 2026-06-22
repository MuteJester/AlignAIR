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

- **More accurate than IgBLAST across the board.** On a 4,400‑case / 22‑stratum benchmark (bootstrap CIs, Bonferroni‑corrected) AlignAIR wins **23 of 24 metrics** — biggest on short fragments, reverse‑complement / arbitrary orientation, and D/J calling, where classical seed‑and‑extend degrades.
- **Dynamic genotype.** The allele reference is an **input, not baked into the weights.** Supply a genotype as YAML or FASTA — a subset of the trained reference and/or **novel alleles never seen in training** — and the model conditions on exactly what you provide. (Validated: novel alleles are called as accurately as trained ones.)
- **Calibrated uncertainty.** When a read can't distinguish alleles (e.g. short fragments), AlignAIR reports a calibrated **equivalence set** and degrades gracefully to gene/family level instead of guessing.
- **AIRR output.** Standard AIRR rearrangement TSV (V/D/J calls, coordinates, junction, productivity) plus uncertainty columns.

## Install

```bash
pip install "AlignAIR[cli]"            # core + CLI
pip install "AlignAIR[cli,reader]"     # + parasail (faster, sharper V calling)
alignair doctor                        # verify Python / PyTorch+CUDA / GenAIRR
```

Or Docker:

```bash
docker pull thomask90/alignair:latest
docker run --rm thomask90/alignair:latest doctor
```

> GPU is auto‑detected for inference; the default Docker image is CPU‑only (small + portable).

## Quick start

You need a model bundle (a pretrained one, or one you train — see below). Then:

```bash
# align reads -> AIRR rearrangement TSV
alignair predict examples/reads.fasta -o out.tsv --model <bundle_or_checkpoint>

# align against a donor genotype (fewer alleles and/or NOVEL alleles) — YAML or FASTA
alignair predict examples/reads.fasta -o out.tsv \
  --model <bundle_or_checkpoint> --genotype examples/donor_genotype.yaml
```

`--genotype` simply *becomes* the reference for the run — no retraining, no extra flags. See
[`examples/`](examples/) for runnable data.

## Bring your own reference

Because the reference is an input, AlignAIR works on any reference you hand it:

- **A donor's genotype** (a subset of known alleles) — `--genotype donor.yaml` or `donor.fasta`.
- **Novel alleles** — list them in the genotype file with their sequence; they're embedded and aligned at predict time.
- **A new species / locus** — train your own AlignAIR model on your reference (training workflow in [docs/dnalignair.md](docs/dnalignair.md); a first‑class `alignair train` command is on the roadmap — see [docs/architecture/adoption_roadmap.md](docs/architecture/adoption_roadmap.md)).

## Output

`alignair predict` writes an AIRR rearrangement TSV: `sequence_id`, `sequence`, `locus`,
`v_call`/`d_call`/`j_call`, per‑gene sequence/germline coordinates, `junction`/`junction_aa`,
`productive`, `rev_comp`, plus calibrated uncertainty columns (`*_call_set`, `*_call_level`,
`*_set_confidence`).

## Commands

| Command | Purpose |
| --- | --- |
| `alignair predict` | align reads → AIRR rearrangement TSV |
| `alignair doctor` | check the environment (Python, PyTorch+CUDA, GenAIRR, parasail) |
| `alignair bundle` | package a raw checkpoint into a versioned, fingerprinted bundle |

Run `alignair <command> --help` for options.

## Documentation

- [Getting started](docs/getting_started.md)
- [DNAlignAIR design & benchmarks](docs/dnalignair.md)
- [Adoption roadmap](docs/architecture/adoption_roadmap.md)

## Development

```bash
git clone https://github.com/MuteJester/AlignAIR && cd AlignAIR
pip install -e ".[dev]"
pytest
```

## Citation

```
doi:10.5281/zenodo.15687939
```

## License

GPL‑3.0‑or‑later (see [`LICENSE`](LICENSE)).

## Contact

Issues: [GitHub issues](https://github.com/MuteJester/AlignAIR/issues) · Email: thomaskon90@gmail.com · Site: https://alignair.ai
