# AlignAIR

[AlignAIR](https://github.com/MuteJester/AlignAIR) is a **neural aligner** for Adaptive Immune
Receptor Repertoire (AIRR) sequences — immunoglobulin (IG) and T‑cell receptor (TCR). It assigns
V/D/J alleles, segment coordinates, junction/CDR3, productivity, and mutation rate, and writes
standard AIRR rearrangement output. It is licensed **GPL‑3.0‑or‑later**.

## What makes it different

- **More accurate than IgBLAST across the board** (23/24 metrics on a 4,400‑case / 22‑stratum
  benchmark with bootstrap CIs) — especially on short fragments, arbitrary orientation, and D/J.
- **Dynamic genotype**: the allele reference is an **input**, not memorized in the weights. Supply
  a genotype (YAML or FASTA) that is a subset of the trained reference and/or contains **novel
  alleles**, and the model conditions on exactly that reference — no retraining.
- **Calibrated uncertainty**: reports an equivalence set and degrades to gene/family level when a
  read can't distinguish alleles, instead of guessing.

## Install

```bash
pip install "AlignAIR[cli]"            # core + CLI
pip install "AlignAIR[cli,reader]"     # + parasail (faster, sharper V calling)
alignair doctor                        # verify Python / PyTorch+CUDA / GenAIRR
```

Docker:

```bash
docker pull thomask90/alignair:latest
docker run --rm thomask90/alignair:latest doctor
```

## Quick start

```bash
# align reads -> AIRR rearrangement TSV
alignair predict reads.fasta -o out.tsv --model <bundle_or_checkpoint>

# align against a donor genotype (fewer alleles and/or NOVEL alleles)
alignair predict reads.fasta -o out.tsv --model <bundle_or_checkpoint> --genotype donor.yaml
```

See [Getting started](getting_started.md) for a full walk‑through and
[DNAlignAIR design & benchmarks](dnalignair.md) for the architecture and results.

## Loci

AlignAIR supports IG and TCR loci (IGH, IGK, IGL, TCRB, …). A model is trained for a given
locus/reference; you can also train your own model for a new species or reference (see the
training workflow in [dnalignair.md](dnalignair.md)).

## Project

- Source & issues: <https://github.com/MuteJester/AlignAIR>
- Citation: `doi:10.5281/zenodo.15687939`
- License: GPL‑3.0‑or‑later
