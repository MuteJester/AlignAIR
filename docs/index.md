# AlignAIR

**AlignAIR** is a neural aligner for **Adaptive Immune Receptor Repertoire (AIRR)** sequences —
immunoglobulin (IG) and T‑cell receptor (TCR). It assigns V/D/J alleles, segment coordinates,
junction / CDR3, productivity, and mutation rate, and writes standard AIRR rearrangement output. It is
more accurate than IgBLAST across a broad benchmark, and it is free software (GPL‑3.0‑or‑later).

[Get started](getting_started.md){ .md-button .md-button--primary }
[Benchmarks](benchmarks.md){ .md-button }
[View on GitHub](https://github.com/MuteJester/AlignAIR){ .md-button }

## Why AlignAIR

<div class="grid cards" markdown>

-   :material-target: __More accurate than IgBLAST__

    On a 4,400‑case / 22‑stratum benchmark (bootstrap CIs, Bonferroni‑corrected) AlignAIR wins
    **23 of 24 metrics** — biggest on short fragments, reverse‑complement / arbitrary orientation,
    and D/J calling. [See the benchmarks →](benchmarks.md)

-   :material-download: __Pretrained models, one command__

    `alignair predict --model alignair-igh-human` downloads a model from the public hub, verifies it,
    and aligns — no login. Human IGH, IGK+IGL, and TRB are available today.
    [Browse the models →](models.md)

-   :material-tune: __Donor‑genotype aware__

    Constrain calls to a donor's alleles (a subset of the model's reference) with `--genotype` — no
    retraining — to sharpen accuracy on ambiguous reads. [How it works →](model_contract.md)

-   :material-file-table: __Standard AIRR output__

    A schema‑valid AIRR rearrangement TSV (validates against the official `airr` library; reads back
    with Change‑O / Scirpy / Immcantation), plus an equivalence‑set column for ambiguous calls.
    [Output fields →](airr_fields.md)

</div>

## Install

=== "pip"

    ```bash
    pip install "AlignAIR[cli]"            # core + CLI
    pip install "AlignAIR[cli,reader]"     # + parasail (faster, sharper V calling)
    alignair doctor                        # verify Python / PyTorch+CUDA / GenAIRR
    ```

=== "Docker"

    ```bash
    docker pull thomask90/alignair:latest
    docker run --rm thomask90/alignair:latest doctor
    ```

PyTorch is auto‑detected for GPU; for a CPU‑only install,
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

## Quick start

```bash
# align reads against a pretrained model -> AIRR rearrangement TSV
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human

# constrain to a donor genotype (a subset of the model's reference)
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human --genotype donor.yaml
```

!!! tip "See it work with no setup"
    `alignair demo` trains a tiny model, aligns simulated reads, validates the AIRR output, and runs
    the donor‑genotype path — entirely offline, in one command.

Prefer your own reference or species? [Train a model](getting_started.md#2-get-a-model) — AlignAIR
embeds the germline reference into the model file, so the result is self‑contained.

## Loci & species

AlignAIR handles IG and TCR loci (IGH, IGK, IGL, TRB, …). Each model is trained for a given
reference; the pretrained catalog covers human IGH / IGK+IGL / TRB, and you can train your own model
for a new species or reference. A model is a **fixed‑reference classifier** — the embedded catalog is
exactly the set of alleles it can call ([model contract](model_contract.md)).

## Project

- **Source & issues:** <https://github.com/MuteJester/AlignAIR>
- **Pretrained models:** <https://huggingface.co/AlignAIR/AlignAIR-pretrained>
- **Citation:** `doi:10.5281/zenodo.15687939`
- **License:** GPL‑3.0‑or‑later
