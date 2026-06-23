# Getting started

A 5‑minute walk‑through of AlignAIR: install, check your environment, and align reads — including
against your own donor/novel genotype.

## 1. Install

```bash
pip install "AlignAIR[cli]"            # core + CLI
pip install "AlignAIR[cli,reader]"     # optional: parasail (faster, sharper V calling)
```

Check the environment (Python, PyTorch + CUDA, GenAIRR, optional parasail):

```bash
alignair doctor
```

GPU is auto‑detected; pass `--device cpu` or `--device cuda` to `alignair predict` to force one.

## 1b. See it work in one command (offline)

```bash
alignair demo
```

Trains a tiny demo model (not production quality), aligns simulated reads, validates the AIRR
output, and runs the dynamic-genotype path — proving the full pipeline with no model download.

## 2. Get a model

`alignair predict` needs a model — either a **bundle** directory or a raw `.pt` checkpoint. Options:

- **Pretrained bundles** are not published yet. `alignair model list` shows the catalog and its
  status; `alignair model download <id>` / `--model <id>` will work once a model is published.
- **Train your own** for any reference or species (the path to use today):
  ```bash
  # any of GenAIRR's ~90 built-in references
  alignair train --reference HUMAN_IGH_OGRDB -o runs/my_igh --preset desktop
  # or your own germline FASTAs (custom/novel reference)
  alignair train --v-fasta v.fa --d-fasta d.fa --j-fasta j.fa --chain-type BCR_HEAVY \
    -o runs/custom --preset desktop --allow-curatable
  ```
  This writes `runs/.../bundle/` (self-contained — custom references are embedded), plus a
  `model_card.md` and `validation_report.json`. Use it with `--model runs/.../bundle`.
- Package a raw checkpoint into a versioned bundle: `alignair bundle --model ckpt.pt -o my_bundle/`.

## 3. Align reads

Input may be FASTA, FASTQ, CSV/TSV, or TXT (optionally `.gz`):

```bash
alignair predict examples/reads.fasta -o out.tsv --model <bundle_or_checkpoint>
```

The output is an AIRR rearrangement TSV with `v_call`/`d_call`/`j_call`, per‑gene sequence and
germline coordinates, `junction`/`junction_aa`, `productive`, `rev_comp`, and calibrated
uncertainty columns (`*_call_set`, `*_call_level`, `*_set_confidence`).

## 4. Use your own reference (dynamic genotype)

The reference is an **input**. Supply a genotype as YAML or FASTA — it can contain **fewer alleles**
than the trained reference and/or **novel alleles** the model has never seen:

```bash
# YAML: top-level v/d/j, each {allele_name: dna_sequence}
alignair predict examples/reads.fasta -o out.tsv \
  --model <bundle_or_checkpoint> --genotype examples/donor_genotype.yaml

# FASTA: >allele_name headers (gene type inferred from the AIRR/IMGT name)
alignair predict examples/reads.fasta -o out.tsv \
  --model <bundle_or_checkpoint> --genotype donor.fasta
```

No retraining is required — the model conditions on exactly the alleles you provide, and every call
is guaranteed to be within that genotype.

## 4b. Single-cell / 10x and preserving metadata

AlignAIR fits existing workflows by carrying per-read metadata into its output. Point `--metadata`
at a side table (joined by read id) — e.g. 10x `filtered_contig_annotations.csv`, or an AIRR TSV
that already has `cell_id`/`duplicate_count`:

```bash
# 10x: align the contigs, preserve barcode / UMI / chain / etc. for downstream single-cell tools
alignair predict filtered_contig.fasta -o out.tsv --model <bundle> \
  --metadata filtered_contig_annotations.csv

# carry specific columns (default: a known 10x/AIRR metadata set present in the file)
alignair predict reads.tsv -o out.tsv --model <bundle> \
  --metadata reads.tsv --keep-columns cell_id,duplicate_count,sample_id
```

The kept columns are appended to the AIRR TSV (still schema-valid), so Change-O / Scirpy /
Immcantation single-cell workflows get `cell_id`, barcodes, and counts without custom glue.

## 4c. Acting on calibrated uncertainty

For each gene AlignAIR reports more than a single call, so you can decide how much to trust it:

| column | meaning | how to act |
| --- | --- | --- |
| `*_call` | top-1 allele | use it when `*_call_level` is `allele` |
| `*_call_set` | calibrated equivalence set (alleles the read can't distinguish) | report the set when it's small |
| `*_resolved_call` | the most-specific *safe* call | use directly — it already degrades when uncertain |
| `*_call_level` | `allele` / `gene` / `family` / `none` | `allele`→accept; `gene`/`family`→use that level; `none`→abstain / inspect |
| `*_set_confidence` | probability mass inside the set | threshold for stricter calling |

Rule of thumb: take `*_resolved_call` (it collapses to gene/family or abstains when the evidence
can't support an allele) instead of forcing a possibly-wrong allele like classical tools do.

## 4d. Try a donor genotype (before/after)

```bash
alignair reference template HUMAN_IGH_OGRDB -o donor.yaml   # start from the full reference
#   ... edit donor.yaml down to the donor's alleles and/or add novel alleles ...
alignair predict reads.fasta -o full.tsv  --model <bundle>
alignair predict reads.fasta -o donor.tsv --model <bundle> --genotype donor.yaml
alignair compare --a donor.tsv --b full.tsv --a-name donor --b-name full --out genotype_effect.md
```

The report shows how conditioning on the donor reference changes calls and shrinks uncertainty sets.

## 4e. Many samples at once (cohorts)

Process a whole study with the model loaded once, instead of a shell loop:

```bash
# manifest columns: sample_id, input (+ optional genotype, metadata per row)
alignair batch --manifest samples.tsv -o results/ --model my_model/bundle
# -> results/<sample_id>.tsv per sample + results/manifest_summary.tsv (per-sample stats)
```

A failing sample is recorded as `status=error` in the summary and the run continues;
the command exits non-zero only if no sample aligned. See
[examples/batch/](https://github.com/MuteJester/AlignAIR/tree/main/examples/batch).

## 4f. Scripting & tooling

```bash
alignair doctor --json            # machine-readable environment report (CI-friendly)
alignair model list --json        # the catalog as JSON
alignair model inspect <bundle> --json
alignair reference list           # built-in GenAIRR references + valid custom chain types
alignair completion bash          # print the line to enable tab completion (needs AlignAIR[cli])
```

Before spending GPU hours, dry-run a training plan — it reports the model size, a timed
wall-clock and peak-memory estimate, the expected validation accuracy for the preset, and
any anchor/GPU warnings:

```bash
alignair train --reference HUMAN_IGH_OGRDB -o my_model --preset standard --plan
```

## 5. Common options

| Flag | Meaning |
| --- | --- |
| `--genotype FILE` | YAML/FASTA reference for this run (subset and/or novel alleles) |
| `--metadata FILE` | per-read metadata (CSV/TSV, e.g. 10x annotations) preserved into output |
| `--chunk-size N` | reads per streaming chunk (repertoire-scale, bounded memory) |
| `--calibration FILE` | allele‑set calibration JSON (overrides a bundled one) |
| `--v-reader parasail` | use the fast classical V reader (needs `AlignAIR[reader]`) |
| `--batch N` | batch size (default 64) |
| `--device cuda|cpu` | force a device (auto if unset) |
| `--quiet` | suppress progress output |
| `--json` | machine-readable output (on `doctor`, `model list/inspect`, `compare`) |

## Next steps

- [DNAlignAIR design & benchmarks](dnalignair.md) — architecture and head‑to‑head results.
- [Adoption roadmap](architecture/adoption_roadmap.md) — model zoo, `alignair train`, packaging.
