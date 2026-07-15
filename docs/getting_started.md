# Getting started

A 5‑minute walk‑through of AlignAIR: install, check your environment, and align reads — including
against a donor genotype (a subset of the model's reference).

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
output, and runs the donor-genotype path — proving the full pipeline with no model download.

## 2. Get a model

`alignair predict` needs a model — a `.alignair` model file (from `alignair train`, inside a
`bundle/` directory), a catalog id, or a Hugging Face repo id. Options:

- **Use a pretrained model** (no login, downloaded on first use):
  ```bash
  alignair models list                      # human IGH, IGK+IGL, TRB (fetched live from the hub)
  alignair predict --input examples/reads.fasta --out out.tsv --model alignair-igh-human
  ```
  `--model <id>` resolves from the public [model hub](https://huggingface.co/AlignAIR/AlignAIR-pretrained),
  hash-verifies, and caches it; pin a version with `--model <id>@<version>`.
- **Train your own** for any reference or species:
  ```bash
  # any of GenAIRR's ~90 built-in references
  alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/my_igh --preset desktop
  # or your own germline FASTAs (custom reference)
  alignair train --v-fasta v.fa --d-fasta d.fa --j-fasta j.fa --chain-type BCR_HEAVY \
    --out runs/custom --preset desktop
  ```
  This writes `runs/.../bundle/` (self-contained — custom references are embedded), including
  `model.alignair`, a `model_card.md`, a `reference_manifest.json`, and a `validation_report.json`.
  Use it with `--model runs/.../bundle/model.alignair`.
- Package a raw `.pt` checkpoint into a safe, pickle-free `.alignair`:
  `alignair convert ckpt.pt model.alignair --dataconfig HUMAN_IGH_OGRDB --trust-pickle`.

## 3. Align reads

Input may be FASTA, FASTQ, CSV/TSV, or TXT (optionally `.gz`, or `-` for stdin):

```bash
alignair predict --input examples/reads.fasta --out out.tsv --model runs/my_igh/bundle/model.alignair
```

The output is an AIRR rearrangement TSV with `v_call`/`d_call`/`j_call`, per‑gene sequence and
germline coordinates, `junction`/`junction_aa`, `productive`, `rev_comp`, and a per‑gene
equivalence‑set column (`*_call_set`) listing the alleles a read cannot distinguish.

## 4. Constrain to a donor genotype

Supply a genotype as YAML or FASTA to restrict calls to a **subset of the model's reference** (a
donor's alleles) — no retraining. Alleles the model was not trained on are not callable (train a new
model to add them):

```bash
# YAML: top-level v/d/j, each {allele_name: dna_sequence}
alignair predict --input examples/reads.fasta --out out.tsv \
  --model runs/my_igh/bundle/model.alignair --genotype examples/donor_genotype.yaml

# FASTA: >allele_name headers (gene type inferred from the AIRR/IMGT name)
alignair predict --input examples/reads.fasta --out out.tsv \
  --model runs/my_igh/bundle/model.alignair --genotype donor.fasta
```

No retraining is required — the model conditions on exactly the alleles you provide, and every call
is guaranteed to be within that genotype.

## 4b. Single-cell / 10x and preserving metadata

AlignAIR fits existing workflows by carrying per-read metadata into its output. Point `--metadata`
at a side table (joined by read id) — e.g. 10x `filtered_contig_annotations.csv`, or an AIRR TSV
that already has `cell_id`/`duplicate_count`:

```bash
# 10x: align the contigs, preserve barcode / UMI / chain / etc. for downstream single-cell tools
alignair predict --input filtered_contig.fasta --out out.tsv --model <model.alignair> \
  --metadata filtered_contig_annotations.csv

# carry specific columns (default: a known 10x/AIRR metadata set present in the file)
alignair predict --input reads.tsv --out out.tsv --model <model.alignair> \
  --metadata reads.tsv --keep-columns cell_id,duplicate_count,sample_id
```

The kept columns are appended to the AIRR TSV (still schema-valid), so Change-O / Scirpy /
Immcantation single-cell workflows get `cell_id`, barcodes, and counts without custom glue. The
join key must be unique; if a column repeats, pass `--metadata-id-column` to pick a unique one.

## 4c. Handling uncertain calls

When a read can't pin down a single allele, AlignAIR reports the ambiguity instead of guessing:

| column | meaning |
| --- | --- |
| `*_call` | the top‑1 allele |
| `*_call_set` | the set of alleles the read cannot distinguish (comma‑separated) |

When `*_call_set` holds more than one allele, treat the result as gene/family‑level rather than a
confident single allele. Constraining the run to a donor genotype (below) shrinks these sets. (Optional
per‑allele confidence calibration is available as a separate step but is not applied by default.)

## 4d. Try a donor genotype (before/after)

```bash
alignair reference export runs/my_igh/bundle/model.alignair --fasta donor.fasta  # the full reference
#   ... edit donor.fasta down to the donor's alleles ...
alignair predict --input reads.fasta --out full.tsv  --model runs/my_igh/bundle/model.alignair
alignair predict --input reads.fasta --out donor.tsv --model runs/my_igh/bundle/model.alignair --genotype donor.fasta
alignair compare --a donor.tsv --b full.tsv --a-name donor --b-name full --out genotype_effect.md
```

The report shows how conditioning on the donor reference changes calls and shrinks uncertainty sets.

## 4e. Many samples at once (cohorts)

Process a cohort by looping `alignair predict` (or use a workflow engine — see `workflows/`):

```bash
# samples.tsv: sample_id<TAB>reads_path
while IFS=$'\t' read -r sample reads; do
    alignair predict --model runs/my_igh/bundle/model.alignair --input "$reads" \
        --out "results/${sample}.tsv" --quiet
done < samples.tsv
```

Each run writes an AIRR TSV + a `<out>.run.json` provenance sidecar. See
[examples/batch/](https://github.com/MuteJester/AlignAIR/tree/main/examples/batch).

## 4f. Scripting & tooling

```bash
alignair doctor --json                          # machine-readable environment report (CI-friendly)
alignair models list                            # the model catalog + install/update status
alignair info runs/my_igh/bundle/model.alignair --json   # a model file's metadata / card as JSON
alignair reference list                         # built-in GenAIRR references + valid custom chain types
```

Before spending GPU hours, dry-run a training plan — it reports the model size and the resolved
schedule, and validates the reference/config, without training:

```bash
alignair train --dataconfig HUMAN_IGH_OGRDB --out my_model --preset desktop --plan
```

## 5. Common `predict` options

| Flag | Meaning |
| --- | --- |
| `--genotype FILE` | YAML/FASTA genotype for this run (a subset of the model's reference) |
| `--genotype-method M` | how to apply it: `mask` (default) / `softmax` / `renormalize` / `redistribute` |
| `--metadata FILE` | per-read metadata (CSV/TSV, e.g. 10x annotations) preserved into output |
| `--keep-columns LIST` | comma-separated metadata columns to carry through |
| `--chunk-size N` | reads per streaming chunk (repertoire-scale, bounded memory; default 20000) |
| `--columns SPEC` | output columns: a preset (`full`/`core`/`minimal`/`airr`) or a field list |
| `--rejects-out FILE` | write dropped/invalid input records here (id, position, reason, sequence) |
| `--batch-size N` | batch size (default 64) |
| `--device cuda\|cpu` | force a device (auto if unset) |
| `--quiet` | suppress progress / update output |

## Next steps

- [DNAlignAIR design & benchmarks](dnalignair.md) — architecture and head‑to‑head results.
- [Adoption roadmap](architecture/adoption_roadmap.md) — model zoo, `alignair train`, packaging.
