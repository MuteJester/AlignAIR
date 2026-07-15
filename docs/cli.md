# Command-line reference

Every command supports `alignair <command> --help` for its full option list. This page covers the
common ones; the two workhorses are [`predict`](#alignair-predict) and [`train`](#alignair-train).

| Command | Purpose |
| --- | --- |
| [`predict`](#alignair-predict) | align reads → AIRR rearrangement TSV |
| [`train`](#alignair-train) | train a model on a built-in dataconfig or your own FASTAs |
| [`demo`](#alignair-demo) | offline end-to-end trial (train → predict → validate → genotype) |
| [`models`](#alignair-models) | list / download / manage pretrained models |
| [`reference`](#alignair-reference) | list built-in references, or export a model's reference |
| [`convert`](#alignair-convert) | package a legacy `.pt` into a safe, pickle-free `.alignair` |
| [`validate-airr`](#alignair-validate-airr) | validate an AIRR TSV against the AIRR-C schema |
| [`compare`](#alignair-compare) | agreement report between two AIRR TSVs |
| `analyze` | summarize an AIRR TSV (repertoire + QC) |
| `genotype` | infer an individual's genotype from a repertoire (experimental) |
| `doctor` | check the environment (Python, PyTorch+CUDA, GenAIRR, parasail) |

## `alignair predict`

Align reads with a trained model and write an AIRR rearrangement TSV.

```bash
alignair predict --input reads.fasta --out out.tsv --model alignair-igh-human
```

- `--input` FASTA / FASTQ / CSV / TSV (`.gz` ok; `-` reads stdin). For a table, pass
  `--sequence-column` (and `--id-column`).
- `--out` output AIRR TSV path.
- `--model` a pretrained id (`alignair-igh-human`), a local `.alignair`/`.pt` path, or an `org/name`
  Hugging Face repo id. Pin a catalog version with `id@version`.
- `--genotype FILE` constrain calls to a donor genotype — a YAML/FASTA subset of the model's reference
  (`--genotype-method mask|softmax|renormalize|redistribute`, default `mask`).
- `--metadata FILE` join a per-read side table (e.g. 10x `filtered_contig_annotations.csv`) into the
  output by read id; `--keep-columns a,b,c` selects which columns to carry.
- `--columns full|core|minimal|airr` choose the output field set (lighter presets skip the gapped-
  alignment assembly and run faster).
- `--chunk-size N` stream the input in chunks for bounded memory (repertoire-scale; default 20000).
- `--device cpu|cuda` force a device (auto by default). `--batch-size N` (default 64).
- `--rejects-out FILE` write dropped/invalid input records for inspection.

Each run also writes a `<out>.run.json` provenance sidecar (model + fingerprint, command, versions).

## `alignair train`

Train a model on a built-in GenAIRR dataconfig **or** your own germline FASTAs, and export a
self-contained, pickle-free bundle.

```bash
# built-in reference
alignair train --dataconfig HUMAN_IGH_OGRDB --out runs/my_igh --preset desktop

# your own germline FASTAs (custom reference / species)
alignair train --v-fasta v.fa --d-fasta d.fa --j-fasta j.fa --chain-type BCR_HEAVY \
  --out runs/custom --preset desktop
```

- `--dataconfig NAME [NAME ...]` one or more built-in references (`alignair reference list`); several
  names train a multi-chain model.
- `--v-fasta` / `--j-fasta` / `--d-fasta` + `--chain-type` train from custom germline FASTAs (D only for
  heavy / D-bearing loci).
- `--preset quick|desktop|full` resource-tuned defaults; override with `--steps` / `--batch-size` / `--lr`.
- `--plan` validate the reference/config and print the plan **without training**.
- `--overwrite` replace an existing `bundle/` in `--out` (refused by default, to protect a published bundle).

Writes `runs/.../bundle/model.alignair` plus a `model_card.md`, `reference_manifest.json`, and
`validation_report.json`. Align with it via `--model runs/.../bundle/model.alignair`.

## `alignair demo`

```bash
alignair demo
```

Trains a tiny model, aligns simulated reads, validates the AIRR output, and runs the donor-genotype
path — entirely offline, no download. Writes to a temp directory (or `--out DIR`). The demo model is
**not** accurate; it only proves the pipeline works.

## `alignair models`

```bash
alignair models list                         # catalog + install status
alignair models get alignair-igh-human       # pre-download (id or id@version)
alignair models info alignair-igh-human      # model card
```

Also `path`, `verify`, `update`, `prune`. See [Pretrained models](models.md).

## `alignair reference`

```bash
alignair reference list                          # built-in GenAIRR dataconfigs + custom chain types
alignair reference export model.alignair --fasta ref.fasta   # extract a model's germline reference
```

## `alignair convert`

```bash
alignair convert model.pt model.alignair --dataconfig HUMAN_IGH_OGRDB --trust-pickle
```

Package a legacy `.pt` checkpoint into a safe, versioned, pickle-free `.alignair` (`--trust-pickle`
is required consent, since reading a `.pt` runs `torch.load`).

## `alignair validate-airr`

```bash
alignair validate-airr out.tsv
```

Validate a rearrangement TSV against the AIRR-C schema (columns + coordinate/CIGAR bounds).

## `alignair compare`

```bash
alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md
```

Per-gene agreement between two AIRR TSVs on the same reads (no ground truth needed), including
**set-rescue**: how often the other tool's call falls inside AlignAIR's equivalence set.
