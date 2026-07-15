# Troubleshooting

Run `alignair doctor` first — it reports Python, PyTorch + CUDA, GenAIRR, and parasail status.

## Install

- **`torch` install is huge / pulls CUDA.** For a CPU-only environment install CPU torch first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install "AlignAIR[cli]"`.
- **`No module named GenAIRR`.** Install the CLI extra (`pip install "AlignAIR[cli]"`); GenAIRR is a
  core dependency on PyPI.
- **CUDA not used.** `alignair doctor` shows `CUDA available: False` → you have a CPU torch build;
  reinstall a CUDA wheel matching your driver, or run with `--device cpu` intentionally.

## Prediction

- **`error: model not found`.** `--model` takes a pretrained id (`alignair models list`), a local
  `.alignair` file (e.g. `runs/.../bundle/model.alignair`) or `.pt` checkpoint, or an `org/name`
  Hugging Face repo id — not a bundle *directory*; point at the `.alignair` file inside it.
- **`no sequence column in <file>`.** For CSV/TSV, pass `--sequence-column` (and `--id-column`).
- **Reverse-complement reads.** Handled automatically — `rev_comp=T` marks reoriented reads, and
  coordinates/`sequence` are in the canonical forward frame.

## Docker

- **`Permission denied` writing output.** The image runs as a non-root user; mount a writable
  output dir and add `--user $(id -u):$(id -g)`. See [`examples/README.md`](https://github.com/MuteJester/AlignAIR/blob/main/examples/README.md).
- **No model in the image.** Models aren't baked in — mount one with `-v` or use a catalog id.

## Training

- **`V allele has no anchor`.** Your custom FASTA reference has alleles GenAIRR can't anchor; those
  reads still get calls but no junction. Prefer a curated reference, or accept junction-free rows for
  the unanchored alleles.
- **Junction is empty in output.** Junction needs conserved anchors. Built-in dataconfigs and the
  trained reference carry them; plain YAML/FASTA genotypes do not, so junction is omitted there.

## AIRR output

- **Validate any TSV:** `alignair validate-airr out.tsv`. The output is AIRR-C schema-valid and
  reads back via the `airr` library / Change-O.
- **`sequence_alignment` / `germline_alignment` / `*_cigar` / `*_identity`** are produced by a real
  gapped alignment (parasail) when it is installed (it's in `[cli]`). Without parasail, AlignAIR
  falls back to coordinate-derived ungapped cigars and an empty `germline_alignment`; a lighter
  `--columns core`/`minimal` preset skips the gapped-alignment assembly entirely for speed.
