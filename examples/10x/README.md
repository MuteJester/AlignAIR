# 10x Genomics (Cell Ranger) BCR/TCR workflow

Cell Ranger `vdj` writes two files per sample:

- `filtered_contig.fasta` — the assembled contig nucleotide sequences
- `filtered_contig_annotations.csv` — per-contig metadata (barcode, UMIs, chain, clonotype, …)

AlignAIR aligns the contigs and **carries the 10x per-cell metadata into the AIRR
output**, so the result drops straight into single-cell tooling (Scirpy, Change-O,
Immcantation) where rows are grouped by cell.

```bash
alignair predict --input examples/10x/filtered_contig.fasta --out tenx_airr.tsv \
  --model my_model/bundle \
  --metadata examples/10x/filtered_contig_annotations.csv
```

The contig ids in the FASTA header (`AAACCTGAGAAACCAT-1_contig_1`) are matched to the
`contig_id` column of the annotations file. By default AlignAIR preserves the common
10x/AIRR columns it finds (`barcode`, `umis`, `reads`, `chain`, `raw_clonotype_id`,
`raw_consensus_id`, `is_cell`, `high_confidence`, …); pass `--keep-columns a,b,c` to
choose your own set.

Every output row is a valid AIRR-C rearrangement, plus the preserved 10x columns —
so you can group by `barcode` to reconstruct each cell's heavy/light chains.

> The example contigs are real human IGH reads. A paired BCR cell also has a light
> (IGK/IGL) contig; align those with the matching light-chain model. AlignAIR always
> preserves the 10x annotations regardless of which model produced the call.
