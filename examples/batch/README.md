# Multi-sample manifest (cohorts)

AIRR studies usually process a cohort, not one file. `alignair batch` aligns many
samples from a manifest with the model **loaded once**, writes one AIRR TSV per
sample, and emits a per-sample run summary — easier to wrap (Nextflow/Snakemake)
and audit than a shell loop that reloads the model every time.

```bash
alignair batch --manifest examples/batch/manifest.tsv -o results/ --model my_model/bundle
```

## Manifest format

A CSV or TSV with these columns:

| Column | Required | Meaning |
|--------|----------|---------|
| `sample_id` | yes | output file name (`<sample_id>.tsv`) and summary key |
| `input` | yes | reads file (FASTA/FASTQ/CSV/TSV, optionally `.gz`) |
| `genotype` | no | per-sample donor genotype (YAML/FASTA); falls back to `--genotype` then the model default |
| `metadata` | no | per-sample metadata table to join into the output (e.g. a 10x `filtered_contig_annotations.csv`) |

Relative paths are tried as-is, then resolved against the manifest's own directory
(so a manifest can sit next to its data and use bare file names).

## Outputs

```
results/
  donorA.tsv              # AIRR rearrangement TSV per sample (+ donorA.tsv.run.json)
  donorB_10x.tsv
  manifest_summary.tsv    # one row per sample: status, n_aligned, n_productive, dropped, seconds, ...
  manifest_summary.json   # same, machine-readable
```

A sample that fails (missing file, bad reference) is recorded as `status=error` in
the summary and the run continues; the command exits non-zero only if **no** sample
aligned. Flags like `--genotype`, `--keep-columns`, `--device`, `--batch`,
`--v-reader`, and `--no-full-alignment` apply to every sample.
