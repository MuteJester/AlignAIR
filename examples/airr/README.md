# AIRR input with sample metadata

[reads.tsv](reads.tsv) contains sequence IDs and sequences.
[sample_metadata.tsv](sample_metadata.tsv) supplies sample, subject, tissue, and timepoint
values keyed by `sequence_id`.

Run from the repository root:

```bash
alignair predict --input examples/airr/reads.tsv --out out.tsv \
  --model alignair-igh-human --metadata examples/airr/sample_metadata.tsv \
  --keep-columns sample_id,subject_id,tissue,timepoint
```

The selected metadata columns are appended to the output.
See [integrations](https://mutejester.github.io/AlignAIR/#/docs/integrations) and
[AIRR output fields](https://mutejester.github.io/AlignAIR/#/docs/airr-fields).
