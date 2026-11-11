# AIRR input with sample metadata

[reads.tsv](reads.tsv) contains sequence IDs and sequences.
[sample_metadata.tsv](sample_metadata.tsv) supplies sample, subject, tissue, and timepoint
values keyed by `sequence_id`.

Run from the repository root:

```bash
alignair predict --input examples/airr/reads.tsv --out out.tsv \
  --model alignair4-igh-human --metadata examples/airr/sample_metadata.tsv \
  --keep-columns sample_id,subject_id,tissue,timepoint
```

`alignair4-igh-human` is the AlignAIR 4 human IGH model's id in the shared catalog. It is not published yet;
until it is, pass an AlignAIR 4 `.alignair` path to `--model`. <!-- PENDING-V4-PUBLISH: alignair4-igh-human -->

The selected metadata columns are appended to the output.
See [integrations](https://mutejester.github.io/AlignAIR/#/docs/integrations) and
[AIRR output fields](https://mutejester.github.io/AlignAIR/#/docs/airr-fields).
