# AIRR input with sample metadata

[reads.tsv](reads.tsv) contains sequence IDs and sequences.
[sample_metadata.tsv](sample_metadata.tsv) supplies sample, subject, tissue, and timepoint
values keyed by `sequence_id`.

Run from the repository root:

```bash
alignair predict --input examples/airr/reads.tsv --out out.tsv \
  --model <v4-igh-model> --metadata examples/airr/sample_metadata.tsv \
  --keep-columns sample_id,subject_id,tissue,timepoint
```

`<v4-igh-model>` is a placeholder for an AlignAIR 4 human IGH model (a `.alignair` path, a Hugging Face repo
`hf://org/repo`, or a catalog id); the id is not published yet. <!-- PENDING-V4-MODEL-ID -->

The selected metadata columns are appended to the output.
See [integrations](https://mutejester.github.io/AlignAIR/#/docs/integrations) and
[AIRR output fields](https://mutejester.github.io/AlignAIR/#/docs/airr-fields).
