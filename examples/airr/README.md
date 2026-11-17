# Reads in a table, with sample metadata

[reads.tsv](reads.tsv) holds three human IGH reads, in a `sequence_id` and a `sequence` column.
[sample_metadata.tsv](sample_metadata.tsv) gives each read's sample, subject, tissue and time point, keyed by
`sequence_id`.

Run from the repository root:

<!-- PENDING-V4-PUBLISH: the model id alignair4-igh-human is confirmed when the models are published. -->
```bash
alignair predict --model alignair4-igh-human --input examples/airr/reads.tsv --out out.tsv \
  --metadata examples/airr/sample_metadata.tsv --keep-columns sample_id,subject_id,tissue,timepoint
```

AlignAIR reads a table's sequence from its `sequence` column and the id from `sequence_id`; a table with other column
names takes `--sequence-column` and `--id-column`. `--metadata` joins the metadata table by read id, and
`--keep-columns` names the columns to carry: they follow AlignAIR's 99 columns in `out.tsv`.

The other columns of the reads table itself are not carried into the output. To carry them, give the same table to
`--metadata` as well.

More: [Prepare your input](https://mutejester.github.io/AlignAIR/#/docs/prepare-input) (formats and tables) and
[Join sample and cell metadata](https://mutejester.github.io/AlignAIR/#/docs/metadata-join).
