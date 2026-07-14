# AIRR input with sample metadata

Bulk repertoire pipelines often carry sample/subject metadata alongside the reads.
AlignAIR can read an AIRR/CSV/TSV table of sequences and join an arbitrary metadata
table onto the output by `sequence_id`.

- `reads.tsv` — input sequences (`sequence_id` + `sequence` columns)
- `sample_metadata.tsv` — per-read metadata (`sample_id`, `subject_id`, `tissue`, `timepoint`)

```bash
alignair predict --input examples/airr/reads.tsv --out out.tsv \
  --model my_model/bundle \
  --metadata examples/airr/sample_metadata.tsv \
  --keep-columns sample_id,subject_id,tissue,timepoint
```

The output is a valid AIRR-C rearrangement TSV with the four metadata columns appended,
so downstream grouping by `sample_id` / `subject_id` works without a second join step.
Without `--keep-columns`, AlignAIR carries the known AIRR/10x metadata fields it finds
(here, `sample_id`).
