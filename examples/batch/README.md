# Many samples

[manifest.tsv](manifest.tsv) has a header and three tab-separated columns: `sample_id`, `input` and an optional
`metadata`. Its paths are relative to this folder. Its two rows use the example reads and the 10x contigs from
elsewhere in `examples/`.

Run this Bash block from the repository root:

```bash
set -euo pipefail
alignair models get alignair4-igh-human
out_dir="${TMPDIR:-/tmp}/alignair-batch-example"
mkdir -p "$out_dir"
tail -n +2 examples/batch/manifest.tsv |
while IFS=$'\t' read -r sample input metadata; do
    [[ -n "$sample" ]] || continue
    if [[ -n "$metadata" ]]; then
        alignair predict --model alignair4-igh-human --input "examples/batch/$input" \
          --metadata "examples/batch/$metadata" --out "$out_dir/${sample}.tsv" --offline --quiet
    else
        alignair predict --model alignair4-igh-human --input "examples/batch/$input" \
          --out "$out_dir/${sample}.tsv" --offline --quiet
    fi
done
```

The model is downloaded once, before the loop, and every run reads it from the cache with `--offline`, so parallel
jobs never fetch it at the same time. The loop skips the header, passes the metadata table when a row names one, and
writes one AIRR table per sample, each with its own `.run.json`, under `$out_dir`.

Each run loads the model again. For many small samples, a Python loop loads it once:
[Run at scale](https://mutejester.github.io/AlignAIR/#/docs/run-at-scale) shows it, with `--threads` and GPU runs.
The [workflow wrappers](../../workflows/) run the same command from Nextflow, Snakemake or Galaxy.
