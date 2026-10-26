# Predict multiple samples

[manifest.tsv](manifest.tsv) has a header and three tab-separated columns:
`sample_id`, `input`, and optional `metadata`. Input paths are relative to this folder.
The two rows use the basic reads and the 10x contigs included elsewhere in `examples/`.

Run this Bash block from the repository root:

```bash
set -euo pipefail
out_dir="${TMPDIR:-/tmp}/alignair-batch-example"
mkdir -p "$out_dir"
tail -n +2 examples/batch/manifest.tsv |
while IFS=$'\t' read -r sample input metadata; do
    [[ -n "$sample" ]] || continue
    if [[ -n "$metadata" ]]; then
        alignair predict --model alignair-igh-human --input "examples/batch/$input" \
          --metadata "examples/batch/$metadata" --out "$out_dir/${sample}.tsv" --quiet
    else
        alignair predict --model alignair-igh-human --input "examples/batch/$input" \
          --out "$out_dir/${sample}.tsv" --quiet
    fi
done
```

This skips the header, forwards metadata when supplied, and writes one AIRR table and
run-provenance sidecar per sample under `$out_dir`. Each sample loads the model separately.
For parallel execution, see the [workflow wrappers](../../workflows/) and
[integrations guide](https://mutejester.github.io/AlignAIR/#/docs/integrations).
