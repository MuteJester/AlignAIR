# Multiple samples (cohorts)

AlignAIR aligns one input at a time; run a cohort by looping `alignair predict` over your samples with
your workflow engine (Nextflow / Snakemake / Galaxy — see [`../../workflows/`](../../workflows/)) or a
plain shell loop. Each run writes an AIRR rearrangement TSV plus a `<out>.run.json` provenance sidecar.

```bash
# one AIRR TSV per sample from a simple sample list (sample_id<TAB>reads_path)
while IFS=$'\t' read -r sample reads; do
    alignair predict --model my_model/bundle/model.alignair --input "$reads" --out "results/${sample}.tsv" --quiet
done < samples.tsv
```

Per-sample options (`--genotype`, `--metadata` / `--keep-columns`, `--device`, `--rejects-out`) are
passed on each `predict` call.

> A native `alignair batch` verb (one model load across samples, manifest validation, consolidated
> provenance) is on the roadmap but not yet implemented — use a workflow engine or the loop above.
