# Workflow wrappers (DRAFTS — internal, not yet stable)

Thin wrappers that run AlignAIR inside common bioinformatics workflow managers. They are
**drafts kept here to surface the integration rough edges** (manifests, path resolution,
exit codes, logs, containers, multi-sample output) before AlignAIR is published. They are
**not** advertised entry points and will change.

Prerequisites that are not met yet (the project is pre-release):

- a **published model** (`alignair model list` currently shows none) — until then, point
  `--model` at a locally trained bundle (`alignair train ... -o my_model`), and
- a **published container** — the wrappers reference `thomask90/alignair:latest`; build it
  locally (`docker build -t alignair:local .`) and substitute that tag, or run the wrappers
  against a local `alignair` already on `PATH` (Nextflow `-profile local`).

## One samplesheet, several runners

[`samplesheet.csv`](samplesheet.csv) has the same columns as an `alignair batch` manifest
(`sample_id, input` and optional `genotype, metadata`), so the simplest "wrapper" is the
built-in batch mode itself — no workflow manager required:

```bash
alignair batch --manifest workflows/samplesheet.csv -o results/ --model my_model/bundle
```

### Nextflow ([`nextflow/`](nextflow))

```bash
nextflow run workflows/nextflow/main.nf -profile docker \
    --model my_model/bundle --samplesheet workflows/samplesheet.csv
# or, with a local install on PATH:
nextflow run workflows/nextflow/main.nf -profile local --model my_model/bundle
```

One process per sample (parallel across cores/nodes). Note: the model reloads per process;
for a single-node cohort, `alignair batch` (one model load) is faster.

### Snakemake ([`snakemake/`](snakemake))

```bash
cd workflows/snakemake
# edit config.yaml: set `model`
snakemake -n                       # dry run first
snakemake -j4 --use-singularity    # or drop --use-singularity to use a local alignair
```

### Galaxy ([`galaxy/alignair_predict.xml`](galaxy/alignair_predict.xml))

A single-sample `predict` tool. `detect_errors="exit_code"` maps AlignAIR's non-zero exits
to Galaxy job failures. Load it into a local Galaxy `tool_conf.xml` to try it; Tool Shed
submission waits on a published container/conda package.

## Known rough edges (the point of these drafts)

- **Per-sample genotype/metadata** — the Nextflow draft applies a single `--genotype` to all
  samples; per-row staging of `genotype`/`metadata` files needs proper Nextflow `path()`
  inputs. `alignair batch` already supports per-row `genotype`/`metadata`.
- **Path resolution** — `alignair batch` resolves relative manifest paths against the
  manifest's directory; Nextflow/Snakemake resolve against the launch dir. Prefer absolute
  paths in the samplesheet when mixing runners.
- **Containers/model** — pinned to an unpublished image/model; substitute your own until release.
- **Resources** — `cpus`/`memory` hints are placeholders; tune for your reads and reference size.
