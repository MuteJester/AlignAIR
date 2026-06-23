# AlignAIR examples

Tiny, self-contained data for a first run.

| File | What it is |
| --- | --- |
| `reads.fasta` | 8 simulated human IGH reads (FASTA header notes the true V allele) |
| `donor_genotype.yaml` | a small "donor" genotype (a subset of the human IGH reference) — used to demonstrate the dynamic-genotype feature |
| `10x/` | a 10x Genomics (Cell Ranger) BCR workflow: contig FASTA + `filtered_contig_annotations.csv`, with per-cell metadata carried into the AIRR output — see [`10x/README.md`](10x/README.md) |
| `airr/` | a bulk AIRR/TSV input with a separate sample-metadata table joined onto the output — see [`airr/README.md`](airr/README.md) |
| `batch/` | a multi-sample manifest for `alignair batch` (cohorts): one model load, one AIRR TSV per sample + a run summary — see [`batch/README.md`](batch/README.md) |
| `novel_allele/` | a donor genotype with an allele **subset + a novel allele** — the dynamic-genotype feature — see [`novel_allele/README.md`](novel_allele/README.md) |
| `custom_reference/` | germline V/D/J FASTAs to **train on your own reference / species** (`alignair train --v-fasta ...`) — see [`custom_reference/README.md`](custom_reference/README.md) |
| `compare/` | two tiny AIRR TSVs for `alignair compare` (AlignAIR vs IgBLAST, with set-rescue) — see [`compare/README.md`](compare/README.md) |

## Gallery by scenario

| Scenario | Where |
| --- | --- |
| First run, offline | `alignair demo` |
| Single-sample IGH predict | [`reads.fasta`](reads.fasta) |
| Donor genotype (subset) | [`donor_genotype.yaml`](donor_genotype.yaml) |
| Novel allele | [`novel_allele/`](novel_allele/) |
| Light chain (IGK/IGL) / TCR | train or predict with the matching reference, e.g. `--reference HUMAN_IGK_OGRDB` / `HUMAN_TRB_IMGT` (`alignair reference list`) — same commands, different reference |
| Custom reference / species | [`custom_reference/`](custom_reference/) |
| Single-cell 10x BCR/TCR | [`10x/`](10x/) |
| Bulk + sample metadata | [`airr/`](airr/) |
| Cohort (many samples) | [`batch/`](batch/) |
| Tool comparison | [`compare/`](compare/) |

## Run

The fastest way to see everything work (no model needed — it trains a tiny one offline):

```bash
alignair doctor      # environment check
alignair demo        # tiny train -> predict -> validate AIRR -> dynamic genotype
```

To run on these example reads with your own model (pretrained bundles aren't published yet — train
one with `alignair train`):

```bash
alignair train --reference HUMAN_IGH_OGRDB -o my_model --preset desktop

# align against the model's default (full) reference
alignair predict examples/reads.fasta -o out.tsv --model my_model/bundle

# align against a donor genotype (fewer alleles and/or novel alleles) — the dynamic-genotype feature
alignair predict examples/reads.fasta -o out_donor.tsv \
  --model my_model/bundle --genotype examples/donor_genotype.yaml
```

The output is an AIRR rearrangement TSV (V/D/J calls + coordinates + junction + calibrated
equivalence-set columns). A genotype file may be YAML or FASTA, and may contain fewer alleles
than the trained reference and/or novel alleles — the model conditions on exactly what you provide.

## Run with Docker

Mount your reads, your model, and a writable output directory. Add `--user $(id -u):$(id -g)`
so output files are owned by you (the image runs as a non-root user):

```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "$PWD/examples:/data:ro" \
  -v "/path/to/models:/models:ro" \
  -v "$PWD/out:/out" \
  thomask90/alignair:latest \
  predict /data/reads.fasta -o /out/result.tsv --model /models/<bundle_or_checkpoint>
```

Add `--genotype /data/donor_genotype.yaml` for the dynamic-genotype path. For GPU, add
`--gpus all` and `--device cuda` (requires a CUDA-enabled image/host).
