# AlignAIR examples

Tiny, self-contained data for a first run.

| File | What it is |
| --- | --- |
| `reads.fasta` | 8 simulated human IGH reads (FASTA header notes the true V allele) |
| `donor_genotype.yaml` | a small "donor" genotype (a subset of the human IGH reference) |
| `10x/` | a 10x Genomics (Cell Ranger) BCR workflow: contig FASTA + `filtered_contig_annotations.csv`, with per-cell metadata carried into the AIRR output — see [`10x/README.md`](10x/README.md) |
| `airr/` | a bulk AIRR/TSV input with a separate sample-metadata table joined onto the output — see [`airr/README.md`](airr/README.md) |
| `batch/` | running a cohort by looping `alignair predict` (per-sample AIRR TSV) — see [`batch/README.md`](batch/README.md) |
| `novel_allele/` | a donor genotype (a subset of the model's reference) — see [`novel_allele/README.md`](novel_allele/README.md) |
| `custom_reference/` | germline V/D/J FASTAs to **train on your own reference / species** (`alignair train --v-fasta ...`) — see [`custom_reference/README.md`](custom_reference/README.md) |
| `compare/` | two tiny AIRR TSVs for `alignair compare` (AlignAIR vs IgBLAST, with set-rescue) — see [`compare/README.md`](compare/README.md) |

## Gallery by scenario

| Scenario | Where |
| --- | --- |
| First run, offline | `alignair demo` |
| Single-sample IGH predict | [`reads.fasta`](reads.fasta) |
| Donor genotype (subset) | [`donor_genotype.yaml`](donor_genotype.yaml) |
| Novel allele | [`novel_allele/`](novel_allele/) |
| Light chain (IGK/IGL) / TCR | train or predict with the matching reference, e.g. `--dataconfig HUMAN_IGK_OGRDB` / `HUMAN_TRB_OGRDB` (`alignair reference list`) — same commands, different reference |
| Custom reference / species | [`custom_reference/`](custom_reference/) |
| Single-cell 10x BCR/TCR | [`10x/`](10x/) |
| Bulk + sample metadata | [`airr/`](airr/) |
| Cohort (many samples) | [`batch/`](batch/) |
| Tool comparison | [`compare/`](compare/) |

## Run

The fastest way to see everything work (no model needed — it trains a tiny one offline):

```bash
alignair doctor      # environment check
alignair demo        # tiny train -> predict -> validate AIRR -> donor genotype
```

To run on these example reads with your own model (pretrained bundles aren't published yet — train
one with `alignair train`):

```bash
alignair train --dataconfig HUMAN_IGH_OGRDB --out my_model --preset desktop

# align against the model's default (full) reference
alignair predict --input examples/reads.fasta --out out.tsv --model my_model/bundle/model.alignair

# constrain to a donor genotype (a subset of the model's reference)
alignair predict --input examples/reads.fasta --out out_donor.tsv \
  --model my_model/bundle/model.alignair --genotype examples/donor_genotype.yaml
```

The output is an AIRR rearrangement TSV (V/D/J calls + coordinates + junction + a per-gene
equivalence-set column `*_call_set`). A genotype file may be YAML or FASTA, and may contain a subset
of the trained reference's alleles — calls are then restricted to those alleles.

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
  predict --input /data/reads.fasta --out /out/result.tsv --model /models/<model.alignair>
```

Add `--genotype /data/donor_genotype.yaml` to constrain to a donor subset. For GPU, add
`--gpus all` and `--device cuda` (requires a CUDA-enabled image/host).
