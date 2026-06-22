# AlignAIR examples

Tiny, self-contained data for a first run.

| File | What it is |
| --- | --- |
| `reads.fasta` | 8 simulated human IGH reads (FASTA header notes the true V allele) |
| `donor_genotype.yaml` | a small "donor" genotype (a subset of the human IGH reference) — used to demonstrate the dynamic-genotype feature |

## Run

You need a model bundle. Get one from the model zoo (`alignair model download human-igh-ogrdb-v1`,
coming soon), train your own (`alignair train`, see the docs), or point `--model` at a checkpoint.

```bash
# environment check
alignair doctor

# align against the model's default (full) reference
alignair predict examples/reads.fasta -o out.tsv --model <bundle_or_checkpoint>

# align against a donor genotype (fewer alleles and/or novel alleles) — the dynamic-genotype feature
alignair predict examples/reads.fasta -o out_donor.tsv \
  --model <bundle_or_checkpoint> --genotype examples/donor_genotype.yaml
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
