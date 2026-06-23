# Train on your own reference (custom FASTA / non-human species)

When no built-in GenAIRR DataConfig fits (a non-model organism, an in-house reference),
train AlignAIR from germline FASTA files. [`v.fasta`](v.fasta) / [`j.fasta`](j.fasta) /
[`d.fasta`](d.fasta) here are a small human-IGH slice standing in for your own reference.

```bash
# preflight (no training) — checks the reference, model size, time/memory, warnings
alignair train --v-fasta examples/custom_reference/v.fasta \
    --j-fasta examples/custom_reference/j.fasta \
    --d-fasta examples/custom_reference/d.fasta \
    --chain-type BCR_HEAVY -o my_model --preset desktop --plan

# train (drop --plan); add --allow-curatable if alleles have no detected anchor
alignair train --v-fasta examples/custom_reference/v.fasta \
    --j-fasta examples/custom_reference/j.fasta \
    --d-fasta examples/custom_reference/d.fasta \
    --chain-type BCR_HEAVY -o my_model --preset desktop --allow-curatable
```

`--chain-type` must be one of the GenAIRR chain types (`alignair reference list` shows them:
BCR_HEAVY, BCR_LIGHT_KAPPA, BCR_LIGHT_LAMBDA, TCR_ALPHA/BETA/GAMMA/DELTA). Light/TCR loci
omit `--d-fasta`. The resulting bundle embeds your reference, so `alignair predict` needs no
extra flags.
