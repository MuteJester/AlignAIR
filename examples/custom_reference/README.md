# Train from a custom reference

These human IGH FASTAs provide a small reference: [v.fasta](v.fasta) has eight V alleles,
[d.fasta](d.fasta) has six D alleles, and [j.fasta](j.fasta) has seven J alleles.
The files are also used by the custom-reference tests.

Run a preflight from the repository root:

```bash
alignair train --v-fasta examples/custom_reference/v.fasta \
  --d-fasta examples/custom_reference/d.fasta --j-fasta examples/custom_reference/j.fasta \
  --chain-type BCR_HEAVY --out my_model --preset desktop --plan
```

The plan reports configuration, model size, and reference checks. Remove `--plan` to
train. The exported inference model is written to `my_model/bundle/model.alignair`.
See [custom-reference training](https://mutejester.github.io/AlignAIR/#/docs/training)
for other loci, presets, and reference requirements.
