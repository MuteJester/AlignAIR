# Donor genotype subsets

These YAML files restrict prediction to alleles already present in the model's reference.
They contain allele names and germline sequences for human IGH.

| File | V alleles | D alleles | J alleles |
| --- | --- | --- | --- |
| [donor.yaml](donor.yaml) | 17 | 15 | 7 |
| [small_donor.yaml](small_donor.yaml) | 8 | 6 | 7 |

Run from the repository root:

```bash
alignair predict --input examples/reads.fasta --out out_donor.tsv \
  --model alignair-igh-human --genotype examples/donor_genotype/donor.yaml
```

Use `small_donor.yaml` in the same command to try the smaller subset. Both files require
a model with matching allele identities. A subset cannot add novel alleles to a model.
See the [model contract](https://mutejester.github.io/AlignAIR/#/docs/model-contract)
and [custom-reference training](https://mutejester.github.io/AlignAIR/#/docs/training).
