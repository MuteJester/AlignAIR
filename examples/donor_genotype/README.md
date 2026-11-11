# Donor genotype subsets

These YAML files restrict prediction to a subset of the model's reference. Their keys are
the segments (`v`, `d`, `j`); each maps human IGH allele names (OGRDB) to germline sequences.
`--genotype` reads only the names; a plain list of names works too.

| File | V alleles | D alleles | J alleles |
| --- | --- | --- | --- |
| [donor.yaml](donor.yaml) | 17 | 15 | 7 |
| [small_donor.yaml](small_donor.yaml) | 8 | 6 | 7 |

Run from the repository root:

```bash
alignair predict --input examples/reads.fasta --out out_donor.tsv \
  --model <v4-igh-model> --genotype examples/donor_genotype/donor.yaml
```

`<v4-igh-model>` is a placeholder for an AlignAIR 4 human IGH model (a `.alignair` path, a Hugging Face repo
`hf://org/repo`, or a catalog id); the id is not published yet. <!-- PENDING-V4-MODEL-ID -->

Use `small_donor.yaml` in the same command to try the smaller subset: its V alleles are all
of one family, so most of the example reads get no V call (their rows are tagged `partial`,
reason `no_v`). Both files require a model whose reference holds these allele names; an
unknown name is refused (`ALIGNAIR_GENOTYPE_UNKNOWN_ALLELE`). A genotype cannot add
alleles: bind a FASTA with `--reference` that holds them (for example the model's own
reference from `alignair reference export`, plus the novel alleles), and the genotype can then
name them. To infer the genotype from the reads instead, use `--infer-genotype`.
See the [model contract](https://mutejester.github.io/AlignAIR/#/docs/model-contract)
and [custom-reference training](https://mutejester.github.io/AlignAIR/#/docs/training).
