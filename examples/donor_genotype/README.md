# Donor genotypes

These YAML files limit the calls to a donor's alleles: a subset of the IGH model's germline set. Their keys are the
segments (`v`, `d`, `j`), each mapping OGRDB allele names to germline sequences. `--genotype` reads only the names, so
a plain list of names per segment works too.

| File | V alleles | D alleles | J alleles |
| --- | --- | --- | --- |
| [donor.yaml](donor.yaml) | 17 | 15 | 7 |
| [small_donor.yaml](small_donor.yaml) | 8 | 6 | 7 |

Run from the repository root:

<!-- PENDING-V4-PUBLISH: the model id alignair4-igh-human is confirmed when the models are published. -->
```bash
alignair predict --model alignair4-igh-human --input examples/reads.fasta --out out_donor.tsv \
  --genotype examples/donor_genotype/donor.yaml
```

Every call now comes from the donor's alleles; one read (`read08`) matches none of its V alleles and gets no V call.
Run the same command with `small_donor.yaml` to see a genotype that leaves most reads without their gene: its eight V
alleles are all of one family (IGHVF1), so 15 of the 20 example reads get no V call, and their rows are tagged
`airr_assembly_status` = `partial`, `airr_assembly_reason` = `no_v`.

A genotype names alleles of the model's germline set: the IG models' OGRDB names. A name the set does not hold is
refused (`ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND`). A genotype cannot add alleles: to name a novel allele, bind a FASTA that
holds it with `--reference` (for example the model's own germline set, plus your alleles), and the genotype can then
name it. The model's own set is one command away:

```bash
alignair reference export "$(alignair models path alignair4-igh-human)" --fasta igh.fasta
```

To infer the genotype from the reads themselves, use `--infer-genotype`, or `alignair genotype` on a table you already
have.

More: [Align with a donor genotype](https://mutejester.github.io/AlignAIR/#/docs/donor-genotype) and
[Use your own germline reference](https://mutejester.github.io/AlignAIR/#/docs/own-reference).
