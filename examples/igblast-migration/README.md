# From IgBLAST

AlignAIR takes the place of `igblastn -outfmt 19`: reads in, an AIRR rearrangement table out. To make AlignAIR's
table from the example reads, run from the repository root:

<!-- PENDING-V4-PUBLISH: the model id alignair4-igh-human is confirmed when the models are published. -->
```bash
alignair predict --model alignair4-igh-human --input examples/reads.fasta --out alignair.tsv
```

`alignair.tsv` has the AIRR columns IgBLAST's table has (`v_call`, `d_call`, `j_call`, `junction`, `productive`,
`sequence_alignment`, the CIGARs and coordinates), and a few of AlignAIR's own, such as each gene's candidate set
(`v_call_set`). The IGH model names alleles as OGRDB does (`IGHVF10-G50*02`), where IgBLAST with IMGT's database names
them the IMGT way.

To compare the two tools fairly, give IgBLAST the germline set the model calls from. The first step is the model's
germlines in one FASTA:

```bash
alignair reference export "$(alignair models path alignair4-igh-human)" --fasta igh.fasta
```

Then split it per segment, build IgBLAST's databases, run both tools on the same reads, and compare:

```bash
alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md
```

[Coming from IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/from-igblast) has every step, IgBLAST's options
one by one and its columns side by side with AlignAIR's. For a small `alignair compare` example that needs no model,
see [compare/](../compare/).
