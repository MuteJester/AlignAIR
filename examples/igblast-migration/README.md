# IgBLAST migration example

The [migration guide](https://mutejester.github.io/AlignAIR/#/docs/migrating-from-igblast)
covers reference selection, field mappings, coordinate conventions, and output filtering.

To produce an AlignAIR table from the included reads, run from the repository root:

```bash
alignair predict --input examples/reads.fasta --out alignair.tsv --model alignair4-igh-human
```

`alignair4-igh-human` is the AlignAIR 4 human IGH model's id in the shared catalog. It is not published yet;
until it is, pass an AlignAIR 4 `.alignair` path to `--model`. <!-- PENDING-V4-PUBLISH: alignair4-igh-human -->

For a small tool-agreement example that requires no model download, see [compare/](../compare/).
