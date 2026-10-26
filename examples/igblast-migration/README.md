# IgBLAST migration example

The [migration guide](https://mutejester.github.io/AlignAIR/#/docs/migrating-from-igblast)
covers reference selection, field mappings, coordinate conventions, and output filtering.

To produce an AlignAIR table from the included reads, run from the repository root:

```bash
alignair predict --input examples/reads.fasta --out alignair.tsv --model alignair-igh-human@1.0.0
```

For a small tool-agreement example that requires no model download, see [compare/](../compare/).
