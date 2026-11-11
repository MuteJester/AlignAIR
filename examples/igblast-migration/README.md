# IgBLAST migration example

The [migration guide](https://mutejester.github.io/AlignAIR/#/docs/migrating-from-igblast)
covers reference selection, field mappings, coordinate conventions, and output filtering.

To produce an AlignAIR table from the included reads, run from the repository root:

```bash
alignair predict --input examples/reads.fasta --out alignair.tsv --model <v4-igh-model>
```

`<v4-igh-model>` is a placeholder for an AlignAIR 4 human IGH model (a `.alignair` path, a Hugging Face repo
`hf://org/repo`, or a catalog id); the id is not published yet. <!-- PENDING-V4-MODEL-ID -->

For a small tool-agreement example that requires no model download, see [compare/](../compare/).
