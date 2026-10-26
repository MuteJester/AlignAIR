# Compare tool agreement

[alignair.tsv](alignair.tsv) and [igblast.tsv](igblast.tsv) are hand-written AIRR tables
with four shared sequence IDs. They demonstrate agreement and candidate-set rescue.

Run from the repository root:

```bash
alignair compare --a examples/compare/alignair.tsv --b examples/compare/igblast.tsv \
  --a-name AlignAIR --b-name IgBLAST
```

Agreement is not an accuracy measurement without ground truth.
See [evaluation workflows](https://mutejester.github.io/AlignAIR/#/docs/benchmarks)
for report interpretation and comparisons on real data.
