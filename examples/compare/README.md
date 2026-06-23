# Compare AlignAIR against another tool (no ground truth)

`alignair compare` reports agreement between two AIRR rearrangement TSVs on the **same
reads**, joined by `sequence_id` — useful for sanity-checking against IgBLAST/MiXCR without
ground truth. [`alignair.tsv`](alignair.tsv) and [`igblast.tsv`](igblast.tsv) are tiny
hand-made examples (4 shared reads).

```bash
alignair compare --a examples/compare/alignair.tsv --b examples/compare/igblast.tsv \
    --a-name AlignAIR --b-name IgBLAST
```

The report gives per-gene allele/gene agreement and, crucially, **set-rescue**: when the
top-1 alleles differ, how often the other tool's call falls inside AlignAIR's calibrated
equivalence set (`v_call_set`) — i.e. shared ambiguity rather than a real conflict. In this
example r2's V alleles differ (`*01` vs `*04`) but both are in AlignAIR's set, so it is
rescued; r3's D is a genuine disagreement.

To produce an AIRR TSV from another tool, use its AIRR exporter (e.g. IgBLAST
`-outfmt 19`, `MiXCR exportAirr`), then point `--b` at it.
