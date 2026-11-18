# Compare two tools

[alignair.tsv](alignair.tsv) and [igblast.tsv](igblast.tsv) are two small hand-written AIRR tables, with IMGT
allele names, for the same four reads. They show what `alignair compare` reports: where two tools agree, and where a
disagreement is shared ambiguity rather than a conflict.

Run from the repository root (no model needed):

```bash
alignair compare --a examples/compare/alignair.tsv --b examples/compare/igblast.tsv \
  --a-name AlignAIR --b-name IgBLAST
```

It prints a Markdown report (`--out report.md` writes it to a file, `--json` gives the raw numbers):

```text
# AIRR agreement: AlignAIR vs IgBLAST

Matched **4** sequences by `sequence_id` (AlignAIR: 4, IgBLAST: 4; AlignAIR-only 0, IgBLAST-only 0).

## Call agreement (rows where both tools called the gene)

| gene | n | allele agreement | gene agreement | set-rescue of disagreements |
| --- | --- | --- | --- | --- |
| V | 4 | 75.0% | 100.0% | 100.0% |
| D | 4 | 75.0% | 75.0% | 0.0% |
| J | 4 | 100.0% | 100.0% | n/a |
```

The set rescue counts, of the reads where the two calls differ, how often IgBLAST's call is inside AlignAIR's
candidate set (`v_call_set` and the others). Here read `r2` is called `IGHV3-23*01` by AlignAIR and `IGHV3-23*04` by
IgBLAST, and AlignAIR's `v_call_set` holds both.

Agreement is not accuracy: without known truth, neither tool is the reference.
[Coming from IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/from-igblast) shows how to run both tools on the
same reads and the same germline set; the measured comparison, on simulated reads with known truth, is on
[Comparison with IgBLAST](https://mutejester.github.io/AlignAIR/#/docs/comparison).
