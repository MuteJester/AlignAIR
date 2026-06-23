# AlignAIR vs IgBLAST — benchmarks

Head-to-head on a frozen, simulated benchmark with ground truth: **4,400 cases across 22 strata**
(clean, heavy SHM, indels, fragments down to 40 bp, arbitrary orientation, D-inversion,
contaminants, ambiguous), scored with **paired bootstrap confidence intervals** and Bonferroni
correction via `alignair`'s benchmark module. Reference: human IGH (OGRDB). Model: `scaled_long`.

## Headline

**AlignAIR beats IgBLAST on 23 of 24 scored metrics.** It wins every boundary and germline-
coordinate metric, and wins D and J allele calling by large margins — biggest on the degraded
reads where classical seed-and-extend breaks down (short fragments, reverse-complement/arbitrary
orientation). The one metric IgBLAST leads is exact junction-nucleotide recovery (a boundary-
precision item).

## Allele calling (top-1 in truth set)

| Gene | IgBLAST | AlignAIR | |
| --- | --- | --- | --- |
| V | 0.745 | **0.776** | win |
| D | 0.538 | **0.694** | win (+0.16) |
| J | 0.713 | **0.842** | win (+0.13) |

- V is the contested axis: IgBLAST leads on full-length heavy-SHM V, AlignAIR wins hugely on
  fragments and on reverse-complement/arbitrary orientation (which IgBLAST does not handle), for a
  net win overall. The fast classical V reader (`--v-reader parasail`) lifts V to 0.776.
- D and J: AlignAIR dominates across strata — e.g. on ~80 bp fragments D ≈ 0.72 vs 0.34 and
  J ≈ 0.88 vs 0.47.
- Equivalence sets: when a read genuinely cannot distinguish alleles, AlignAIR reports a calibrated
  set and degrades to gene/family level instead of guessing — uncertainty behavior IgBLAST lacks.

## Coordinates

AlignAIR wins **every** V/D/J in-read and germline start/end MAE; the largest gaps are on V germline
coordinates. Exact junction-nucleotide match is the one place IgBLAST currently leads (±1–2 nt
boundary jitter); the AIRR output is otherwise schema-valid (`validate-airr`).

## Throughput (honest)

On an RTX 3090 Ti, AlignAIR runs ~100–125 reads/s (GPU); IgBLAST runs ~240 reads/s (8-thread CPU)
on the same set. **AlignAIR's advantage is accuracy on degraded reads plus the dynamic-genotype
capability, at competitive throughput** — not raw speed today. Architectural work to remove the
soft-DP bottleneck (the dominant cost) is ongoing.

## Reproduce

```bash
# 1. build a frozen benchmark + export inputs
python -m alignair.benchmark.cli build --out cases.jsonl --recipe assay \
  --n-per-stratum 200 --n-per-focus 200 --export-dir export --export-prefix h2h --airr-metadata

# 2. run IgBLAST + AlignAIR on the same cases (needs IgBLAST installed)
python scripts/run_h2h_benchmark.py --export-dir export --prefix h2h --out preds

# 3. paired comparison with bootstrap CIs
python -m alignair.benchmark.cli compare --cases cases.jsonl \
  --a-predictions preds/igblast_airr.tsv --a-prediction-format airr-tsv \
  --b-predictions preds/dnalignair_predictions.jsonl --b-prediction-format jsonl \
  --model-a-name igblast --model-b-name alignair \
  --policy igh_allele_calling_core --bootstrap 500 --multiple-comparison-correction bonferroni
```

See [`src/alignair/benchmark/README.md`](../src/alignair/benchmark/README.md) for the full
methodology (readiness profiles, per-allele/stratum diagnostics, performance metrics).
