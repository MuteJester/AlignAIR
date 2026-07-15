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
  net win overall.
- D and J: AlignAIR dominates across strata — e.g. on ~80 bp fragments D ≈ 0.72 vs 0.34 and
  J ≈ 0.88 vs 0.47.
- Equivalence sets: when a read genuinely cannot distinguish alleles, AlignAIR reports the set of
  candidate alleles (`*_call_set`) instead of forcing a single guess — behavior IgBLAST lacks.

## Coordinates

AlignAIR wins **every** V/D/J in-read and germline start/end MAE; the largest gaps are on V germline
coordinates. Exact junction-nucleotide match is the one place IgBLAST currently leads (±1–2 nt
boundary jitter); the AIRR output is otherwise schema-valid (`validate-airr`).

## Throughput (honest)

On an RTX 3090 Ti, AlignAIR runs ~100–125 reads/s (GPU); IgBLAST runs ~240 reads/s (8-thread CPU)
on the same set. **AlignAIR's advantage is accuracy on degraded reads plus donor-genotype
constraint, at competitive throughput** — not raw speed today. A lighter `--columns` preset (which
skips the gapped-alignment assembly) recovers throughput when the full alignment fields aren't needed.

## Reproduce

Evaluate any model against freshly-generated, ground-truth reads, broken down per stratum:

```bash
alignair benchmark --model alignair-igh-human --n 200 --out benchmark.json
```

The full head-to-head-vs-IgBLAST methodology (frozen case sets, paired bootstrap CIs, Bonferroni
correction, per-allele/stratum diagnostics) lives in the `alignair_benchmark` package — see its
[README](https://github.com/MuteJester/AlignAIR/blob/main/src/alignair_benchmark/README.md).

## Prove it on your own data (no ground truth needed)

Run both tools on the same reads and compare their AIRR output directly:

```bash
alignair predict --input my_reads.fastq --out alignair.tsv --model alignair-igh-human
# ... produce igblast.tsv (e.g. IgBLAST AIRR outfmt 19) or mixcr.tsv (mixcr exportAirr) ...
alignair compare --a alignair.tsv --b igblast.tsv --a-name AlignAIR --b-name IgBLAST --out report.md
```

The report shows per-gene allele/gene agreement, junction and productivity concordance, coverage,
and **set-rescue** — how often the other tool's call falls inside AlignAIR's equivalence
set (shared ambiguity, not a real conflict) — plus example disagreements to inspect.
