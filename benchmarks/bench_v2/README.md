# bench_v2 - ARCHIVAL / pre-v3.0.0 (not the canonical v3 benchmark)

> **Status: archival, incomplete provenance.** The comparison in this bundle does **not** identify
> which AlignAIR model produced the predictions (no model id / version / fingerprint / commit) or the
> IgBLAST version, so its results **cannot be attributed to the v3.0.0 product** and must not be
> published as v3 claims. It is retained for its frozen ground-truth **cases** (which are canonical
> and reusable) and as a record of the earlier run. The verified v3.0.0 head-to-head - the released
> model plus a recorded IgBLAST version, run against these same cases - is prepared under
> `comparisons/` (see [`comparisons/RERUN.md`](comparisons/RERUN.md) and "Provenance gaps" below).

The frozen ground-truth dataset and the earlier AlignAIR-vs-IgBLAST comparison, as an auditable
bundle: the comparison report, the case manifest, the generated cases, and their checksums.

## Dataset

- **2,600 cases across 13 strata** (200 each): clean / moderate / hard full-length, high SHM, high
  indel, noisy/ambiguous, trimmed, five 5'- and 3'-anchored amplicon and short-fragment shapes, and
  arbitrary orientation.
- **Simulator:** GenAIRR, dataconfig `HUMAN_IGH_OGRDB`, **seed 123** (per-stratum `seed_offset` 0).
- **Germline reference** (SHA-256, from `manifest.json`): combined
  `ea5e9b9332020c65e135e193d7ecee93bc484c7a18840811e33fbb0fb05fe627`; V (198 alleles)
  `ea6a5a32...`, D (33) `407322c5...`, J (7) `53c8788d...`.
- **Scoring:** paired-case bootstrap, 95% intervals, Bonferroni-corrected across the 24 metrics.

## Result (of the archived run - NOT a v3.0.0 claim)

Under the Bonferroni-corrected paired bootstrap, AlignAIR is statistically **better on 18 of 24
metrics, IgBLAST on 4, with 2 inconclusive** (point estimates: AlignAIR higher on 18, IgBLAST on 6).

Per-gene allele accuracy (top-1 in truth set) and key globals (IgBLAST vs AlignAIR):

| Metric | IgBLAST | AlignAIR | Verdict |
| --- | --- | --- | --- |
| V call top-1 | **0.856** | 0.828 | IgBLAST better |
| D call top-1 | 0.534 | **0.722** | AlignAIR better |
| J call top-1 | 0.669 | **0.821** | AlignAIR better |
| orientation acc | 0.931 | **0.958** | AlignAIR better |
| junction nt exact | **0.958** | 0.676 | IgBLAST better |

AlignAIR's gains concentrate in D/J and orientation; IgBLAST leads on V (from short fragments and
heavy-SHM full-length reads) and on exact junction recovery. The two inconclusive metrics are the
J-segment start/end coordinate MAEs. Full per-metric point estimates, intervals, and verdicts are in
`comparison_report.json` (`overall` and `by_stratum`).

## Files

| File | What |
| --- | --- |
| `comparison_report.json` | Full model-comparison report - all 24 metrics, per stratum and overall, with CIs and verdicts. |
| `manifest.json` | Benchmark manifest - strata, case IDs, seed, germline reference SHAs, generation and software versions. |
| `cases/igh.fasta`, `cases/igh_airr_input.tsv` | The exact 2,600 generated cases (FASTA and AIRR-input TSV). |
| `SHA256SUMS.txt` | SHA-256 of every file above. |

Case files are stored with LF line endings (the source `igh_airr_input.tsv` used CRLF; normalized for the
repo and pinned via `.gitattributes`). The records are unchanged, so re-running reproduces the same result.

## Validate

The report and manifest validate against the `alignair_benchmark` artifact contracts:

```bash
PYTHONPATH=src python -m alignair_benchmark.cli validate-artifact \
  --path benchmarks/bench_v2/comparison_report.json --kind model_comparison_report
PYTHONPATH=src python -m alignair_benchmark.cli validate-artifact \
  --path benchmarks/bench_v2/manifest.json --kind benchmark_manifest
```

CI runs both on every change to this directory (`.github/workflows/benchmark-artifact.yml`).

## Reproduce

The head-to-head is produced by the `alignair_benchmark` harness (build the cases, evaluate each
tool, compare) - **not** by `alignair benchmark`, which is a separate AlignAIR-only self-check on four
strata. The canonical flow (see `python -m alignair_benchmark.cli <command> --help` for exact flags):

```bash
# 1. build the frozen cases (dataconfig HUMAN_IGH_OGRDB, seed 123, 13 strata x 200)
PYTHONPATH=src python -m alignair_benchmark.cli build   ...  --seed 123 --out bench_v2/
# 2. AlignAIR predictions -> score
alignair predict --input bench_v2/igh_airr_input.tsv --out alignair_airr.tsv --model <MODEL>
PYTHONPATH=src python -m alignair_benchmark.cli evaluate     --predictions alignair_airr.tsv ... --out alignair_report.json
# 3. IgBLAST (-outfmt 19) -> normalize -> score
igblastn ... -outfmt 19 bench_v2/igh.fasta > igblast_airr.tsv
PYTHONPATH=src python -m alignair_benchmark.cli normalize-predictions --input igblast_airr.tsv --out igblast.jsonl
PYTHONPATH=src python -m alignair_benchmark.cli evaluate     --predictions igblast.jsonl ... --out igblast_report.json
# 4. compare -> comparison_report.json
PYTHONPATH=src python -m alignair_benchmark.cli compare --a alignair_report.json --b igblast_report.json \
  --a-name AlignAIR --b-name IgBLAST --out comparison_report.json
```

## Provenance gaps (read before citing)

This bundle captures the generated cases, the germline reference, the seed, the scoring, and the full
comparison report. The following provenance is **not recorded in the artifacts** and must be supplied
or regenerated before this is treated as a complete v3.0.0 provenance record:

- **The evaluated AlignAIR model is not identified.** No model id / version / fingerprint / training
  commit is stored with the predictions. These numbers were produced during development; the exact
  model file that generated the AlignAIR predictions is not pinned here.
- **IgBLAST version and its germline-DB build are not captured.**
- **The cases were generated with `alignair` 2.0.2** (a pre-release development version;
  `manifest.json` -> `software.alignair_version`), not 3.0.0.
- **The exact original build/evaluate/compare invocation is not recorded** (the flow above is the
  canonical reconstruction, not a verbatim command log).

To close these, either add the recorded provenance (model fingerprint/commit, IgBLAST version) here,
or re-run the head-to-head against the released `alignair-igh-human` model with IgBLAST's version
recorded, and refresh this bundle.
