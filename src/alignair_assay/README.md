# alignair_assay

## Purpose

`alignair_assay` scores immune-receptor aligners (AlignAIR, IgBLAST, ...) against **observable truth**: what a
perfect aligner could recover from each simulated read. It is a development package, excluded from the wheel, and
the instrument for improving AlignAIR's architecture, wiring and loss. GenAIRR is the only simulator.

The flow is: simulated cases → observable truth → each tool's AIRR rearrangement TSV → per-case scores →
coverage-aware summaries and paired, Holm-corrected comparisons.

Units everywhere: coordinates are 0-based and half-open, rates run from 0 to 1, and wall times are in seconds.

## Layers and dependency rule

Each subpackage is a layer and may import only the layers in its row.

| Layer | Holds | May import from `alignair_assay.` |
|---|---|---|
| `core` | shared primitives: `reference` (the germline reference), `cigar`, `seqops`, `prediction` (normalized prediction records), `jsonl`, `provenance`; the run-directory path schema `core/layout.py` (from P2 Task 3) | core |
| `truth` | observable truth (pure): `model`, `segment`, `boundaries`, `alleles`, `genairr` (record decoding), `build`, `bench`, `io`, `oracle` | core, truth |
| `airr` | AIRR TSV `reader` and `writer` | core, airr |
| `metrics` | per-case scorers (pure) and the metric registries | core, truth, metrics |
| `stats` | aggregation with coverage and the paired comparison (pure) | core, metrics, stats |
| `report` | JSON reports and markdown summaries | core, stats, report |
| `scoring` | `pipeline` (score one tool against a truth file) and `slices` | core, truth, airr, metrics, stats, scoring |
| `suites` | the GenAIRR suite catalog (from P2 Task 3) | core, suites |
| `runners` | one runner per tool (from P2 Task 6) | core, airr, truth, suites, runners |
| `workflow` | build → run → score → report orchestration (from P2 Task 3) | everything except cli |
| `cli` | the `python -m alignair_assay` commands | everything |
| `testing.py` | test-only helpers | everything |

Further rules:

- `alignair` is imported only by `core/reference.py`, and `alignair_benchmark` is never imported.
- `truth`, `metrics` and `stats` are pure: no file IO, no subprocess, no CLI, no network. `truth/io.py` reads and
  writes truth files through `core.jsonl`.
- GenAIRR is imported only from `core`, `truth` (record decoding), `suites` and `workflow`.

`tests/alignair_assay/test_architecture.py` enforces the table and the first two rules by reading every module's
imports, function-level ones included. A new subpackage fails that test until it has a row in the test's `ALLOWED`
map. Tests live in `tests/alignair_assay/<layer>/`, mirroring the package; `tests/alignair_assay/conftest.py`
provides the `genairr` guard, the session `reference` (`HUMAN_IGH_OGRDB`) and `load_fixture(name)` for the
committed bench cases in `tests/alignair_assay/fixtures/`.

## Truth conventions

**Observable truth.** Each simulated record is projected onto what the read shows (`truth.build`):

- A segment is observed when at least one of its germline bases is in the read. An unobserved segment has no call
  or coordinate truth; a tool that calls it scores `{g}.spurious_call`.
- The observed alignment drops edge gaps: a clipped or edge-deleted germline base cannot be told apart from extra
  trimming.
- The allele truth is a set: every reference allele whose sequence contains the observed germline, plus the
  simulator's own calls that are in the reference. Disagreements with the simulator's tie list are counted.
- Each segment edge is an inclusive interval of equally defensible positions. An edge is exact when the prediction
  lies inside its interval, and its error is the distance to the interval.
- The junction exists only when both conserved anchors (V Cys, J Trp/Phe) are aligned read bases. Productivity
  flags exist only with a junction. The junction translation and the stop-codon flag are scored only on
  frame-unambiguous reads: an in-frame junction and no V or J frameshift.
- The clone key is (V genes, J genes, junction length).
- Truth is in the canonical forward frame. bench_v2 orientation ids 2 and 3 (complement only, reverse only) are
  excluded (`non_biological_orientation`) and never scored, as are genes whose truth could not be built
  (`ObservableTruth.unscorable`).
- `truth.oracle.oracle_prediction` is the prediction a perfect aligner makes. It must score 1.0 on every metric that
  is neither lower-is-better nor informational, 0.0 on every lower-is-better metric, and no `nan` anywhere.

**`CaseScores` NaN rules.** `CaseScores` maps a metric name (`<scope>.<metric>`, e.g. `v.top1_in_set`) to a float.

- An absent key means the metric does not apply to the case.
- `nan` means the metric applies but the tool did not cover it.
- Accuracy metrics score 0.0 when the prediction is missing. The exception is germline-frame edges, which stay
  `nan` without a top-1 call in the truth set, because the frames cannot be compared.
- Summaries report coverage next to every mean.
- `metrics.lower_is_better` and `metrics.INFORMATIONAL` are the registries. A comparison runs Holm over every
  non-informational metric either tool produced.

**Minimum observed lengths.** `metrics.case.MIN_OBSERVED = {"v": 20, "d": 8, "j": 10}` observed germline bases.
Below its minimum a segment gets only `{g}_short.top1_in_set` and `{g}_short.abstain_ok`, for which abstaining is
as correct as a right call. A missing row is a failure, not an abstention.

## Commands

Run from the repository root with `PYTHONPATH=src`:

```
python -m alignair_assay truth --bench-cases CASES.jsonl[.gz] [--reference HUMAN_IGH_OGRDB] --out TRUTH.jsonl.gz
python -m alignair_assay score --truth TRUTH --predictions AIRR.tsv --tool NAME --out-dir DIR
python -m alignair_assay compare --truth TRUTH --a A.tsv --a-name A --b B.tsv --b-name B --out-dir DIR [--alpha 0.05]
```

- `truth` writes the truth file. Its header records `kind`, `version`, `reference`, `reference_digest` and the
  GenAIRR provenance: `genairr_version`, `genairr_source`, `genairr_git_commit` and `genairr_git_dirty`. The
  command then prints the excluded cases, warnings, unscorable genes and simulator disagreements.
- `score` writes `<tool>.scores.jsonl.gz`, `<tool>.report.json` and `<tool>.report.md`.
- `compare` writes `comparison.json` and `comparison.md`, and prints the headline table. Its 95% and simultaneous
  CIs of 0/1 metrics are clipped to [-1, 1].
- Predictions are matched to truth by AIRR `sequence_id` == truth `case_id`. A truth case without a prediction is
  scored as missing, never dropped.
- A missing file, an unknown reference, a truth file built against another reference, or a prediction file
  matching no case exits with status 2.

To add a command, write one module in `cli/` with `register(subparsers)` and `run(args) -> int`, and add it to
`cli.COMMANDS`.

## Adding a metric

1. Write the scorer in the `metrics/` module of its kind (`alleles`, `boundaries`, `junction`, `shm`, `flags`). It
   returns `CaseScores` named `<scope>.<metric>` and follows the NaN rules above.
2. Call it from `metrics.case.score_case`. `scoring.pipeline.score_predictions` scores every case through it, so the
   pipeline and the oracle check see the same metrics.
3. Declare its status in `metrics/__init__.py`. A name ending in one of the `_LOWER_IS_BETTER` suffixes is
   lower-is-better. A metric that is reported but not compared goes into `INFORMATIONAL`.
4. Extend the oracle tests in `tests/alignair_assay/metrics/` so the oracle scores perfectly on it, and add a test
   for a wrong prediction.
5. For a headline metric, add it to `report.markdown.HEADLINE`.

## Adding a suite (from P2 Task 3)

To be written with the suite engine.

## Adding a runner (from P2 Task 6)

To be written with the runner infrastructure.
