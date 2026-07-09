# `alignair.benchmark`

GenAIRR-backed benchmark utilities for evaluating AIRR/IG alignment tools.

This is a submodule of the `alignair` package. It is not a standalone package.
It generates simulated IG rearrangement cases with GenAIRR truth, runs or scores
aligner predictions, and returns JSON reports that expose aggregate accuracy,
scenario-specific failures, allele-level errors, coverage, readiness, audit,
runtime/memory statistics, single-model uncertainty intervals, and paired
model-vs-model comparisons.

## What It Is For

Use this benchmark when you want to answer questions such as:

- Does an aligner recover V/D/J allele calls, including ambiguous multi-call
  truth sets?
- Does it put V/D/J starts and ends in the right query and germline positions?
- Does it preserve junction/CDR3, productivity, mutation, indel, orientation,
  CIGAR/trim, and metadata annotations?
- Which alleles, genes, gene families, strata, lengths, mutation burdens, or
  hard biological cases fail?
- How much wall time, per-sequence latency, throughput, memory, candidate
  enumeration, and reranking cost does the aligner use?
- Is a generated benchmark broad enough to trust for serious model evaluation?

The benchmark can diagnose one model deeply and compare two models on the same
frozen case set with paired deltas, win/loss/tie tables, bootstrap intervals,
and metric-level verdicts. Treat the verdicts as benchmark evidence, not as a
universal leaderboard claim: readiness, audit, assay grades, allele diagnostics,
context slices, and uncertainty still determine how strong the claim is.

## End-To-End CLI Workflow

### 1. Generate Benchmark Cases

For normal development:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_benchmark.jsonl \
  --recipe broad \
  --n-per-stratum 200
```

For a broader assay-style benchmark with focused stress cases:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_assay.jsonl \
  --recipe assay \
  --n-per-stratum 200 \
  --n-per-focus 200 \
  --workers 4
```

For focused hard cases only:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_focused.jsonl \
  --recipe focused \
  --n-per-focus 200
```

The case JSONL contains the input sequence plus full GenAIRR-derived truth. Keep
it as the frozen benchmark artifact for repeated model comparisons.

`--workers` enables deterministic multiprocessing during generation. For
fixed-size builds, strata are generated in worker processes and collected in
stratum order so case IDs and output order remain stable. For coverage-planned
builds, workers generate deterministic candidate chunks, but the main process
still owns coverage acceptance. The command prints `generation_profile` with
wall time, worker count, generated candidates, accepted cases, acceptance rate,
and cases per second.

### 2. Check Benchmark Coverage And Readiness

Print a coverage summary:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli summary \
  experiments/human_igh_assay.jsonl
```

Run a readiness preflight:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli readiness \
  --cases experiments/human_igh_assay.jsonl \
  --config HUMAN_IGH_OGRDB \
  --profile assay \
  --out experiments/human_igh_readiness.json
```

Readiness does not score a model. It checks whether the generated GenAIRR cases
cover enough cases, strata, contexts, orientations, and reference alleles for
the selected profile.

Profiles:

| Profile | Use |
| --- | --- |
| `smoke` | Fast sanity check for tests and local iteration. |
| `development` | Small but useful benchmark for model development. |
| `assay` | Stricter profile intended before serious model comparison. |
| `allele_complete` | Serious allele-level benchmark: every DataConfig reference allele must appear repeatedly. |
| `allele_stratified` | Stronger benchmark: every reference allele must also appear across key strata and stress contexts. |

### 3. Export Inputs For External Aligners

If the aligner consumes FASTA or AIRR-like input tables, export those from the
case JSONL:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli export \
  --cases experiments/human_igh_assay.jsonl \
  --out-dir experiments/human_igh_export \
  --prefix human_igh_assay \
  --config HUMAN_IGH_OGRDB \
  --airr-metadata
```

This writes:

| File | Purpose |
| --- | --- |
| `human_igh_assay.fasta` | FASTA input. The record id is the benchmark `case_id`. |
| `human_igh_assay_airr_input.tsv` | AIRR-style input table with `sequence_id` and `sequence`. |
| `human_igh_assay_manifest.json` | Benchmark spec, reference summary, hashes, coverage, readiness, and file paths. |

You can also build and export in one command:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_assay.jsonl \
  --recipe assay \
  --n-per-stratum 200 \
  --n-per-focus 200 \
  --export-dir experiments/human_igh_export \
  --export-prefix human_igh_assay \
  --airr-metadata
```

### 4. Run The Aligner

Run your model or external tool on the exported FASTA/AIRR input. The output must
either:

- already be normalized benchmark JSONL, or
- be an AIRR/IgBLAST-style TSV/CSV that can be normalized by the benchmark.

The safest matching key is `sequence_id`. For exported benchmark inputs,
`sequence_id` is the benchmark `case_id`.

### 5. Score Predictions

Score normalized prediction JSONL:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli evaluate \
  --cases experiments/human_igh_assay.jsonl \
  --predictions experiments/my_aligner_predictions.jsonl \
  --prediction-format jsonl \
  --contract-level core \
  --out experiments/my_aligner_report.json
```

Score an AIRR/IgBLAST TSV directly:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli evaluate \
  --cases experiments/human_igh_assay.jsonl \
  --predictions experiments/igblast_airr.tsv \
  --prediction-format airr-tsv \
  --contract-level core \
  --out experiments/igblast_report.json
```

Or normalize the table first:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli normalize-predictions \
  --input experiments/igblast_airr.tsv \
  --format airr-tsv \
  --out experiments/igblast_predictions.jsonl
```

Then score the normalized JSONL:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli evaluate \
  --cases experiments/human_igh_assay.jsonl \
  --predictions experiments/igblast_predictions.jsonl \
  --prediction-format jsonl \
  --contract-level core \
  --out experiments/igblast_report.json
```

Add bootstrap confidence intervals for key single-model metrics:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli evaluate \
  --cases experiments/human_igh_assay.jsonl \
  --predictions experiments/my_aligner_predictions.jsonl \
  --contract-level core \
  --bootstrap 500 \
  --confidence 0.95 \
  --bootstrap-seed 123 \
  --no-bootstrap-strata \
  --out experiments/my_aligner_report.json
```

For large allele-complete cohorts, prefer `--no-bootstrap-strata` unless you
specifically need confidence intervals for every stratum. The report still
contains point metrics for every stratum/context; the flag only skips the
expensive per-stratum resampling loop.

By default, offline evaluation aligns predictions to cases by `sequence_id`.
Use `--match-by order` only when predictions are guaranteed to be in the exact
same order as the case JSONL.

### 6. Validate Report Contracts

Before using a saved report as CI evidence or as input to downstream tooling,
validate its artifact metadata, metric registry, criteria mappings, and observed
metric keys:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli validate-report \
  --report experiments/my_aligner_report.json \
  --require-current-version \
  --require-metric-registry
```

The command prints validation JSON and exits nonzero for hard contract failures.
Add `--fail-on-warning` when CI should also reject legacy reports or observed
metrics that are outside the embedded registry.

For other standalone benchmark JSON artifacts, such as readiness or model
comparison reports, validate the registered artifact contract:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli validate-artifact \
  --path experiments/baseline_vs_new_model.json \
  --kind model_comparison_report \
  --require-current-version
```

### 7. Add Runtime And Memory Statistics

Runtime and memory are part of the benchmark report. For Python predictors run
through `run_benchmark_report` or `run_online_benchmark`, wall time, throughput,
per-sequence latency, process RSS high-water memory, RSS delta, and Python
allocation peak are measured automatically.

For external tools scored with `evaluate`, either add performance fields to each
prediction row or pass a report-level sidecar JSON:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli evaluate \
  --cases experiments/human_igh_assay.jsonl \
  --predictions experiments/igblast_airr.tsv \
  --prediction-format airr-tsv \
  --contract-level core \
  --performance-json experiments/igblast_performance.json \
  --out experiments/igblast_report.json
```

Example sidecar:

```json
{
  "wall_time_seconds": 38.4,
  "n_sequences": 10000,
  "peak_memory_mb": 1536.0,
  "candidate_count": 250,
  "rerank_count": 25
}
```

Accepted per-prediction fields include `runtime_seconds`, `runtime_ms`,
`seconds_per_read`, `milliseconds_per_read`, `reads_per_second`,
`peak_memory_mb`, `peak_memory_bytes`, `peak_memory_delta_mb`,
`python_tracemalloc_peak_mb`, `candidate_count`, and `rerank_count`.

The report writes normalized values under `performance` and mirrors assay-level
metrics such as `seconds_per_read`, `reads_per_second`, and `peak_memory_mb`
into `results.overall.global`.

### 8. Compare Two Prediction Sets

Compare two model outputs on exactly the same cases:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli compare \
  --cases experiments/human_igh_assay.jsonl \
  --a-predictions experiments/baseline_predictions.jsonl \
  --b-predictions experiments/new_model_predictions.jsonl \
  --model-a-name baseline \
  --model-b-name new_model \
  --policy allele_calling_core \
  --metric genes.v.call_top1_in_set \
  --metric genes.v.ss_mae \
  --minimum-primary-advantage 0.001 \
  --maximum-guardrail-regression 1.0 \
  --bootstrap 500 \
  --confidence 0.95 \
  --multiple-comparison-correction bonferroni \
  --no-bootstrap-strata \
  --out experiments/baseline_vs_new_model.json
```

The comparison report is paired at the case level. For each metric it includes
model A and B paired means, `raw_delta_model_b_minus_model_a`, a
direction-adjusted `model_b_advantage`, win/loss/tie counts, optional bootstrap
intervals, and a verdict. Positive `model_b_advantage` always favors model B;
for lower-is-better metrics such as `*_mae`, it is the sign-flipped raw delta.
Use `--practical-delta` to require a minimum aggregate effect size before a
metric is called better or worse.

Use a named `--policy` for reproducible endpoint and guardrail choices:

| Policy | Use |
| --- | --- |
| `allele_calling_core` | Mixed-chain V/J allele-call endpoint with set-call, boundary, junction, and output guardrails. |
| `igh_allele_calling_core` | IGH V/D/J allele-call endpoint with D-aware guardrails. |
| `boundary_core` | V/J query-boundary endpoint with allele-call and output guardrails. |
| `airr_core` | General AIRR-style endpoint over V/J calls, junction recovery, productivity, and output completeness. |

Print the built-in policy catalog:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli comparison-policies
```

For custom decisions, pass `--primary-metric` and `--guardrail-metric` directly.
The report then includes a `decision` section. Primary metrics must clear
`--minimum-primary-advantage`; guardrails must stay within
`--maximum-guardrail-regression`. With bootstrap enabled, gates use confidence
intervals instead of only point estimates. Typical decision verdicts are
`model_b_superior`, `blocked_by_guardrail_regression`, `no_primary_improvement`,
`no_regression_pass`, `inconclusive`, and `not_scored`.

When a decision checks multiple primary and guardrail metrics, use
`--multiple-comparison-correction bonferroni` or `sidak` to control the
familywise decision interval. The ordinary metric rows still report the requested
`--confidence`; the `decision.multiple_comparison` section records the stricter
per-metric confidence used by the gate.

## Prediction Contract

Inspect the full normalized prediction contract:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli contract
```

Contract levels:

| Level | Meaning |
| --- | --- |
| `minimal` | Top V/D/J calls only. |
| `core` | Allele sets, orientation, productivity, and V/D/J query/germline coordinates. |
| `assay` | Rich AIRR-style output: CIGAR, identity, trim, junction, NP/P regions, labels, metadata, and robustness annotations. |

Normalized JSONL predictions are one dictionary per sequence. Important fields:

```json
{
  "sequence_id": "tiny_clean_000001",
  "v_call": "IGHV1-69*01",
  "v_calls": ["IGHV1-69*01"],
  "d_call": "IGHD3-10*01",
  "d_calls": ["IGHD3-10*01"],
  "j_call": "IGHJ4*02",
  "j_calls": ["IGHJ4*02"],
  "v_sequence_start": 0,
  "v_sequence_end": 296,
  "v_germline_start": 0,
  "v_germline_end": 296,
  "productive": true,
  "orientation_id": 0,
  "junction": "TGTGCGAGAGATTACTACGGT",
  "junction_aa": "CARDYYG"
}
```

Coordinate convention:

- normalized benchmark starts are 0-based;
- normalized benchmark ends are end positions in the evaluated frame;
- AIRR/IgBLAST starts are converted by the adapter from 1-based to 0-based;
- choose `--frame canonical` when predictions are reported on the canonical
  forward sequence;
- choose `--frame presented` when predictions are reported on the exact input
  sequence after benchmark orientation transforms.

For AIRR/IgBLAST tables, fields such as `v_call`, `d_call`, `j_call`,
`*_sequence_start`, `*_sequence_end`, `productive`, `junction`, and
`sequence_id` are converted automatically.

## What The Report Returns

`evaluate` writes one JSON report. The main sections are:

| Section | What it answers |
| --- | --- |
| `artifact` | Versioned benchmark report schema metadata. |
| `benchmark` | Which case file was scored and how many cases it contains. |
| `frame` | Whether scoring used canonical or presented-frame truth. |
| `criteria` | The full benchmark criteria catalog. |
| `catalog_validation` | Structural validation of embedded criteria and scenario-axis catalogs. |
| `metric_registry` | Metric directions, pass/warn thresholds, and criterion/category mappings used by the report. |
| `prediction_contract` | Accepted normalized prediction fields. |
| `scenario_axes` | Stress axes used to interpret contexts. |
| `coverage` | Case, stratum, allele, ambiguity, length, and stressor coverage. |
| `results.overall` | Aggregate scalar metrics across all cases. |
| `results.by_context` | Metrics sliced by strata and contexts. |
| `performance` | Normalized wall time, per-sequence latency, throughput, and memory statistics when measured or supplied. |
| `diagnostics.allele_calling` | Per-allele, per-gene, per-family, and confusion-pair tables. |
| `diagnostics.boundaries` | V/D/J query and germline coordinate failure decomposition. |
| `prediction_matching` | Present when predictions are matched by id; shows missing, extra, and duplicate ids. |
| `prediction_validation` | Present when `--contract-level` is set; shows missing/malformed prediction fields using per-case D/no-D requirements. |
| `uncertainty` | Present when `--bootstrap` is set; bootstrap intervals for selected metrics. |
| `criteria_audit` | Checks whether observed metrics cover the criteria catalog and GenAIRR truth fields. |
| `assay` | Pass/warn/fail grading by criterion and category. |

Minimal report shape:

```json
{
  "benchmark": {"n_cases": 1000, "strata": ["clean_full", "hard_full"]},
  "frame": "canonical",
  "coverage": {"n_cases": 1000, "by_stratum": {"clean_full": 200}},
  "performance": {"seconds_per_read": 0.004, "reads_per_second": 250.0, "peak_memory_mb": 1024.0},
  "results": {
    "overall": {
      "global": {"productive_acc": 0.99, "orientation_acc": 1.0},
      "genes": {
        "v": {"call_top1_in_set": 0.98, "call_set_f1": 0.97},
        "d": {"call_top1_in_set": 0.92},
        "j": {"call_top1_in_set": 0.99}
      }
    },
    "by_context": {
      "stratum:hard_full": {
        "genes": {"v": {"call_top1_in_set": 0.91}}
      }
    }
  },
  "diagnostics": {"allele_calling": {"genes": {"v": {}}}},
  "assay": {"summary": {"grade": "warn", "n_failed_criteria": 0}}
}
```

## Interpreting The Main Outputs

### `results`

`results.overall` contains direct metric averages. It is useful for dashboards
and quick regression checks.

High-value metrics include:

| Metric | Meaning |
| --- | --- |
| `call_top1_in_set` | Top predicted allele is in the GenAIRR truth set. |
| `gene_top1_in_set` | Top predicted allele has the correct allele-less gene. |
| `call_set_precision`, `call_set_recall`, `call_set_f1` | Set-valued allele-call quality. |
| `call_exact_set` | Predicted allele set exactly equals the GenAIRR truth set. |
| `ss_mae`, `se_mae` | Query sequence start/end absolute error. |
| `gs_mae`, `ge_mae` | Germline start/end absolute error. |
| `seq_span_iou` | Segment span overlap in the query sequence. |
| `junction_nt_exact`, `junction_aa_exact` | Junction/CDR3 sequence recovery. |
| `productive_acc` | Productive/nonproductive status accuracy. |
| `orientation_acc` | Presented-read orientation accuracy. |
| `required_field_presence` | AIRR-required output field completeness. |

Top-k candidate metrics (`top1_recall`, `top3_recall`, `top5_recall`,
`top10_recall`, and `topk_truth_set_recall`) are emitted only when predictions
contain explicit ranked candidate outputs or scores, such as `v_ranked_calls`,
`v_topk`, `v_scores`, or the equivalent `d_*`/`j_*` fields. Final call fields
like `v_call` and `v_calls` are not reused as candidate rankings, so these
metrics measure retrieval coverage rather than ordinary final-call accuracy.

### `diagnostics.allele_calling`

This is the most important section for allele-call failures. It is grouped by
gene type: `v`, `d`, and `j`.

Each gene contains:

| Field | Meaning |
| --- | --- |
| `summary` | Overall allele/gene/family rates for that segment type. |
| `per_allele` | One row per observed GenAIRR truth allele. |
| `per_gene` | One row per allele-less gene, for example `IGHV1-69`. |
| `per_gene_family` | One row per family, for example `IGHV1`. |
| `allele_confusions` | Most common truth allele -> predicted allele errors. |
| `gene_confusions` | Most common truth gene -> predicted gene errors. |
| `family_confusions` | Most common truth family -> predicted family errors. |

Error kinds:

| Error kind | Meaning |
| --- | --- |
| `accepted_allele` | Top call is one of the GenAIRR truth alleles. |
| `same_gene_wrong_allele` | Correct allele-less gene, wrong allele. |
| `same_family_wrong_gene` | Correct family, wrong allele-less gene. |
| `wrong_family` | Wrong family. |
| `missing_prediction` | No call was returned. |

Example confusion row:

```json
{
  "truth_allele": "IGHV1-1*02",
  "truth_gene": "IGHV1-1",
  "truth_family": "IGHV1",
  "pred_call": "IGHV1-1*01",
  "pred_gene": "IGHV1-1",
  "pred_family": "IGHV1",
  "n": 14,
  "rate_among_truth_allele_cases": 0.28,
  "error_kind": "same_gene_wrong_allele",
  "example_case_ids": ["case_00017", "case_00291"]
}
```

Use these tables to find errors hidden by aggregate accuracy, especially rare
alleles, sibling alleles, family-level collapse, ambiguous truth sets, and
systematic no-call behavior.

### `diagnostics.boundaries`

This section explains how V/D/J coordinate predictions failed. It is grouped by
gene type: `v`, `d`, and `j`, with a small global section for V/D/J order and
overlap errors.

Each gene contains:

| Field | Meaning |
| --- | --- |
| `summary` | Parse rates, exact span rates, MAE values, and failure-type rates. |
| `failure_types` | Most common coordinate failure classes with example case IDs. |
| `by_context` | Boundary accuracy and failure classes by stratum, orientation, length, fragment, and segment visibility. |

Important failure classes:

| Failure class | Meaning |
| --- | --- |
| `missing_coordinates` | Query start/end coordinates were not reported. |
| `missing_germline_coordinates` | Germline start/end coordinates were not reported. |
| `start_only_error` / `end_only_error` | Only one query boundary is wrong. |
| `off_by_one` | At least one boundary is one base away. |
| `systematic_plus_one_shift` / `systematic_minus_one_shift` | Both query boundaries are shifted by exactly one base. |
| `correct_length_shifted_span` | Segment length is right but the span is shifted. |
| `wrong_length` | Predicted query segment length is wrong. |
| `canonical_presented_frame_confusion` | Prediction matches the other coordinate frame. |
| `wrong_germline_span` | Query span may be usable, but germline coordinates are wrong. |
| `correct_query_span_wrong_germline_span` | Query span is exact while germline span is wrong. |
| `correct_allele_wrong_trim` | Allele is right, query span is right, but implied trimming/germline span is wrong. |
| `fragment_limited_boundary` | Boundary failure occurs on a fragment/short visible segment. |
| `vdj_order_or_overlap_error` | Predicted V/D/J spans are biologically out of order or overlapping. |

This is the coordinate analogue of allele diagnostics: it prevents a good
average MAE from hiding systematic 1-based/0-based shifts, frame confusion,
wrong trimming, and fragment-specific boundary collapse.

### `assay`

`assay` turns raw metrics into criterion/category-level grades:

| Field | Meaning |
| --- | --- |
| `assay.summary.grade` | Overall `pass`, `warn`, `fail`, `planned`, or `not_scored`. |
| `assay.by_category` | Category-level grades, for example allele calling or segmentation. |
| `assay.criteria` | Criterion-level observed metrics, missing metrics, grade reasons, and thresholds. |
| `assay.completeness_gate` | Audit-backed check that scoreable available core criteria were actually measured. |
| `assay.critical_failures` | Failed core criteria that should block trust in aggregate accuracy. |
| `assay.weak_contexts` | Worst context/metric slices. |

Treat a `pass` as "passed the currently implemented criteria for this
benchmark, with no scoreable available core criteria left unmeasured." Criteria
that are planned, partial, or lack available truth are still reported separately;
they should guide the next benchmark expansion rather than be hidden by the
aggregate grade.

### `criteria_audit`

The audit answers whether the report actually measured the criteria it claims
to measure. It highlights:

- criteria with missing metric keys;
- observed metrics not mapped to any criterion;
- catalog metrics missing from the registry and observed metrics outside the registry;
- planned criteria that unexpectedly have observed metrics;
- GenAIRR truth fields that are unavailable in the cases.

If many important criteria are unscored, do not make strong model claims from
the assay grade alone.

### `uncertainty`

When `--bootstrap` is set, uncertainty uses paired case-level bootstrap
resampling over the same GenAIRR case/prediction pairs. These are single-model
confidence intervals. Use the `compare` command for paired model-vs-model
deltas, win/loss/tie tables, bootstrap intervals, and metric-level verdicts.
Use `--no-bootstrap-strata` for large benchmark cohorts when overall confidence
intervals are enough and stratum-level point estimates are sufficient.

### `coverage` And `readiness`

`coverage` describes what the generated benchmark contains. `readiness` decides
whether that coverage meets a named profile. A benchmark can have good model
metrics but still be too small or too narrow for serious claims.

## Coverage-Planned Generation

Coverage planning keeps sampling GenAIRR cases until requested quotas are met or
the candidate budget is exhausted:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_planned.jsonl \
  --recipe assay \
  --n-per-stratum 200 \
  --n-per-focus 200 \
  --coverage-planned \
  --min-per-orientation 25 \
  --min-per-context 25 \
  --min-per-allele 1 \
  --max-candidates 50000 \
  --workers 4
```

The command prints `generation_coverage` with accepted cases, generated
candidates, target counts, and unmet quotas, plus `generation_profile` with
runtime and throughput.

For serious allele-level model comparison, use the allele-complete profile. This
starts from the full DataConfig reference set and requires every V/D/J reference
allele to be represented in the GenAIRR truth, not just a large aggregate case
count:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_allele_complete/cases.jsonl \
  --recipe assay \
  --n-per-stratum 200 \
  --n-per-focus 200 \
  --coverage-planned \
  --readiness-profile allele_complete \
  --min-per-allele 100 \
  --min-per-orientation 200 \
  --min-per-context 200 \
  --min-per-stratum 200 \
  --max-candidates 1000000 \
  --export-dir experiments/human_igh_allele_complete \
  --export-prefix human_igh_allele_complete \
  --export-frame presented \
  --airr-metadata
```

If `--coverage-planned` is used with `--readiness-profile allele_complete` and
`--min-per-allele` is omitted, the planner uses the profile's per-reference-
allele minimum. The manifest readiness check still verifies the final cohort,
including `reference_allele_fraction` and `reference_allele_min_counts`.
Coverage-planned builds also use the selected readiness profile's minimum case
count unless `--min-cases` is provided.

For a more faithful allele-level assay, use the allele-stratified profile. This
adds axis-marginal allele/context coverage: each reference allele must appear in
important benchmark contexts such as assay strata, orientation transforms,
fragment/full-read tags, mutation/noise/indel bins, ambiguity bins, and segment
visibility bins. This is stronger than `allele_complete` because it catches
cases where an allele appears 100 times overall but only in easy identity reads.

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_allele_stratified/cases.jsonl \
  --recipe assay \
  --n-per-stratum 200 \
  --n-per-focus 200 \
  --coverage-planned \
  --readiness-profile allele_stratified \
  --min-cases 25000 \
  --min-per-allele 100 \
  --min-per-allele-context 1 \
  --min-per-orientation 200 \
  --min-per-context 200 \
  --min-per-stratum 200 \
  --max-candidates 2000000 \
  --export-dir experiments/human_igh_allele_stratified \
  --export-prefix human_igh_allele_stratified \
  --export-frame presented \
  --airr-metadata
```

`--min-per-allele-context` is deliberately separate from `--min-per-allele`.
The first controls allele/context matrix cells; the second controls total
examples per reference allele. Raising the matrix-cell count from `1` to `5` or
`10` can make a public benchmark much stronger, but it will substantially
increase generation time and case count.

## Python Usage

### Saved-Case Workflow

```python
from alignair.benchmark import (
    build_benchmark_report,
    compact_summary,
    default_igh_assay_spec,
    export_benchmark_inputs,
    generate_benchmark_with_report,
    load_airr_predictions,
    run_benchmark_report,
    save_jsonl,
    score_cases,
)

generated = generate_benchmark_with_report(
    default_igh_assay_spec(n_per_stratum=200, n_per_focus=200),
    workers=4,
)
cases = generated.cases
print(generated.report)
save_jsonl(cases, "experiments/human_igh_assay.jsonl")
export_benchmark_inputs(cases, "experiments/human_igh_export", prefix="human_igh_assay")

preds = my_aligner([case.sequence for case in cases])
scores = score_cases(cases, preds, frame="canonical")
print(compact_summary(scores))

report = build_benchmark_report(cases, preds, contract_level="core", n_bootstrap=500)
print(report["assay"]["summary"])
print(report["diagnostics"]["allele_calling"]["genes"]["v"]["summary"])
print(report["diagnostics"]["boundaries"]["genes"]["v"]["summary"])

igblast_preds = load_airr_predictions("experiments/igblast_airr.tsv")
igblast_report = build_benchmark_report(
    cases,
    igblast_preds,
    contract_level="core",
    match_by="sequence_id",
)
print(igblast_report["prediction_matching"])

profiled_report = run_benchmark_report(cases, my_aligner, contract_level="core")
print(profiled_report["performance"])
```

### Online Workflow

Use online benchmarking when you do not want to save generated cases first:

```python
from alignair.benchmark import default_igh_assay_spec, run_online_benchmark

spec = default_igh_assay_spec(n_per_stratum=200, n_per_focus=200)
report = run_online_benchmark(
    spec,
    predictor=my_aligner,
    batch_size=64,
    contract_level="core",
)
print(report["assay"]["summary"])
print(report["performance"])
```

The predictor receives a list of input sequences and must return one prediction
dictionary per sequence. Online and profiled predictor workflows measure runtime
and memory by default; pass `profile_runtime=False` to disable that overhead.

### Coverage-Planned Python Workflow

```python
from alignair.benchmark import (
    coverage_plan_from_spec,
    default_igh_assay_spec,
    generate_coverage_benchmark,
)

spec = default_igh_assay_spec(n_per_stratum=200, n_per_focus=200)
plan = coverage_plan_from_spec(
    spec,
    min_per_orientation=25,
    min_per_context=25,
    required_labels={"orientation:reverse_complement": 25},
)
result = generate_coverage_benchmark(spec, plan=plan)
print(result.report["satisfied"], result.report["unmet"])
```

## Command Reference

| Command | Purpose |
| --- | --- |
| `build` | Generate benchmark case JSONL, optionally with multiprocessing, coverage planning, and export. |
| `summary` | Print coverage summary for a case JSONL. |
| `readiness` | Assess whether generated cases meet a readiness profile. |
| `export` | Export FASTA, AIRR input TSV, and manifest for external tools. |
| `normalize-predictions` | Convert AIRR/IgBLAST TSV/CSV to normalized prediction JSONL. |
| `evaluate` | Score predictions against benchmark cases and write a full report. |
| `validate-report` | Validate report artifact metadata, metric registry, scoring manifest, criteria mappings, and observed metric keys. |
| `validate-artifact` | Validate any registered benchmark artifact JSON by kind. |
| `compare` | Compare two prediction files with paired deltas, win/loss/tie counts, bootstrap intervals, and optional endpoint/guardrail decisions. |
| `comparison-policies` | Print built-in named endpoint/guardrail policies for `compare --policy`. |
| `validate-comparison-policies` | Validate built-in comparison policy metrics against the metric registry. |
| `assay` | Build an assay view from saved score/report JSON. |
| `audit` | Audit observed metrics against the criteria catalog and optional case truth. |
| `criteria` | Print criteria and scenario-axis catalogs. |
| `validate-catalogs` | Validate criteria and scenario-axis catalog structure against registry-backed contracts. |
| `metrics` | Print the metric registry with directions, thresholds, and criterion mappings. |
| `scoring-manifest` | Print scorer-owned metric emission metadata by component. |
| `contract` | Print the normalized prediction contract. |
| `artifact-contracts` | Print registered artifact contracts and schema metadata. |

## Current Interpretation Limits

The benchmark is strong enough to identify model weaknesses and compare models
on the same generated case set. It is not yet a complete superiority adjudicator
because the package does not yet provide validation against independent
real-read cohorts.

For serious claims, use the report as an assay-style evidence package:
readiness first, then prediction validation, criteria audit, assay grades,
allele diagnostics, context slices, single-model uncertainty, and paired
comparison decisions.
