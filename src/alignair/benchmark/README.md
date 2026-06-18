# `alignair.benchmark`

GenAIRR-backed benchmark utilities for evaluating AIRR alignment tools.

The benchmark is part of the `alignair` package, not a standalone package. It
provides:

- named benchmark specs and broad default human IGH strata;
- deterministic GenAIRR generation into JSONL benchmark cases;
- coverage-driven generation that keeps sampling until requested assay quotas
  are met or reported as unmet;
- canonical and presented-frame ground truth for orientation-aware evaluation;
- top-1, set-valued allele, coordinate, junction/NP/P-region, CIGAR/trim,
  region/state, metadata, scalar, and orientation metrics;
- adapters for common prediction formats such as IgBLAST AIRR rows.

Build a small benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_benchmark.jsonl \
  --recipe broad \
  --n-per-stratum 200
```

Build a focused stress benchmark:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli build \
  --out experiments/human_igh_focused.jsonl \
  --recipe focused \
  --n-per-focus 200
```

Build with coverage planning:

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
  --max-candidates 50000
```

Inspect coverage:

```bash
PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli summary \
  experiments/human_igh_benchmark.jsonl
```

Use from Python:

```python
from alignair.benchmark import (
    default_igh_assay_spec, default_igh_spec, generate_benchmark, save_jsonl,
    score_cases, compact_summary,
)

cases = generate_benchmark(default_igh_spec(n_per_stratum=200))
assay_cases = generate_benchmark(default_igh_assay_spec(n_per_stratum=200, n_per_focus=200))
save_jsonl(cases, "experiments/human_igh_benchmark.jsonl")
preds = my_aligner([case.sequence for case in cases])
scores = score_cases(cases, preds, frame="canonical")
print(compact_summary(scores))
```

Coverage-planned generation from Python:

```python
from alignair.benchmark import (
    coverage_plan_from_spec,
    default_igh_spec,
    generate_coverage_benchmark,
)

spec = default_igh_spec(n_per_stratum=200)
plan = coverage_plan_from_spec(
    spec,
    min_per_orientation=25,
    min_per_context=25,
    required_labels={"orientation:reverse_complement": 25},
)
result = generate_coverage_benchmark(spec, plan=plan)
print(result.report["satisfied"], result.report["unmet"])
```

Run online without saving generated cases:

```python
from alignair.benchmark import default_igh_spec, run_online_benchmark

spec = default_igh_spec(n_per_stratum=200)
report = run_online_benchmark(spec, predictor=my_aligner, batch_size=64)
```

Run online with coverage planning:

```python
from alignair.benchmark import coverage_plan_from_spec, default_igh_spec, run_online_benchmark

spec = default_igh_spec(n_per_stratum=200)
plan = coverage_plan_from_spec(spec, min_per_orientation=25, min_per_context=25)
report = run_online_benchmark(spec, predictor=my_aligner, coverage_plan=plan)
```

The online report is JSON-serializable and includes:

- `criteria`: the assay criteria being evaluated;
- `coverage`: stratum/context/allele/ambiguity coverage;
- `results.overall`: aggregate benchmark metrics;
- `results.by_context`: slices by stratum, corruption type, length bin,
  mutation bin, indel bin, ambiguity, orientation, productivity, locus/chain,
  D orientation, segment visibility, read layout, junction biology, and
  revision/contaminant flags when present.
- `generation_coverage`: when coverage planning is enabled, the accepted cases,
  generated candidates, target counts, and unmet quotas.
