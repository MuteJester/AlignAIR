# Hardened Gym — Phase 1 (Instrument) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the measurement instrument for the hardened gym — a difficulty parameter box, a frozen seeded evaluation lattice, an external competence metric with bootstrap confidence intervals, and a lattice evaluator — then validate that gym competence correlates with the canonical IgBLAST benchmark.

**Architecture:** New package `src/alignair/gym/instrument/`. Pure, GPU-free units (`bootstrap_ci`, `TaskSpace`, `CompetenceMetric`, lattice cell specs + fingerprinting) are unit-tested; the model-dependent `LatticeEvaluator` and the benchmark-correlation validation are integration-tested small/CPU. Reuses `gym/gym.py::build_experiment` for generation and the existing `CompetenceMetric` composite aligns to the canonical `benchmark.cli compare` metrics.

**Tech Stack:** Python 3.12, PyTorch, pytest 7.4.4, GenAIRR, existing `alignair.gym` + `alignair.benchmark`.

## Global Constraints

- venv `./.venv/bin/python`; run tests `./.venv/bin/python -m pytest <path> -v` from repo root (conftest puts `src/` on path). `alignair` not pip-installed — scripts use `PYTHONPATH=src`.
- Difficulty axes max = real deployment 99th percentile (SHM→0.30, end-loss→120, etc.) — the terminal/eval distribution must cover the hard tail, NOT cap at the scalar curriculum's 0.15.
- `CompetenceMetric` is EXTERNAL and fixed (not the Kendall training loss) so it is comparable across architectures.
- Hard constraints inherited: dynamic genotype, segmentation-first. Phase 1 changes only measurement, not the model.
- Commit after each task. No `Co-Authored-By`/Claude mentions in commit messages.

## File Structure

- Create `src/alignair/gym/instrument/__init__.py`
- Create `src/alignair/gym/instrument/stats.py` — `bootstrap_ci`.
- Create `src/alignair/gym/instrument/task_space.py` — `Axis`, `TaskSpace`.
- Create `src/alignair/gym/instrument/competence.py` — `CompetenceMetric`.
- Create `src/alignair/gym/instrument/lattice.py` — `LatticeCell`, `FrozenLattice`.
- Create `src/alignair/gym/instrument/evaluator.py` — `LatticeEvaluator`.
- Create `scripts/validate_competence_vs_igblast.py` — validation script.
- Tests under `tests/alignair/gym/instrument/`.

---

### Task 1: bootstrap_ci

**Files:**
- Create: `src/alignair/gym/instrument/__init__.py`, `src/alignair/gym/instrument/stats.py`
- Test: `tests/alignair/gym/instrument/test_stats.py`

**Interfaces:**
- Produces: `bootstrap_ci(values: Sequence[float], n_boot=1000, alpha=0.05, seed=0) -> tuple[float,float,float]` returning `(mean, lo, hi)` percentile-bootstrap CI. Deterministic given `seed`. Empty input → `(0.0, 0.0, 0.0)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_stats.py
from alignair.gym.instrument.stats import bootstrap_ci


def test_ci_brackets_mean_and_is_deterministic():
    vals = [1.0] * 50 + [0.0] * 50      # mean 0.5
    m, lo, hi = bootstrap_ci(vals, n_boot=500, seed=7)
    assert abs(m - 0.5) < 1e-9
    assert lo < m < hi
    assert (lo, hi) == bootstrap_ci(vals, n_boot=500, seed=7)[1:]   # deterministic


def test_constant_values_give_zero_width():
    m, lo, hi = bootstrap_ci([0.8] * 20, n_boot=200, seed=1)
    assert m == 0.8 and lo == 0.8 and hi == 0.8


def test_empty_is_zero():
    assert bootstrap_ci([], n_boot=10, seed=1) == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/instrument/__init__.py
"""Hardened-gym measurement instrument: task space, frozen lattice, competence."""
```

```python
# src/alignair/gym/instrument/stats.py
"""Percentile-bootstrap confidence intervals for competence aggregation."""
import random
from typing import Sequence


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (mean, lo, hi)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_stats.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/__init__.py src/alignair/gym/instrument/stats.py tests/alignair/gym/instrument/test_stats.py
git commit -m "feat(gym): add percentile bootstrap_ci for competence aggregation"
```

---

### Task 2: TaskSpace (difficulty parameter box)

**Files:**
- Create: `src/alignair/gym/instrument/task_space.py`
- Test: `tests/alignair/gym/instrument/test_task_space.py`

**Interfaces:**
- Produces:
  - `Axis(name: str, lo: float, hi: float, kind: str)` — `kind ∈ {"rate","count","prob","len"}`.
  - `TaskSpace(axes: tuple[Axis,...])` with `.deployment()` classmethod (the canonical IGH axis box with deployment-99th-pct maxes), `.sample(rng, frac: dict|None) -> dict` (θ as `{axis: value}`; `frac` optionally fixes an axis to a fraction of its range), `.to_genairr_params(theta: dict) -> dict` (maps θ to the dict `build_experiment` consumes: `mutation_rate`, `end_loss_5`, `end_loss_3`, `indel_count`, `seq_error_rate`, `ambiguous_count`, `crop_prob`, `crop_len_min`, `crop_len_max`, `orient_prob`).
- Note: SHM max is **0.30** (deployment hard tail), end-loss max **120**, indel max **5**, N max **10**, orient_prob max **0.5** — these EXCEED the scalar `Curriculum`'s caps by design.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_task_space.py
import random
from alignair.gym.instrument.task_space import Axis, TaskSpace


def test_deployment_box_covers_hard_tail():
    ts = TaskSpace.deployment()
    by = {a.name: a for a in ts.axes}
    assert by["mutation_rate"].hi >= 0.25      # hard tail, NOT capped at 0.15
    assert by["end_loss_5"].hi >= 100
    assert "crop_len" in by and "orient_prob" in by


def test_sample_in_range_and_seeded():
    ts = TaskSpace.deployment()
    a = ts.sample(random.Random(0))
    b = ts.sample(random.Random(0))
    assert a == b                               # deterministic per seed
    for ax in ts.axes:
        assert ax.lo <= a[ax.name] <= ax.hi


def test_to_genairr_params_has_required_keys():
    ts = TaskSpace.deployment()
    theta = ts.sample(random.Random(1))
    p = ts.to_genairr_params(theta)
    for k in ("mutation_rate", "end_loss_5", "end_loss_3", "indel_count",
              "seq_error_rate", "ambiguous_count", "crop_prob", "crop_len_min",
              "crop_len_max", "orient_prob"):
        assert k in p
    assert isinstance(p["end_loss_5"], tuple) and len(p["end_loss_5"]) == 2


def test_frac_fixes_axis_to_fraction_of_range():
    ts = TaskSpace.deployment()
    theta = ts.sample(random.Random(2), frac={"mutation_rate": 1.0})
    by = {a.name: a for a in ts.axes}
    assert abs(theta["mutation_rate"] - by["mutation_rate"].hi) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_task_space.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/instrument/task_space.py
"""TaskSpace: the difficulty parameter box Θ (one axis per GenAIRR knob), with
deployment-99th-percentile maxes so the eval/terminal distribution covers the hard
tail. Maps a sampled θ to the params dict gym.build_experiment consumes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    name: str
    lo: float
    hi: float
    kind: str          # "rate" | "count" | "prob" | "len"


# crop_len is expressed as the MIN length of the (junction-centered) window; harder =
# shorter, so its "value" is the shortest allowed window. We invert at param time.
_DEPLOY_AXES = (
    Axis("mutation_rate", 0.005, 0.30, "rate"),    # hard tail well past 0.15
    Axis("end_loss_5", 0.0, 120.0, "count"),
    Axis("end_loss_3", 0.0, 45.0, "count"),
    Axis("indel_count", 0.0, 5.0, "count"),
    Axis("seq_error_rate", 0.0, 0.02, "rate"),
    Axis("ambiguous_count", 0.0, 10.0, "count"),
    Axis("crop_len", 50.0, 576.0, "len"),          # shortest junction window allowed
    Axis("orient_prob", 0.0, 0.5, "prob"),
)


class TaskSpace:
    def __init__(self, axes):
        self.axes = tuple(axes)

    @classmethod
    def deployment(cls):
        return cls(_DEPLOY_AXES)

    def sample(self, rng, frac: dict | None = None) -> dict:
        out = {}
        for ax in self.axes:
            if frac is not None and ax.name in frac:
                f = max(0.0, min(1.0, frac[ax.name]))
                out[ax.name] = ax.lo + (ax.hi - ax.lo) * f
            else:
                out[ax.name] = rng.uniform(ax.lo, ax.hi)
        return out

    def to_genairr_params(self, theta: dict) -> dict:
        def _ct(v):     # count axis -> (0, n) GenAIRR length-range tuple
            return (0, int(round(v)))
        crop_min = int(round(theta["crop_len"]))
        # below the full read length => some reads cropped to a window >= crop_min
        cropped = crop_min < 576
        return {
            "mutation_rate": float(theta["mutation_rate"]),
            "end_loss_5": _ct(theta["end_loss_5"]),
            "end_loss_3": _ct(theta["end_loss_3"]),
            "indel_count": _ct(theta["indel_count"]),
            "seq_error_rate": float(theta["seq_error_rate"]),
            "ambiguous_count": _ct(theta["ambiguous_count"]),
            "crop_prob": 0.6 if cropped else 0.0,
            "crop_len_min": crop_min,
            "crop_len_max": 576 if not cropped else max(crop_min + 1, crop_min),
            "orient_prob": float(theta["orient_prob"]),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_task_space.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/task_space.py tests/alignair/gym/instrument/test_task_space.py
git commit -m "feat(gym): add TaskSpace difficulty box (deployment-99th-pct maxes)"
```

---

### Task 3: CompetenceMetric

**Files:**
- Create: `src/alignair/gym/instrument/competence.py`
- Test: `tests/alignair/gym/instrument/test_competence.py`

**Interfaces:**
- Consumes: `bootstrap_ci` (Task 1).
- Produces:
  - `CompetenceMetric(weights: dict | None = None, coord_tol: float = 2.0)` — pre-registered external score. Default weights `{"v_call":0.2,"d_call":0.1,"j_call":0.15,"coords":0.25,"region":0.15,"junction":0.15}` (sum 1).
  - `.score(rec: dict) -> float` — per-read S∈[0,1] from a normalized eval record with keys: `v_call_correct,d_call_correct,j_call_correct ∈ {0,1}`, `coord_errs: list[float]` (per-boundary abs nt error), `region_acc ∈ [0,1]`, `junction_exact ∈ {0,1}`. Missing keys contribute 0. Coordinate sub-score = fraction of boundaries within `coord_tol` nt.
  - `.aggregate(recs: Sequence[dict], seed=0) -> dict` — `{"S": mean, "lo":, "hi":, "n":}` via `bootstrap_ci` over per-read scores.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_competence.py
from alignair.gym.instrument.competence import CompetenceMetric


def _perfect():
    return {"v_call_correct": 1, "d_call_correct": 1, "j_call_correct": 1,
            "coord_errs": [0.0, 0.0, 1.0], "region_acc": 1.0, "junction_exact": 1}


def test_perfect_read_scores_one():
    assert abs(CompetenceMetric().score(_perfect()) - 1.0) < 1e-9


def test_zero_read_scores_zero():
    rec = {"v_call_correct": 0, "d_call_correct": 0, "j_call_correct": 0,
           "coord_errs": [50.0, 50.0], "region_acc": 0.0, "junction_exact": 0}
    assert CompetenceMetric(coord_tol=2.0).score(rec) == 0.0


def test_coord_tolerance_counts_within_band():
    m = CompetenceMetric(weights={"coords": 1.0}, coord_tol=2.0)
    # 2 of 4 boundaries within 2nt -> coord sub-score 0.5
    rec = {"coord_errs": [0.0, 1.0, 5.0, 9.0]}
    assert abs(m.score(rec) - 0.5) < 1e-9


def test_aggregate_returns_ci():
    m = CompetenceMetric()
    out = m.aggregate([_perfect(), _perfect()], seed=1)
    assert out["n"] == 2 and out["S"] == 1.0 and out["lo"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_competence.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/instrument/competence.py
"""CompetenceMetric: the single pre-registered EXTERNAL deployment-alignment score
(NOT the Kendall training loss), so it is comparable across architectures. A weighted
composite of allele calls, coordinate accuracy (within a fixed nt tolerance), region
accuracy, and junction exact-match. Aggregated with bootstrap CIs."""
from typing import Sequence

from .stats import bootstrap_ci

_DEFAULT_WEIGHTS = {"v_call": 0.2, "d_call": 0.1, "j_call": 0.15,
                    "coords": 0.25, "region": 0.15, "junction": 0.15}


class CompetenceMetric:
    def __init__(self, weights: dict | None = None, coord_tol: float = 2.0):
        self.weights = dict(weights) if weights is not None else dict(_DEFAULT_WEIGHTS)
        self.coord_tol = coord_tol

    def _coord_subscore(self, errs) -> float:
        errs = list(errs or [])
        if not errs:
            return 0.0
        within = sum(1 for e in errs if abs(e) <= self.coord_tol)
        return within / len(errs)

    def score(self, rec: dict) -> float:
        parts = {
            "v_call": float(rec.get("v_call_correct", 0)),
            "d_call": float(rec.get("d_call_correct", 0)),
            "j_call": float(rec.get("j_call_correct", 0)),
            "coords": self._coord_subscore(rec.get("coord_errs")),
            "region": float(rec.get("region_acc", 0.0)),
            "junction": float(rec.get("junction_exact", 0)),
        }
        wsum = sum(self.weights.get(k, 0.0) for k in parts)
        if wsum == 0:
            return 0.0
        return sum(self.weights.get(k, 0.0) * v for k, v in parts.items()) / wsum

    def aggregate(self, recs: Sequence[dict], seed: int = 0) -> dict:
        scores = [self.score(r) for r in recs]
        mean, lo, hi = bootstrap_ci(scores, seed=seed)
        return {"S": mean, "lo": lo, "hi": hi, "n": len(scores)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_competence.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/competence.py tests/alignair/gym/instrument/test_competence.py
git commit -m "feat(gym): add external CompetenceMetric composite (+ bootstrap aggregate)"
```

---

### Task 4: FrozenLattice (cells + fingerprint + serialize)

**Files:**
- Create: `src/alignair/gym/instrument/lattice.py`
- Test: `tests/alignair/gym/instrument/test_lattice.py`

**Interfaces:**
- Consumes: `TaskSpace` (Task 2).
- Produces:
  - `LatticeCell(name: str, frac: dict, n: int)` — a named eval cell fixing some axes to fractions of their range (e.g. heavy-SHM-full-length), `n` reads.
  - `FrozenLattice(task_space, cells: tuple[LatticeCell,...], seed: int)` with `.standard()` classmethod (the canonical stratified cells incl. `heavy_shm`, `heavy_shm_fulllen`, `junction_boundary`, `clean`, `fragment`), `.fingerprint() -> str` (stable hash of cells+seed+axis box — identifies the instrument version), `.cell_params(cell) -> dict` (GenAIRR params for that cell via TaskSpace).
- Note: generation of the actual reads (calls GenAIRR) lives in `LatticeEvaluator` (Task 5) so this stays GPU/sim-free and unit-testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_lattice.py
from alignair.gym.instrument.task_space import TaskSpace
from alignair.gym.instrument.lattice import LatticeCell, FrozenLattice


def test_standard_lattice_has_hard_cells():
    lat = FrozenLattice.standard(seed=0)
    names = {c.name for c in lat.cells}
    assert {"clean", "heavy_shm", "heavy_shm_fulllen", "junction_boundary"} <= names


def test_fingerprint_is_stable_and_sensitive():
    a = FrozenLattice.standard(seed=0).fingerprint()
    assert a == FrozenLattice.standard(seed=0).fingerprint()      # stable
    assert a != FrozenLattice.standard(seed=1).fingerprint()      # seed changes it


def test_heavy_shm_fulllen_cell_is_high_shm_and_uncropped():
    lat = FrozenLattice.standard(seed=0)
    cell = next(c for c in lat.cells if c.name == "heavy_shm_fulllen")
    p = lat.cell_params(cell)
    assert p["mutation_rate"] >= 0.25          # hard tail
    assert p["crop_prob"] == 0.0               # full length (the excluded corner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_lattice.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/instrument/lattice.py
"""FrozenLattice: the seeded, stratified, never-trained evaluation cells over the
TaskSpace. Includes explicit deployment-hard cells (heavy-SHM, full-length heavy-SHM,
junction-boundary) — the corners the coupled scalar ramp structurally never visits.
Fingerprinted so the instrument version is identifiable and comparable across runs."""
import hashlib
import json
from dataclasses import dataclass

from .task_space import TaskSpace


@dataclass(frozen=True)
class LatticeCell:
    name: str
    frac: dict          # axis -> fraction of range to FIX (others default mid)
    n: int


# fractions are of each axis's [lo,hi] range. full-length => crop_len frac 1.0 (=576).
_STANDARD = (
    LatticeCell("clean", {"mutation_rate": 0.0, "end_loss_5": 0.0, "end_loss_3": 0.0,
                          "indel_count": 0.0, "crop_len": 1.0, "orient_prob": 0.0}, 2000),
    LatticeCell("heavy_shm", {"mutation_rate": 0.85, "crop_len": 0.4}, 2000),
    LatticeCell("heavy_shm_fulllen", {"mutation_rate": 0.85, "crop_len": 1.0,
                                      "end_loss_5": 0.0, "end_loss_3": 0.0}, 2000),
    LatticeCell("junction_boundary", {"mutation_rate": 0.4, "indel_count": 0.5,
                                      "crop_len": 0.2}, 2000),
    LatticeCell("fragment", {"mutation_rate": 0.3, "crop_len": 0.0}, 2000),
)


class FrozenLattice:
    def __init__(self, task_space: TaskSpace, cells, seed: int):
        self.task_space = task_space
        self.cells = tuple(cells)
        self.seed = seed

    @classmethod
    def standard(cls, seed: int = 0):
        return cls(TaskSpace.deployment(), _STANDARD, seed)

    def cell_params(self, cell: LatticeCell) -> dict:
        import random
        theta = self.task_space.sample(random.Random(self.seed), frac=cell.frac)
        return self.task_space.to_genairr_params(theta)

    def fingerprint(self) -> str:
        payload = {
            "seed": self.seed,
            "axes": [(a.name, a.lo, a.hi, a.kind) for a in self.task_space.axes],
            "cells": [(c.name, sorted(c.frac.items()), c.n) for c in self.cells],
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_lattice.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/lattice.py tests/alignair/gym/instrument/test_lattice.py
git commit -m "feat(gym): add FrozenLattice (stratified hard cells + fingerprint)"
```

---

### Task 5: LatticeEvaluator (model-side, integration)

**Files:**
- Create: `src/alignair/gym/instrument/evaluator.py`
- Test: `tests/alignair/gym/instrument/test_evaluator.py`

**Interfaces:**
- Consumes: `FrozenLattice`, `CompetenceMetric`, `gym.build_experiment`, an inference path that yields per-read predictions vs truth.
- Produces: `LatticeEvaluator(model, reference_set, lattice, metric, device=None)` with `.eval_cell(cell, n=None) -> dict` (`{"S","lo","hi","n"}` from `metric.aggregate` over the cell's reads) and `.eval_all(n_per_cell=None) -> dict[str, dict]` (cell name → competence dict). Builds a per-read normalized eval record (the keys `CompetenceMetric.score` expects) from model predictions and GenAIRR ground truth.
- Note: keep the per-read record builder small and reuse existing prediction helpers where possible; the integration test asserts SHAPE/RANGE (S∈[0,1], CI ordered, n>0), not accuracy, to stay fast on a tiny untrained model.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_evaluator.py
import pytest
import torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.evaluator import LatticeEvaluator


def test_eval_cell_returns_bounded_competence_with_ci():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    lat = FrozenLattice.standard(seed=0)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(), device="cpu")
    cell = next(c for c in lat.cells if c.name == "clean")
    out = ev.eval_cell(cell, n=16)
    assert out["n"] > 0
    assert 0.0 <= out["lo"] <= out["S"] <= out["hi"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError` (evaluator missing)

- [ ] **Step 3: Implement `LatticeEvaluator`**

Implement against the model's existing eval/prediction surface (mirror `GymTrainer.evaluate_records` for the forward pass + ground-truth tags, and `compute_germline_logits`/`decode_germline_coords` for coordinate errors). Build one normalized record per read with keys `v_call_correct,d_call_correct,j_call_correct,coord_errs,region_acc,junction_exact`, then `return self.metric.aggregate(records)`. Generate the cell's reads via `build_experiment(dataconfig, lattice.cell_params(cell))` and `stream_records(n, seed=lattice.seed)`. Keep `j`/`d` correctness via the multi-hot truth (`allele[pred]==1`), `coord_errs` from predicted-vs-true germline+in-read boundaries, `region_acc` from per-position argmax, `junction_exact` left 0 until junction emission is wired (documented TODO — it becomes the junction gate later).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_evaluator.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/evaluator.py tests/alignair/gym/instrument/test_evaluator.py
git commit -m "feat(gym): add LatticeEvaluator (per-cell competence over frozen lattice)"
```

---

### Task 6: Validate competence vs IgBLAST benchmark

**Files:**
- Create: `scripts/validate_competence_vs_igblast.py`
- Test: `tests/alignair/gym/instrument/test_validation_smoke.py`

**Interfaces:**
- Produces: a script that loads `scaled_long`, evaluates the frozen lattice, and prints per-cell competence `S` alongside the canonical benchmark; a smoke test asserting the script's core function runs on a tiny model and returns a per-cell competence dict whose keys match the lattice cells.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_validation_smoke.py
import pytest
import torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator


def test_competence_field_covers_all_cells():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    lat = FrozenLattice.standard(seed=0)
    field = LatticeEvaluator(model, rs, lat, CompetenceMetric(), device="cpu").eval_all(n_per_cell=8)
    assert {c.name for c in lat.cells} <= set(field)
    assert all(0.0 <= v["S"] <= 1.0 for v in field.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_validation_smoke.py -v`
Expected: FAIL (`eval_all` missing if not yet added in Task 5 — add it now)

- [ ] **Step 3: Implement `eval_all` + the validation script**

Add `LatticeEvaluator.eval_all(n_per_cell=None)` looping `eval_cell` over `lattice.cells`. Write `scripts/validate_competence_vs_igblast.py` that: loads `.private/models/scaled_long.pt`, builds the IGH deployment lattice, computes the competence field, and prints a table of cell → `S ± CI`; documents that the operator compares this against the canonical `scripts/run_h2h_benchmark.py` output to confirm the proxy tracks the real benchmark (the Phase-1 validation gate).

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/test_validation_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full instrument suite + commit**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/instrument/ -v`
Expected: PASS (all Phase-1 tests)

```bash
git add scripts/validate_competence_vs_igblast.py src/alignair/gym/instrument/evaluator.py tests/alignair/gym/instrument/test_validation_smoke.py
git commit -m "feat(gym): competence-vs-IgBLAST validation (eval_all + script)"
```

---

## Self-Review

**Spec coverage (Phase 1 only):** `TaskSpace` → Task 2; `FrozenLattice` → Task 4; `CompetenceMetric` → Task 3; `LatticeEvaluator` (TF + e2e per-cell competence) → Tasks 5–6; bootstrap CIs → Task 1; deployment-99th-pct maxes → Task 2; hard cells incl. full-length-heavy-SHM → Task 4; benchmark-correlation validation gate → Task 6. ✓ (Later phases — sampler, Kendall fix, ALP-GMM, ceiling/anti-forgetting — are separate plans.)

**Placeholder scan:** Tasks 5–6 describe the model-side record builder in prose rather than full code because it must mirror existing prediction helpers (`evaluate_records`, `compute_germline_logits`, `decode_germline_coords`) whose exact call shapes the implementer will read in-repo; the interfaces, inputs, outputs, generation path, and the normalized record keys are fully specified. `junction_exact` is explicitly stubbed to 0 with a documented reason (junction emission wired in a later phase). No silent gaps.

**Type consistency:** `bootstrap_ci -> (mean,lo,hi)` (T1) consumed by `CompetenceMetric.aggregate` (T3). `TaskSpace.to_genairr_params` keys (T2) match `build_experiment`'s consumed params. `FrozenLattice.cell_params` (T4) feeds `LatticeEvaluator` generation (T5). `CompetenceMetric.score` record keys (T3) are exactly what `LatticeEvaluator` builds (T5). ✓

## Notes for the implementer

- The competence weights and `coord_tol` are pre-registered defaults; once Task 6 validates against IgBLAST, freeze them for the whole project (changing them invalidates cross-time/architecture comparison).
- Phase 2 (factored sampler) consumes `TaskSpace` directly; Phase 4 (ALP-GMM) consumes the `LatticeEvaluator` competence field. Keep both stable.
