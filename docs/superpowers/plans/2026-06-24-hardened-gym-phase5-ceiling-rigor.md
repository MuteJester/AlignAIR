# Hardened Gym — Phase 5 (Ceiling / Anti-Forgetting Rigor) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Replace the scaffold's point-estimate promotion + patience-counter plateau with statistically rigorous mechanics — promote on the bootstrap-CI **lower bound** (conservative), declare a ceiling only via a **Mann–Kendall** trend test + an irreducible-floor check (true capacity vs sampler stall), and guard against **forgetting** won cells (feeding the Phase-4 regret targeting).

**Architecture:** Pure statistical units in a new `src/alignair/gym/control/rigor.py` (`mann_kendall_trend`, `HardenedCeiling`, `RegressionGuard`), plus an LCB-aware option on `axis_competence_from_field` so the factored curriculum advances on the conservative lower bound, not a lucky point estimate. The training-side promotion gates (σ² settled, trunk-gradient conflict, LR-probe) are **Phase 5b** — they need a training run to validate, like Phases 3b/4.

**Tech Stack:** Python 3.12, pytest. Files: `src/alignair/gym/control/rigor.py`, `src/alignair/gym/factored.py`, `src/alignair/training/gym_trainer.py`.

## Global Constraints

- venv `./.venv/bin/python`; tests from repo root. Pure-Python stats (no scipy dependency).
- Competence field entries are `{"S","lo","hi","n"}` from `CompetenceMetric.aggregate` (Phase 1). LCB = `lo`.
- Backward compatible: new options default off (`use_lcb=False`, `floor=None`).
- Commit per task; no Claude mentions in commit messages.

---

### Task 1: Mann–Kendall trend test

**Files:** `src/alignair/gym/control/rigor.py`; test `tests/alignair/gym/control/test_rigor.py`

**Interfaces:** `mann_kendall_trend(series, eps=0.05) -> str` ∈ {"up","flat","down"} — sign of Kendall's tau = `S / (n(n-1)/2)` where `S = Σ_{i<j} sign(x_j−x_i)`; `|tau|<eps → "flat"`. `<2` points → "flat".

- [ ] **Step 1: failing test**

```python
# tests/alignair/gym/control/test_rigor.py
from alignair.gym.control.rigor import mann_kendall_trend


def test_trend_up_down_flat():
    assert mann_kendall_trend([0.1, 0.2, 0.35, 0.5, 0.7]) == "up"
    assert mann_kendall_trend([0.7, 0.5, 0.35, 0.2, 0.1]) == "down"
    assert mann_kendall_trend([0.50, 0.51, 0.49, 0.50, 0.505]) == "flat"


def test_short_series_is_flat():
    assert mann_kendall_trend([0.5]) == "flat"
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement**

```python
# src/alignair/gym/control/rigor.py
"""Phase-5 statistical rigor for promotion / ceiling / anti-forgetting."""


def mann_kendall_trend(series, eps: float = 0.05) -> str:
    xs = list(series)
    n = len(xs)
    if n < 2:
        return "flat"
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = xs[j] - xs[i]
            s += (d > 0) - (d < 0)
    tau = s / (n * (n - 1) / 2)
    if tau > eps:
        return "up"
    if tau < -eps:
        return "down"
    return "flat"
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat(gym): add Mann-Kendall trend test (pure)`

---

### Task 2: HardenedCeiling (trend + irreducible floor)

**Files:** `src/alignair/gym/control/rigor.py`; test `tests/alignair/gym/control/test_rigor.py`

**Interfaces:** `HardenedCeiling(window=6, eps=0.05, floor=None)`; `.update(composite: float) -> str` ∈ {"improving","ceiling","stall"}. Improving while the trend over the last `window` is "up" or history `< window`. When flat/down: "ceiling" if `floor is None or composite >= floor` (true capacity), else "stall" (plateaued BELOW the achievable floor → sampler stalled, re-explore — do NOT stop).

- [ ] **Step 1: failing test (append)**

```python
from alignair.gym.control.rigor import HardenedCeiling


def test_ceiling_improving_then_ceiling():
    c = HardenedCeiling(window=4, eps=0.05)
    for v in [0.4, 0.5, 0.6, 0.7]:
        assert c.update(v) == "improving"
    for v in [0.70, 0.701, 0.699, 0.70]:           # flat
        last = c.update(v)
    assert last == "ceiling"


def test_stall_below_floor_is_not_ceiling():
    c = HardenedCeiling(window=4, eps=0.05, floor=0.9)
    for v in [0.5, 0.5, 0.5, 0.5, 0.5]:            # flat, but below floor 0.9
        last = c.update(v)
    assert last == "stall"                         # sampler stalled, not a capacity ceiling
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement (append)**

```python
class HardenedCeiling:
    def __init__(self, window: int = 6, eps: float = 0.05, floor=None):
        self.window = window
        self.eps = eps
        self.floor = floor
        self._hist = []

    def update(self, composite: float) -> str:
        self._hist.append(float(composite))
        if len(self._hist) < self.window:
            return "improving"
        if mann_kendall_trend(self._hist[-self.window:], self.eps) == "up":
            return "improving"
        if self.floor is not None and composite < self.floor:
            return "stall"
        return "ceiling"
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat(gym): add HardenedCeiling (Mann-Kendall + irreducible-floor: ceiling vs stall)`

---

### Task 3: RegressionGuard (anti-forgetting)

**Files:** `src/alignair/gym/control/rigor.py`; test `tests/alignair/gym/control/test_rigor.py`

**Interfaces:** `RegressionGuard(margin=0.03)`; `.check(field: dict) -> list[str]` returns the cells whose LCB dropped more than `margin` below their best-seen LCB (a forgetting alarm). Tracks best LCB per cell internally.

- [ ] **Step 1: failing test (append)**

```python
from alignair.gym.control.rigor import RegressionGuard


def test_regression_guard_flags_drops():
    g = RegressionGuard(margin=0.03)
    assert g.check({"a": {"lo": 0.80}, "b": {"lo": 0.50}}) == []   # first sight, no baseline
    assert g.check({"a": {"lo": 0.81}, "b": {"lo": 0.52}}) == []   # both improved
    drop = g.check({"a": {"lo": 0.70}, "b": {"lo": 0.52}})         # a fell 0.81->0.70 (>0.03)
    assert drop == ["a"]
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement (append)**

```python
class RegressionGuard:
    def __init__(self, margin: float = 0.03):
        self.margin = margin
        self._best = {}

    def check(self, field: dict) -> list:
        regressed = []
        for cell, v in field.items():
            lcb = float(v["lo"] if isinstance(v, dict) else v)
            best = self._best.get(cell)
            if best is not None and lcb < best - self.margin:
                regressed.append(cell)
            self._best[cell] = lcb if best is None else max(best, lcb)
        return regressed
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat(gym): add RegressionGuard (anti-forgetting: best-LCB drop alarm)`

---

### Task 4: LCB-aware promotion in axis_competence_from_field

**Files:** `src/alignair/gym/factored.py`; `src/alignair/training/gym_trainer.py`; test `tests/alignair/gym/test_factored_pacing.py`

**Interfaces:** `axis_competence_from_field(field, fallback_cell="clean", use_lcb=False)` — when `use_lcb=True`, read each cell's `lo` (bootstrap CI lower bound) instead of `S`, so a per-axis pace advances only when the *conservative* competence clears the bar (no lucky-exam promotion). `GymTrainer(..., promote_on_lcb=False)`; `advance_curriculum` passes `use_lcb=self.promote_on_lcb`.

- [ ] **Step 1: failing test (append to test_factored_pacing.py)**

```python
def test_axis_competence_uses_lcb_when_requested():
    field = {"clean": {"S": 0.9, "lo": 0.85}, "heavy_shm_fulllen": {"S": 0.8, "lo": 0.72}}
    ac_s = axis_competence_from_field(field)                       # point S
    ac_l = axis_competence_from_field(field, use_lcb=True)         # conservative LCB
    assert ac_s["mutation_count"] == 0.8 and ac_l["mutation_count"] == 0.72
    assert ac_l["indel_count"] == 0.85                            # clean fallback uses its lo
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** — in `axis_competence_from_field` add `use_lcb=False`; in `_S`, read `"lo"` when `use_lcb and "lo" in field[cell]` else `"S"`:

```python
def axis_competence_from_field(field: dict, fallback_cell: str = "clean",
                               use_lcb: bool = False) -> dict:
    key = "lo" if use_lcb else "S"

    def _S(cell):
        if cell in field:
            return float(field[cell].get(key, field[cell].get("S", 0.0)))
        if fallback_cell in field:
            return float(field[fallback_cell].get(key, field[fallback_cell].get("S", 0.0)))
        return 0.0
    fb = _S(fallback_cell)
    out = {}
    for axis in (*_MIX_AXES, *_SCALAR_AXES):
        out[axis] = _S(_AXIS_CELL[axis]) if axis in _AXIS_CELL else fb
    return out
```
In `GymTrainer.__init__` add `promote_on_lcb=False` (store). In `advance_curriculum`, change the `axis_competence_from_field(field)` call to `axis_competence_from_field(field, use_lcb=self.promote_on_lcb)`.

- [ ] **Step 4:** run the pacing test + `test_targeted_training.py` → PASS. **Step 5:** commit `feat(gym): LCB-aware promotion (advance on conservative competence bound)`

---

### Task 5: Phase-5b note + full-suite green

**Files:** `docs/superpowers/specs/2026-06-23-hardened-genairr-gym-design.md`; all gym tests

- [ ] **Step 1:** append a "Phase 5 status" note to section D of the spec: σ²/CI promotion mechanics SHIPPED (`rigor.py`: Mann–Kendall ceiling, RegressionGuard, LCB promotion); DEFERRED to Phase 5b — the training-side promotion gates (per-task loss-slope settled via Mann–Kendall, σ² settled, trunk-gradient-conflict check, LR-probe restart) and SGDR, because they need a training run to validate (soft-DP-bound) and belong with the Phase-3b trunk-balancing work.
- [ ] **Step 2:** run full gym + control + nn + losses suite: `./.venv/bin/python -m pytest tests/alignair/gym tests/alignair/nn/test_weighting.py tests/alignair/losses -q` → all PASS.
- [ ] **Step 3:** commit `docs(gym): Phase 5 ceiling/anti-forgetting rigor shipped; training-side gates -> 5b`

---

## Self-Review

Spec coverage (Phase 5 / section D): bootstrap-CI lower-bound promotion → T4 (use_lcb); Mann–Kendall ceiling → T1+T2; irreducible-floor (ceiling vs stall) → T2; no-regression / anti-forgetting → T3 (RegressionGuard; its output feeds the Phase-4 regret tracker — the drop also raises regret automatically since regret = target−S). Training-side gates (σ² settled / gradient conflict / LR-probe) + BH-FDR explicitly deferred to 5b with rationale → T5. Types: `field` entries `{"S","lo",...}` consumed consistently by RegressionGuard (lo) and axis_competence (lo/S). No placeholders.
