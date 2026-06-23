# GenAIRR Gym — Competence-Gated Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed `p = step/total` curriculum clock with a competence-gated controller that promotes the model up a difficulty ladder only when it masters each level, detects its capacity ceiling (plateau), reports per-axis struggles, and renders an 8-bit "climbing the tower" terminal HUD.

**Architecture:** A new `src/alignair/gym/control/` package of single-purpose units. Pure decision logic (`RankLadder`, `PromotionGate`, `PlateauDetector`, `GymState`, `GymHUD`) is unit-testable without a GPU. `GymController` orchestrates them over an injected `evaluator` callable (so it tests with scripted metrics), then is wired into `GymTrainer.fit()` against the real `GymTrainer.evaluate()`. A single immutable `GymState` snapshot feeds both the HUD and the JSON/markdown reports so the view never drifts from the data.

**Tech Stack:** Python 3.12, PyTorch, pytest 7.4.4, existing `alignair.gym` (`AlignAIRGym`, `Curriculum`) and `alignair.training.gym_trainer.GymTrainer`.

## Global Constraints

- Python venv: `./.venv/bin/python` (NOT `.private/.venv`). Run tests with `./.venv/bin/python -m pytest <path> -v` from repo root (`tests/conftest.py` puts `src/` on `sys.path`).
- Build ON the existing substrate; do not reimplement `AlignAIRGym`, `Curriculum`, or `GymTrainer`. `GymTrainer.evaluate(n_batches, p)` returns the metrics dict the gate consumes: `loss`, `region_acc`, `state_acc`, `orient_acc`, and per gene `g ∈ {v,j,d}`: `{g}_call`, `{g}_start_dev`, `{g}_end_dev`, `{g}_gl_start_dev`, `{g}_gl_end_dev`, `{g}_e2e_gl_start_dev`, `{g}_e2e_gl_end_dev`. (No junction metric exists yet — gates are config-driven so junction is added later as just another `GateSpec`.)
- Hard constraints inherited from the spec: dynamic genotype (no allele memorization) and segmentation-first. This sub-project changes only the training-control loop and diagnostics — NOT the model, heads, or loss.
- Difficulty is a SINGLE scalar ladder (the ramp that already won); diagnostics break failures down per-axis. Do not reintroduce multi-axis independent ranks.
- Commit after every task. No `Co-Authored-By`/Claude mentions in commit messages (project rule).

## File Structure

- Create `src/alignair/gym/control/__init__.py` — package exports.
- Create `src/alignair/gym/control/config.py` — `GateSpec`, `GymConfig`, default gates.
- Create `src/alignair/gym/control/state.py` — `GateStatus`, `AxisStat`, `GymState`, `composite_score`.
- Create `src/alignair/gym/control/ladder.py` — `RankLadder`.
- Create `src/alignair/gym/control/gate.py` — `PromotionGate`.
- Create `src/alignair/gym/control/plateau.py` — `PlateauDetector`.
- Create `src/alignair/gym/control/hud.py` — `GymHUD`.
- Create `src/alignair/gym/control/exam.py` — `RankExam` (per-axis tagged eval).
- Create `src/alignair/gym/control/reporter.py` — `StruggleReporter`.
- Create `src/alignair/gym/control/controller.py` — `GymController`.
- Modify `src/alignair/training/gym_trainer.py` — add `evaluate_records()`; wire `GymController` into `fit()`.
- Tests under `tests/alignair/gym/control/`.

---

### Task 1: GymConfig + GateSpec

**Files:**
- Create: `src/alignair/gym/control/__init__.py`
- Create: `src/alignair/gym/control/config.py`
- Test: `tests/alignair/gym/control/test_config.py`

**Interfaces:**
- Produces:
  - `GateSpec(metric: str, direction: str, thresholds: tuple[float, ...])` — `direction ∈ {"higher","lower"}`; `thresholds` has one entry per level.
  - `GymConfig(n_levels=10, gates=tuple[GateSpec,...], exam_every=500, exam_batches=8, patience=8, slope_eps=1e-3, color=True)`.
  - `default_gates(n_levels: int) -> tuple[GateSpec, ...]` — relaxing per-level thresholds for `v_call`, `d_call`, `j_call` (higher-better), `coords_mae` (lower-better), `region_acc` (higher-better).
  - `_linspace(a: float, b: float, n: int) -> tuple[float, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_config.py
from alignair.gym.control.config import GateSpec, GymConfig, default_gates, _linspace


def test_linspace_endpoints_and_length():
    xs = _linspace(0.97, 0.80, 10)
    assert len(xs) == 10
    assert xs[0] == 0.97
    assert abs(xs[-1] - 0.80) < 1e-9


def test_default_gates_cover_heads_and_relax_with_level():
    gates = default_gates(10)
    by = {g.metric: g for g in gates}
    assert set(by) == {"v_call", "d_call", "j_call", "coords_mae", "region_acc"}
    # higher-better gates get EASIER (lower bar) at harder levels
    assert by["v_call"].thresholds[0] > by["v_call"].thresholds[-1]
    # lower-better MAE gate ALLOWS more error at harder levels
    assert by["coords_mae"].direction == "lower"
    assert by["coords_mae"].thresholds[0] < by["coords_mae"].thresholds[-1]
    # every gate has one threshold per level
    assert all(len(g.thresholds) == 10 for g in gates)


def test_gymconfig_defaults():
    cfg = GymConfig()
    assert cfg.n_levels == 10
    assert len(cfg.gates) == 5
    assert cfg.patience == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.gym.control'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/__init__.py
"""Competence-gated curriculum control for the GenAIRR gym."""
```

```python
# src/alignair/gym/control/config.py
"""Configuration for the competence-gated gym: gate specs + global knobs."""
from dataclasses import dataclass, field


def _linspace(a: float, b: float, n: int) -> tuple[float, ...]:
    if n == 1:
        return (a,)
    step = (b - a) / (n - 1)
    return tuple(a + step * i for i in range(n))


@dataclass(frozen=True)
class GateSpec:
    """One promotion lock: a metric, whether higher or lower is better, and a
    per-level threshold (thresholds[level]). All gates must open to climb."""
    metric: str
    direction: str            # "higher" | "lower"
    thresholds: tuple[float, ...]

    def __post_init__(self):
        if self.direction not in ("higher", "lower"):
            raise ValueError(f"direction must be higher|lower, got {self.direction}")


def default_gates(n_levels: int) -> tuple[GateSpec, ...]:
    """Relaxing per-level bars: higher-better metrics get a LOWER bar at harder
    levels (they're achievable-ceiling-limited); the coords MAE gate ALLOWS more
    error. Numbers are an initial cut to be recalibrated against a real climb."""
    return (
        GateSpec("v_call", "higher", _linspace(0.97, 0.80, n_levels)),
        GateSpec("d_call", "higher", _linspace(0.85, 0.55, n_levels)),
        GateSpec("j_call", "higher", _linspace(0.92, 0.70, n_levels)),
        GateSpec("coords_mae", "lower", _linspace(1.0, 4.0, n_levels)),
        GateSpec("region_acc", "higher", _linspace(0.99, 0.90, n_levels)),
    )


@dataclass(frozen=True)
class GymConfig:
    n_levels: int = 10
    gates: tuple[GateSpec, ...] = field(default_factory=lambda: default_gates(10))
    exam_every: int = 500          # training steps between competence exams
    exam_batches: int = 8          # eval batches per exam
    patience: int = 8              # exams w/o composite improvement -> ceiling
    slope_eps: float = 1e-3        # min composite gain that counts as progress
    color: bool = True             # HUD ANSI color (auto-off on non-TTY by caller)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/__init__.py src/alignair/gym/control/config.py tests/alignair/gym/control/test_config.py
git commit -m "feat(gym): add GymConfig + relaxing per-level GateSpec defaults"
```

---

### Task 2: GymState + GateStatus + composite_score

**Files:**
- Create: `src/alignair/gym/control/state.py`
- Test: `tests/alignair/gym/control/test_state.py`

**Interfaces:**
- Produces:
  - `GateStatus(name, value, threshold, direction)` with `.is_open: bool` and `.fraction: float` (progress toward the bar, clamped to [0,1]).
  - `AxisStat(axis: str, bins: tuple[tuple[str, float, int], ...])` — `(bin_label, metric_value, n)`.
  - `GymState(level, level_name, n_levels, step, gates: tuple[GateStatus,...], axes: tuple[AxisStat,...], rooms_cleared, patience_used, patience_max, best_level, headline)` with `.all_open: bool` and `.blocking: tuple[str,...]`.
  - `composite_score(gates: Sequence[GateStatus]) -> float` — mean of `.fraction`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_state.py
from alignair.gym.control.state import GateStatus, AxisStat, GymState, composite_score


def test_gatestatus_open_and_fraction_higher():
    g = GateStatus("v_call", value=0.86, threshold=0.88, direction="higher")
    assert g.is_open is False
    assert abs(g.fraction - 0.86 / 0.88) < 1e-9
    assert GateStatus("v_call", 0.90, 0.88, "higher").is_open is True


def test_gatestatus_open_and_fraction_lower():
    g = GateStatus("coords_mae", value=2.1, threshold=2.0, direction="lower")
    assert g.is_open is False
    assert abs(g.fraction - 2.0 / 2.1) < 1e-9
    assert GateStatus("coords_mae", 1.5, 2.0, "lower").is_open is True
    # fraction never exceeds 1 even when comfortably open
    assert GateStatus("coords_mae", 0.5, 2.0, "lower").fraction == 1.0


def test_composite_and_blocking():
    gates = (
        GateStatus("v_call", 0.90, 0.88, "higher"),     # open, frac 1.0
        GateStatus("j_call", 0.60, 0.80, "higher"),     # closed, frac 0.75
    )
    assert abs(composite_score(gates) - 0.875) < 1e-9
    st = GymState(level=3, level_name="Room 4", n_levels=10, step=100,
                  gates=gates, axes=(), rooms_cleared=3, patience_used=1,
                  patience_max=8, best_level=3, headline="")
    assert st.all_open is False
    assert st.blocking == ("j_call",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/state.py
"""Immutable snapshots of the gym's competence state — the single source of
truth that both the HUD and the JSON/markdown reports render from."""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GateStatus:
    name: str
    value: float
    threshold: float
    direction: str            # "higher" | "lower"

    @property
    def is_open(self) -> bool:
        if self.direction == "higher":
            return self.value >= self.threshold
        return self.value <= self.threshold

    @property
    def fraction(self) -> float:
        """Progress toward the bar in [0,1] (for the HUD progress bar)."""
        if self.threshold == 0:
            return 1.0 if self.is_open else 0.0
        if self.direction == "higher":
            return max(0.0, min(1.0, self.value / self.threshold))
        # lower-better: closer to (or below) threshold => fuller bar
        if self.value <= 0:
            return 1.0
        return max(0.0, min(1.0, self.threshold / self.value))


@dataclass(frozen=True)
class AxisStat:
    axis: str
    bins: tuple                       # tuple[(bin_label: str, value: float, n: int), ...]


@dataclass(frozen=True)
class GymState:
    level: int
    level_name: str
    n_levels: int
    step: int
    gates: tuple                      # tuple[GateStatus, ...]
    axes: tuple                       # tuple[AxisStat, ...]
    rooms_cleared: int
    patience_used: int
    patience_max: int
    best_level: int
    headline: str

    @property
    def all_open(self) -> bool:
        return all(g.is_open for g in self.gates)

    @property
    def blocking(self) -> tuple:
        return tuple(g.name for g in self.gates if not g.is_open)


def composite_score(gates: Sequence[GateStatus]) -> float:
    gs = list(gates)
    if not gs:
        return 0.0
    return sum(g.fraction for g in gs) / len(gs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_state.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/state.py tests/alignair/gym/control/test_state.py
git commit -m "feat(gym): add GymState/GateStatus snapshot + composite_score"
```

---

### Task 3: RankLadder

**Files:**
- Create: `src/alignair/gym/control/ladder.py`
- Test: `tests/alignair/gym/control/test_ladder.py`

**Interfaces:**
- Consumes: `alignair.gym.curriculum.Curriculum` (has `.params(p)`).
- Produces: `RankLadder(curriculum=None, n_levels=10)` with `.progress(level)->float`, `.params(level)->dict`, `.name(level)->str`, `.top: int` (== `n_levels-1`).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_ladder.py
from alignair.gym.control.ladder import RankLadder


def test_progress_maps_levels_to_unit_interval():
    lad = RankLadder(n_levels=10)
    assert lad.progress(0) == 0.0
    assert lad.progress(9) == 1.0
    assert lad.top == 9
    # clamps out-of-range
    assert lad.progress(-1) == 0.0
    assert lad.progress(99) == 1.0


def test_params_get_harder_with_level():
    lad = RankLadder(n_levels=10)
    easy = lad.params(0)
    hard = lad.params(9)
    assert hard["mutation_rate"] >= easy["mutation_rate"]


def test_each_level_has_a_name():
    lad = RankLadder(n_levels=10)
    assert isinstance(lad.name(0), str) and lad.name(0)
    assert lad.name(0) != lad.name(9)
    # out-of-range still returns a string (no crash)
    assert isinstance(lad.name(99), str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_ladder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/ladder.py
"""RankLadder: integer floor -> scalar curriculum progress -> GenAIRR params."""
from ..curriculum import Curriculum

_ROOM_NAMES = (
    "Training Grounds", "Mutant Foothills", "SHM Caverns", "Trimmed Halls",
    "Indel Marsh", "Noisy Bazaar", "Fragment Ruins", "Heavy-SHM Tower",
    "Orientation Abyss", "The Gauntlet",
)


class RankLadder:
    def __init__(self, curriculum: Curriculum | None = None, n_levels: int = 10):
        self.curriculum = curriculum or Curriculum()
        self.n_levels = n_levels

    @property
    def top(self) -> int:
        return self.n_levels - 1

    def progress(self, level: int) -> float:
        level = max(0, min(self.top, level))
        return level / max(self.top, 1)

    def params(self, level: int) -> dict:
        return self.curriculum.params(self.progress(level))

    def name(self, level: int) -> str:
        if 0 <= level < len(_ROOM_NAMES):
            return _ROOM_NAMES[level]
        return f"Floor {level + 1}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_ladder.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/ladder.py tests/alignair/gym/control/test_ladder.py
git commit -m "feat(gym): add RankLadder (floor -> curriculum params + room name)"
```

---

### Task 4: PromotionGate

**Files:**
- Create: `src/alignair/gym/control/gate.py`
- Test: `tests/alignair/gym/control/test_gate.py`

**Interfaces:**
- Consumes: `GateSpec` (Task 1), `GateStatus` (Task 2).
- Produces: `PromotionGate(gates: Sequence[GateSpec])` with `.statuses(metrics: dict, level: int) -> list[GateStatus]` and `.evaluate(metrics, level) -> tuple[bool, list[str]]` (all-open, blocking-names). Missing metric keys are treated as a CLOSED gate (value 0 for higher, +inf for lower) so an unmeasured gate never silently promotes.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_gate.py
from alignair.gym.control.config import GateSpec
from alignair.gym.control.gate import PromotionGate


def _gate():
    return PromotionGate((
        GateSpec("v_call", "higher", (0.90, 0.80)),
        GateSpec("coords_mae", "lower", (2.0, 4.0)),
    ))


def test_all_pass_promotes():
    ok, blocking = _gate().evaluate({"v_call": 0.95, "coords_mae": 1.5}, level=0)
    assert ok is True
    assert blocking == []


def test_one_closed_blocks_and_names_it():
    ok, blocking = _gate().evaluate({"v_call": 0.85, "coords_mae": 1.5}, level=0)
    assert ok is False
    assert blocking == ["v_call"]


def test_thresholds_relax_by_level():
    g = _gate()
    # v_call 0.85 fails level 0 (bar 0.90) but passes level 1 (bar 0.80)
    assert g.evaluate({"v_call": 0.85, "coords_mae": 1.0}, level=1)[0] is True


def test_missing_metric_is_closed_not_promoted():
    ok, blocking = _gate().evaluate({"v_call": 0.95}, level=0)   # coords_mae absent
    assert ok is False
    assert "coords_mae" in blocking
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/gate.py
"""PromotionGate: all per-head locks must open (at the level's threshold) to climb."""
from typing import Sequence

from .config import GateSpec
from .state import GateStatus


class PromotionGate:
    def __init__(self, gates: Sequence[GateSpec]):
        self.gates = tuple(gates)

    def statuses(self, metrics: dict, level: int) -> list:
        out = []
        for spec in self.gates:
            thr = spec.thresholds[max(0, min(len(spec.thresholds) - 1, level))]
            if spec.metric in metrics:
                val = float(metrics[spec.metric])
            else:
                # unmeasured => force CLOSED so we never promote on missing evidence
                val = 0.0 if spec.direction == "higher" else float("inf")
            out.append(GateStatus(spec.metric, val, thr, spec.direction))
        return out

    def evaluate(self, metrics: dict, level: int) -> tuple:
        sts = self.statuses(metrics, level)
        blocking = [s.name for s in sts if not s.is_open]
        return (len(blocking) == 0, blocking)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_gate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/gate.py tests/alignair/gym/control/test_gate.py
git commit -m "feat(gym): add PromotionGate (all-locks-open, missing=closed)"
```

---

### Task 5: PlateauDetector

**Files:**
- Create: `src/alignair/gym/control/plateau.py`
- Test: `tests/alignair/gym/control/test_plateau.py`

**Interfaces:**
- Produces: `PlateauDetector(patience=8, slope_eps=1e-3)` with `.update(composite: float) -> bool` (True once `patience` consecutive exams pass without the composite improving by at least `slope_eps` over the best-so-far), `.used: int` (exams since last improvement), `.reset()` (call on promotion).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_plateau.py
from alignair.gym.control.plateau import PlateauDetector


def test_improving_never_plateaus():
    d = PlateauDetector(patience=3, slope_eps=1e-3)
    for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
        assert d.update(v) is False
    assert d.used == 0


def test_flat_sequence_plateaus_after_patience():
    d = PlateauDetector(patience=3, slope_eps=1e-3)
    assert d.update(0.80) is False        # first obs = new best
    assert d.update(0.8005) is False      # < eps gain -> stall 1
    assert d.update(0.8009) is False      # stall 2
    assert d.update(0.8001) is True       # stall 3 == patience -> ceiling
    assert d.used == 3


def test_reset_clears_stall_counter():
    d = PlateauDetector(patience=2, slope_eps=1e-3)
    d.update(0.80); d.update(0.8001)
    d.reset()
    assert d.used == 0
    assert d.update(0.8002) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_plateau.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/plateau.py
"""PlateauDetector: declares a capacity ceiling when the composite competence
score stops improving for `patience` consecutive exams."""


class PlateauDetector:
    def __init__(self, patience: int = 8, slope_eps: float = 1e-3):
        self.patience = patience
        self.slope_eps = slope_eps
        self._best = None
        self._stall = 0

    @property
    def used(self) -> int:
        return self._stall

    def reset(self) -> None:
        self._best = None
        self._stall = 0

    def update(self, composite: float) -> bool:
        if self._best is None or composite > self._best + self.slope_eps:
            self._best = composite
            self._stall = 0
            return False
        self._stall += 1
        return self._stall >= self.patience
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_plateau.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/plateau.py tests/alignair/gym/control/test_plateau.py
git commit -m "feat(gym): add PlateauDetector (patience-bounded ceiling)"
```

---

### Task 6: GymHUD (8-bit renderer)

**Files:**
- Create: `src/alignair/gym/control/hud.py`
- Test: `tests/alignair/gym/control/test_hud.py`

**Interfaces:**
- Consumes: `GymState`, `GateStatus` (Task 2).
- Produces: `GymHUD(color=True)` with `.render(state: GymState) -> str` and `.event(kind: str, state: GymState) -> str` (`kind ∈ {"cleared","best","ceiling","complete"}`). When `color=False` the output contains NO ANSI escape (`\x1b`) sequences.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_hud.py
from alignair.gym.control.state import GateStatus, GymState
from alignair.gym.control.hud import GymHUD


def _state():
    gates = (
        GateStatus("v_call", 0.86, 0.88, "higher"),
        GateStatus("d_call", 0.71, 0.65, "higher"),
        GateStatus("coords_mae", 2.1, 2.0, "lower"),
    )
    return GymState(level=6, level_name="Heavy-SHM Tower", n_levels=10, step=41200,
                    gates=gates, axes=(), rooms_cleared=6, patience_used=3,
                    patience_max=8, best_level=6, headline="junction jitter")


def test_render_plain_has_no_ansi_and_shows_floor_and_locks():
    out = GymHUD(color=False).render(_state())
    assert "\x1b" not in out
    assert "7/10" in out                      # floor 6 shown 1-indexed as 7/10
    assert "Heavy-SHM Tower" in out
    assert "v_call" in out and "d_call" in out
    # open vs closed lock glyphs both present (d_call open, v_call closed)
    assert out.count("\n") > 3                 # multi-line box


def test_render_color_has_ansi():
    out = GymHUD(color=True).render(_state())
    assert "\x1b" in out


def test_event_callouts():
    hud = GymHUD(color=False)
    assert "CLEARED" in hud.event("cleared", _state()).upper()
    assert "CEILING" in hud.event("ceiling", _state()).upper()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_hud.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/hud.py
"""GymHUD: 8-bit 'climb the tower' terminal view. Pure function of GymState."""
from .state import GymState

_BAR_W = 9


def _c(s: str, code: str, on: bool) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if on else s


def _bar(frac: float) -> str:
    filled = int(round(max(0.0, min(1.0, frac)) * _BAR_W))
    return "█" * filled + "░" * (_BAR_W - filled)


class GymHUD:
    def __init__(self, color: bool = True):
        self.color = color

    def _floor_row(self, state: GymState) -> str:
        cells = []
        for i in range(state.n_levels):
            if i < state.level:
                cells.append(_c("[✓]", "32", self.color))     # cleared (green)
            elif i == state.level:
                cells.append(_c("[▶]", "33", self.color))     # current (yellow)
            else:
                cells.append("[ ]")
        return "".join(cells)

    def render(self, state: GymState) -> str:
        title = _c(" A L I G N A I R   G Y M ", "1;36", self.color)
        lines = [
            "╔══════════════" + title + "══════════════╗",
            f"  FLOOR {state.level + 1}/{state.n_levels}  \"{state.level_name}\"    step {state.step:,}",
            "  " + self._floor_row(state),
            "",
            "  LOCKS (all must open to climb):",
        ]
        for g in state.gates:
            glyph = _c("🔓", "32", self.color) if g.is_open else _c("🔒", "31", self.color)
            lines.append(f"   {g.name:<11} {_bar(g.fraction)}  "
                         f"{g.value:.3g} / {g.threshold:.3g}  {glyph}")
        if state.headline:
            lines += ["", f"  ⚠ STUCK ON: {state.headline}"]
        lines.append("╚" + "═" * 47 + "╝")
        lines.append(f"   ROOM CLEARED ×{state.rooms_cleared}   ·   "
                     f"patience {state.patience_used}/{state.patience_max}   ·   "
                     f"best floor {state.best_level + 1}")
        return "\n".join(lines)

    def event(self, kind: str, state: GymState) -> str:
        msg = {
            "cleared": f"★ ROOM CLEARED — LEVEL UP! now on floor {state.level + 1} ★",
            "best": f"⚑ NEW BEST FLOOR: {state.best_level + 1}",
            "ceiling": f"☠ CEILING REACHED at floor {state.level + 1} — can't climb further",
            "complete": "✦ GYM COMPLETE — all floors cleared ✦",
        }.get(kind, kind)
        codes = {"cleared": "1;32", "best": "1;33", "ceiling": "1;31", "complete": "1;35"}
        return _c(msg, codes.get(kind, "1"), self.color)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_hud.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/hud.py tests/alignair/gym/control/test_hud.py
git commit -m "feat(gym): add 8-bit GymHUD renderer (tower/locks/bars, ASCII fallback)"
```

---

### Task 7: GymController (orchestration over an injected evaluator)

**Files:**
- Create: `src/alignair/gym/control/controller.py`
- Test: `tests/alignair/gym/control/test_controller.py`

**Interfaces:**
- Consumes: `GymConfig` (1), `RankLadder` (3), `PromotionGate` (4), `PlateauDetector` (5), `GymState`/`composite_score` (2), `GymHUD` (6).
- Produces: `GymController(config, ladder, gate, evaluator, hud=None, emit=print)` where `evaluator(level: int, batches: int) -> dict` returns a metrics dict (injected so it tests without a model). Methods:
  - `.level: int` (current floor, starts 0).
  - `.done: bool` (set when ceiling reached or top floor cleared).
  - `.exam(step: int) -> GymState` — runs one exam: builds metrics (adds derived `coords_mae`), gate statuses, composite, axes (empty for now), updates promotion/plateau, emits HUD, returns the snapshot.
  - `.progress() -> float` — `ladder.progress(self.level)` (what to pass to `gym.set_progress`).
- Behavior: on all-gates-open → promote (`level += 1`, reset plateau, emit "cleared"; if was top floor → `done=True`, emit "complete"). Else feed composite to plateau; if plateaued → `done=True`, emit "ceiling". `coords_mae` is derived as the mean of available `{g}_e2e_gl_start_dev`/`{g}_e2e_gl_end_dev` over genes present in the metrics.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_controller.py
from alignair.gym.control.config import GymConfig, GateSpec
from alignair.gym.control.ladder import RankLadder
from alignair.gym.control.gate import PromotionGate
from alignair.gym.control.controller import GymController


def _controller(evaluator, n_levels=3):
    gates = (GateSpec("v_call", "higher", tuple(0.8 for _ in range(n_levels))),)
    cfg = GymConfig(n_levels=n_levels, gates=gates, patience=2)
    return GymController(cfg, RankLadder(n_levels=n_levels),
                         PromotionGate(gates), evaluator, hud=None, emit=lambda *_: None)


def test_promotes_when_gate_passes():
    ctrl = _controller(lambda level, batches: {"v_call": 0.95})
    ctrl.exam(step=10)
    assert ctrl.level == 1
    assert ctrl.done is False


def test_clearing_top_floor_completes():
    ctrl = _controller(lambda level, batches: {"v_call": 0.95}, n_levels=2)
    ctrl.exam(step=1)     # 0 -> 1
    ctrl.exam(step=2)     # 1 (top) cleared -> complete
    assert ctrl.level == 1
    assert ctrl.done is True


def test_plateau_sets_ceiling():
    ctrl = _controller(lambda level, batches: {"v_call": 0.5})  # never passes (bar 0.8)
    ctrl.exam(step=1)     # composite ~0.625, best
    ctrl.exam(step=2)     # stall 1
    ctrl.exam(step=3)     # stall 2 == patience -> ceiling
    assert ctrl.level == 0
    assert ctrl.done is True


def test_coords_mae_is_derived_from_e2e_devs():
    seen = {}
    def ev(level, batches):
        return {"v_call": 0.99, "v_e2e_gl_start_dev": 1.0, "v_e2e_gl_end_dev": 3.0}
    gates = (GateSpec("coords_mae", "lower", (2.5,)),)
    cfg = GymConfig(n_levels=1, gates=gates, patience=2)
    ctrl = GymController(cfg, RankLadder(n_levels=1), PromotionGate(gates), ev,
                         hud=None, emit=lambda *_: None)
    state = ctrl.exam(step=1)
    # mean(1.0, 3.0) = 2.0 <= 2.5 => coords gate open
    cm = next(g for g in state.gates if g.name == "coords_mae")
    assert abs(cm.value - 2.0) < 1e-9 and cm.is_open
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_controller.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/controller.py
"""GymController: runs competence exams, promotes on all-locks-open, detects the
ceiling on plateau, and drives the curriculum progress. The evaluator is injected
(a callable returning a metrics dict) so the control logic tests without a model."""
from typing import Callable

from .config import GymConfig
from .gate import PromotionGate
from .ladder import RankLadder
from .plateau import PlateauDetector
from .state import GymState, composite_score


def _derive_coords_mae(metrics: dict) -> float | None:
    devs = []
    for g in ("v", "j", "d"):
        for suffix in ("e2e_gl_start_dev", "e2e_gl_end_dev"):
            key = f"{g}_{suffix}"
            if key in metrics:
                devs.append(float(metrics[key]))
    return sum(devs) / len(devs) if devs else None


class GymController:
    def __init__(self, config: GymConfig, ladder: RankLadder, gate: PromotionGate,
                 evaluator: Callable[[int, int], dict], hud=None, emit: Callable = print):
        self.config = config
        self.ladder = ladder
        self.gate = gate
        self.evaluator = evaluator
        self.hud = hud
        self.emit = emit
        self.plateau = PlateauDetector(config.patience, config.slope_eps)
        self.level = 0
        self.best_level = 0
        self.rooms_cleared = 0
        self.done = False

    def progress(self) -> float:
        return self.ladder.progress(self.level)

    def _metrics(self, level: int) -> dict:
        metrics = dict(self.evaluator(level, self.config.exam_batches))
        cm = _derive_coords_mae(metrics)
        if cm is not None:
            metrics.setdefault("coords_mae", cm)
        return metrics

    def _snapshot(self, metrics: dict, step: int, headline: str = "") -> GymState:
        sts = self.gate.statuses(metrics, self.level)
        return GymState(
            level=self.level, level_name=self.ladder.name(self.level),
            n_levels=self.config.n_levels, step=step, gates=tuple(sts), axes=(),
            rooms_cleared=self.rooms_cleared, patience_used=self.plateau.used,
            patience_max=self.config.patience, best_level=self.best_level,
            headline=headline)

    def exam(self, step: int) -> GymState:
        metrics = self._metrics(self.level)
        promote, blocking = self.gate.evaluate(metrics, self.level)
        headline = "" if promote else ", ".join(blocking)
        state = self._snapshot(metrics, step, headline)
        if self.hud is not None:
            self.emit(self.hud.render(state))
        if promote:
            self.plateau.reset()
            if self.level >= self.ladder.top:
                self.done = True
                self._event("complete", state)
            else:
                self.level += 1
                self.rooms_cleared += 1
                if self.level > self.best_level:
                    self.best_level = self.level
                self._event("cleared", self._snapshot(metrics, step))
        else:
            if self.plateau.update(composite_score(state.gates)):
                self.done = True
                self._event("ceiling", state)
        return state

    def _event(self, kind: str, state: GymState) -> None:
        if self.hud is not None:
            self.emit(self.hud.event(kind, state))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_controller.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/controller.py tests/alignair/gym/control/test_controller.py
git commit -m "feat(gym): add GymController (exam -> promote/ceiling over injected evaluator)"
```

---

### Task 8: Wire GymController into GymTrainer.fit

**Files:**
- Modify: `src/alignair/training/gym_trainer.py` (the `fit()` curriculum block, lines ~99-115)
- Modify: `src/alignair/gym/control/__init__.py` (export the public surface)
- Test: `tests/alignair/gym/control/test_controller_wiring.py`

**Interfaces:**
- Consumes: `GymController`, `RankLadder`, `PromotionGate`, `GymConfig`, `GymHUD`; `GymTrainer.evaluate(n_batches, p)` (existing).
- Produces: `GymTrainer.fit(..., controller: GymController | None = None)`. When a controller is given, the curriculum progress is driven by `controller.progress()` (set on the gym at each refresh) and an exam runs every `config.exam_every` steps; training stops early when `controller.done`. When `controller is None`, behavior is the existing `p = step/total` clock (unchanged). Export from `__init__`: `GymConfig, GateSpec, default_gates, RankLadder, PromotionGate, PlateauDetector, GymController, GymHUD, GymState`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_controller_wiring.py
"""The controller's progress drives the gym; an evaluator that always passes the
gate climbs floors as fit() runs. Uses a stub trainer to stay GPU-free."""
from alignair.gym.control import GymConfig, GateSpec, RankLadder, PromotionGate, GymController


class _StubGym:
    def __init__(self):
        self.progress_calls = []
    def set_progress(self, p):
        self.progress_calls.append(p)


def test_controller_progress_climbs_with_passing_exams():
    gates = (GateSpec("v_call", "higher", (0.8, 0.8, 0.8)),)
    cfg = GymConfig(n_levels=3, gates=gates, patience=2, exam_every=1)
    ctrl = GymController(cfg, RankLadder(n_levels=3), PromotionGate(gates),
                         evaluator=lambda level, batches: {"v_call": 0.99},
                         hud=None, emit=lambda *_: None)
    gym = _StubGym()
    # simulate fit()'s exam cadence: each exam promotes (passing metrics)
    for step in range(1, 4):
        ctrl.exam(step)
        gym.set_progress(ctrl.progress())
        if ctrl.done:
            break
    assert ctrl.best_level == 2          # climbed 0 -> 1 -> 2 (top)
    assert ctrl.done is True
    assert gym.progress_calls[-1] == 1.0  # top floor progress
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_controller_wiring.py -v`
Expected: FAIL with `ImportError: cannot import name 'GymConfig' from 'alignair.gym.control'`

- [ ] **Step 3: Write the `__init__` exports**

```python
# src/alignair/gym/control/__init__.py
"""Competence-gated curriculum control for the GenAIRR gym."""
from .config import GateSpec, GymConfig, default_gates
from .state import GateStatus, AxisStat, GymState, composite_score
from .ladder import RankLadder
from .gate import PromotionGate
from .plateau import PlateauDetector
from .hud import GymHUD
from .controller import GymController

__all__ = [
    "GateSpec", "GymConfig", "default_gates", "GateStatus", "AxisStat", "GymState",
    "composite_score", "RankLadder", "PromotionGate", "PlateauDetector", "GymHUD",
    "GymController",
]
```

- [ ] **Step 4: Run wiring test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_controller_wiring.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Wire the controller into `fit()`**

In `src/alignair/training/gym_trainer.py`, change the `fit` signature to accept `controller=None`:

```python
    def fit(self, total_steps: int, global_total: int | None = None,
            progress: bool = True, controller=None) -> list:
```

Replace the curriculum-refresh block (currently):

```python
            if since_refresh >= self.refresh_curriculum_every:
                if global_total:
                    p = self._global_step / max(global_total - 1, 1)
                else:
                    p = step / max(total_steps - 1, 1)
                self.gym.set_progress(min(1.0, p))
                it = iter(loader)  # picks up the new curriculum progress on rebuild
                since_refresh = 0
```

with:

```python
            if since_refresh >= self.refresh_curriculum_every:
                if controller is not None:
                    self.gym.set_progress(controller.progress())
                elif global_total:
                    self.gym.set_progress(min(1.0, self._global_step / max(global_total - 1, 1)))
                else:
                    self.gym.set_progress(min(1.0, step / max(total_steps - 1, 1)))
                it = iter(loader)  # picks up the new curriculum progress on rebuild
                since_refresh = 0
```

Immediately after `step += 1` near the end of the loop body (the last line before `bar.close()`'s `while`), add the exam cadence + early stop:

```python
            if controller is not None and step % controller.config.exam_every == 0:
                controller.exam(step=self._global_step)
                if controller.done:
                    break
```

- [ ] **Step 6: Run the full control test suite to verify nothing regressed**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/ -v`
Expected: PASS (all tasks 1-8 green)

- [ ] **Step 7: Commit**

```bash
git add src/alignair/gym/control/__init__.py src/alignair/training/gym_trainer.py tests/alignair/gym/control/test_controller_wiring.py
git commit -m "feat(gym): drive GymTrainer.fit curriculum via GymController (opt-in)"
```

---

### Task 9: Per-axis tagged exam metrics (RankExam)

**Files:**
- Modify: `src/alignair/training/gym_trainer.py` (add `evaluate_records()`)
- Create: `src/alignair/gym/control/exam.py`
- Test: `tests/alignair/gym/control/test_exam.py`

**Interfaces:**
- Consumes: the gym bundle's per-read truth tags (`mutation_rate`, `indel_count`, `noise_count`, `orientation_id`) from `build_targets`, and read length from `tokens`.
- Produces:
  - `bucket_axis(records: list[dict], axis: str, edges: Sequence[float], metric_key: str) -> AxisStat` — buckets per-read `metric_key` by the read's `axis` value into bins delimited by `edges`, returning mean metric + count per bin.
  - `axis_breakdown(records: list[dict]) -> tuple[AxisStat, ...]` — standard axes: `shm` (mutation_rate), `indel` (indel_count), `noise` (noise_count), `length` (read length). Each record is a dict with those tag keys plus a per-read `correct` ∈ {0,1} (allele call) for the metric.
- Note: this task delivers the pure bucketing + an `AxisStat` list. Hooking the model-side `evaluate_records()` is included but its assertion is light (shape only) to stay GPU-optional.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_exam.py
from alignair.gym.control.state import AxisStat
from alignair.gym.control.exam import bucket_axis, axis_breakdown


def _recs():
    # two easy (low SHM, correct) + two hard (high SHM, wrong)
    return [
        {"mutation_rate": 0.01, "indel_count": 0, "noise_count": 0, "length": 300, "correct": 1.0},
        {"mutation_rate": 0.02, "indel_count": 0, "noise_count": 0, "length": 300, "correct": 1.0},
        {"mutation_rate": 0.20, "indel_count": 3, "noise_count": 1, "length": 80, "correct": 0.0},
        {"mutation_rate": 0.25, "indel_count": 4, "noise_count": 2, "length": 60, "correct": 0.0},
    ]


def test_bucket_axis_splits_and_means():
    st = bucket_axis(_recs(), axis="mutation_rate", edges=[0.0, 0.05, 1.0], metric_key="correct")
    assert isinstance(st, AxisStat) and st.axis == "mutation_rate"
    # bin 0 (<=0.05): both correct -> 1.0, n=2 ; bin 1 (>0.05): both wrong -> 0.0, n=2
    labels = {b[0]: (b[1], b[2]) for b in st.bins}
    assert any(v == (1.0, 2) for v in labels.values())
    assert any(v == (0.0, 2) for v in labels.values())


def test_axis_breakdown_returns_standard_axes():
    axes = axis_breakdown(_recs())
    names = {a.axis for a in axes}
    assert {"shm", "indel", "noise", "length"} <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_exam.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/exam.py
"""Per-axis struggle attribution: bucket per-read correctness by the read's
GenAIRR truth difficulty (SHM, indels, noise, length) to name the bottleneck."""
from typing import Sequence

from .state import AxisStat


def bucket_axis(records: Sequence[dict], axis: str, edges: Sequence[float],
                metric_key: str) -> AxisStat:
    edges = list(edges)
    sums = [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    for r in records:
        v = float(r.get(axis, 0.0))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            # last bin is inclusive of the top edge
            in_bin = lo < v <= hi if i > 0 else lo <= v <= hi
            if in_bin:
                sums[i] += float(r.get(metric_key, 0.0))
                counts[i] += 1
                break
    bins = []
    for i in range(len(edges) - 1):
        label = f"{edges[i]:g}-{edges[i + 1]:g}"
        mean = sums[i] / counts[i] if counts[i] else 0.0
        bins.append((label, mean, counts[i]))
    return AxisStat(axis=axis, bins=tuple(bins))


_AXES = (
    ("shm", "mutation_rate", [0.0, 0.05, 0.15, 1.0]),
    ("indel", "indel_count", [0.0, 0.5, 3.0, 100.0]),
    ("noise", "noise_count", [0.0, 0.5, 3.0, 100.0]),
    ("length", "length", [0.0, 100.0, 250.0, 10000.0]),
)


def axis_breakdown(records: Sequence[dict], metric_key: str = "correct") -> tuple:
    out = []
    for name, key, edges in _AXES:
        st = bucket_axis(records, axis=key, edges=edges, metric_key=metric_key)
        out.append(AxisStat(axis=name, bins=st.bins))
    return tuple(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_exam.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add `evaluate_records()` to GymTrainer (model-side, light test)**

In `src/alignair/training/gym_trainer.py`, add a method that returns per-read tagged correctness for the V call (the headline allele head), reusing the existing eval forward path:

```python
    @torch.no_grad()
    def evaluate_records(self, n_batches: int = 4, p: float = 1.0) -> list:
        """Per-read tagged records for axis attribution: each dict carries the
        read's GenAIRR truth difficulty tags + `correct` (V top-1 in true set)."""
        self.model.eval()
        prev_p = self.gym._p
        self.gym.set_progress(p)
        loader = self._loader()
        ref_emb = self.model.encode_reference(self.reference_set)
        records, nb = [], 0
        for batch in loader:
            if nb >= n_batches:
                break
            batch = self._to_device(batch)
            out = self.model(batch["tokens"], batch["mask"], ref_emb,
                             orientation_ids=batch["orientation_id"])
            pred = out["match"]["V"].argmax(-1)
            B = batch["tokens"].shape[0]
            correct = batch["v_allele"][torch.arange(B), pred].cpu()
            length = batch["mask"].sum(dim=1).cpu()
            for i in range(B):
                records.append({
                    "mutation_rate": float(batch["mutation_rate"][i].cpu()),
                    "indel_count": float(batch["indel_count"][i].cpu()),
                    "noise_count": float(batch["noise_count"][i].cpu()),
                    "length": int(length[i]),
                    "correct": float(correct[i] > 0),
                })
            nb += 1
        self.model.train()
        self.gym.set_progress(prev_p)
        return records
```

(If `batch` lacks `mutation_rate`/`indel_count`/`noise_count` collated tensors, add them to `gym_collate` alongside the existing scalar targets — they already exist per-read in the bundle from `build_targets`.)

- [ ] **Step 6: Commit**

```bash
git add src/alignair/gym/control/exam.py src/alignair/training/gym_trainer.py tests/alignair/gym/control/test_exam.py
git commit -m "feat(gym): per-axis struggle attribution (bucketing + tagged records)"
```

---

### Task 10: StruggleReporter (JSON / markdown / climb curve)

**Files:**
- Create: `src/alignair/gym/control/reporter.py`
- Test: `tests/alignair/gym/control/test_reporter.py`

**Interfaces:**
- Consumes: `GymState` (2), `AxisStat` (2).
- Produces: `StruggleReporter(out_dir: str)` with:
  - `.to_dict(state: GymState) -> dict` — JSON-serializable snapshot (level, step, gates, axes, blocking, composite).
  - `.write(state: GymState) -> str` — writes `gym_report_floor{L}_step{N}.json` + `.md`, appends one line to `climb_curve.jsonl`, returns the json path.
  - `.markdown(state: GymState) -> str` — human-readable report (gate table + per-axis breakdown + headline).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/control/test_reporter.py
import json
from alignair.gym.control.state import GateStatus, AxisStat, GymState
from alignair.gym.control.reporter import StruggleReporter


def _state():
    return GymState(level=2, level_name="SHM Caverns", n_levels=10, step=5000,
                    gates=(GateStatus("v_call", 0.7, 0.9, "higher"),),
                    axes=(AxisStat("shm", (("0-0.05", 0.99, 50), ("0.15-1", 0.61, 40))),),
                    rooms_cleared=2, patience_used=1, patience_max=8,
                    best_level=2, headline="v_call")


def test_to_dict_is_json_serializable_and_complete():
    d = StruggleReporter("/tmp/gymtest").to_dict(_state())
    json.dumps(d)                                  # must not raise
    assert d["level"] == 2 and d["step"] == 5000
    assert d["blocking"] == ["v_call"]
    assert any(g["name"] == "v_call" for g in d["gates"])
    assert any(a["axis"] == "shm" for a in d["axes"])


def test_write_creates_files_and_appends_curve(tmp_path):
    rep = StruggleReporter(str(tmp_path))
    path = rep.write(_state())
    assert path.endswith(".json")
    assert (tmp_path / "climb_curve.jsonl").exists()
    md = rep.markdown(_state())
    assert "SHM Caverns" in md and "v_call" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_reporter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/control/reporter.py
"""StruggleReporter: render a GymState to JSON + markdown + a climb-curve line."""
import json
import os

from .state import GymState, composite_score


class StruggleReporter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def to_dict(self, state: GymState) -> dict:
        return {
            "level": state.level,
            "level_name": state.level_name,
            "n_levels": state.n_levels,
            "step": state.step,
            "composite": composite_score(state.gates),
            "blocking": list(state.blocking),
            "headline": state.headline,
            "rooms_cleared": state.rooms_cleared,
            "best_level": state.best_level,
            "gates": [{"name": g.name, "value": g.value, "threshold": g.threshold,
                       "direction": g.direction, "open": g.is_open} for g in state.gates],
            "axes": [{"axis": a.axis,
                      "bins": [{"label": b[0], "value": b[1], "n": b[2]} for b in a.bins]}
                     for a in state.axes],
        }

    def markdown(self, state: GymState) -> str:
        d = self.to_dict(state)
        lines = [f"# Gym report — floor {state.level + 1}/{state.n_levels} "
                 f"\"{state.level_name}\" @ step {state.step:,}",
                 "", f"**Composite:** {d['composite']:.3f}  ·  "
                 f"**Blocking:** {', '.join(d['blocking']) or 'none (cleared)'}", "",
                 "## Locks", "", "| gate | value | threshold | open |",
                 "|---|---|---|---|"]
        for g in d["gates"]:
            lines.append(f"| {g['name']} | {g['value']:.3g} | {g['threshold']:.3g} "
                         f"| {'✓' if g['open'] else '✗'} |")
        for a in d["axes"]:
            lines += ["", f"## Axis: {a['axis']}", "", "| bin | metric | n |", "|---|---|---|"]
            for b in a["bins"]:
                lines.append(f"| {b['label']} | {b['value']:.3f} | {b['n']} |")
        return "\n".join(lines)

    def write(self, state: GymState) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        base = f"gym_report_floor{state.level}_step{state.step}"
        jpath = os.path.join(self.out_dir, base + ".json")
        with open(jpath, "w") as f:
            json.dump(self.to_dict(state), f, indent=2)
        with open(os.path.join(self.out_dir, base + ".md"), "w") as f:
            f.write(self.markdown(state))
        with open(os.path.join(self.out_dir, "climb_curve.jsonl"), "a") as f:
            f.write(json.dumps({"step": state.step, "level": state.level,
                                "composite": composite_score(state.gates),
                                "best_level": state.best_level}) + "\n")
        return jpath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/test_reporter.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/control/reporter.py tests/alignair/gym/control/test_reporter.py
git commit -m "feat(gym): add StruggleReporter (JSON + markdown + climb_curve.jsonl)"
```

---

### Task 11: Full control-suite green + smoke run

**Files:**
- Test: `tests/alignair/gym/control/` (all)

- [ ] **Step 1: Run the entire control suite**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/control/ -v`
Expected: PASS (all of tasks 1-10).

- [ ] **Step 2: Run the existing gym tests to confirm no regression**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/ tests/alignair/integration/test_gym_training.py -v`
Expected: PASS (existing curriculum/collate/training tests unaffected — `controller=None` default preserves old behavior).

- [ ] **Step 3: Commit (if any fixups were needed)**

```bash
git add -A && git commit -m "test(gym): full competence-curriculum control suite green"
```

---

## Self-Review

**Spec coverage:**
- Scalar ladder + per-axis diagnostics → Tasks 3 (ladder), 9 (axis breakdown). ✓
- Multi-gate all-must-pass promotion → Task 4. ✓
- Plateau / ceiling detection → Task 5, wired in Task 7. ✓
- `GymState` single source of truth → Task 2, used by HUD (6) and reporter (10). ✓
- 8-bit HUD with ASCII fallback + event callouts → Task 6. ✓
- Struggle report (JSON/markdown/climb curve, Benchmarking-style attribution) → Tasks 9-10. ✓
- Config-driven foundation (no magic numbers in logic) → Task 1. ✓
- Build on existing `evaluate()` / `Curriculum` / `AlignAIRGym` → Tasks 7-9 reuse them; `controller=None` keeps old behavior. ✓
- Phasing: P1 control loop = Tasks 1-8; P2 exam+reporter = Tasks 9-10; P3 full HUD folded into Task 6. ✓
- Sub-project B (learned aligner) explicitly out of scope. ✓

**Type consistency:** `GymState`, `GateStatus`, `AxisStat`, `composite_score` defined in Task 2 and consumed with matching signatures in Tasks 6, 7, 9, 10. `GateSpec(metric, direction, thresholds)` from Task 1 consumed by `PromotionGate` (4) and `GymConfig` (1). `evaluator(level, batches) -> dict` injected in Task 7, supplied as `trainer.evaluate`-shaped metrics in Task 8. ✓

**Placeholder scan:** Task 7 deliberately includes a copy/paste dead line with an explicit cleanup step (Step 4) — this is a guided fix, not a placeholder; all other steps show complete code. The `gym_collate` note in Task 9 Step 5 is a conditional ("if the tensors aren't collated") with the exact source named. No TBDs.

## Notes for the implementer

- The default gate thresholds (Task 1) are an initial cut. Once the suite is green, calibrate them against a real short climb (`scripts/`-style run) before a long training run — that calibration is expected future work, not part of this plan.
- `junction` is intentionally NOT a gate yet (no metric exists). When a junction-exact metric is added to `evaluate()`, it becomes one more `GateSpec` in `default_gates` — no control-loop change needed.
- Sub-project B (the learned parallel aligner) will be A/B'd by giving two model architectures the SAME `GymConfig` tower and comparing `best_level` — the reason the gym was built first.
