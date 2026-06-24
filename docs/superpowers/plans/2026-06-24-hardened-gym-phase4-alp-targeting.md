# Hardened Gym — Phase 4 (ALP / Regret Targeting) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Automatically concentrate sampling on the regimes where the model is failing or still improving (heavy-SHM-V, junction band) — Absolute Learning Progress + regret targeting — on top of the Phase-2 factored ramp, with a permanent regret-pinned hard-corner floor (anti-forgetting).

**Architecture:** Model a curriculum as a set of weighted **difficulty components** `[(weight, params), …]`; producers draw one component per epoch, so the long-run sampling distribution is the mixture. A `TargetedCurriculum` wraps the Phase-2 `FactoredCurriculum` and emits three components: the factored ramp (majority), an **ALP/regret-targeted** lattice cell (the current highest-priority weakness), and a fixed **hard-corner floor** (`heavy_shm_fulllen`). A `ProgressTracker` turns the Phase-1 lattice competence field over time into per-cell priorities = `alp·|ΔS| + regret·max(0, target−S)`.

**Tech Stack:** Python 3.12, pytest, Phase-1 `instrument` + Phase-2 `factored`.

## Global Constraints

- venv `./.venv/bin/python`; tests from repo root.
- A "component" `params` dict is anything `build_experiment` accepts — both the factored per-read-distribution form and the lattice `to_genairr_params` (rate + count-tuple) form are valid, so components may mix forms.
- Backward compatible: curricula WITHOUT a `components()` method behave exactly as today (the gym wraps `params()` as a single `[(1.0, params)]` component). `sigma`/targeting defaults keep current behavior.
- Commit per task; no Claude mentions in commit messages.

## File Structure

- Create `src/alignair/gym/targeting.py` — `ProgressTracker`, `TargetedCurriculum`.
- Modify `src/alignair/gym/gym.py` — per-epoch component sampling (both simple + shared paths).
- Modify `src/alignair/training/gym_trainer.py` — `advance_curriculum` updates targeting + pushes.
- Tests under `tests/alignair/gym/`.

---

### Task 1: ProgressTracker (ALP + regret → per-cell priorities)

**Files:** `src/alignair/gym/targeting.py`; test `tests/alignair/gym/test_progress_tracker.py`

**Interfaces:** `ProgressTracker(target=0.95, alp_weight=1.0, regret_weight=0.5)`; `.update(field: dict)` records per-cell `S` and updates ALP=`|S_now−S_prev|`; `.priorities() -> dict[cell,float]` normalized over cells (alp·ALP + regret·max(0,target−S)); `.top_cell() -> str|None` highest-priority cell. Before two updates exist ALP is 0 (regret still ranks); empty → `{}`/`None`.

- [ ] **Step 1: failing test**

```python
# tests/alignair/gym/test_progress_tracker.py
from alignair.gym.targeting import ProgressTracker


def test_regret_ranks_low_competence_cells_first():
    t = ProgressTracker(target=0.95, alp_weight=0.0, regret_weight=1.0)
    t.update({"clean": {"S": 0.97}, "heavy_shm": {"S": 0.50}})
    pr = t.priorities()
    assert pr["heavy_shm"] > pr["clean"]              # bigger regret -> higher priority
    assert t.top_cell() == "heavy_shm"
    assert abs(sum(pr.values()) - 1.0) < 1e-9         # normalized


def test_alp_rewards_moving_cells():
    t = ProgressTracker(target=0.95, alp_weight=1.0, regret_weight=0.0)
    t.update({"a": {"S": 0.40}, "b": {"S": 0.40}})
    t.update({"a": {"S": 0.60}, "b": {"S": 0.41}})    # a moved +0.20, b +0.01
    assert t.top_cell() == "a"


def test_empty_tracker():
    t = ProgressTracker()
    assert t.priorities() == {} and t.top_cell() is None
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement**

```python
# src/alignair/gym/targeting.py
"""Phase-4 automatic targeting: turn the lattice competence field over time into
per-cell priorities (Absolute Learning Progress + regret) and express the curriculum
as a mixture of weighted difficulty components (ramp + targeted cell + hard-corner)."""


class ProgressTracker:
    def __init__(self, target: float = 0.95, alp_weight: float = 1.0,
                 regret_weight: float = 0.5):
        self.target = target
        self.alp_weight = alp_weight
        self.regret_weight = regret_weight
        self._prev: dict = {}
        self._alp: dict = {}

    def update(self, field: dict) -> None:
        for cell, v in field.items():
            s = float(v["S"] if isinstance(v, dict) else v)
            if cell in self._prev:
                self._alp[cell] = abs(s - self._prev[cell])
            self._prev[cell] = s

    def _raw(self) -> dict:
        out = {}
        for c, s in self._prev.items():
            alp = self._alp.get(c, 0.0)
            regret = max(0.0, self.target - s)
            out[c] = self.alp_weight * alp + self.regret_weight * regret
        return out

    def priorities(self) -> dict:
        raw = self._raw()
        tot = sum(raw.values())
        if not raw:
            return {}
        if tot <= 0:
            return {c: 1.0 / len(raw) for c in raw}
        return {c: w / tot for c, w in raw.items()}

    def top_cell(self):
        raw = self._raw()
        return max(raw, key=raw.get) if raw else None
```

- [ ] **Step 4:** run → PASS (3). **Step 5:** commit `feat(gym): add ProgressTracker (ALP + regret per-cell priorities)`

---

### Task 2: TargetedCurriculum (weighted difficulty-component mixture)

**Files:** `src/alignair/gym/targeting.py`; test `tests/alignair/gym/test_targeted_curriculum.py`

**Interfaces:** `TargetedCurriculum(factored, lattice, tracker=None, p_ramp=0.6, p_alp=0.25, p_floor=0.15, floor_cell="heavy_shm_fulllen")`. Delegates the curriculum interface to `factored` (`params`/`describe`/`stage`/`pace`/`axes`/`advance`). `update_targets(field)` feeds the tracker. `components() -> list[(weight, params)]`: ramp (`p_ramp`, factored.params()) + targeted (`p_alp`, top-priority cell params) + floor (`p_floor`, floor_cell params); when no targets yet, the ramp absorbs `p_alp`. Cell params are cached from `lattice.cell_params`.

- [ ] **Step 1: failing test**

```python
# tests/alignair/gym/test_targeted_curriculum.py
from alignair.gym.factored import FactoredCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.targeting import TargetedCurriculum


def _tc():
    return TargetedCurriculum(FactoredCurriculum(start_pace=0.3), FrozenLattice.standard(seed=0))


def test_components_sum_to_one_and_delegate_interface():
    tc = _tc()
    comps = tc.components()
    assert abs(sum(w for w, _ in comps) - 1.0) < 1e-9
    assert tc.params()["mutation_count"] == tc.factored.params()["mutation_count"]
    assert tc.advance({"heavy_shm_fulllen": {"S": 0.9}})           # delegates to factored


def test_targeted_component_tracks_top_priority_cell():
    tc = _tc()
    tc.update_targets({"clean": {"S": 0.97}, "heavy_shm": {"S": 0.40},
                       "heavy_shm_fulllen": {"S": 0.95}})
    comps = tc.components()
    # the p_alp component should carry heavy_shm's params (lowest competence => top regret)
    lat = FrozenLattice.standard(seed=0)
    hs = lat.cell_params(next(c for c in lat.cells if c.name == "heavy_shm"))
    assert any(abs(w - tc.p_alp) < 1e-9 and p == hs for w, p in comps)


def test_no_targets_folds_alp_into_ramp():
    tc = _tc()
    comps = tc.components()                                         # tracker empty
    ramp_w = sum(w for w, _ in comps[:-1])                          # all but floor
    assert abs(ramp_w - (tc.p_ramp + tc.p_alp)) < 1e-9
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** (append to `targeting.py`)

```python
class TargetedCurriculum:
    def __init__(self, factored, lattice, tracker=None, p_ramp=0.6, p_alp=0.25,
                 p_floor=0.15, floor_cell="heavy_shm_fulllen"):
        self.factored = factored
        self.lattice = lattice
        self.tracker = tracker or ProgressTracker()
        self.p_ramp, self.p_alp, self.p_floor = p_ramp, p_alp, p_floor
        self.floor_cell = floor_cell
        self._cell_params = {c.name: lattice.cell_params(c) for c in lattice.cells}

    # ---- curriculum interface delegated to the factored ramp ----
    def params(self, p=None):
        return self.factored.params(p)

    def describe(self, p=None):
        return "targeted | " + self.factored.describe(p)

    def stage(self, p=None):
        return self.factored.stage(p)

    @property
    def pace(self):
        return self.factored.pace

    @property
    def axes(self):
        return self.factored.axes

    def advance(self, axis_competence, **kw):
        return self.factored.advance(axis_competence, **kw)

    def update_targets(self, field: dict) -> None:
        self.tracker.update(field)

    def components(self):
        ramp = (self.p_ramp, self.factored.params())
        floor = (self.p_floor, self._cell_params.get(
            self.floor_cell, self.factored.params()))
        top = self.tracker.top_cell()
        if top is None or top not in self._cell_params:
            return [(self.p_ramp + self.p_alp, self.factored.params()), floor]
        return [ramp, (self.p_alp, self._cell_params[top]), floor]
```

- [ ] **Step 4:** run → PASS (3). **Step 5:** commit `feat(gym): add TargetedCurriculum (ramp + ALP-targeted cell + hard-corner floor mixture)`

---

### Task 3: Gym draws a component per epoch

**Files:** `src/alignair/gym/gym.py`; test `tests/alignair/gym/test_component_sampling.py`

**Interfaces:** Add `AlignAIRGym._components()` → the current component list (`curriculum.components()` if present, else `[(1.0, curriculum.params(self._p))]`; in the shared path read from the shared dict). Per epoch, both `_iter_simple` and `_iter_shared` pick one component's params by weight (`_pick_params(components, rng)`). `_push_params` pushes `components` (not a single params) to the shared dict.

- [ ] **Step 1: failing test**

```python
# tests/alignair/gym/test_component_sampling.py
import numpy as np
from alignair.gym.gym import _pick_params


def test_pick_params_respects_weights():
    comps = [(0.0, {"a": 1}), (1.0, {"a": 2})]
    rng = np.random.default_rng(0)
    picks = [_pick_params(comps, rng)["a"] for _ in range(20)]
    assert set(picks) == {2}                       # zero-weight component never chosen


def test_pick_params_single_component():
    assert _pick_params([(1.0, {"x": 9})], np.random.default_rng(0)) == {"x": 9}
```

- [ ] **Step 2:** run → FAIL (`_pick_params` missing). **Step 3: implement** — in `gym.py` add a module-level helper and wire both iter paths:

```python
def _pick_params(components, rng):
    """Weighted choice of one difficulty component's params (per epoch)."""
    weights = [max(0.0, float(w)) for w, _ in components]
    tot = sum(weights)
    if tot <= 0:
        return components[0][1]
    r = float(rng.random()) * tot
    upto = 0.0
    for (w, params), wn in zip(components, weights):
        upto += wn
        if r <= upto:
            return params
    return components[-1][1]
```
Add a method:
```python
    def _components(self):
        if self._shared_params is not None:
            return self._shared_params["components"]
        if hasattr(self.curriculum, "components"):
            return self.curriculum.components()
        return [(1.0, self.curriculum.params(self._p))]
```
In `enable_sharing` and `_push_params`, store COMPONENTS instead of a single params:
```python
        comps = (self.curriculum.components() if hasattr(self.curriculum, "components")
                 else [(1.0, self.curriculum.params(self._p))])
        self._shared_params["components"] = comps
```
In `_iter_simple`, replace `params = self.curriculum.params(self._p)` with
`params = _pick_params(self._components(), rng)` (move `rng` creation above it).
In `_iter_shared`, replace `params = dict(self._shared_params["params"])` with
`params = _pick_params(self._shared_params["components"], np.random.default_rng(seed))`.

- [ ] **Step 4:** run new test + the existing `test_parallel_producers.py` + `test_gym_stream.py` → PASS (component plumbing didn't break single-component curricula). **Step 5:** commit `feat(gym): producers draw one difficulty component per epoch (mixture sampling)`

---

### Task 4: Trainer updates targeting on advance

**Files:** `src/alignair/training/gym_trainer.py`; test `tests/alignair/integration/test_targeted_training.py`

**Interfaces:** `GymTrainer.advance_curriculum(field, …)` — if the gym's curriculum is a `TargetedCurriculum`, call `curriculum.update_targets(field)` (and `gym.refresh_params()` to push the new mixture) IN ADDITION to the existing factored advance. Works whether the curriculum is a bare `FactoredCurriculum` or a `TargetedCurriculum` wrapping one.

- [ ] **Step 1: failing test**

```python
# tests/alignair/integration/test_targeted_training.py
import pytest, torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.factored import FactoredCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.targeting import TargetedCurriculum
from alignair.training.gym_trainer import GymTrainer


def test_targeted_curriculum_trains_and_updates_targets():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128))
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    tc = TargetedCurriculum(FactoredCurriculum(start_pace=0.3), FrozenLattice.standard(seed=0))
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0, curriculum=tc)
    trainer = GymTrainer(model, loss_fn, rs, gym, batch_size=8)
    trainer.fit(total_steps=6, progress=False)               # mixture sampling works in training
    trainer.advance_curriculum({"clean": {"S": 0.97}, "heavy_shm": {"S": 0.40},
                                "heavy_shm_fulllen": {"S": 0.95}})
    assert tc.tracker.top_cell() == "heavy_shm"              # targeting updated from the field
    assert any(abs(w - tc.p_alp) < 1e-9 for w, _ in tc.components())
```

- [ ] **Step 2:** run → FAIL. **Step 3: implement** — in `advance_curriculum`, after resolving `cur = self.gym.curriculum`:

```python
        from ..gym.targeting import TargetedCurriculum
        target = cur if isinstance(cur, TargetedCurriculum) else None
        factored = cur.factored if target is not None else cur
        if not isinstance(factored, FactoredCurriculum):
            return []
        moved = factored.advance(axis_competence_from_field(field), threshold=threshold, step=step)
        if target is not None:
            target.update_targets(field)
        if moved or target is not None:
            self.gym.refresh_params()
            if moved and self.sigma_freeze_steps > 0:
                self.freeze_uncertainty()
        return moved
```
(Replace the existing `FactoredCurriculum`-only body; keep the `import FactoredCurriculum, axis_competence_from_field` line.)

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat(train): advance_curriculum updates ALP/regret targeting (TargetedCurriculum)`

---

## Self-Review

Spec coverage (Phase 4): ALP-GMM-style learning-progress targeting → ProgressTracker (ALP) + TargetedCurriculum p_alp component (T1,T2); regret + hard-corner floor (anti-forgetting) → regret in tracker + p_floor component (T1,T2); mixture realized in generation → per-epoch component sampling (T3); wired to the competence field → advance_curriculum (T4). The continuous GMM-over-Θ is approximated by a discrete ALP/regret over the frozen lattice cells (documented design choice; the lattice is the measurement substrate we already trust). Backward-compat: single-component wrapping keeps all existing curricula working. Types: `components()->[(w,params)]` consistent across TargetedCurriculum (T2), gym `_pick_params`/`_components` (T3), trainer push (T4). No placeholders.

## Notes

- The regret `target=0.95` is a fixed stand-in for the per-cell irreducible-error floor that Phase 5 estimates from the simulator; swap it there.
- Like Phases 2–3, the *targeting behavior's* benefit is only visible in a training run (soft-DP-bound today) — the unit tests prove the mixture mechanics; empirical validation rides on sub-project B's speedup.
