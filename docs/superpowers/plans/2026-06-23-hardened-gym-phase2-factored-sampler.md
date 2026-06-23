# Hardened Gym — Phase 2 (Factored Sampler) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the single coupled difficulty scalar with a **factored** sampler — each difficulty axis gets its own competence-paced clock — so the axes decouple (no shortcut learning, and the full-length-heavy-SHM corner becomes reachable), with a permanent easy floor and a deployment-distribution endpoint.

**Architecture:** New `src/alignair/gym/factored.py::FactoredCurriculum` — a per-axis-paced generalization of the existing `StratifiedCurriculum` (reuses its GenAIRR-faithful `_MIX` bin profiles and `_blend`). It implements the `AlignAIRGym` curriculum interface (`params`/`describe`/`stage`) so it drops in. Per-axis pacing (Platanios competence function) is driven by the Phase-1 `LatticeEvaluator` competence field via an axis↔cell mapping. Pure logic is unit-tested; the gym wiring is integration-tested small.

**Tech Stack:** Python 3.12, PyTorch, pytest 7.4.4, GenAIRR, Phase-1 `alignair.gym.instrument`.

## Global Constraints

- venv `./.venv/bin/python`; tests `./.venv/bin/python -m pytest <path> -v` from repo root.
- Reuse `StratifiedCurriculum._MIX` (per-axis easy/hard bin profiles) + `._blend` + `_lerp` from `gym/curriculum.py` — do NOT redefine the GenAIRR bin shapes (their hard profiles already retain easy-bin mass = built-in anti-forgetting).
- Output `params()` must match the dict shape `build_experiment` consumes (`mutation_count`, `end_loss_5/3`, `indel_count`, `ambiguous_count` as `(value,prob)` lists; `seq_error_rate`, `crop_prob`, `crop_len_min/max`, `crop_log_uniform`, `orient_prob` scalars).
- Difficulty stays a per-axis ramp (factored); NOT a single coupled scalar. Terminal (all paces=1) must equal the deployment distribution incl. the hard tail.
- Commit after each task. No `Co-Authored-By`/Claude mentions in commit messages.

## File Structure

- Create `src/alignair/gym/factored.py` — `FactoredCurriculum`, `axis_competence_from_field`.
- Modify `src/alignair/training/gym_trainer.py` — optional `advance_curriculum(field)` hook used when the curriculum is a `FactoredCurriculum`.
- Create `scripts/exp_ramp_vs_factored.py` — the decisive ramp-vs-mixture-vs-factored experiment.
- Tests under `tests/alignair/gym/`.

---

### Task 1: FactoredCurriculum core (per-axis paced params)

**Files:**
- Create: `src/alignair/gym/factored.py`
- Test: `tests/alignair/gym/test_factored.py`

**Interfaces:**
- Consumes: `StratifiedCurriculum._MIX`, `StratifiedCurriculum._blend`, `_lerp` (from `gym/curriculum.py`).
- Produces: `FactoredCurriculum(start_pace=0.1)` with:
  - `.axes -> tuple[str,...]` — the paced axes: `("mutation_count","end_loss_5","end_loss_3","indel_count","ambiguous_count","seq_error_rate","crop","orient")`.
  - `.pace: dict[str,float]` — per-axis pace ∈ [0,1] (mutable; starts `start_pace`).
  - `.params(p=None) -> dict` — each `_MIX` axis blended at ITS OWN pace; scalar axes (`seq_error_rate`,`crop`,`orient`) lerped at their own pace; shape matches `build_experiment`.
  - `.describe(p=None) -> str`, `.stage(p=None) -> int` (max pace × 5, capped 4) — the `AlignAIRGym` interface.
  - `.is_terminal() -> bool` — all paces ≥ 1.0.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/test_factored.py
from alignair.gym.factored import FactoredCurriculum
from alignair.gym.curriculum import StratifiedCurriculum


def _hard_mass(dist):           # prob mass on the hardest two bins of a (value,prob) list
    return sum(p for _, p in dist[-2:])


def test_advancing_one_axis_only_changes_that_axis():
    fc = FactoredCurriculum(start_pace=0.1)
    before = fc.params()
    fc.pace["mutation_count"] = 1.0          # unlock SHM only
    after = fc.params()
    # SHM distribution shifted toward hard...
    assert _hard_mass(after["mutation_count"]) > _hard_mass(before["mutation_count"])
    # ...while a DECOUPLED axis (indel_count) is unchanged
    assert after["indel_count"] == before["indel_count"]


def test_terminal_equals_full_stratified_distribution():
    fc = FactoredCurriculum()
    for a in fc.axes:
        fc.pace[a] = 1.0
    assert fc.is_terminal()
    # at all-paces-1 the SHM mixture equals StratifiedCurriculum at tau=1
    assert fc.params()["mutation_count"] == StratifiedCurriculum().params(1.0)["mutation_count"]


def test_params_shape_matches_build_experiment_keys():
    p = FactoredCurriculum().params()
    for k in ("mutation_count", "end_loss_5", "end_loss_3", "indel_count",
              "ambiguous_count", "seq_error_rate", "crop_prob", "crop_len_min",
              "crop_len_max", "orient_prob"):
        assert k in p
    assert isinstance(p["mutation_count"], list) and isinstance(p["mutation_count"][0], tuple)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_factored.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/factored.py
"""FactoredCurriculum: a per-axis competence-paced generalization of
StratifiedCurriculum. Each difficulty axis advances on its OWN clock (pace), so the
axes decouple — no shortcut learning, and the full-length-heavy-SHM corner becomes
reachable. Reuses StratifiedCurriculum's GenAIRR-faithful bin profiles; terminal
(all paces=1) equals the deployment distribution incl. the hard tail."""
from .curriculum import StratifiedCurriculum, _lerp

_MIX_AXES = ("mutation_count", "end_loss_5", "end_loss_3", "indel_count", "ambiguous_count")
_SCALAR_AXES = ("seq_error_rate", "crop", "orient")


class FactoredCurriculum:
    def __init__(self, start_pace: float = 0.1):
        self.axes = _MIX_AXES + _SCALAR_AXES
        self.pace = {a: float(start_pace) for a in self.axes}

    def _p(self, axis: str) -> float:
        return max(0.0, min(1.0, self.pace[axis]))

    def params(self, p=None) -> dict:
        mix = StratifiedCurriculum._MIX
        out = {k: StratifiedCurriculum._blend(*mix[k], self._p(k)) for k in _MIX_AXES}
        out.update({
            "seq_error_rate": _lerp(0.001, 0.01, self._p("seq_error_rate")),
            "crop_prob": _lerp(0.1, 0.5, self._p("crop")),
            "crop_len_min": 50, "crop_len_max": 576, "crop_log_uniform": True,
            "orient_prob": _lerp(0.1, 0.35, self._p("orient")),
        })
        return out

    def is_terminal(self) -> bool:
        return all(self.pace[a] >= 1.0 for a in self.axes)

    def stage(self, p=None) -> int:
        return min(4, int(max(self._p(a) for a in self.axes) * 5))

    def describe(self, p=None) -> str:
        worst = min(self.pace.values())
        hot = min(self.pace, key=self.pace.get)
        return (f"factored curriculum (per-axis pace; min={worst:.2f} on '{hot}'; "
                f"terminal={self.is_terminal()})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_factored.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/factored.py tests/alignair/gym/test_factored.py
git commit -m "feat(gym): add FactoredCurriculum (per-axis paced, decoupled difficulty)"
```

---

### Task 2: Per-axis competence pacing

**Files:**
- Modify: `src/alignair/gym/factored.py`
- Test: `tests/alignair/gym/test_factored_pacing.py`

**Interfaces:**
- Produces:
  - `FactoredCurriculum.advance(axis_competence: dict, threshold=0.7, step=0.1) -> list[str]` — Platanios competence pacing: for each axis, if `axis_competence[axis] >= threshold`, raise its pace by `step` (capped 1.0). Returns the list of axes that advanced. Axes absent from `axis_competence` do not move.
  - `axis_competence_from_field(field: dict) -> dict` — maps the Phase-1 `LatticeEvaluator` competence field (cell name → `{"S",...}`) to per-axis competence signals. Mapping: SHM-stressing axes (`mutation_count`) ← `heavy_shm_fulllen`; `crop` ← `fragment`; everything else ← `clean` (the overall-competence fallback until axis-isolated eval cells are added). Missing cells fall back to `clean`, then to 0.0.
- Note: the cell→axis mapping is an approximation pending axis-isolated lattice cells (documented); the MECHANISM (independent per-axis pace advanced by measured competence) is the Phase-2 deliverable.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/test_factored_pacing.py
from alignair.gym.factored import FactoredCurriculum, axis_competence_from_field


def test_advance_only_raises_axes_above_threshold():
    fc = FactoredCurriculum(start_pace=0.2)
    moved = fc.advance({"mutation_count": 0.9, "indel_count": 0.5}, threshold=0.7, step=0.1)
    assert moved == ["mutation_count"]
    assert abs(fc.pace["mutation_count"] - 0.3) < 1e-9
    assert abs(fc.pace["indel_count"] - 0.2) < 1e-9      # below threshold, unmoved


def test_pace_caps_at_one():
    fc = FactoredCurriculum(start_pace=0.95)
    fc.advance({"crop": 1.0}, threshold=0.7, step=0.2)
    assert fc.pace["crop"] == 1.0


def test_axis_competence_mapping_uses_hard_cells():
    field = {"clean": {"S": 0.8}, "heavy_shm_fulllen": {"S": 0.4}, "fragment": {"S": 0.5}}
    ac = axis_competence_from_field(field)
    assert ac["mutation_count"] == 0.4      # SHM axis <- heavy_shm_fulllen
    assert ac["crop"] == 0.5                # crop axis <- fragment
    assert ac["indel_count"] == 0.8         # fallback <- clean
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_factored_pacing.py -v`
Expected: FAIL (`advance`/`axis_competence_from_field` missing)

- [ ] **Step 3: Add pacing to `factored.py`**

```python
# append to src/alignair/gym/factored.py

    def advance(self, axis_competence: dict, threshold: float = 0.7,
                step: float = 0.1) -> list:
        moved = []
        for axis in self.axes:
            c = axis_competence.get(axis)
            if c is not None and c >= threshold and self.pace[axis] < 1.0:
                self.pace[axis] = min(1.0, self.pace[axis] + step)
                moved.append(axis)
        return moved


# cell that most stresses each axis; fallback "clean" = overall competence (until
# axis-isolated eval cells exist). Axes not listed use the fallback.
_AXIS_CELL = {"mutation_count": "heavy_shm_fulllen", "crop": "fragment"}


def axis_competence_from_field(field: dict, fallback_cell: str = "clean") -> dict:
    def _S(cell):
        if cell in field:
            return float(field[cell].get("S", 0.0))
        if fallback_cell in field:
            return float(field[fallback_cell].get("S", 0.0))
        return 0.0
    fb = _S(fallback_cell)
    out = {}
    for axis in (*_MIX_AXES, *_SCALAR_AXES):
        out[axis] = _S(_AXIS_CELL[axis]) if axis in _AXIS_CELL else fb
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_factored_pacing.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/factored.py tests/alignair/gym/test_factored_pacing.py
git commit -m "feat(gym): per-axis Platanios pacing + competence-field axis mapping"
```

---

### Task 3: Wire FactoredCurriculum into the gym + advance hook

**Files:**
- Modify: `src/alignair/training/gym_trainer.py`
- Test: `tests/alignair/integration/test_factored_training.py`

**Interfaces:**
- Produces: `GymTrainer.advance_curriculum(field: dict, threshold=0.7, step=0.1) -> list` — if `self.gym.curriculum` is a `FactoredCurriculum`, calls `axis_competence_from_field(field)` then `curriculum.advance(...)`, returns advanced axes (else `[]`). Used by a training loop to advance the factored curriculum from a Phase-1 lattice competence field. `AlignAIRGym` already accepts any curriculum with `params/describe/stage`, so a `FactoredCurriculum` drops in via the existing constructor.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/integration/test_factored_training.py
import pytest
import torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.factored import FactoredCurriculum
from alignair.training.gym_trainer import GymTrainer


def test_factored_curriculum_drives_training_and_advances():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    fc = FactoredCurriculum(start_pace=0.2)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0, curriculum=fc)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)
    history = trainer.fit(total_steps=10)          # runs with the factored curriculum
    assert len(history) == 10
    # a high competence field advances the SHM axis (its mapped cell clears threshold)
    moved = trainer.advance_curriculum({"heavy_shm_fulllen": {"S": 0.9}, "clean": {"S": 0.9},
                                        "fragment": {"S": 0.9}}, threshold=0.7, step=0.1)
    assert "mutation_count" in moved
    assert fc.pace["mutation_count"] > 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/integration/test_factored_training.py -v`
Expected: FAIL (`advance_curriculum` missing)

- [ ] **Step 3: Add `advance_curriculum` to `GymTrainer`**

Add the method to the `GymTrainer` class in `src/alignair/training/gym_trainer.py`:

```python
    def advance_curriculum(self, field: dict, threshold: float = 0.7,
                           step: float = 0.1) -> list:
        """Advance a FactoredCurriculum from a Phase-1 lattice competence field.
        No-op (returns []) for non-factored curricula."""
        from ..gym.factored import FactoredCurriculum, axis_competence_from_field
        cur = self.gym.curriculum
        if not isinstance(cur, FactoredCurriculum):
            return []
        return cur.advance(axis_competence_from_field(field), threshold=threshold, step=step)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/integration/test_factored_training.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/training/gym_trainer.py tests/alignair/integration/test_factored_training.py
git commit -m "feat(gym): GymTrainer.advance_curriculum drives FactoredCurriculum from competence field"
```

---

### Task 4: Decisive experiment (ramp vs mixture vs factored) + smoke

**Files:**
- Create: `scripts/exp_ramp_vs_factored.py`
- Test: `tests/alignair/gym/test_exp_smoke.py`

**Interfaces:**
- Produces: a script that trains three short identical models — `Curriculum` (scalar ramp), `StratifiedCurriculum` (coupled mixture), and `FactoredCurriculum` (competence-paced, advanced from the lattice each eval) — and prints each one's final FrozenLattice competence field, so the operator can confirm the factored arm reaches the heavy-SHM-full-length cell. A smoke test runs the experiment's core for a couple of steps per arm on a tiny model and asserts it returns a competence field per arm.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/test_exp_smoke.py
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("GenAIRR")
from scripts.exp_ramp_vs_factored import run_arm


def test_run_arm_returns_competence_field():
    field = run_arm("factored", steps=2, n_per_cell=4, batch_size=8, seed=0)
    assert "heavy_shm_fulllen" in field
    assert 0.0 <= field["heavy_shm_fulllen"]["S"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_exp_smoke.py -v`
Expected: FAIL (`scripts.exp_ramp_vs_factored` missing)

- [ ] **Step 3: Write the experiment script**

```python
# scripts/exp_ramp_vs_factored.py
"""Decisive experiment: scalar ramp vs coupled mixture vs FACTORED per-axis pacing.
Trains three short identical models and reports each one's FrozenLattice competence
field. The factored arm is expected to reach the heavy_shm_fulllen corner the coupled
ramp structurally excludes. Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_ramp_vs_factored.py --steps 4000
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.curriculum import Curriculum, StratifiedCurriculum
from alignair.gym.factored import FactoredCurriculum
from alignair.training.gym_trainer import GymTrainer
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator

_DC = gdata.HUMAN_IGK_OGRDB     # V/J only -> fast for the smoke/dev run; use IGH for the real run


def _curriculum(arm):
    return {"ramp": Curriculum(), "mixture": StratifiedCurriculum(),
            "factored": FactoredCurriculum(start_pace=0.1)}[arm]


def run_arm(arm: str, steps: int, n_per_cell: int, batch_size: int = 16, seed: int = 0,
            device=None) -> dict:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(_DC)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    cur = _curriculum(arm)
    gym = AlignAIRGym([_DC], rs, n=batch_size * 4, seed=seed, curriculum=cur)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=batch_size, device=device)
    lat = FrozenLattice.standard(seed=seed)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(), [_DC], device=device)
    chunk = max(1, steps // 4)
    done = 0
    while done < steps:
        trainer.fit(total_steps=min(chunk, steps - done), global_total=steps, progress=False)
        done += chunk
        if isinstance(cur, FactoredCurriculum):
            trainer.advance_curriculum(ev.eval_all(n_per_cell=n_per_cell))
    return ev.eval_all(n_per_cell=n_per_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-per-cell", type=int, default=500)
    args = ap.parse_args()
    for arm in ("ramp", "mixture", "factored"):
        field = run_arm(arm, args.steps, args.n_per_cell)
        print(f"\n=== {arm} ===")
        for name, v in field.items():
            print(f"  {name:22s} S={v['S']:.3f}  [{v['lo']:.3f},{v['hi']:.3f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_exp_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run full Phase-2 suite + commit**

Run: `./.venv/bin/python -m pytest tests/alignair/gym/test_factored.py tests/alignair/gym/test_factored_pacing.py tests/alignair/gym/test_exp_smoke.py tests/alignair/integration/test_factored_training.py -v`
Expected: PASS

```bash
git add scripts/exp_ramp_vs_factored.py tests/alignair/gym/test_exp_smoke.py
git commit -m "feat(gym): ramp-vs-mixture-vs-factored decisive experiment + smoke"
```

---

## Self-Review

**Spec coverage (Phase 2):** factored per-axis sampler → Task 1; competence pacing (Platanios) → Task 2; easy-floor/anti-forgetting → inherited from `_MIX` hard profiles (documented in Task 1); deployment endpoint (terminal=deployment) → Task 1 `is_terminal`/terminal test; gym wiring → Task 3; decisive ramp-vs-mixture-vs-factored experiment → Task 4. ALP-GMM/regret (the ~25% targeting component) is explicitly Phase 4, not here. ✓

**Placeholder scan:** the axis↔cell mapping in Task 2 is an explicit, documented approximation (fallback to `clean`) pending axis-isolated lattice cells — not a silent gap; the mechanism is complete. No TBDs.

**Type consistency:** `FactoredCurriculum.params()` returns the `build_experiment` shape (T1); `advance(axis_competence)` consumes the dict from `axis_competence_from_field(field)` (T2) where `field` is the Phase-1 `LatticeEvaluator.eval_all()` output; `GymTrainer.advance_curriculum(field)` (T3) bridges them; the experiment (T4) uses all three. ✓

## Notes for the implementer

- The decisive experiment (Task 4) uses IGK for speed in dev/smoke; switch `_DC` to `HUMAN_IGH_OGRDB` and raise `--steps`/`--n-per-cell` for the real run that decides ramp-vs-factored.
- Phase 3 (Kendall–curriculum coupling fix) consumes this: it adds σ² floors / GradNorm so the factored sampler's hard-axis concentration isn't cancelled by the loss balancer.
