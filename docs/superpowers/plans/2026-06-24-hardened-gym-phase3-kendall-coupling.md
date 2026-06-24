# Hardened Gym — Phase 3 (Kendall–Curriculum Coupling Fix) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Stop the Kendall uncertainty weighting from abandoning the hard heads (V-call, junction, coords) as the curriculum raises difficulty — the mechanism both experts named as the likely cause of the heavy-SHM-V weakness.

**Architecture:** Two levers in the loss + one trainer hook. (1) Per-head **σ² floor** — give the protected heads a tighter `max_log_var` so their precision weight `exp(-log_var)` cannot collapse to the global floor. (2) **σ² freeze during transients** — after a difficulty advance, hold all `log_var`s constant for a short window so Kendall doesn't react to the post-shift loss spike by down-weighting the newly-hard head. Trainer freezes σ² for K steps whenever the factored curriculum advances. GradNorm/PCGrad trunk balancing is **Phase 3b** (needs training to validate; out of scope here).

**Tech Stack:** Python 3.12, PyTorch, pytest. Files: `src/alignair/nn/weighting.py`, `src/alignair/losses/dnalignair_loss.py`, `src/alignair/training/gym_trainer.py`.

## Global Constraints

- venv `./.venv/bin/python`; tests from repo root.
- `weight = exp(-clamp(log_var, min, max))`; higher `log_var` → lower weight. A TIGHTER `max_log_var` raises the minimum weight (floor): `max_log_var=1.5 → min weight exp(-1.5)≈0.223` vs default `exp(-3)≈0.050`.
- Default behavior must be preserved when the new knobs are off (`protected=None` keeps a sensible default set; `sigma_freeze_steps=0` disables freezing).
- Commit per task; no Claude mentions in commit messages.

---

### Task 1: UncertaintyWeight freeze support

**Files:** `src/alignair/nn/weighting.py`; test `tests/alignair/nn/test_weighting.py`

**Interfaces:** `UncertaintyWeight.set_frozen(frozen: bool)`; when frozen, `forward()`/`penalty()` use a DETACHED clamped log_var so no gradient reaches it (it can't move) while the current weight value is still applied.

- [ ] **Step 1: failing test**

```python
# tests/alignair/nn/test_weighting.py
import torch
from alignair.nn.weighting import UncertaintyWeight


def test_frozen_log_var_does_not_update():
    w = UncertaintyWeight(initial_value=1.0)
    opt = torch.optim.SGD(w.parameters(), lr=1.0)
    w.set_frozen(True)
    before = float(w.log_var)
    # a loss that would normally push log_var: raw_loss*weight + penalty
    loss = 5.0 * w() + w.penalty()
    opt.zero_grad(); loss.backward(); opt.step()
    assert float(w.log_var) == before            # frozen -> unchanged
    assert w.log_var.grad is None or float(w.log_var.grad) == 0.0


def test_unfrozen_log_var_updates():
    w = UncertaintyWeight(initial_value=1.0)
    opt = torch.optim.SGD(w.parameters(), lr=1.0)
    before = float(w.log_var)
    loss = 5.0 * w() + w.penalty()
    opt.zero_grad(); loss.backward(); opt.step()
    assert float(w.log_var) != before            # learns normally
```

- [ ] **Step 2:** run → FAIL (`set_frozen` missing). `./.venv/bin/python -m pytest tests/alignair/nn/test_weighting.py -v`

- [ ] **Step 3: implement**

```python
# in UncertaintyWeight.__init__ add:
        self._frozen = False

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = bool(frozen)

    def _s(self) -> torch.Tensor:
        s = torch.clamp(self.log_var, self.min_log_var, self.max_log_var)
        return s.detach() if self._frozen else s
```
Replace `forward()` body with `return torch.exp(-self._s())` and `penalty()` body with `return 0.5 * self._s()`.

- [ ] **Step 4:** run → PASS (2). **Step 5:** commit `feat(loss): UncertaintyWeight.set_frozen (hold log_var during transients)`

---

### Task 2: Per-head σ² floor + freeze-all in DNAlignAIRLoss

**Files:** `src/alignair/losses/dnalignair_loss.py`; test `tests/alignair/losses/test_loss_protection.py`

**Interfaces:** `DNAlignAIRLoss(..., protected_max_log_var=1.5, protected=None)`. Protected heads default to `{"v_match"} ∪ {f"{g}_germline"} ∪ ({f"{g}_boundary"} if use_boundary)` — the V-call, coordinate, and junction heads. Each protected head's `UncertaintyWeight` gets `max_log_var=protected_max_log_var` (a higher weight floor); others keep `3.0`. `set_log_vars_frozen(frozen)` toggles freeze on all heads. `protected_heads` property exposes the set.

- [ ] **Step 1: failing test**

```python
# tests/alignair/losses/test_loss_protection.py
from alignair.losses.dnalignair_loss import DNAlignAIRLoss


def test_protected_heads_have_tighter_cap_and_higher_weight_floor():
    loss = DNAlignAIRLoss(has_d=True, protected_max_log_var=1.5)
    assert "v_match" in loss.protected_heads
    assert "v_germline" in loss.protected_heads and "d_germline" in loss.protected_heads
    assert loss.weights["v_match"].max_log_var == 1.5         # protected -> tighter
    assert loss.weights["region"].max_log_var == 3.0          # unprotected -> default
    # tighter cap => strictly higher minimum precision weight (can't be abandoned)
    import math
    assert math.exp(-1.5) > math.exp(-3.0)


def test_freeze_toggles_all_heads():
    loss = DNAlignAIRLoss(has_d=False)
    loss.set_log_vars_frozen(True)
    assert all(w._frozen for w in loss.weights.values())
    loss.set_log_vars_frozen(False)
    assert not any(w._frozen for w in loss.weights.values())
```

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: implement** — in `DNAlignAIRLoss.__init__`, after `names` is built and before `self.weights`:

```python
        genes_l = ["v", "j"] + (["d"] if has_d else [])
        default_protected = {"v_match"} | {f"{g}_germline" for g in genes_l}
        if use_boundary:
            default_protected |= {f"{g}_boundary" for g in genes_l}
        self._protected = set(default_protected if protected is None else protected)
        self.weights = nn.ModuleDict({
            n: UncertaintyWeight(max_log_var=(protected_max_log_var if n in self._protected else 3.0))
            for n in names})
```
Add to the signature: `protected_max_log_var: float = 1.5, protected=None`. Add:
```python
    @property
    def protected_heads(self):
        return set(self._protected)

    def set_log_vars_frozen(self, frozen: bool) -> None:
        for w in self.weights.values():
            w.set_frozen(frozen)
```

- [ ] **Step 4:** run → PASS (2). **Step 5:** commit `feat(loss): per-head sigma floor (protect V-call/coords/junction) + freeze-all`

---

### Task 3: Trainer freezes σ² across difficulty transients

**Files:** `src/alignair/training/gym_trainer.py`; test `tests/alignair/integration/test_sigma_freeze.py`

**Interfaces:** `GymTrainer(..., sigma_freeze_steps=0)`. `freeze_uncertainty(steps=None)` sets a countdown and freezes the loss's log_vars. `advance_curriculum(...)` — when axes advance — triggers `freeze_uncertainty()`. `fit()` decrements the countdown each step and unfreezes at 0. Default `sigma_freeze_steps=0` => no freezing (behavior unchanged).

- [ ] **Step 1: failing test**

```python
# tests/alignair/integration/test_sigma_freeze.py
import pytest, torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.gym.factored import FactoredCurriculum
from alignair.training.gym_trainer import GymTrainer


def test_advance_triggers_sigma_freeze():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128))
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    fc = FactoredCurriculum(start_pace=0.2)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=0, curriculum=fc)
    trainer = GymTrainer(model, loss_fn, rs, gym, batch_size=8, sigma_freeze_steps=5)
    moved = trainer.advance_curriculum({"heavy_shm_fulllen": {"S": 0.9}, "clean": {"S": 0.9}})
    assert moved and trainer._freeze_remaining == 5
    assert all(w._frozen for w in loss_fn.weights.values())   # frozen now
    trainer.fit(total_steps=6, progress=False)                # 5 frozen steps then unfreeze
    assert trainer._freeze_remaining == 0
    assert not any(w._frozen for w in loss_fn.weights.values())
```

- [ ] **Step 2:** run → FAIL.

- [ ] **Step 3: implement** — in `GymTrainer.__init__` add param `sigma_freeze_steps=0`, store it and `self._freeze_remaining = 0`. Add:

```python
    def freeze_uncertainty(self, steps: int | None = None) -> None:
        self._freeze_remaining = self.sigma_freeze_steps if steps is None else steps
        self.loss_fn.set_log_vars_frozen(self._freeze_remaining > 0)
```
In `advance_curriculum`, after computing `moved` and before `return moved`:
```python
        if moved and self.sigma_freeze_steps > 0:
            self.freeze_uncertainty()
```
In `fit()`, right after `step += 1`, add the countdown:
```python
            if self._freeze_remaining > 0:
                self._freeze_remaining -= 1
                if self._freeze_remaining == 0:
                    self.loss_fn.set_log_vars_frozen(False)
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `feat(train): freeze sigma across difficulty transients (curriculum advance)`

---

### Task 4: Phase-3b note (GradNorm/PCGrad deferred)

**Files:** `docs/architecture/dnalignair_coordinate_redesign.md` or the gym spec — append a short "Phase 3b" note: trunk gradient balancing (GradNorm/PCGrad) is the second Kendall-coupling lever (stops saturated easy heads starving the trunk); deferred because its benefit can only be validated by a training run (soft-DP-bound today), and the σ²-floor + freeze address the primary abandonment mechanism. Add after the σ²-floor lands and the gym can measure whether it suffices.

- [ ] **Step 1:** append the note. **Step 2:** commit `docs(gym): note GradNorm/PCGrad as Phase 3b (trunk balancing)`

---

## Self-Review

Spec coverage: σ² floor → T2; σ² freeze → T1+T3; trainer transient hook → T3; GradNorm/PCGrad explicitly deferred with rationale → T4. Schedule-staggering (scheduled-sampling/EMA/SGDR) is noted in the spec but deferred with GradNorm to Phase 3b (also training-validation-bound). Types: `set_frozen`(T1) ⊂ `set_log_vars_frozen`(T2) ⊂ `freeze_uncertainty`(T3); `protected`/`protected_max_log_var` consistent T2↔spec. No placeholders.
