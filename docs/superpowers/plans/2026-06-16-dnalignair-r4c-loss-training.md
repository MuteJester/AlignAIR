# DNAlignAIR R4c — Composite Loss + Verbose Training + Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train the unified `DNAlignAIR` model end-to-end on the gym with a Kendall-weighted composite loss over every head, a **verbose** training loop (live tqdm with loss components + curriculum stage), the germline-coordinate teacher-forcing term, and an eval harness.

**Architecture:** `DNAlignAIRLoss` sums per-task terms (orientation CE, region/state per-position CE, multi-label allele BCE, scalar regressions), each scaled by a learned uncertainty weight, and optionally a germline-coordinate CE computed by teacher-forcing the true allele's germline reps. `GymTrainer` drives the gym (ramping curriculum), shows a tqdm bar with live metrics, applies AMP + constraints, and logs the curriculum stage. An `evaluate` routine reports boundary/germline deviation, call agreement, per-position accuracy, and count MAEs.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), GenAIRR 2.2.0, tqdm, pytest. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs have NO `__init__.py`; unique test basenames. GenAIRR-dependent tests `pytest.importorskip("GenAIRR")`.

---

## File structure (R4c)

```
src/alignair/losses/dnalignair_loss.py   DNAlignAIRLoss
src/alignair/training/germline_tf.py     compute_germline_logits (teacher-forced)
src/alignair/training/gym_trainer.py     GymTrainer (verbose) + evaluate
tests/alignair/losses/test_dnalignair_loss.py
tests/alignair/training/test_germline_tf.py
tests/alignair/integration/test_gym_training.py
```

---

## Task 1: `losses/dnalignair_loss.py` — composite loss (dense + matching + scalars)

**Files:** Create `src/alignair/losses/dnalignair_loss.py`; Test `tests/alignair/losses/test_dnalignair_loss.py`

Terms (each Kendall-weighted via `UncertaintyWeight`): orientation CE, region per-position CE
(ignore_index=-100), state per-position CE (ignore -100), per-gene multi-label allele BCE
(`multilabel_match_loss`), scalar losses (noise/indel L1, mutation MSE, productive BCE). Germline term is
added in Task 3.

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.losses.dnalignair_loss import DNAlignAIRLoss


def _outputs(B, L, nV, nJ, nD):
    return {
        "orientation_logits": torch.randn(B, 4, requires_grad=True),
        "region_logits": torch.randn(B, L, 8, requires_grad=True),
        "state_logits": torch.randn(B, L, 4, requires_grad=True),
        "noise_count": torch.rand(B, 1, requires_grad=True),
        "mutation_rate": torch.rand(B, 1, requires_grad=True),
        "indel_count": torch.rand(B, 1, requires_grad=True),
        "productive": torch.rand(B, 1, requires_grad=True),
        "match": {"V": torch.randn(B, nV, requires_grad=True),
                  "J": torch.randn(B, nJ, requires_grad=True),
                  "D": torch.randn(B, nD, requires_grad=True)},
    }


def _batch(B, L, nV, nJ, nD):
    region = torch.randint(0, 8, (B, L)); region[:, L // 2:] = -100
    state = torch.randint(0, 4, (B, L)); state[:, L // 2:] = -100
    vA = torch.zeros(B, nV); vA[:, 0] = 1.0
    jA = torch.zeros(B, nJ); jA[:, 0] = 1.0
    dA = torch.zeros(B, nD); dA[:, 0] = 1.0
    return {
        "orientation_id": torch.zeros(B, dtype=torch.long),
        "region_labels": region, "state_labels": state,
        "noise_count": torch.rand(B, 1), "mutation_rate": torch.rand(B, 1),
        "indel_count": torch.rand(B, 1), "productive": torch.ones(B, 1),
        "v_allele": vA, "j_allele": jA, "d_allele": dA,
    }


def test_loss_finite_backprops_and_has_components():
    B, L, nV, nJ, nD = 3, 12, 5, 3, 4
    loss_fn = DNAlignAIRLoss(has_d=True)
    out, batch = _outputs(B, L, nV, nJ, nD), _batch(B, L, nV, nJ, nD)
    total, comp = loss_fn(out, batch)
    assert torch.isfinite(total)
    total.backward()
    assert out["region_logits"].grad is not None
    for k in ("orientation", "region", "state", "v_match", "j_match", "d_match",
              "noise", "mutation", "indel", "productive"):
        assert k in comp


def test_loss_no_d_omits_d_match():
    B, L, nV, nJ = 2, 10, 4, 2
    loss_fn = DNAlignAIRLoss(has_d=False)
    out = _outputs(B, L, nV, nJ, 1)
    del out["match"]["D"]
    batch = _batch(B, L, nV, nJ, 1)
    total, comp = loss_fn(out, batch)
    assert "d_match" not in comp and torch.isfinite(total)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Composite Kendall-weighted loss for the unified DNAlignAIR model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.weighting import UncertaintyWeight
from ..nn.matching import multilabel_match_loss

IGNORE = -100


class DNAlignAIRLoss(nn.Module):
    def __init__(self, has_d: bool = True):
        super().__init__()
        self.has_d = has_d
        names = ["orientation", "region", "state", "v_match", "j_match",
                 "noise", "mutation", "indel", "productive"]
        if has_d:
            names += ["d_match"]
        # germline weights are created lazily in Task 3 if germline_logits provided
        names += ["v_germline", "j_germline"] + (["d_germline"] if has_d else [])
        self.weights = nn.ModuleDict({n: UncertaintyWeight() for n in names})

    def forward(self, outputs: dict, batch: dict, germline_logits: dict | None = None):
        comp = {}

        def add(name, raw):
            return raw * self.weights[name]()

        orientation = F.cross_entropy(outputs["orientation_logits"], batch["orientation_id"])
        region = F.cross_entropy(outputs["region_logits"].reshape(-1, outputs["region_logits"].shape[-1]),
                                 batch["region_labels"].reshape(-1), ignore_index=IGNORE)
        state = F.cross_entropy(outputs["state_logits"].reshape(-1, outputs["state_logits"].shape[-1]),
                                batch["state_labels"].reshape(-1), ignore_index=IGNORE)

        genes = ["v", "j"] + (["d"] if self.has_d else [])
        match_terms = {}
        for g in genes:
            match_terms[g] = multilabel_match_loss(outputs["match"][g.upper()], batch[f"{g}_allele"])

        noise = F.l1_loss(outputs["noise_count"], batch["noise_count"])
        mutation = F.mse_loss(outputs["mutation_rate"], batch["mutation_rate"])
        indel = F.l1_loss(outputs["indel_count"], batch["indel_count"])
        productive = F.binary_cross_entropy(outputs["productive"].clamp(1e-7, 1 - 1e-7),
                                            batch["productive"])

        total = (add("orientation", orientation) + add("region", region) + add("state", state)
                 + add("noise", noise) + add("mutation", mutation) + add("indel", indel)
                 + add("productive", productive))
        comp.update({"orientation": orientation.detach(), "region": region.detach(),
                     "state": state.detach(), "noise": noise.detach(),
                     "mutation": mutation.detach(), "indel": indel.detach(),
                     "productive": productive.detach()})
        for g in genes:
            total = total + add(f"{g}_match", match_terms[g])
            comp[f"{g}_match"] = match_terms[g].detach()

        if germline_logits is not None:
            for g in genes:
                if g not in germline_logits:
                    continue
                sl, el = germline_logits[g]
                gs = batch[f"{g}_germline_start"].clamp(min=0, max=sl.shape[-1] - 1)
                ge = (batch[f"{g}_germline_end"] - 1).clamp(min=0, max=el.shape[-1] - 1)
                gl = F.cross_entropy(sl, gs) + F.cross_entropy(el, ge)
                total = total + add(f"{g}_germline", gl)
                comp[f"{g}_germline"] = gl.detach()

        total = total + sum(w.regularization() for w in self.weights.values())
        comp["total"] = total.detach()
        return total, comp

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for w in self.weights.values():
            w.apply_constraints()
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/losses/dnalignair_loss.py tests/alignair/losses/test_dnalignair_loss.py
git commit -m "feat(alignair): DNAlignAIR composite Kendall-weighted loss"
```

---

## Task 2: `training/gym_trainer.py` — verbose training loop

**Files:** Create `src/alignair/training/gym_trainer.py`; Test `tests/alignair/integration/test_gym_training.py`

`GymTrainer.fit(total_steps, batch_size)` ramps the gym curriculum, runs forward → loss → step, applies
constraints, and shows a **tqdm** bar with live total loss + key components + curriculum stage; logs the
stage each ramp. (Germline term wired in Task 3.)

- [ ] **Step 1: Write the integration test**

```python
import logging
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.losses.dnalignair_loss import DNAlignAIRLoss
from alignair.gym.gym import AlignAIRGym
from alignair.training.gym_trainer import GymTrainer


def test_train_few_steps_loss_decreases(caplog):
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)  # V/J only -> smaller/faster
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, seed=0)
    trainer = GymTrainer(model, loss_fn, rs, gym, lr=1e-3, batch_size=8)
    with caplog.at_level(logging.INFO):
        history = trainer.fit(total_steps=20)
    assert len(history) == 20
    first = sum(h["total"] for h in history[:5]) / 5
    last = sum(h["total"] for h in history[-5:]) / 5
    assert last < first, f"loss did not decrease: {first} -> {last}"
    assert any("curriculum stage" in m.lower() for m in caplog.messages)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""GymTrainer: verbose curriculum training loop for the unified DNAlignAIR model."""
import itertools
import logging

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..gym.collate import gym_collate

logger = logging.getLogger(__name__)


class GymTrainer:
    def __init__(self, model, loss_fn, reference_set, gym, lr=1e-3, batch_size=16,
                 device=None, grad_clip=10.0, refresh_reference_every=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.reference_set = reference_set
        self.gym = gym
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.refresh_reference_every = refresh_reference_every
        self.has_d = reference_set.has_d
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def fit(self, total_steps: int) -> list:
        self.model.train()
        loader = DataLoader(self.gym, batch_size=self.batch_size,
                            collate_fn=lambda b: gym_collate(b, self.reference_set, self.has_d))
        history = []
        ref_emb = None
        bar = tqdm(total=total_steps, desc="gym-train")
        step = 0
        while step < total_steps:
            self.gym.set_progress(step / max(total_steps - 1, 1))
            for batch in loader:
                if step >= total_steps:
                    break
                batch = self._to_device(batch)
                if ref_emb is None or step % self.refresh_reference_every == 0:
                    ref_emb = self.model.encode_reference(self.reference_set)
                out = self.model(batch["tokens"], batch["mask"], ref_emb)
                total, comp = self.loss_fn(out, batch)
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.loss_fn.apply_constraints()
                logs = {k: float(v.cpu()) for k, v in comp.items()}
                history.append(logs)
                bar.update(1)
                bar.set_postfix(loss=f"{logs['total']:.3f}", region=f"{logs['region']:.3f}",
                                stage=self.gym.curriculum.stage(self.gym._p) + 1)
                step += 1
        bar.close()
        return history
```

- [ ] **Step 4: Run — expect PASS** (the model must learn — loss decreases over 20 steps). If it doesn't
decrease under seeding, raise to 30 steps or lr to 2e-3.
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/integration/test_gym_training.py -q`

- [ ] **Step 5: Commit**
```bash
git add src/alignair/training/gym_trainer.py tests/alignair/integration/test_gym_training.py
git commit -m "feat(alignair): verbose GymTrainer (curriculum + tqdm live metrics)"
```

---

## Task 3: germline teacher-forcing term

**Files:** Create `src/alignair/training/germline_tf.py`; Modify `src/alignair/training/gym_trainer.py`;
Test `tests/alignair/training/test_germline_tf.py`

`compute_germline_logits` extracts each gene's true segment from the backbone reps and aligns it to the
true allele's per-position germline reps (gathered by the first positive in the multi-hot label).

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.training.germline_tf import compute_germline_logits


class _Gene:
    def __init__(self, n):
        self.names = [f"a{i}" for i in range(n)]
        self.index = {nm: i for i, nm in enumerate(self.names)}
        self.sequences = ["ACGT" * 10 for _ in range(n)]


class _RS:
    def __init__(self):
        self.genes = {"V": _Gene(4), "J": _Gene(3)}
        self.has_d = False

    def gene(self, g):
        return self.genes[g.upper()]


def test_compute_germline_logits_shapes():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=2, nhead=4, dim_feedforward=64)
    model = DNAlignAIR(cfg)
    rs = _RS()
    ref_emb = model.encode_reference(rs)
    B, L = 2, 16
    reps = torch.randn(B, L, cfg.d_model)
    mask = torch.ones(B, L, dtype=torch.bool)
    region_labels = torch.zeros(B, L, dtype=torch.long)
    from alignair.nn.region_head import REGION_INDEX
    region_labels[:, 2:8] = REGION_INDEX["V"]
    region_labels[:, 8:12] = REGION_INDEX["J"]
    batch = {"region_labels": region_labels,
             "v_allele": torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
             "j_allele": torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])}
    gl = compute_germline_logits(model, reps, mask, batch, ref_emb, has_d=False)
    Lg_v = ref_emb["V"]["pos_reps"].shape[1]
    assert gl["v"][0].shape == (B, Lg_v) and gl["v"][1].shape == (B, Lg_v)
    assert "j" in gl
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `germline_tf.py`**

```python
"""Teacher-forced germline-coordinate logits for training.

For each gene, gather the true segment reps (from the GT region labels) and the
true allele's per-position germline reps (first positive in the multi-hot label),
then run the model's germline aligner."""
import torch

from ..core.dnalignair import extract_segment


def compute_germline_logits(model, reps, mask, batch, ref_emb, has_d: bool):
    genes = ["v", "j"] + (["d"] if has_d else [])
    out = {}
    for g in genes:
        seg, seg_mask = extract_segment(reps, mask, batch["region_labels"], g.upper())
        # true allele index per sample = first positive in the multi-hot label (fallback 0)
        multihot = batch[f"{g}_allele"]                       # (B, K)
        has_pos = multihot.sum(dim=1) > 0
        idx = multihot.argmax(dim=1)                          # (B,)
        idx = torch.where(has_pos, idx, torch.zeros_like(idx))
        germ_reps = ref_emb[g.upper()]["pos_reps"][idx]       # (B, Lg, d)
        germ_mask = ref_emb[g.upper()]["pos_mask"][idx]       # (B, Lg)
        out[g] = model.germline_coords(seg, seg_mask, germ_reps, germ_mask)
    return out
```

- [ ] **Step 4: Wire into the trainer** — in `gym_trainer.py`, import and use it so the loss gets the
germline term. Change the forward/loss block in `fit`:
```python
                out = self.model(batch["tokens"], batch["mask"], ref_emb)
                from .germline_tf import compute_germline_logits
                germline_logits = compute_germline_logits(
                    self.model, out["reps"], batch["mask"], batch, ref_emb, self.has_d)
                total, comp = self.loss_fn(out, batch, germline_logits=germline_logits)
```

- [ ] **Step 5: Run — expect PASS** (1 passed) and re-run the training integration test (it now includes
the germline term):
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_germline_tf.py tests/alignair/integration/test_gym_training.py -q`
Expected: PASS (loss still decreases). If the germline term destabilizes early training, the Kendall
weight will down-weight it; if the test flakes, raise total_steps to 30.

- [ ] **Step 6: Commit**
```bash
git add src/alignair/training/germline_tf.py src/alignair/training/gym_trainer.py tests/alignair/training/test_germline_tf.py
git commit -m "feat(alignair): teacher-forced germline-coordinate loss term in training"
```

---

## Task 4: eval harness + exports

**Files:** Modify `src/alignair/training/gym_trainer.py` (add `evaluate`); Modify
`src/alignair/training/__init__.py`, `src/alignair/losses/__init__.py`;
Test `tests/alignair/integration/test_gym_training.py` (add)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_evaluate_reports_metrics():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    loss_fn = DNAlignAIRLoss(has_d=rs.has_d)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=16, seed=3)
    trainer = GymTrainer(model, loss_fn, rs, gym, batch_size=8)
    metrics = trainer.evaluate(n_batches=2)
    for k in ("region_acc", "v_call_agreement", "loss"):
        assert k in metrics
    assert 0.0 <= metrics["region_acc"] <= 1.0
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `evaluate`** — append to `GymTrainer`:
```python
    @torch.no_grad()
    def evaluate(self, n_batches: int = 4) -> dict:
        self.model.eval()
        from ..gym.collate import gym_collate
        loader = DataLoader(self.gym, batch_size=self.batch_size,
                            collate_fn=lambda b: gym_collate(b, self.reference_set, self.has_d))
        ref_emb = self.model.encode_reference(self.reference_set)
        tot_loss, region_correct, region_total = 0.0, 0, 0
        v_hits, v_total = 0, 0
        nb = 0
        for batch in loader:
            if nb >= n_batches:
                break
            batch = self._to_device(batch)
            out = self.model(batch["tokens"], batch["mask"], ref_emb)
            total, _ = self.loss_fn(out, batch)
            tot_loss += float(total.cpu())
            # region accuracy over valid positions
            valid = batch["region_labels"] != -100
            pred = out["region_logits"].argmax(-1)
            region_correct += int(((pred == batch["region_labels"]) & valid).sum().cpu())
            region_total += int(valid.sum().cpu())
            # top-1 V call agreement with the true multi-hot set
            v_pred = out["match"]["V"].argmax(-1)
            v_hits += int(batch["v_allele"][torch.arange(v_pred.shape[0]), v_pred].sum().cpu())
            v_total += v_pred.shape[0]
            nb += 1
        self.model.train()
        return {
            "loss": tot_loss / max(nb, 1),
            "region_acc": region_correct / max(region_total, 1),
            "v_call_agreement": v_hits / max(v_total, 1),
        }
```

- [ ] **Step 4: Run — expect PASS** (1 passed).

- [ ] **Step 5: Add exports** — append to `src/alignair/training/__init__.py`:
```python
from .gym_trainer import GymTrainer
```
and `"GymTrainer"` to `__all__`. Append to `src/alignair/losses/__init__.py`:
```python
from .dnalignair_loss import DNAlignAIRLoss
```
and `"DNAlignAIRLoss"` to `__all__`.

- [ ] **Step 6: Run the full suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 7: Commit**
```bash
git add src/alignair/training/gym_trainer.py src/alignair/training/__init__.py src/alignair/losses/__init__.py tests/alignair/integration/test_gym_training.py
git commit -m "feat(alignair): eval harness (region acc, call agreement) + exports"
```

---

## Self-Review

**Spec coverage (R4 design §5):** composite Kendall-weighted loss (orientation/region/state/match/
scalars) → Task 1; verbose curriculum training loop (tqdm + stage logging) → Task 2; germline-coordinate
teacher-forced term → Task 3; eval harness + exports → Task 4. "Done when the full model trains on the
gym and metrics improve" → Task 2 (loss decreases) + Task 4 (metrics).

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `DNAlignAIRLoss(has_d).forward(outputs, batch, germline_logits=None) ->
(total, comp)` consistent (Tasks 1/3); `compute_germline_logits(model, reps, mask, batch, ref_emb, has_d)
-> {gene: (start_logits, end_logits)}` consumed by the loss's germline branch (Tasks 3); `GymTrainer(
model, loss_fn, reference_set, gym, ...)` / `.fit(total_steps)` / `.evaluate(n_batches)` consistent
(Tasks 2/4); uses `model.encode_reference`, `model.germline_coords`, `extract_segment`, `gym_collate`,
`UncertaintyWeight`, `multilabel_match_loss` from earlier phases.

**Known notes:** germline teacher-forcing uses the TRUE region labels + first-positive allele; inference
will use predicted regions + top-1 match. Reference embeddings are re-encoded per `refresh_reference_every`
steps (default 1; raise to amortize cost on large references). Forward-only orientation (target 0) for now.
The training test asserts the loss *decreases* — the real convergence quality is the empirical work that
follows once the full loop runs. Verbose: tqdm postfix (loss/region/stage) + curriculum stage logs.
```
