# XAttnAligner Training Losses Implementation Plan (LLM-Encoder Aligner — Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `xattn_losses` — the multi-task supervised loss that trains `XAttnAligner` on a GenAIRR gym batch (orientation CE + region CE + allele set-NCE + germline-span CE), and prove the model trains (loss decreases) on the gym.

**Architecture:** A teacher-forced training forward: encode with the TRUE orientation, tag regions, and for each gene build a teacher-forced candidate pool (true allele + sibling + random via `build_candidates`), run the matcher, and supervise allele scores (set-NCE) + the true candidate's germline pointers (CE) against GenAIRR truth. Reuses `training/reader.py` and `nn/heads/cross_attn_matcher.py`.

**Tech Stack:** Python 3.12, PyTorch, GenAIRR, `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- Reuse: `XAttnAligner` (Plan 3), `xattn_match` (Plan 2), `build_sibling_index`, `build_candidates(primary_idx, multihot, sib_index_G, rng, n_sib, n_rand) -> (cand_idx (B,C), pos_mask (B,C))`, `reader_set_nce(scores, pos_mask)` (`training/reader.py`), `apply_orientation` (`nn/heads/orientation.py`), `extract_segment` (`core/dnalignair.py`).
- Gym batch keys (from `gym_collate`): `tokens`, `mask`, `region_labels`, `orientation_id`, and per gene `{g}_allele` (B,K multihot), `{g}_primary_idx` (B,), `{g}_germline_start` (B,), `{g}_germline_end` (B,). Genes lower-case `v/j` (+`d` if `has_d`).
- `build_candidates` places the example's primary at candidate column 0 → the true candidate's germline pointers are `gstart_logits[:, 0]` / `gend_logits[:, 0]`.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Create `src/alignair/training/xattn_loss.py` — `xattn_losses(...)`.
- Create `tests/alignair/training/test_xattn_loss.py` — finite/differentiable + trains-a-few-steps.

---

### Task 1: `xattn_losses` multi-task loss

**Files:**
- Create: `src/alignair/training/xattn_loss.py`
- Test: `tests/alignair/training/test_xattn_loss.py`

**Interfaces:**
- Produces: `xattn_losses(model, batch, ref_emb, sib_index, rng, n_sib=6, n_rand=6) -> (total: Tensor, parts: dict)` where `parts` has `orientation`, `region`, `allele`, `gstart`, `gend` scalar tensors. `sib_index = build_sibling_index(reference_set)`; `rng = random.Random(seed)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/training/test_xattn_loss.py
import random
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.reference.reference_set import ReferenceSet
from alignair.core.xattn_aligner import XAttnAligner
from alignair.training.reader import build_sibling_index
from alignair.training.xattn_loss import xattn_losses
from alignair.gym import AlignAIRGym, gym_collate
from torch.utils.data import DataLoader


def _batch_and_model():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = XAttnAligner(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64))
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=8, seed=0)
    loader = DataLoader(gym, batch_size=8, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    batch = next(iter(loader))
    return model, rs, batch


def test_losses_finite_and_have_parts():
    model, rs, batch = _batch_and_model()
    ref_emb = model.encode_reference(rs)
    sib = build_sibling_index(rs)
    total, parts = xattn_losses(model, batch, ref_emb, sib, random.Random(0))
    assert torch.isfinite(total)
    for k in ("orientation", "region", "allele", "gstart", "gend"):
        assert k in parts and torch.isfinite(parts[k])


def test_loss_is_differentiable():
    model, rs, batch = _batch_and_model()
    ref_emb = model.encode_reference(rs)
    sib = build_sibling_index(rs)
    total, _ = xattn_losses(model, batch, ref_emb, sib, random.Random(0))
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_xattn_loss.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/training/xattn_loss.py
from __future__ import annotations
import torch
import torch.nn.functional as F

from ..nn.heads.orientation import apply_orientation
from ..nn.heads.cross_attn_matcher import xattn_match
from ..core.dnalignair import extract_segment
from .reader import build_candidates, reader_set_nce


def _coord_ce(logits, target, Lg):
    # logits (B,Lg) over germline positions; target (B,) true coordinate, clamped into range
    tgt = target.long().clamp(0, Lg - 1)
    return F.cross_entropy(logits, tgt)


def xattn_losses(model, batch, ref_emb, sib_index, rng, n_sib: int = 6, n_rand: int = 6):
    tokens, mask = batch["tokens"], batch["mask"]
    ori_logits = model.orientation_head(tokens, mask)
    L_ori = F.cross_entropy(ori_logits, batch["orientation_id"].long())

    canon = apply_orientation(tokens, mask, batch["orientation_id"].long())   # teacher orientation
    reps = model.backbone.forward_positions(canon, mask)
    rdec = model.region_tagger(reps, mask)
    rl = rdec["region_logits"]                                                # (B,L,R)
    B, L, R = rl.shape
    reg_ce = F.cross_entropy(rl.reshape(B * L, R), batch["region_labels"].reshape(B * L).long(),
                             reduction="none").reshape(B, L)
    L_reg = (reg_ce * mask.float()).sum() / mask.float().sum().clamp(min=1)

    genes = ["v", "j"] + (["d"] if "d_allele" in batch else [])
    L_allele = L_gs = L_ge = reps.new_zeros(())
    for g in genes:
        G = g.upper()
        emb = ref_emb[G]
        seg, seg_mask = extract_segment(reps, mask, batch["region_labels"].long(), G)
        cand_idx, pos_mask = build_candidates(batch[f"{g}_primary_idx"], batch[f"{g}_allele"],
                                              sib_index[G], rng, n_sib=n_sib, n_rand=n_rand)
        out = xattn_match(model.matcher, seg, seg_mask, emb["pos_reps"], emb["pos_mask"], cand_idx)
        Lg = emb["pos_reps"].shape[1]
        L_allele = L_allele + reader_set_nce(out["allele_logits"], pos_mask)
        L_gs = L_gs + _coord_ce(out["gstart_logits"][:, 0], batch[f"{g}_germline_start"], Lg)
        L_ge = L_ge + _coord_ce(out["gend_logits"][:, 0], batch[f"{g}_germline_end"], Lg)

    total = L_ori + L_reg + L_allele + 0.5 * (L_gs + L_ge)
    return total, {"orientation": L_ori.detach(), "region": L_reg.detach(),
                   "allele": L_allele.detach(), "gstart": L_gs.detach(), "gend": L_ge.detach()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_xattn_loss.py -q`
Expected: both PASS. (If `build_candidates` returns `pos_mask` as float, `reader_set_nce` already handles it; if a gene's segment is empty for some rows the coord CE still runs against clamped targets — acceptable for the loss to be defined.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/training/xattn_loss.py tests/alignair/training/test_xattn_loss.py
git commit -m "training: xattn_losses (orientation + region CE + allele set-NCE + germline-span CE)"
```

---

### Task 2: Trains-a-few-steps on the gym (loss decreases)

**Files:**
- Test: `tests/alignair/training/test_xattn_loss.py`

**Interfaces:**
- Consumes: `xattn_losses`, `XAttnAligner`, `AlignAIRGym`. Locks that the assembled model + losses actually optimize on real gym data.

- [ ] **Step 1: Write the test**

```python
# append to tests/alignair/training/test_xattn_loss.py
def test_model_trains_a_few_steps_and_loss_decreases():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = XAttnAligner(DNAlignAIRConfig(d_model=32, n_layers=2, nhead=4, dim_feedforward=64)).train()
    ref_emb = model.encode_reference(rs)
    sib = build_sibling_index(rs)
    gym = AlignAIRGym([gdata.HUMAN_IGK_OGRDB], rs, n=8, seed=0)
    loader = DataLoader(gym, batch_size=8, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    it = iter(loader)
    rng = random.Random(0)
    losses = []
    for step in range(24):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        ref_emb = model.encode_reference(rs)          # re-encode (weights changed)
        total, _ = xattn_losses(model, batch, ref_emb, sib, rng)
        opt.zero_grad(); total.backward(); opt.step()
        losses.append(float(total))
    first = sum(losses[:6]) / 6
    last = sum(losses[-6:]) / 6
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_xattn_loss.py::test_model_trains_a_few_steps_and_loss_decreases -q`
Expected: PASS (loss decreases over 24 steps).

- [ ] **Step 3: Commit**

```bash
git add tests/alignair/training/test_xattn_loss.py
git commit -m "training: XAttnAligner trains a few steps on the gym (loss decreases)"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_xattn_loss.py -q` is green.
- `xattn_losses` returns a finite, differentiable total with the five named parts; the model trains and the loss decreases on real gym data.

## Follow-on plans (not this plan)

1. **Full training script** — `scripts/train_xattn.py`: long run on the stratified gym with embargo'd alleles, checkpoint/resume, bf16, the lattice exams (the LLM best-practices regime from the spec).
2. **Derivations module** — four heads → full GenAIRR record (vectorized).
3. **Eval/gates** — assay, IgBLAST head-to-head, embargo, throughput.
