# Seed-and-Extend Gate 1 (Band-Recall Geometry) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the structural diagonal-offset band head and measure its band recall on the frozen 8h soft-DP model, to decide (HARD STOP) whether the Structural Seed-and-Extend Neural DP is worth building.

**Architecture:** A small band head aggregates per-offset features (raw base-match diagonal [dominant] + learned token cosine [additive]) into a distribution over germline start offsets, trained with offset cross-entropy. A metrics module computes top-1 recall, top-m union recall, fail-open rate, and effective DP cell budget. A Gate-1 experiment harness freezes the soft-DP model, trains ONLY the band head on true-region/true-allele V segments, and reports per-lattice-cell results with bootstrap CIs.

**Tech Stack:** PyTorch (`.venv/bin/python`, `PYTHONPATH=src`), pytest, the GenAIRR gym + FrozenLattice, the 8h-trained model `.private/models/scaled_long.pt`.

**Spec:** `docs/superpowers/specs/2026-06-25-structural-seed-and-extend-neural-dp-design.md` (this plan implements Gate 1 / build step 1 ONLY).

## Global Constraints

- Run everything with `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`. `alignair` is NOT pip-installed. NEVER bare `python`/`python3`.
- Package under test is `src/alignair` (lowercase), not `src/AlignAIR` (legacy TF).
- Git commit messages: NEVER add Co-Authored-By or any Claude/AI mention (project rule).
- **HARD STOP:** this plan ends at the Gate-1 decision. **No kernel, encoder-refactor, or DP work happens in this plan** — those are gated behind a passing Gate 1 (spec §5–§6). Do not start them here.
- Band-head features deliberately weight the **representation-independent** channel (raw base-match) to dominate the learned cosine, so the head survives the later encoder refactor (spec §4.4).
- Training objective is **offset cross-entropy** (recall is the decision metric, NOT the loss). `w`, `top-m`, fail-open threshold are values to VALIDATE, not bake in (spec rule 6).
- TDD: failing test first, watch it fail, minimal implementation, watch it pass, commit. One logical change per commit.
- The true band center (offset where read V-segment position 0 aligns to germline) is `v_germline_start`.

## File Structure

- `src/alignair/nn/band_head.py` — NEW. `base_match_matrix`, `BandHead(nn.Module)`, `band_offset_loss`. Reuses `weighted_leading_diag` from `nn/pointer_aligner.py`.
- `src/alignair/gym/instrument/band_metrics.py` — NEW. `top1_recall`, `topm_union_recall`, `fail_open_rate`, `cell_budget` (pure functions on logits + true start).
- `scripts/exp_band_recall_gate.py` — NEW. Gate-1 harness: freeze `scaled_long.pt`, train the band head, eval per-cell with bootstrap CIs.
- Tests: `tests/alignair/nn/test_band_head.py`, `tests/alignair/gym/instrument/test_band_metrics.py`.

---

## Task 1: Band head module (features + head + loss)

**Files:**
- Create: `src/alignair/nn/band_head.py`
- Test: `tests/alignair/nn/test_band_head.py`
- Reuse: `src/alignair/nn/pointer_aligner.py` (`weighted_leading_diag`)

**Interfaces:**
- Produces:
  - `base_match_matrix(seg_tok[B,S], germ_tok[B,Lg]) -> Tensor[B,S,Lg]` — raw +1 match / −1 mismatch / 0 non-ACGT.
  - `BandHead(d_model)` with `forward(seg_reps[B,S,d], seg_mask[B,S], germ_reps[B,Lg,d], germ_mask[B,Lg], seg_tok[B,S], germ_tok[B,Lg]) -> offset_logits[B,Lg]` over germline start offsets (NEG on invalid columns). Base-match channel dominant by init.
  - `band_offset_loss(offset_logits[B,Lg], true_start[B]) -> scalar` — offset cross-entropy.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/nn/test_band_head.py
import torch
from alignair.nn.band_head import base_match_matrix, BandHead, band_offset_loss


def test_base_match_matrix_signs():
    seg = torch.tensor([[1, 2, 0]])        # A, C, pad
    germ = torch.tensor([[1, 1, 2]])       # A, A, C
    M = base_match_matrix(seg, germ)
    assert M.shape == (1, 3, 3)
    assert M[0, 0, 0] == 1.0                # A vs A match
    assert M[0, 0, 1] == 1.0                # A vs A match
    assert M[0, 0, 2] == -1.0               # A vs C mismatch
    assert M[0, 1, 0] == -1.0               # C vs A mismatch
    assert (M[0, 2] == 0.0).all()           # pad token (0) -> 0 everywhere


def test_bandhead_localizes_clean_offset():
    # base-match alone (zeroed cosine projections) must localize the true offset
    al = BandHead(d_model=16)
    with torch.no_grad():
        al.proj_s.weight.zero_(); al.proj_s.bias.zero_()
        al.proj_g.weight.zero_(); al.proj_g.bias.zero_()
    B, S, Lg, off = 1, 8, 30, 7
    seg_tok = torch.randint(1, 5, (B, S))
    germ_tok = torch.randint(1, 5, (B, Lg))
    germ_tok[0, off:off + S] = seg_tok[0]                 # plant exact match at offset 7
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    seg = torch.zeros(B, S, 16); germ = torch.zeros(B, Lg, 16)
    logits = al(seg, sm, germ, gm, seg_tok, germ_tok)
    assert logits.shape == (B, Lg)
    assert logits.argmax(-1).item() == off


def test_bandhead_masks_invalid_columns():
    al = BandHead(d_model=16)
    B, S, Lg = 2, 6, 20
    seg = torch.randn(B, S, 16); germ = torch.randn(B, Lg, 16)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    gm[:, 15:] = False
    st = torch.randint(1, 5, (B, S)); gt = torch.randint(1, 5, (B, Lg))
    logits = al(seg, sm, germ, gm, st, gt)
    assert (logits[:, 15:] <= -1e3).all()


def test_band_offset_loss_minimized_at_truth():
    B, Lg = 4, 30
    true = torch.tensor([5, 10, 3, 12])
    pos = torch.arange(Lg).float()
    good = -(pos[None] - true[:, None].float()) ** 2          # peaked on truth
    bad = -(pos[None] - (true[:, None].float() + 8)) ** 2
    assert band_offset_loss(good, true) < band_offset_loss(bad, true)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_band_head.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.nn.band_head'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/nn/band_head.py
"""Structural diagonal-offset band head (the "seed" of seed-and-extend).

Predicts a distribution over germline START offsets for a read segment, from
representation-INDEPENDENT raw base-match (dominant) + learned token cosine
(additive), so it survives the later encoder refactor. Trained with offset
cross-entropy; band recall is the decision metric, not the loss (spec §4.4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointer_aligner import weighted_leading_diag

NEG = -1e4


def base_match_matrix(seg_tok: torch.Tensor, germ_tok: torch.Tensor) -> torch.Tensor:
    """Raw +1 match / -1 mismatch / 0 non-ACGT base-match grid (B,S,Lg)."""
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)            # (B,S,1),(B,1,Lg)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)
    return real.float() * (2.0 * (st == gt).float() - 1.0)


class BandHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_s = nn.Linear(d_model, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))           # cosine scale ~5
        # base-match weight DOMINATES at init (representation-independent); cosine is a
        # small additive correction so the head survives the encoder refactor.
        self.w_bm = nn.Parameter(torch.tensor(1.0))
        self.w_cos = nn.Parameter(torch.tensor(0.1))
        self.log_temp = nn.Parameter(torch.tensor(1.6))            # sharpen the offset posterior

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok):
        w = seg_mask.float().unsqueeze(-1)                          # (B,S,1) mask pad rows
        bm = weighted_leading_diag(base_match_matrix(seg_tok, germ_tok).float(), w)  # (B,Lg)
        Sn = F.normalize(self.proj_s(seg_reps).float(), dim=-1)
        Gn = F.normalize(self.proj_g(germ_reps).float(), dim=-1)
        cos_M = self.log_scale.clamp(-2.0, 3.0).exp() * torch.einsum("bid,bjd->bij", Sn, Gn)
        cos = weighted_leading_diag(cos_M, w)                       # (B,Lg)
        temp = self.log_temp.clamp(0.0, 4.5).exp()
        logit = temp * (F.softplus(self.w_bm) * bm + self.w_cos * cos)
        return logit.masked_fill(~germ_mask, NEG)


def band_offset_loss(offset_logits: torch.Tensor, true_start: torch.Tensor) -> torch.Tensor:
    """Offset cross-entropy of the start-offset posterior against the true germline_start."""
    Lg = offset_logits.shape[-1]
    tgt = true_start.clamp(min=0, max=Lg - 1)
    return F.cross_entropy(offset_logits, tgt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_band_head.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/band_head.py tests/alignair/nn/test_band_head.py
git commit -m "Add structural diagonal-offset band head (base-match dominant) + offset loss"
```

---

## Task 2: Band metrics (recall, top-m union, fail-open, cell budget)

**Files:**
- Create: `src/alignair/gym/instrument/band_metrics.py`
- Test: `tests/alignair/gym/instrument/test_band_metrics.py`

**Interfaces:**
- Consumes: band `offset_logits[B,Lg]`, `true_start[B]`.
- Produces (all return python floats over the batch):
  - `top1_recall(offset_logits, true_start, w) -> float` — fraction with `|argmax−true| ≤ w`.
  - `topm_union_recall(offset_logits, true_start, w, m) -> float` — fraction where ANY of m NMS-spaced peaks is within w.
  - `fail_open_rate(offset_logits, threshold) -> float` — fraction with `max softmax prob < threshold`.
  - `cell_budget(offset_logits, w, threshold, seg_len[B]) -> float` — mean DP cells: banded rows contribute `(2w+1)·seg_len`, fail-open rows `Lg·seg_len`.
  - `_topm_centers(offset_logits, w, m) -> LongTensor[B,m]` — helper: greedy NMS peak centers spaced > 2w apart.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/instrument/test_band_metrics.py
import torch
from alignair.gym.instrument.band_metrics import (
    top1_recall, topm_union_recall, fail_open_rate, cell_budget)


def _two_peaks(Lg=40, a=5, b=25):
    logits = torch.full((1, Lg), -1e4)
    logits[0, a] = 4.0; logits[0, b] = 5.0      # higher peak at b
    return logits


def test_top1_recall_within_tol():
    logits = _two_peaks()
    assert top1_recall(logits, torch.tensor([25]), w=1) == 1.0   # argmax=25
    assert top1_recall(logits, torch.tensor([5]), w=1) == 0.0    # argmax!=5


def test_topm_union_recovers_secondary_peak():
    logits = _two_peaks()
    # true=5 is the SECONDARY peak; top-1 misses, top-2 union recovers it
    assert top1_recall(logits, torch.tensor([5]), w=1) == 0.0
    assert topm_union_recall(logits, torch.tensor([5]), w=1, m=2) == 1.0


def test_fail_open_rate_thresholds_low_confidence():
    flat = torch.zeros(1, 50)                    # uniform -> max prob 0.02, low confidence
    peaked = _two_peaks()                        # confident
    assert fail_open_rate(flat, threshold=0.1) == 1.0
    assert fail_open_rate(peaked, threshold=0.1) == 0.0


def test_cell_budget_counts_band_vs_fullopen():
    logits = _two_peaks()                        # confident -> banded
    seg_len = torch.tensor([100])
    # confident: (2*4+1)*100 = 900 cells
    assert cell_budget(logits, w=4, threshold=0.1, seg_len=seg_len) == 900.0
    flat = torch.zeros(1, 40)                     # fail-open -> Lg*seg_len = 40*100 = 4000
    assert cell_budget(flat, w=4, threshold=0.1, seg_len=seg_len) == 4000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/instrument/test_band_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.gym.instrument.band_metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/gym/instrument/band_metrics.py
"""Gate-1 band-head metrics: recall (top-1 + top-m union), fail-open rate, and the
effective DP cell budget (speed proxy). Recall and budget are gated TOGETHER so a
broad/always-open predictor cannot pass recall while destroying speed (spec §5)."""
import torch


def top1_recall(offset_logits, true_start, w):
    pred = offset_logits.argmax(dim=-1)
    return float(((pred - true_start).abs() <= w).float().mean())


def _topm_centers(offset_logits, w, m):
    """Greedy NMS: repeatedly take the max, suppress +-2w around it, m times. (B,m)."""
    x = offset_logits.clone()
    B, Lg = x.shape
    pos = torch.arange(Lg, device=x.device)
    centers = []
    for _ in range(m):
        c = x.argmax(dim=-1)                                   # (B,)
        centers.append(c)
        near = (pos.unsqueeze(0) - c.unsqueeze(1)).abs() <= 2 * w
        x = x.masked_fill(near, -1e9)
    return torch.stack(centers, dim=1)                         # (B,m)


def topm_union_recall(offset_logits, true_start, w, m):
    centers = _topm_centers(offset_logits, w, m)               # (B,m)
    hit = ((centers - true_start.unsqueeze(1)).abs() <= w).any(dim=1)
    return float(hit.float().mean())


def fail_open_rate(offset_logits, threshold):
    maxp = torch.softmax(offset_logits.float(), dim=-1).max(dim=-1).values
    return float((maxp < threshold).float().mean())


def cell_budget(offset_logits, w, threshold, seg_len):
    Lg = offset_logits.shape[-1]
    maxp = torch.softmax(offset_logits.float(), dim=-1).max(dim=-1).values
    open_ = maxp < threshold
    cols = torch.where(open_, torch.full_like(seg_len, Lg), torch.full_like(seg_len, 2 * w + 1))
    return float((cols.float() * seg_len.float()).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/instrument/test_band_metrics.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/band_metrics.py tests/alignair/gym/instrument/test_band_metrics.py
git commit -m "Add Gate-1 band metrics: top-1/top-m recall, fail-open rate, DP cell budget"
```

---

## Task 3: Gate-1 experiment harness

**Files:**
- Create: `scripts/exp_band_recall_gate.py`

**Interfaces:**
- Consumes: `BandHead`, `band_offset_loss` (Task 1); the band metrics (Task 2); `FrozenLattice`, `AlignAIRGym`, `gym_collate`, `bootstrap_ci`; the frozen model `.private/models/scaled_long.pt`.
- Produces: a CLI that trains ONLY the band head on true-region/true-allele V segments of the frozen model, and prints per-lattice-cell `top1`, `topm-union`, `fail-open`, `cell-budget`, each with bootstrap-CI lower bounds, at `--widths`.

- [ ] **Step 1: Write the harness**

```python
# scripts/exp_band_recall_gate.py
"""Gate 1 (geometry gate, spec §5): freeze the 8h soft-DP model, train ONLY the structural
band head on TRUE-region / TRUE-allele V segments, then report per-lattice-cell band recall
(top-1 + top-m union), fail-open rate, and DP cell budget with bootstrap CIs at w=8,16.

HARD STOP: if top-m union recall misses >0.5-1% at w=16, OR the cell budget erases the speed
win, do NOT proceed to kernel/encoder work (spec build order step 1).

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_band_recall_gate.py --train-steps 2000 --n 400
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.stats import bootstrap_ci
from alignair.nn.band_head import BandHead, band_offset_loss
from alignair.gym.instrument import band_metrics as BM
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def _segment_inputs(model, batch, ref_emb, device):
    """True-region/true-allele V segment reps + tokens + germline reps + true start."""
    out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
    canon = out["canon_tokens"]
    seg_tok, seg_mask = extract_segment_tokens(canon, batch["mask"], batch["region_labels"], "V")
    seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
    idx = batch["v_primary_idx"]
    germ_reps = ref_emb["V"]["pos_reps"][idx]; germ_mask = ref_emb["V"]["pos_mask"][idx]
    germ_tok = ref_emb["V"]["pos_tok"][idx]
    return (seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok,
            batch["v_germline_start"], seg_mask.sum(1))


def _cell_loader(dc, rs, cell_params, n, batch_size, seed):
    cur = type("C", (), {"params": lambda s, p=0.0: dict(cell_params),
                         "describe": lambda s, p=0.0: "cell", "stage": lambda s, p=0.0: 0})()
    gym = AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=cur)
    return DataLoader(gym, batch_size=batch_size,
                      collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def _mixture_loader(dc, rs, n, batch_size, seed):
    """Training stream: the StratifiedCurriculum mixture (covers all difficulty regimes) so
    the band head GENERALIZES across cells rather than overfitting one regime."""
    gym = AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=StratifiedCurriculum())
    return DataLoader(gym, batch_size=batch_size,
                      collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--topm", type=int, default=2)
    ap.add_argument("--fail-open-thresh", type=float, default=0.1)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    widths = [int(x) for x in a.widths.split(",")]
    dc = gdata.HUMAN_IGH_OGRDB

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)                                  # FREEZE the 8h model
    rs = ReferenceSet.from_dataconfigs(dc)
    ref_emb = model.encode_reference(rs)
    lat = FrozenLattice.standard(seed=0)
    cells = {c.name: c for c in lat.cells}

    head = BandHead(d_model=ck["config"]["d_model"]).to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    # train the head on the StratifiedCurriculum MIXTURE (generalizes across regimes)
    train_loader = _mixture_loader(dc, rs, a.n * 50, a.batch_size, seed=1)
    it = iter(train_loader); step = 0
    while step < a.train_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            inp = _segment_inputs(model, batch, ref_emb, device)
        logits = head(*inp[:6])
        loss = band_offset_loss(logits, inp[6])
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 250 == 0:
            print(f"[train] step {step}/{a.train_steps} loss {float(loss):.3f}", flush=True)

    # evaluate per frozen-lattice cell
    head.eval()
    print(f"\nGate 1 band recall (frozen model, true region/allele) | top-m={a.topm} "
          f"fail-open<{a.fail_open_thresh}")
    for w in widths:
        print(f"\n--- w={w} ---")
        print(f"{'cell':18s} {'top1':>14s} {'topm-union':>14s} {'fail-open':>10s} {'cellbudget':>12s}")
        for cname in CELLS:
            loader = _cell_loader(dc, rs, lat.cell_params(cells[cname]), a.n, a.batch_size, seed=0)
            t1, tm, fo, cb = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    inp = _segment_inputs(model, batch, ref_emb, device)
                    logits = head(*inp[:6]); true = inp[6]; slen = inp[7]
                    t1.append(BM.top1_recall(logits, true, w))
                    tm.append(BM.topm_union_recall(logits, true, w, a.topm))
                    fo.append(BM.fail_open_rate(logits, a.fail_open_thresh))
                    cb.append(BM.cell_budget(logits, w, a.fail_open_thresh, slen))
            def lo(xs):
                m, l, h = bootstrap_ci(xs); return m, l
            (m1, l1), (mm, lm), (mf, lf), (mc, lc) = lo(t1), lo(tm), lo(fo), lo(cb)
            print(f"{cname:18s} {m1:.3f}[lo {l1:.3f}] {mm:.3f}[lo {lm:.3f}] "
                  f"{mf:8.3f} {mc:12.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the harness (tiny, CPU-safe, proves wiring)**

Run: `PYTHONPATH=src .venv/bin/python scripts/exp_band_recall_gate.py --train-steps 20 --n 24 --batch-size 8`
Expected: it loads the frozen model, trains 20 steps (prints a loss), and prints a per-cell table for w=8 and w=16 with top1/topm-union/fail-open/cellbudget columns. Numbers will be poor (20 steps) but the table must render without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/exp_band_recall_gate.py
git commit -m "Add Gate-1 band-recall experiment harness (frozen model, true region/allele)"
```

---

## Task 4: Run Gate 1 and record the decision (THE HARD STOP)

**Files:**
- No new code. Runs the experiment, records results, makes the go/no-go decision.

**Interfaces:**
- Consumes: `scripts/exp_band_recall_gate.py` (Task 3).

- [ ] **Step 1: Run Gate 1 at scale**

Run: `PYTHONPATH=src .venv/bin/python scripts/exp_band_recall_gate.py --train-steps 3000 --n 500 --batch-size 32 --widths 8,16 --topm 2`
Expected: per-cell table for w=8 and w=16. Record the full output.

- [ ] **Step 2: Apply the decision rule**

PASS requires, at **w=16**, for every cell (clean, heavy_shm_fulllen, indel, junction_boundary):
- top-m **union** recall (CI lower bound) ≥ **0.99** (misses ≤ 0.5–1%), AND
- the **cell budget** is materially below full-DP (fail-open rate not so high it erases the speed win — sanity: mean cell budget ≪ `Lg·seg_len`).

If any cell fails recall: do NOT proceed to kernel/encoder work. Iterate the band head FIRST — add the k-mer/minimizer seed feature and segment-boundary-confidence feature (spec §4.4), or raise `top-m`, then re-run. If recall passes but cell budget is too high: tighten the fail-open threshold / `top-m` and re-run.

- [ ] **Step 3: Record the outcome to project memory**

Append the per-cell Gate-1 numbers and the PASS/FAIL decision to `/home/thomas/.claude/projects/-home-thomas-Desktop-AlignAIR/memory/aligner-parallel-scan-direction.md`, and update its one-line pointer in `MEMORY.md`. State explicitly whether the kernel/encoder build is UNLOCKED or BLOCKED.

- [ ] **Step 4: Commit any tuning changes**

If Task 2's decision required band-head feature/config changes, commit them:
```bash
git add src/alignair/nn/band_head.py scripts/exp_band_recall_gate.py
git commit -m "Tune band head for Gate-1 recall (k-mer/boundary features)"
```

- [ ] **Step 5: STOP — report to the user**

Report the Gate-1 table and decision. **Do not begin build steps 2–7 (encoder refactor, banded DP, kernel) — those require a NEW plan, written only after Gate 1 passes.** This is the spec's hard stop.

---

## Notes on scope (why this plan stops at Gate 1)

The spec (§5–§6) makes Gate 1 a hard decision point: the entire downstream architecture (shared encoder refactor, banded sequential DP, fused Triton kernel, band predictor wiring, Gate 2/3) is justified ONLY if the structural band head can place a tight band the DP can trust. Writing tasks for that work now would violate the spec's "kernel is the reward for passing the geometry gate." When Gate 1 passes, the next plan covers build step 2 onward (encoder refactor → sequential banded DP parity → Gate-1 repeat on the new encoder → fused kernel → Gate 2 → Gate 3).
