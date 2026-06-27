# Cross-Attention Matcher Implementation Plan (LLM-Encoder Aligner — Plan 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core new component of the LLM-encoder aligner — a token-level cross-attention module that, for each of C candidate germlines per read, produces an allele match score and germline start/end position logits, all from per-token reps (no training, no model assembly yet).

**Architecture:** Multi-head cross-attention with read-segment tokens as queries and a candidate germline's tokens as keys/values. The per-position attention agreement pools to one match score per candidate; the attention distribution of the first/last valid segment token gives germline start/end pointers. Pure tensor module — batched over (read × candidate), shape- and behavior-testable on toy reps.

**Tech Stack:** Python 3.12, PyTorch, `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run everything via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- This module is a pure `nn.Module` over per-token reps — it does NOT encode sequences, retrieve, or know about alleles by name. It consumes reps + masks and returns logits.
- Tensors: `seg_reps (B,S,d)`, `seg_mask (B,S)` bool; `cand_reps (B,C,Lg,d)`, `cand_mask (B,C,Lg)` bool. Returns `match (B,C)`, `gstart_logits (B,C,Lg)`, `gend_logits (B,C,Lg)`.
- Mask convention: `True` = valid (matches `SharedNucleotideEncoder`/existing code). Padded positions are masked out of attention and pooling.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Create `src/alignair/nn/heads/cross_attn_matcher.py` — `CrossAttnMatcher` module.
- Create `tests/alignair/nn/test_cross_attn_matcher.py` — shape + behavior tests.

---

### Task 1: Cross-attention match score

**Files:**
- Create: `src/alignair/nn/heads/cross_attn_matcher.py`
- Test: `tests/alignair/nn/test_cross_attn_matcher.py`

**Interfaces:**
- Produces: `CrossAttnMatcher(d_model: int, nhead: int = 8)`; `forward(seg_reps, seg_mask, cand_reps, cand_mask) -> (match, gstart_logits, gend_logits)` with shapes `match (B,C)`, `gstart_logits (B,C,Lg)`, `gend_logits (B,C,Lg)`. This task implements `match`; the coord logits are added in Task 2 (return zeros for them here).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/nn/test_cross_attn_matcher.py
import torch
from alignair.nn.heads.cross_attn_matcher import CrossAttnMatcher


def _toy(B=2, C=3, S=5, Lg=7, d=16):
    torch.manual_seed(0)
    seg = torch.randn(B, S, d)
    cand = torch.randn(B, C, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool)
    cm = torch.ones(B, C, Lg, dtype=torch.bool)
    return seg, sm, cand, cm


def test_forward_shapes():
    m = CrossAttnMatcher(d_model=16, nhead=4)
    seg, sm, cand, cm = _toy()
    match, gs, ge = m(seg, sm, cand, cm)
    assert match.shape == (2, 3)
    assert gs.shape == (2, 3, 7) and ge.shape == (2, 3, 7)


def test_matching_candidate_scores_highest():
    # candidate 1's germline tokens ARE the segment tokens (a perfect match); the matcher
    # should score candidate 1 above the two random candidates. (Untrained projections, so we
    # assert it on the raw alignment signal by initialising q/k/v near-identity.)
    torch.manual_seed(0)
    m = CrossAttnMatcher(d_model=16, nhead=4)
    with torch.no_grad():                                  # near-identity q/k/v so cosine signal shows
        for lin in (m.q, m.k, m.v):
            lin.weight.copy_(torch.eye(16)); lin.bias.zero_()
    B, C, S, Lg, d = 1, 3, 6, 6, 16
    seg = torch.randn(B, S, d)
    cand = torch.randn(B, C, Lg, d)
    cand[0, 1] = seg[0]                                    # candidate 1 == the segment
    sm = torch.ones(B, S, dtype=torch.bool)
    cm = torch.ones(B, C, Lg, dtype=torch.bool)
    match, _, _ = m(seg, sm, cand, cm)
    assert match.argmax(dim=-1).item() == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_cross_attn_matcher.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/nn/heads/cross_attn_matcher.py
from __future__ import annotations
import torch
import torch.nn as nn


class CrossAttnMatcher(nn.Module):
    """Token-level read-segment x candidate-germline cross-attention. For each of C candidate
    germlines per read, the read-segment tokens (queries) attend to that germline's tokens
    (keys/values); per-position agreement pools to one allele match score, and the boundary
    tokens' attention gives germline start/end pointers."""

    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        assert d_model % nhead == 0
        self.h, self.hd = nhead, d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.match = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                   nn.GELU(), nn.Linear(d_model, 1))

    def _attend(self, seg, sm, germ, gm):
        # seg (N,S,d), germ (N,Lg,d) -> ctx (N,S,d), wmean (N,S,Lg)
        N, S, d = seg.shape
        Lg = germ.shape[1]
        Q = self.q(seg).view(N, S, self.h, self.hd).transpose(1, 2)
        K = self.k(germ).view(N, Lg, self.h, self.hd).transpose(1, 2)
        V = self.v(germ).view(N, Lg, self.h, self.hd).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / (self.hd ** 0.5)              # (N,h,S,Lg)
        att = att.masked_fill(~gm[:, None, None, :], -1e9)
        w = torch.softmax(att, dim=-1)
        ctx = (w @ V).transpose(1, 2).reshape(N, S, d)
        return ctx, w.mean(1)                                          # (N,S,d), (N,S,Lg)

    def forward(self, seg_reps, seg_mask, cand_reps, cand_mask):
        B, S, d = seg_reps.shape
        C, Lg = cand_reps.shape[1], cand_reps.shape[2]
        seg = seg_reps.unsqueeze(1).expand(B, C, S, d).reshape(B * C, S, d)
        sm = seg_mask.unsqueeze(1).expand(B, C, S).reshape(B * C, S)
        germ = cand_reps.reshape(B * C, Lg, d)
        gm = cand_mask.reshape(B * C, Lg)
        ctx, wmean = self._attend(seg, sm, germ, gm)                    # (BC,S,d), (BC,S,Lg)
        pos = self.match(ctx).squeeze(-1).masked_fill(~sm, 0.0)        # (BC,S)
        match = (pos.sum(-1) / sm.sum(-1).clamp(min=1)).reshape(B, C)  # (B,C)
        gstart = torch.zeros(B, C, Lg, device=seg_reps.device)        # filled in Task 2
        gend = torch.zeros(B, C, Lg, device=seg_reps.device)
        return match, gstart, gend
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_cross_attn_matcher.py -q`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/heads/cross_attn_matcher.py tests/alignair/nn/test_cross_attn_matcher.py
git commit -m "nn: CrossAttnMatcher allele match score (read-segment x germline cross-attention)"
```

---

### Task 2: Germline start/end pointers from boundary-token attention

**Files:**
- Modify: `src/alignair/nn/heads/cross_attn_matcher.py`
- Test: `tests/alignair/nn/test_cross_attn_matcher.py`

**Interfaces:**
- Consumes: the `_attend` helper + `wmean (BC,S,Lg)` from Task 1.
- Produces: `gstart_logits`, `gend_logits` `(B,C,Lg)` — the germline-position log-attention of the FIRST and LAST valid segment token (the segment's 5' end maps to germline_start, its 3' end to germline_end). The argmax over Lg is the predicted germline coordinate.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_cross_attn_matcher.py
def test_germline_pointers_track_a_shifted_match():
    # segment of length 4 equals germline positions 2..6 of candidate 0 (a 5' germline trim of 2).
    # The first seg token should point near germline pos 2 (start); the last near pos 5 (end).
    torch.manual_seed(0)
    m = CrossAttnMatcher(d_model=16, nhead=4)
    with torch.no_grad():
        for lin in (m.q, m.k, m.v):
            lin.weight.copy_(torch.eye(16)); lin.bias.zero_()
    B, C, S, Lg, d = 1, 1, 4, 8, 16
    germ = torch.randn(B, C, Lg, d)
    seg = germ[0, 0, 2:6].clone().unsqueeze(0)             # seg == germline[2:6]
    sm = torch.ones(B, S, dtype=torch.bool)
    cm = torch.ones(B, C, Lg, dtype=torch.bool)
    _, gs, ge = m(seg, sm, germ, cm)
    assert gs[0, 0].argmax().item() == 2                  # start pointer -> germline pos 2
    assert ge[0, 0].argmax().item() == 5                  # end pointer -> germline pos 5


def test_pointer_logits_respect_germline_mask():
    m = CrossAttnMatcher(d_model=16, nhead=4)
    seg, sm, cand, cm = _toy()
    cm[:, :, -2:] = False                                  # mask last 2 germline positions
    _, gs, ge = m(seg, sm, cand, cm)
    assert (gs[..., -2:] <= -1e8).all()                   # masked germline positions -> -inf
    assert (ge[..., -2:] <= -1e8).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_cross_attn_matcher.py -q`
Expected: FAIL (`test_germline_pointers_track_a_shifted_match`: pointers are all-zero from Task 1).

- [ ] **Step 3: Write minimal implementation**

Replace the `gstart`/`gend` zero placeholders in `forward` with the boundary-token pointer readout. First add a helper, then use it:

```python
# add to CrossAttnMatcher
    @staticmethod
    def _first_last(mask):
        # mask (N,S) bool -> (first_idx, last_idx) (N,) of valid positions
        N, S = mask.shape
        ar = torch.arange(S, device=mask.device)
        big = S + 1
        first = torch.where(mask, ar, torch.full_like(ar, big)).min(dim=1).values.clamp(max=S - 1)
        last = torch.where(mask, ar, torch.full_like(ar, -1)).max(dim=1).values.clamp(min=0)
        return first, last
```

Then in `forward`, replace the two zero lines with:

```python
        logw = torch.log(wmean.clamp_min(1e-9))                        # (BC,S,Lg) log-attention
        logw = logw.masked_fill(~gm.unsqueeze(1), -1e9)                # respect germline mask
        first, last = self._first_last(sm)                             # (BC,), (BC,)
        bc = torch.arange(B * C, device=seg_reps.device)
        gstart = logw[bc, first].reshape(B, C, Lg)                     # first valid seg token's dist
        gend = logw[bc, last].reshape(B, C, Lg)                        # last valid seg token's dist
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_cross_attn_matcher.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/heads/cross_attn_matcher.py tests/alignair/nn/test_cross_attn_matcher.py
git commit -m "nn: germline start/end pointers from boundary-token cross-attention"
```

---

### Task 3: Batched-candidate parity + gradient sanity

**Files:**
- Test: `tests/alignair/nn/test_cross_attn_matcher.py`

**Interfaces:**
- Consumes: `CrossAttnMatcher` (Tasks 1–2). No new production code — this task locks two correctness properties the training loop will rely on.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_cross_attn_matcher.py
def test_each_candidate_scored_independently():
    # scoring candidate j must not depend on the other candidates in the batch dim C
    torch.manual_seed(1)
    m = CrossAttnMatcher(d_model=16, nhead=4).eval()
    B, C, S, Lg, d = 1, 3, 5, 7, 16
    seg = torch.randn(B, S, d); cand = torch.randn(B, C, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool); cm = torch.ones(B, C, Lg, dtype=torch.bool)
    full, _, _ = m(seg, sm, cand, cm)
    solo, _, _ = m(seg, sm, cand[:, 1:2], cm[:, 1:2])     # score candidate 1 alone
    assert torch.allclose(full[:, 1], solo[:, 0], atol=1e-5)


def test_match_and_pointers_are_differentiable():
    m = CrossAttnMatcher(d_model=16, nhead=4)
    seg, sm, cand, cm = _toy()
    seg.requires_grad_(True)
    match, gs, ge = m(seg, sm, cand, cm)
    (match.sum() + gs[cm.any(-1)].clamp_min(-1e8).sum() * 0 + ge.exp().sum()).backward()
    assert seg.grad is not None and torch.isfinite(seg.grad).all()
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_cross_attn_matcher.py -q`
Expected: PASS if Tasks 1–2 are correct (these are property locks). If `test_each_candidate_scored_independently` FAILS, there is cross-candidate leakage in the reshape — fix the `reshape(B*C, ...)` so each (b,c) row is independent before re-running.

- [ ] **Step 3: Commit**

```bash
git add tests/alignair/nn/test_cross_attn_matcher.py
git commit -m "nn: lock CrossAttnMatcher per-candidate independence + differentiability"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_cross_attn_matcher.py -q` is green.
- `CrossAttnMatcher(d_model, nhead).forward(seg_reps, seg_mask, cand_reps, cand_mask)` returns
  `match (B,C)`, `gstart_logits (B,C,Lg)`, `gend_logits (B,C,Lg)`; the matching candidate scores
  highest; boundary pointers track a shifted match; candidates are scored independently; outputs
  are differentiable.

## Follow-on plans (not this plan)

1. **Full model assembly** — wire `SharedNucleotideEncoder` + segmentation/orientation heads +
   learned retrieval ∪ `align/seed_prefilter.py` + `CrossAttnMatcher` into a `DNAlignAIR`-style
   model that produces the four head outputs end to end.
2. **Derivations module** — predicted-four → full GenAIRR record (trims, np, mutations, junction,
   productivity, cigar) as vectorized tensor ops + the indel cheap-align fallback.
3. **Training loop + losses** — single-phase multi-task on the gym (set-NCE + spans + orientation),
   embargo'd alleles.
4. **Eval/gates** — assay, IgBLAST head-to-head, embargo, throughput.
