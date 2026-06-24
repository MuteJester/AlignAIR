# BandedPointerAligner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sequential differentiable soft-DP germline aligner with a fully-parallel `BandedPointerAligner` (diagonal extraction via `as_strided`) plus a redesigned coordinate loss, killing both soft-DP runtime sites (~66% inference, ~94% training-step) while improving heavy-SHM-V and junction-jitter accuracy.

**Architecture:** A reference-conditioned pointer head computes a score matrix `M = scale·cosine + reliability-gated base-match`, then extracts weighted leading/reverse diagonals in single CUDA launches to produce germline start/end logits; a fast diagonal seed score replaces the soft-DP `alignment_score`. The coordinate loss switches from hard CE to soft-argmax-L1 + ordinal-CDF + start/end-consistency, decoded with soft-argmax everywhere. Each change is gated behind the §9 ablation ladder and A/B'd on the frozen gym lattice.

**Tech Stack:** PyTorch (`.venv/bin/python`, run scripts with `PYTHONPATH=src`), pytest, GenAIRR simulator for GT coords, existing `src/alignair` package.

**Spec:** `docs/superpowers/specs/2026-06-24-banded-pointer-aligner-design.md` (read it; this plan implements §4–§9 and §12).

## Global Constraints

- Run everything with the project venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`. `alignair` is NOT pip-installed. NEVER use bare `python`/`python3`.
- Package under test is `src/alignair` (lowercase). Not `src/AlignAIR` (that is legacy TF).
- Git commit messages: NEVER add Co-Authored-By or any Claude/AI mention (project rule).
- Preserve the **dynamic-genotype** guarantee: germline reps/tokens are runtime inputs; the raw-ACGT base-match channel is floored (`match_floor`) so novel alleles stay callable. No allele memorization in weights.
- Compute the cosine `M` and ALL `logsumexp`/`softmax`/`cumsum` reductions in fp32 even under AMP (`.float()` the inputs). `NEG = -1e4`.
- Soft-argmax / softmax / cumsum over germline columns must exclude `germ_mask`-invalid (NEG-masked) columns.
- The mandatory diagonal correctness tests MUST use **non-uniform random `w`** (a `w≡1` test gives false confidence — see Task 2).
- New aligner selected by `config.aligner == "pointer"`; keep `"softdp"` and `"diagonal"` working unchanged.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit. One logical change per commit.

## File Structure

- `src/alignair/nn/base_match.py` — NEW. `base_match_channel(M, seg_tok, germ_tok, seg_reliability, match_weight, match_floor)` extracted verbatim from `SoftDPAligner._scores` so both aligners share identical code (spec §4.1, B2).
- `src/alignair/nn/pointer_aligner.py` — NEW. Diagonal-extraction helpers (`weighted_leading_diag`, `weighted_reverse_diag`, `banded_start_end`) + `BandedPointerAligner(nn.Module)` (`forward`, `alignment_score`) (spec §4).
- `src/alignair/nn/germline_aligner.py` — MODIFY `decode_germline_coords` to add `soft=` soft-argmax decode (spec §3/§5, S3).
- `src/alignair/losses/dnalignair_loss.py` — MODIFY the germline-coord loss block to soft-argmax-L1 + CDF + consistency, one normalized Kendall term (spec §5, S1/S2).
- `src/alignair/core/dnalignair.py` — MODIFY aligner selection (`"pointer"`) and `germline_coords` signature to thread `seg_tok/germ_tok/seg_reliability` (spec §3, B2).
- `src/alignair/training/germline_tf.py` — MODIFY `compute_germline_logits` to gather + pass `seg_tok/germ_tok/seg_reliability` (spec §3/§12, B2).
- `src/alignair/config/dnalignair_config.py` — MODIFY: add `aligner="pointer"` support, `band_half_width`, loss-term weights, τ schedule fields.
- `src/alignair/inference/dnalignair_infer.py`, `src/alignair/gym/instrument/evaluator.py`, `src/alignair/training/gym_trainer.py` — MODIFY decode call sites to `soft=True` (spec §3, S3).
- `scripts/exp_aligner_ablation.py` — NEW. Runs the §9 ablation ladder on the frozen gym lattice.
- Tests mirror each: `tests/alignair/nn/test_base_match.py`, `test_pointer_aligner.py`, `tests/alignair/losses/test_germline_coord_loss.py`, plus edits to `tests/alignair/nn/test_germline_aligner.py`.

---

## Task 1: Extract the base-match channel into a shared helper

**Files:**
- Create: `src/alignair/nn/base_match.py`
- Test: `tests/alignair/nn/test_base_match.py`
- Reference (do not change yet): `src/alignair/nn/soft_dp_aligner.py:103-126`

**Interfaces:**
- Produces: `base_match_channel(M, seg_tok, germ_tok, seg_reliability, match_weight, match_floor) -> Tensor[B,S,Lg]` — returns `M` with the SNP base-match term added; pass-through (returns `M` unchanged) when `seg_tok is None or germ_tok is None`. `match_weight` is a scalar `nn.Parameter`-like tensor, `match_floor` a python float.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/nn/test_base_match.py
import math
import torch
from alignair.nn.base_match import base_match_channel


def test_passthrough_without_tokens():
    M = torch.randn(2, 4, 6)
    out = base_match_channel(M, None, None, None, torch.tensor(1.0), 1.0)
    assert torch.equal(out, M)


def test_matches_softdp_scores_exactly():
    # identical to SoftDPAligner._scores base-match math (no reliability)
    torch.manual_seed(0)
    B, S, Lg = 2, 5, 7
    M = torch.randn(B, S, Lg)
    seg_tok = torch.randint(0, 5, (B, S))
    germ_tok = torch.randint(0, 5, (B, Lg))
    mw, floor = torch.tensor(1.0), 1.0
    out = base_match_channel(M, seg_tok, germ_tok, None, mw, floor)
    # reference: the exact lines from soft_dp_aligner._scores
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)
    u = real.float() * (2.0 * (st == gt).float() - 1.0)
    lam = floor + torch.nn.functional.softplus(mw)
    ref = M + lam * u
    assert torch.allclose(out, ref, atol=1e-6)


def test_reliability_gates_term_toward_zero():
    # reliability 0 -> base-match contribution collapses (a*u + norm term ~ 0)
    B, S, Lg = 1, 4, 5
    M = torch.zeros(B, S, Lg)
    seg_tok = torch.tensor([[1, 2, 3, 4]])
    germ_tok = torch.tensor([[1, 2, 3, 4, 1]])
    rel0 = torch.zeros(B, S)
    out0 = base_match_channel(M, seg_tok, germ_tok, rel0, torch.tensor(1.0), 1.0)
    # with a=0: a*u = 0 and norm = log(1+3)-log(4) = 0 -> M unchanged
    assert torch.allclose(out0, M, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_base_match.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.nn.base_match'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/nn/base_match.py
"""Shared SNP-sensitive base-match channel for germline aligners.

Extracted verbatim from SoftDPAligner._scores so the soft-DP and the pointer
aligner use identical emission math: a raw-ACGT +1/-1 match term (floored so a
never-seen germline still aligns on real bases — the dynamic-genotype guarantee),
optionally state-conditioned by a per-read-position reliability that down-weights
likely-SHM positions (so a substitution does not penalise the true allele).
"""
import math

import torch
import torch.nn.functional as F


def base_match_channel(M: torch.Tensor, seg_tok, germ_tok, seg_reliability,
                       match_weight: torch.Tensor, match_floor: float) -> torch.Tensor:
    """Add the SNP base-match term to score matrix M (B,S,Lg). Pass-through when
    tokens are absent. seg_tok (B,S), germ_tok (B,Lg), seg_reliability (B,S) in [0,1]."""
    if seg_tok is None or germ_tok is None:
        return M
    st, gt = seg_tok.unsqueeze(2), germ_tok.unsqueeze(1)            # (B,S,1),(B,1,Lg)
    real = (st >= 1) & (st <= 4) & (gt >= 1) & (gt <= 4)           # ACGT only
    u = real.float() * (2.0 * (st == gt).float() - 1.0)           # +1 match / -1 mismatch
    lam = match_floor + F.softplus(match_weight)                   # scalar coefficient
    if seg_reliability is not None:
        a = (lam * seg_reliability.clamp(0.0, 1.0)).unsqueeze(2)   # (B,S,1)
        norm = torch.log(a.exp() + 3.0 * (-a).exp()) - math.log(4.0)  # (B,S,1)
        return M + a * u - norm * real.float()
    return M + lam * u
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_base_match.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Refactor SoftDPAligner._scores to call the helper (no behavior change)**

In `src/alignair/nn/soft_dp_aligner.py`, replace the base-match block inside `_scores` (lines ~108-125, everything after `M = scale * torch.einsum(...)`) with:

```python
        from .base_match import base_match_channel
        M = base_match_channel(M, seg_tok, germ_tok, seg_reliability,
                               self._match_weight, self.match_floor)
        return M
```

- [ ] **Step 6: Run the existing soft-DP tests to confirm no regression**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_soft_dp_aligner.py -v`
Expected: PASS (all existing tests still green).

- [ ] **Step 7: Commit**

```bash
git add src/alignair/nn/base_match.py tests/alignair/nn/test_base_match.py src/alignair/nn/soft_dp_aligner.py
git commit -m "Extract shared base-match channel for germline aligners"
```

---

## Task 2: Diagonal-extraction helpers (the B1-critical reverse diagonal)

**Files:**
- Create: `src/alignair/nn/pointer_aligner.py` (helpers only this task)
- Test: `tests/alignair/nn/test_pointer_aligner.py`

**Interfaces:**
- Produces:
  - `weighted_leading_diag(M[B,S,Lg], w[B,S,1]) -> Tensor[B,Lg]` where `out[b,o] = (Σ_i w[b,i]·M[b,i,o+i]) / Σ_i w[b,i]`, summing only over `i` with `o+i < Lg`.
  - `weighted_reverse_diag(M[B,S,Lg], w[B,S,1]) -> Tensor[B,Lg]` where `out[b,o] = (Σ_i w[b,S-1-i]·M[b,S-1-i,o-i]) / Σ_i w[b, ...]`, summing only over `i` with `0 <= o-i < Lg`. **MUST flip `w` into the flipped-row frame.**

- [ ] **Step 1: Write the failing test (non-uniform w, reference loops)**

```python
# tests/alignair/nn/test_pointer_aligner.py
import torch
from alignair.nn.pointer_aligner import weighted_leading_diag, weighted_reverse_diag


def _ref_leading(M, w):
    B, S, Lg = M.shape
    out = torch.zeros(B, Lg)
    for b in range(B):
        denom = w[b, :, 0].sum().clamp(min=1e-6)
        for o in range(Lg):
            s = 0.0
            for i in range(S):
                if o + i < Lg:
                    s += w[b, i, 0] * M[b, i, o + i]
            out[b, o] = s / denom
    return out


def _ref_reverse(M, w):
    B, S, Lg = M.shape
    out = torch.zeros(B, Lg)
    for b in range(B):
        denom = w[b, :, 0].sum().clamp(min=1e-6)
        for o in range(Lg):
            s = 0.0
            for i in range(S):
                j = o - i
                if 0 <= j < Lg:
                    s += w[b, S - 1 - i, 0] * M[b, S - 1 - i, j]
            out[b, o] = s / denom
    return out


def test_leading_diag_matches_reference_nonuniform_w():
    torch.manual_seed(1)
    B, S, Lg = 2, 5, 7
    M = torch.randn(B, S, Lg)
    w = torch.rand(B, S, 1)                       # NON-UNIFORM (mandatory)
    assert torch.allclose(weighted_leading_diag(M, w), _ref_leading(M, w), atol=1e-5)


def test_reverse_diag_matches_reference_nonuniform_w():
    torch.manual_seed(2)
    B, S, Lg = 2, 5, 7
    M = torch.randn(B, S, Lg)
    w = torch.rand(B, S, 1)                       # NON-UNIFORM (catches the flip-w bug)
    assert torch.allclose(weighted_reverse_diag(M, w), _ref_reverse(M, w), atol=1e-5)


def test_diag_helpers_are_autograd_safe():
    M = torch.randn(1, 4, 6, requires_grad=True)
    w = torch.rand(1, 4, 1)
    (weighted_leading_diag(M, w).sum() + weighted_reverse_diag(M, w).sum()).backward()
    assert M.grad is not None and torch.isfinite(M.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_aligner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.nn.pointer_aligner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/nn/pointer_aligner.py
"""BandedPointerAligner: a fully-parallel, reference-conditioned germline pointer
head that replaces the sequential soft-DP. Start/end logits are weighted leading /
reverse diagonals of the score matrix M, extracted in single CUDA launches via
as_strided (no S-length recurrence). See the design spec §4."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_match import base_match_channel

NEG = -1e4


def weighted_leading_diag(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[b,o] = (Σ_i w[b,i]·M[b,i,o+i]) / Σ_i w[b,i], for o+i < Lg. (B,Lg)."""
    B, S, Lg = M.shape
    Mp = F.pad(M, (0, S))                                  # (B,S,Lg+S)
    bs, ss, es = Mp.stride()
    diag = Mp.as_strided((B, S, Lg), (bs, ss + es, es))    # diag[b,i,o] = Mp[b,i,o+i]
    num = (w * diag).sum(dim=1)                            # (B,Lg)
    return num / w.sum(dim=1).clamp(min=1e-6)


def weighted_reverse_diag(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[b,o] = (Σ_i w[b,S-1-i]·M[b,S-1-i,o-i]) / Σ_i w[b,i], for 0<=o-i<Lg. (B,Lg).
    CRITICAL: w is flipped into the flipped-row frame (wf = flip(w)), else w[i] pairs with
    row S-1-i and the head trains to the wrong coordinate once w is non-uniform (spec §4.2 B1)."""
    B, S, Lg = M.shape
    Mf = torch.flip(M, (1, 2))                             # reverse read rows AND germ cols
    Mfp = F.pad(Mf, (0, S))
    bs, ss, es = Mfp.stride()
    led = Mfp.as_strided((B, S, Lg), (bs, ss + es, es))    # led[b,i,o] = Mf[b,i,o+i]
    wf = torch.flip(w, (1,))                               # weights into led's frame
    num = torch.flip((wf * led).sum(dim=1), (1,))          # (B,Lg)
    return num / w.sum(dim=1).clamp(min=1e-6)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_aligner.py -v`
Expected: PASS (3 passed). If `test_reverse_diag_matches_reference_nonuniform_w` fails, the `wf = torch.flip(w, (1,))` flip is missing — this is the B1 bug; do not "fix" by changing the reference loop.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/pointer_aligner.py tests/alignair/nn/test_pointer_aligner.py
git commit -m "Add diagonal-extraction helpers with flip-w reverse diagonal"
```

---

## Task 3: BandedPointerAligner module (single-diagonal core + fast reader)

**Files:**
- Modify: `src/alignair/nn/pointer_aligner.py`
- Test: `tests/alignair/nn/test_pointer_aligner.py`

**Interfaces:**
- Consumes: `weighted_leading_diag`, `weighted_reverse_diag`, `base_match_channel`.
- Produces: `BandedPointerAligner(d_model, match_floor=1.0, max_len=1024, band_half_width=0)` with:
  - `forward(seg_reps[B,S,d], seg_mask[B,S], germ_reps[B,Lg,d], germ_mask[B,Lg], seg_tok=None, germ_tok=None, seg_reliability=None) -> (start_logits[B,Lg], end_logits[B,Lg])`.
  - `alignment_score(seg_reps, seg_mask, germ_reps, germ_mask, seg_tok=None, germ_tok=None, seg_reliability=None) -> Tensor[B]` (fast diagonal seed score; same signature as `SoftDPAligner.alignment_score`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_pointer_aligner.py
from alignair.nn.pointer_aligner import BandedPointerAligner, NEG


def _toy(B=2, S=8, Lg=20, d=16):
    torch.manual_seed(3)
    seg = torch.randn(B, S, d)
    germ = torch.randn(B, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool)
    gm = torch.ones(B, Lg, dtype=torch.bool)
    return seg, sm, germ, gm


def test_forward_shapes_and_masking():
    al = BandedPointerAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    gm[:, 15:] = False                                  # invalid germline tail
    sl, el = al(seg, sm, germ, gm)
    assert sl.shape == (2, 20) and el.shape == (2, 20)
    assert (sl[:, 15:] <= NEG + 1).all() and (el[:, 15:] <= NEG + 1).all()


def test_forward_localizes_planted_diagonal():
    # plant a high-cosine diagonal: seg rows == germ rows at offset 4
    al = BandedPointerAligner(d_model=16)
    B, S, Lg, d, off = 1, 6, 20, 16, 4
    g = torch.randn(B, Lg, d)
    seg = g[:, off:off + S, :].clone()                  # seg aligns to germ[off:off+S]
    sm = torch.ones(B, S, dtype=torch.bool)
    gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    assert sl.argmax(-1).item() == off
    assert el.argmax(-1).item() == off + S - 1


def test_alignment_score_shape_and_finite():
    al = BandedPointerAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    sc = al.alignment_score(seg, sm, germ, gm)
    assert sc.shape == (2,) and torch.isfinite(sc).all()


def test_novel_allele_floor_keeps_base_match_alive():
    # match_floor>0 means even with zeroed projections the base-match channel scores
    al = BandedPointerAligner(d_model=16, match_floor=1.0)
    with torch.no_grad():
        al.seg_proj.weight.zero_(); al.seg_proj.bias.zero_()
        al.germ_proj.weight.zero_(); al.germ_proj.bias.zero_()
    B, S, Lg = 1, 5, 12
    seg = torch.zeros(B, S, 16); germ = torch.zeros(B, Lg, 16)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = torch.tensor([[1, 2, 3, 4, 1]])
    germ_tok = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])
    sl, el = al(seg, sm, germ, gm, seg_tok=seg_tok, germ_tok=germ_tok)
    assert sl.argmax(-1).item() == 0                    # base match localizes start at 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_aligner.py -k "forward or alignment_score or novel" -v`
Expected: FAIL with `cannot import name 'BandedPointerAligner'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/alignair/nn/pointer_aligner.py`:

```python
class BandedPointerAligner(nn.Module):
    """Drop-in for SoftDPAligner: returns (start_logits, end_logits) over germline
    positions and a fast diagonal alignment_score. Single-diagonal core (band_half_width
    extends it; see Task 7). seg_tok/germ_tok/seg_reliability enable the base-match
    channel + reliability gating (spec §4.1/§4.3)."""

    def __init__(self, d_model: int, match_floor: float = 1.0, max_len: int = 1024,
                 band_half_width: int = 0):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))       # scale ~5
        self.log_temp = nn.Parameter(torch.zeros(()))          # learned softmax temperature
        self._match_weight = nn.Parameter(torch.tensor(1.0))
        self.match_floor = float(match_floor)
        self.diag_bias = nn.Parameter(torch.zeros(max_len))    # per-position weight, init uniform
        self.band_half_width = int(band_half_width)

    def _M(self, seg_reps, germ_reps, germ_mask, seg_tok, germ_tok, seg_reliability):
        S = F.normalize(self.seg_proj(seg_reps).float(), dim=-1)
        G = F.normalize(self.germ_proj(germ_reps).float(), dim=-1)
        scale = self.log_scale.clamp(-2.0, 3.0).exp()
        M = scale * torch.einsum("bid,bjd->bij", S, G)         # (B,S,Lg) fp32
        M = base_match_channel(M, seg_tok, germ_tok, seg_reliability,
                               self._match_weight, self.match_floor)
        return M.masked_fill(~germ_mask.unsqueeze(1), NEG)

    def _weights(self, seg_mask, seg_reliability):
        B, S = seg_mask.shape
        w = F.softmax(self.diag_bias[:S], dim=0)[None, :, None].expand(B, S, 1).clone()
        if seg_reliability is not None:
            w = w * seg_reliability.detach().clamp(0.0, 1.0).unsqueeze(-1)
        return w.masked_fill(~seg_mask.unsqueeze(-1), 0.0)     # mask padded read positions

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask,
                seg_tok=None, germ_tok=None, seg_reliability=None):
        M = self._M(seg_reps, germ_reps, germ_mask, seg_tok, germ_tok, seg_reliability)
        w = self._weights(seg_mask, seg_reliability)
        temp = self.log_temp.clamp(0.0, 3.4).exp()
        start = temp * weighted_leading_diag(M, w)
        end = temp * weighted_reverse_diag(M, w)
        start = start.masked_fill(~germ_mask, NEG)
        end = end.masked_fill(~germ_mask, NEG)
        return start, end

    def alignment_score(self, seg_reps, seg_mask, germ_reps, germ_mask,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Fast diagonal seed score (B,): length-normalized logsumexp over start-offset
        diagonal scores. Replaces the soft-DP alignment_score on the training+seed path."""
        start, _ = self.forward(seg_reps, seg_mask, germ_reps, germ_mask,
                                seg_tok, germ_tok, seg_reliability)
        n_valid = germ_mask.sum(dim=-1).clamp(min=1).to(start.dtype)
        return torch.logsumexp(start, dim=-1) - torch.log(n_valid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_aligner.py -v`
Expected: PASS (all, including the planted-diagonal localization and the novel-allele floor).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/pointer_aligner.py tests/alignair/nn/test_pointer_aligner.py
git commit -m "Add BandedPointerAligner single-diagonal core + fast reader"
```

---

## Task 4: Soft-argmax decode option

**Files:**
- Modify: `src/alignair/nn/germline_aligner.py:72-76` (`decode_germline_coords`)
- Test: `tests/alignair/nn/test_germline_aligner.py`

**Interfaces:**
- Produces: `decode_germline_coords(start_logits, end_logits, soft=False) -> (gs, ge)`. `soft=False` keeps argmax (gs=argmax, ge=argmax+1). `soft=True` returns the rounded soft-argmax expected position over valid (non-NEG) columns (gs=round(E[start]), ge=round(E[end])+1), as long-tensor coords.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_germline_aligner.py
import torch
from alignair.nn.germline_aligner import decode_germline_coords


def test_soft_argmax_decode_centers_between_two_peaks():
    # two equal peaks at columns 4 and 6 -> soft-argmax ~5; argmax picks 4
    logits = torch.full((1, 11), -1e4)
    logits[0, 4] = 2.0
    logits[0, 6] = 2.0
    gs_hard, _ = decode_germline_coords(logits, logits, soft=False)
    gs_soft, _ = decode_germline_coords(logits, logits, soft=True)
    assert gs_hard.item() == 4
    assert gs_soft.item() == 5


def test_soft_decode_ignores_neg_masked_columns():
    logits = torch.full((1, 8), -1e4)
    logits[0, 2] = 5.0                      # only valid peak
    gs, ge = decode_germline_coords(logits, logits, soft=True)
    assert gs.item() == 2 and ge.item() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_germline_aligner.py -k soft -v`
Expected: FAIL with `TypeError: decode_germline_coords() got an unexpected keyword argument 'soft'`.

- [ ] **Step 3: Write minimal implementation**

Replace `decode_germline_coords` in `src/alignair/nn/germline_aligner.py`:

```python
def decode_germline_coords(start_logits: torch.Tensor, end_logits: torch.Tensor,
                           soft: bool = False):
    """Germline_start and germline_end (end-exclusive). soft=False: argmax
    (gs=argmax, ge=argmax+1). soft=True: rounded soft-argmax expected position over
    valid (finite) columns — sub-integer-stable, kills argmax-plateau jitter."""
    if not soft:
        gs = start_logits.argmax(dim=-1)
        ge = end_logits.argmax(dim=-1) + 1
        return gs, ge

    def _expected(logits):
        pos = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
        p = torch.softmax(logits.float(), dim=-1)        # NEG columns -> ~0 weight
        return (p * pos).sum(dim=-1)

    gs = _expected(start_logits).round().long()
    ge = (_expected(end_logits).round().long()) + 1
    return gs, ge
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_germline_aligner.py -v`
Expected: PASS (new tests + all existing germline-aligner tests still green — default `soft=False` is unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/germline_aligner.py tests/alignair/nn/test_germline_aligner.py
git commit -m "Add soft-argmax decode option to decode_germline_coords"
```

---

## Task 5: Redesigned germline coordinate loss

**Files:**
- Modify: `src/alignair/losses/dnalignair_loss.py:84-99` (the `if germline_logits is not None:` block)
- Test: `tests/alignair/losses/test_germline_coord_loss.py`

**Interfaces:**
- Produces: a module-level `germline_coord_loss(start_logits[B,Lg], end_logits[B,Lg], gt_start[B], gt_end[B], tau=1.0, indel_free=None) -> Tensor[B]` (per-row, unreduced) combining `0.3·CE + 1.0·L_exp + 0.5·L_cdf + 0.3·L_cons`, with `L_exp`/`L_cons` normalized by `Lg`. `DNAlignAIRLoss.forward` uses it, masked + meaned, under the single `{g}_germline` Kendall head.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/losses/test_germline_coord_loss.py
import torch
from alignair.losses.dnalignair_loss import germline_coord_loss


def test_loss_minimized_at_truth():
    B, Lg = 4, 30
    gt_s = torch.tensor([5, 10, 3, 12])
    gt_e = torch.tensor([20, 25, 18, 28])
    pos = torch.arange(Lg).float()
    # sharp logits centered on truth vs centered 5nt off
    good_s = -(pos[None] - gt_s[:, None].float()) ** 2
    good_e = -(pos[None] - (gt_e[:, None].float() - 1)) ** 2
    bad_s = -(pos[None] - (gt_s[:, None].float() + 5)) ** 2
    bad_e = -(pos[None] - (gt_e[:, None].float() - 1 + 5)) ** 2
    good = germline_coord_loss(good_s, good_e, gt_s, gt_e).mean()
    bad = germline_coord_loss(bad_s, bad_e, gt_s, gt_e).mean()
    assert good < bad


def test_consistency_term_penalizes_span_mismatch_on_indel_free_rows():
    # end far from start+span should cost more when indel_free=True
    B, Lg = 1, 40
    gt_s = torch.tensor([5]); gt_e = torch.tensor([25])      # span 20
    pos = torch.arange(Lg).float()
    s = -(pos[None] - 5.0) ** 2
    e_ok = -(pos[None] - 24.0) ** 2                          # end 24 -> span 20 (matches)
    e_bad = -(pos[None] - 34.0) ** 2                         # end 34 -> span 30 (mismatch)
    free = torch.tensor([True])
    l_ok = germline_coord_loss(s, e_ok, gt_s, gt_e, indel_free=free).mean()
    l_bad = germline_coord_loss(s, e_bad, gt_s, gt_e, indel_free=free).mean()
    assert l_bad > l_ok


def test_returns_per_row_vector():
    B, Lg = 3, 20
    s = torch.randn(B, Lg); e = torch.randn(B, Lg)
    out = germline_coord_loss(s, e, torch.tensor([1, 2, 3]), torch.tensor([5, 6, 7]))
    assert out.shape == (B,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses/test_germline_coord_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'germline_coord_loss'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/alignair/losses/dnalignair_loss.py` (module level, after imports):

```python
def germline_coord_loss(start_logits, end_logits, gt_start, gt_end,
                        tau: float = 1.0, indel_free=None):
    """Per-row germline coordinate loss (spec §5): 0.3*CE + 1.0*L_exp(soft-argmax L1)
    + 0.5*L_cdf(ordinal soft-step) + 0.3*L_cons(start/end span consistency). L_exp/L_cons
    normalized by Lg so they sit near the CE/CDF scale (one Kendall head, fixed inner
    weights). gt_end is END-EXCLUSIVE; the end target column is gt_end-1."""
    B, Lg = start_logits.shape
    pos = torch.arange(Lg, device=start_logits.device, dtype=torch.float32)
    gs = gt_start.clamp(min=0, max=Lg - 1)
    ge = (gt_end - 1).clamp(min=0, max=Lg - 1)            # inclusive end column

    ce = F.cross_entropy(start_logits, gs, reduction="none") + \
        F.cross_entropy(end_logits, ge, reduction="none")

    ps = torch.softmax(start_logits.float(), dim=-1)
    pe = torch.softmax(end_logits.float(), dim=-1)
    cs = (ps * pos).sum(-1)                               # E[start]
    cee = (pe * pos).sum(-1)                              # E[end] (inclusive)
    l_exp = (F.smooth_l1_loss(cs, gs.float(), reduction="none")
             + F.smooth_l1_loss(cee, ge.float(), reduction="none")) / Lg

    def _cdf(p, y):
        cdf = torch.cumsum(p, dim=-1)
        tgt = torch.sigmoid((pos[None] - y[:, None].float()) / tau)
        return ((cdf - tgt) ** 2).sum(-1)
    l_cdf = _cdf(ps, gs) + _cdf(pe, ge)

    span_pred = cee - cs + 1.0
    span_gt = (gt_end - gt_start).float()
    l_cons = F.smooth_l1_loss(span_pred, span_gt, reduction="none") / Lg
    if indel_free is not None:
        l_cons = l_cons * indel_free.float()

    return 0.3 * ce + 1.0 * l_exp + 0.5 * l_cdf + 0.3 * l_cons
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses/test_germline_coord_loss.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Wire it into DNAlignAIRLoss.forward (replace the CE block)**

In `src/alignair/losses/dnalignair_loss.py`, replace the body of the `if germline_logits is not None:` loop (the `per_row = (F.cross_entropy(sl, gs, ...) + F.cross_entropy(el, ge, ...))` lines) with:

```python
        if germline_logits is not None:
            for g in genes:
                if g not in germline_logits:
                    continue
                sl, el = germline_logits[g]
                gs = batch[f"{g}_germline_start"]
                ge = batch[f"{g}_germline_end"]
                indel_free = None
                if "indel_count" in batch:
                    indel_free = (batch["indel_count"].reshape(-1) < 0.5)
                tau = float(getattr(self, "coord_tau", 1.0))
                per_row = germline_coord_loss(sl, el, gs, ge, tau=tau,
                                              indel_free=indel_free)
                mask = batch.get(f"{g}_supervise")
                if mask is not None:
                    gl = (per_row * mask).sum() / mask.sum().clamp(min=1.0)
                else:
                    gl = per_row.mean()
                total = total + add(f"{g}_germline", gl)
                comp[f"{g}_germline"] = gl.detach()
```

Add `self.coord_tau = 1.0` in `DNAlignAIRLoss.__init__` (the trainer can anneal it via `cosine_sigma_schedule`).

- [ ] **Step 6: Run the loss test suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses/ -v`
Expected: PASS (existing `test_dnalignair_loss.py`, `test_loss_protection.py` still green; the germline term now uses the new loss but the Kendall wiring is unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/alignair/losses/dnalignair_loss.py tests/alignair/losses/test_germline_coord_loss.py
git commit -m "Replace hard-CE germline coord loss with soft-argmax + CDF + consistency"
```

---

## Task 6: Thread tokens + reliability through the coordinate path (B2)

**Files:**
- Modify: `src/alignair/core/dnalignair.py:188-190` (`germline_coords`) and `:92` (aligner selection)
- Modify: `src/alignair/training/germline_tf.py:11-36` (`compute_germline_logits`)
- Modify: `src/alignair/config/dnalignair_config.py`
- Test: `tests/alignair/training/test_germline_tf_pointer.py`

**Interfaces:**
- Consumes: `BandedPointerAligner.forward(..., seg_tok, germ_tok, seg_reliability)`.
- Produces: `germline_coords(seg_reps, seg_mask, germ_reps, germ_mask, seg_tok=None, germ_tok=None, seg_reliability=None)` and `compute_germline_logits(..., state_logits=None)` that gathers and passes the three extra args when the aligner supports them.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/training/test_germline_tf_pointer.py
import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR


def test_pointer_aligner_selected_and_coords_run():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, aligner="pointer")
    model = DNAlignAIR(cfg)
    from alignair.nn.pointer_aligner import BandedPointerAligner
    assert isinstance(model.aligner, BandedPointerAligner)
    B, S, Lg, d = 2, 6, 12, 32
    seg = torch.randn(B, S, d); germ = torch.randn(B, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = torch.randint(0, 5, (B, S)); germ_tok = torch.randint(0, 5, (B, Lg))
    rel = torch.rand(B, S)
    sl, el = model.germline_coords(seg, sm, germ, gm, seg_tok=seg_tok,
                                   germ_tok=germ_tok, seg_reliability=rel)
    assert sl.shape == (B, Lg) and el.shape == (B, Lg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_germline_tf_pointer.py -v`
Expected: FAIL — either `aligner="pointer"` not constructed (still `GermlineAligner`) or `germline_coords` rejects the new kwargs.

- [ ] **Step 3: Implement config + aligner selection + signature**

In `src/alignair/config/dnalignair_config.py`, update the `aligner` field doc and add fields:

```python
    aligner: str = "softdp"  # "softdp" | "diagonal" | "pointer" (fast parallel pointer head)
    band_half_width: int = 0  # pointer aligner indel band (0 = single diagonal)
```

In `src/alignair/core/dnalignair.py`, import and extend selection at line ~92:

```python
from ..nn.pointer_aligner import BandedPointerAligner
...
        _aligner = getattr(config, "aligner", "diagonal")
        if _aligner == "softdp":
            self.aligner = SoftDPAligner(d_model=d)
        elif _aligner == "pointer":
            self.aligner = BandedPointerAligner(
                d_model=d, max_len=config.max_len,
                band_half_width=getattr(config, "band_half_width", 0))
        else:
            self.aligner = GermlineAligner(d_model=d)
```

Update `germline_coords` (line ~188) to forward optional args:

```python
    def germline_coords(self, seg_reps, seg_mask, germ_reps, germ_mask,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Align a gene's segment reps to a chosen allele's per-position germline reps."""
        try:
            return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask,
                                seg_tok=seg_tok, germ_tok=germ_tok,
                                seg_reliability=seg_reliability)
        except TypeError:
            return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask)
```

- [ ] **Step 4: Run the selection test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_germline_tf_pointer.py -v`
Expected: PASS.

- [ ] **Step 5: Thread tokens + reliability in compute_germline_logits**

Replace `src/alignair/training/germline_tf.py` body of the per-gene loop to gather tokens + reliability (lift the gather from `dnalignair_infer.py:299-323`):

```python
from ..core.dnalignair import extract_segment_tokens, extract_segment


def compute_germline_logits(model, tokens, mask, batch, ref_emb, has_d: bool,
                            region_labels=None, allele_idx: dict | None = None,
                            state_logits=None):
    genes = ["v", "j"] + (["d"] if has_d else [])
    rl = region_labels if region_labels is not None else batch["region_labels"]
    out = {}
    for g in genes:
        G = g.upper()
        seg_tok, seg_mask = extract_segment_tokens(tokens, mask, rl, G)
        seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
        if allele_idx is not None and G in allele_idx:
            idx = allele_idx[G]
        elif f"{g}_primary_idx" in batch:
            idx = batch[f"{g}_primary_idx"]
        else:
            multihot = batch[f"{g}_allele"]
            idx = torch.where(multihot.sum(dim=1) > 0, multihot.argmax(dim=1),
                              torch.zeros(multihot.shape[0], dtype=torch.long,
                                          device=multihot.device))
        germ_reps = ref_emb[G]["pos_reps"][idx]
        germ_mask = ref_emb[G]["pos_mask"][idx]
        germ_tok = ref_emb[G]["pos_tok"][idx]
        seg_rel = None
        if state_logits is not None:
            from ..nn.state_head import state_reliability
            seg_state, _ = extract_segment(state_logits, mask, rl, G)
            seg_rel = state_reliability(seg_state)
        out[g] = model.germline_coords(seg_reps, seg_mask, germ_reps, germ_mask,
                                       seg_tok=seg_tok, germ_tok=germ_tok,
                                       seg_reliability=seg_rel)
    return out
```

Note `import torch` must remain at the top of the file.

- [ ] **Step 6: Pass state_logits from the trainer**

In `src/alignair/training/gym_trainer.py:164`, update the call to pass state logits:

```python
            germline_logits = compute_germline_logits(
                self.model, canon, batch["mask"], batch, ref_emb, self.has_d,
                region_labels=sup_regions, state_logits=out["state_logits"])
```

- [ ] **Step 7: Run the full germline_tf + trainer-touching tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/ tests/alignair/gym/ -q`
Expected: PASS (the `state_logits=None` default keeps soft-DP / diagonal aligners unchanged; pointer path now gets reliability).

- [ ] **Step 8: Commit**

```bash
git add src/alignair/core/dnalignair.py src/alignair/training/germline_tf.py src/alignair/training/gym_trainer.py src/alignair/config/dnalignair_config.py tests/alignair/training/test_germline_tf_pointer.py
git commit -m "Thread tokens + reliability through pointer aligner coordinate path"
```

---

## Task 7: Switch all decode sites to soft-argmax (S3)

**Files:**
- Modify: `src/alignair/inference/dnalignair_infer.py:286`
- Modify: `src/alignair/gym/instrument/evaluator.py:72`
- Modify: `src/alignair/training/gym_trainer.py:325,328`
- Test: `tests/alignair/gym/test_evaluator_soft_decode.py`

**Interfaces:**
- Consumes: `decode_germline_coords(..., soft=True)` from Task 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/gym/test_evaluator_soft_decode.py
import inspect
from alignair.gym.instrument import evaluator


def test_evaluator_uses_soft_decode():
    src = inspect.getsource(evaluator)
    assert "soft=True" in src, "lattice eval must decode with soft-argmax (spec S3)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/test_evaluator_soft_decode.py -v`
Expected: FAIL (no `soft=True` yet).

- [ ] **Step 3: Implement the decode switches**

In `src/alignair/gym/instrument/evaluator.py` (line ~72), change:
```python
                gs, ge = decode_germline_coords(gl[g][0], gl[g][1], soft=True)
```
In `src/alignair/training/gym_trainer.py` (lines ~325 and ~328), change both `decode_germline_coords(...)` calls to add `soft=True`.
In `src/alignair/inference/dnalignair_infer.py` (line ~286), change:
```python
        gcoord = {g: decode_germline_coords(gl[g][0], gl[g][1], soft=True) for g in genes}
```

- [ ] **Step 4: Run test + the gym/inference suites**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/gym/test_evaluator_soft_decode.py tests/alignair/gym/ tests/alignair/inference/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/gym/instrument/evaluator.py src/alignair/training/gym_trainer.py src/alignair/inference/dnalignair_infer.py tests/alignair/gym/test_evaluator_soft_decode.py
git commit -m "Decode germline coords with soft-argmax across train/eval/inference"
```

---

## Task 8: Banded indel-tolerant diagonals (ablation #6)

**Files:**
- Modify: `src/alignair/nn/pointer_aligner.py`
- Test: `tests/alignair/nn/test_pointer_aligner.py`

**Interfaces:**
- Produces: `banded_start_end(M[B,S,Lg], w[B,S,1], gamma[2G+1], G) -> (start[B,Lg], end[B,Lg])` — per-Δ leading/reverse diagonals combined by `logsumexp_Δ(diag_Δ + gamma[Δ])`. END band uses the flip-w reverse diagonal per Δ (spec §4.4). `BandedPointerAligner.forward` uses it when `band_half_width > 0`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_pointer_aligner.py
def test_banded_clean_diagonal_still_localizes():
    # with a band but a clean (no-indel) alignment, start/end still localize correctly
    from alignair.nn.pointer_aligner import BandedPointerAligner
    al = BandedPointerAligner(d_model=16, band_half_width=3)
    B, S, Lg, d, off = 1, 6, 20, 16, 4
    g = torch.randn(B, Lg, d)
    seg = g[:, off:off + S, :].clone()
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    assert sl.argmax(-1).item() == off
    assert el.argmax(-1).item() == off + S - 1


def test_banded_localizes_indel_shifted_alignment():
    # germ aligns to seg with a +1 offset jump after position 3 (a 1nt germline deletion)
    from alignair.nn.pointer_aligner import BandedPointerAligner
    al = BandedPointerAligner(d_model=16, band_half_width=3)
    B, S, Lg, d, off = 1, 6, 20, 16, 4
    g = torch.randn(B, Lg, d)
    seg = torch.empty(B, S, d)
    seg[:, :3] = g[:, off:off + 3]
    seg[:, 3:] = g[:, off + 4:off + 4 + 3]              # skip germ col off+3 (deletion)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    sl, el = al(seg, sm, g, gm)
    assert sl.argmax(-1).item() == off                 # start still localizes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_aligner.py -k banded -v`
Expected: FAIL (`band_half_width=3` path not implemented; forward ignores the band).

- [ ] **Step 3: Implement the banded reduction**

Add to `src/alignair/nn/pointer_aligner.py` and use it in `forward` when `band_half_width > 0`. The band shifts germline columns with `torch.roll` (then NEG-masks the wrapped region) rather than hand-rolling `as_strided` storage offsets:

```python
def banded_start_end(M, w, gamma, G):
    """Combine leading/reverse diagonals over offsets Δ∈[-G,G] by logsumexp(diag_Δ + γ_Δ).
    END band uses the flip-w reverse diagonal per Δ (spec §4.4 B1)."""
    B, S, Lg = M.shape
    starts, ends = [], []
    for k, delta in enumerate(range(-G, G + 1)):
        Md = torch.roll(M, shifts=-delta, dims=2)              # column shift by Δ
        if delta > 0:
            Md[:, :, -delta:] = NEG
        elif delta < 0:
            Md[:, :, :(-delta)] = NEG
        starts.append(weighted_leading_diag(Md, w) + gamma[k])
        ends.append(weighted_reverse_diag(Md, w) + gamma[k])
    start = torch.logsumexp(torch.stack(starts, 0), dim=0)
    end = torch.logsumexp(torch.stack(ends, 0), dim=0)
    return start, end
```

Add `self.band_gamma = nn.Parameter(torch.zeros(2 * band_half_width + 1))` in `__init__`, and in `forward` branch:

```python
        if self.band_half_width > 0:
            start, end = banded_start_end(M, w, self.band_gamma, self.band_half_width)
            start, end = temp * start, temp * end
        else:
            start = temp * weighted_leading_diag(M, w)
            end = temp * weighted_reverse_diag(M, w)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_aligner.py -v`
Expected: PASS (G=0 reduces to single diagonal; G=3 localizes the indel-shifted start).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/pointer_aligner.py tests/alignair/nn/test_pointer_aligner.py
git commit -m "Add banded indel-tolerant diagonals to BandedPointerAligner"
```

---

## Task 9: Ablation harness + latency benchmark

**Files:**
- Create: `scripts/exp_aligner_ablation.py`
- Create: `tests/alignair/nn/test_pointer_latency.py`

**Interfaces:**
- Consumes: `BandedPointerAligner`, `SoftDPAligner`, the frozen-lattice gym (`src/alignair/gym/instrument/`).

- [ ] **Step 1: Write the latency smoke test (asserts pointer ≪ soft-DP)**

```python
# tests/alignair/nn/test_pointer_latency.py
import time
import torch
from alignair.nn.pointer_aligner import BandedPointerAligner
from alignair.nn.soft_dp_aligner import SoftDPAligner


def test_pointer_forward_is_much_faster_than_softdp_on_cpu():
    B, S, Lg, d = 8, 64, 80, 32
    seg = torch.randn(B, S, d); germ = torch.randn(B, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    ptr = BandedPointerAligner(d_model=d); sdp = SoftDPAligner(d_model=d)

    def _time(fn, n=3):
        fn()                                  # warmup
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n

    tp = _time(lambda: ptr(seg, sm, germ, gm))
    ts = _time(lambda: sdp(seg, sm, germ, gm))
    assert tp < ts / 5, f"pointer {tp*1e3:.1f}ms not <<= softdp {ts*1e3:.1f}ms"
```

- [ ] **Step 2: Run test to verify it fails-or-passes meaningfully**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_pointer_latency.py -v`
Expected: PASS (pointer is dramatically faster even on CPU at this size). If it does not pass, the `as_strided` path has a hidden Python loop — investigate before proceeding.

- [ ] **Step 3: Write the ablation runner**

```python
# scripts/exp_aligner_ablation.py
"""Run the BandedPointerAligner ablation ladder (spec §9) on the frozen gym lattice.

Each arm trains a small DNAlignAIR for N steps with a given (aligner, loss, decode)
config, then evaluates competence per frozen-lattice cell with bootstrap CIs. Prints a
table so the operator can confirm each step climbs >= the prior arm on heavy_shm_fulllen
+ junction_boundary without regressing clean/indel/fragment.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/exp_aligner_ablation.py --arm pointer_softargmax \
      --steps 3000 --locus IGH --d-model 64
"""
import argparse


def run_arm(arm, steps, locus, d_model, n_per_cell, seed, device):
    # Arms map to config overrides:
    #   softdp_softargmax : aligner=softdp,  new loss   (ablation #1)
    #   pointer_ce        : aligner=pointer, CE-only    (ablation #2, both DP sites gone)
    #   pointer_softargmax: aligner=pointer, new loss   (#3)
    #   pointer_band      : + band_half_width=6         (#6)
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    overrides = {
        "softdp_softargmax": dict(aligner="softdp"),
        "pointer_ce": dict(aligner="pointer"),
        "pointer_softargmax": dict(aligner="pointer"),
        "pointer_band": dict(aligner="pointer", band_half_width=6),
    }[arm]
    cfg = DNAlignAIRConfig(d_model=d_model, n_layers=2, nhead=4, **overrides)
    # NB: the loss form (CE vs soft-argmax) is selected by a trainer flag; see GymTrainer.
    # This script wires GymTrainer + LatticeEvaluator exactly like scripts/exp_ramp_vs_factored.py.
    raise SystemExit("wire to GymTrainer + LatticeEvaluator following exp_ramp_vs_factored.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--locus", default="IGH")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-per-cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_arm(a.arm, a.steps, a.locus, a.d_model, a.n_per_cell, a.seed, a.device)


if __name__ == "__main__":
    main()
```

> The engineer must complete `run_arm` by copying the GymTrainer + LatticeEvaluator wiring from `scripts/exp_ramp_vs_factored.py` (the existing 3-arm experiment harness), substituting the aligner/loss config above. Do this by reading that script and mirroring its structure — do not invent new trainer APIs.

- [ ] **Step 4: Smoke-run the runner's arg parsing**

Run: `PYTHONPATH=src .venv/bin/python scripts/exp_aligner_ablation.py --arm pointer_ce --steps 1 --device cpu`
Expected: exits with the `SystemExit("wire to GymTrainer...")` message (confirms imports + config construction work before the engineer fills in the wiring).

- [ ] **Step 5: Commit**

```bash
git add scripts/exp_aligner_ablation.py tests/alignair/nn/test_pointer_latency.py
git commit -m "Add aligner ablation harness + pointer latency smoke test"
```

---

## Task 10: Full-suite regression + memory record

**Files:**
- No new code; verification + a memory update.

- [ ] **Step 1: Run the full alignair test suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/ -q`
Expected: PASS (no regressions; soft-DP and diagonal aligners unchanged, pointer aligner + new loss covered).

- [ ] **Step 2: Run the latency benchmark on GPU (if available)**

Run: `PYTHONPATH=src .venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"`
If CUDA is available, run a one-step `GymTrainer` with `aligner='pointer'` vs `aligner='softdp'` and record ms/step (target: pointer step ≪ soft-DP step, ~10-15× per spec §8). Use the same profiling approach as the soft-DP profiling that produced the 832/883 ms number.

- [ ] **Step 3: Update project memory**

Append to `/home/thomas/.claude/projects/-home-thomas-Desktop-AlignAIR/memory/soft-dp-bottleneck.md` a dated note: pointer aligner shipped, measured ms/step before/after, and which ablation arms won on the frozen lattice. Add a one-line pointer in `MEMORY.md` under the architecture section.

- [ ] **Step 4: Commit**

```bash
git add /home/thomas/.claude/projects/-home-thomas-Desktop-AlignAIR/memory/
git commit -m "Record BandedPointerAligner results and lattice ablation outcomes"
```

---

## Notes on sequencing (the §9 ablation ladder vs the build order)

The build order above front-loads the **reusable machinery** (helpers → module → loss → plumbing → decode) so that the actual A/B arms in Task 9 are pure config flips. Mapping to spec §9:

- **Ablation #1** (soft-argmax loss on existing soft-DP) is runnable after **Task 5** alone — it needs zero pointer-head code. Run it first to isolate loss-vs-aligner for the jitter.
- **Ablation #2** (pointer head, both DP sites gone, CE-only) needs Tasks 1–3 + 6. The "CE-only" arm uses a trainer flag to bypass the Task-5 loss for a clean latency isolation.
- **Ablations #3–#5** layer the new loss + reliability gating (Tasks 5–6) — config/flag flips.
- **Ablation #6** (banded) needs Task 8.
- **Ablation #7** (learned substitution matrix / pairwise reader to drop parasail) is intentionally NOT in this plan — it is a follow-up gated on #1–#6 landing; the current fast reader (Task 3 `alignment_score`) already removes the soft-DP from training, which is what unblocks the gym loop.

A trainer flag `coord_loss="soft"|"ce"` (default `"soft"`) is needed for the CE-only ablation arm; add it to `GymTrainer.__init__` and gate the Task-5 wiring on it (when `"ce"`, fall back to the original `F.cross_entropy(sl,gs)+F.cross_entropy(el,ge)` per-row). This is a 4-line addition folded into Task 5 Step 5 — implement both branches there.
