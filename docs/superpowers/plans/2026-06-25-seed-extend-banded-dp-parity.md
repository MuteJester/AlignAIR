# Seed-and-Extend: Sequential Banded Exact DP + Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the exact banded soft-DP as a clean, reusable sequential PyTorch module that replicates `soft_dp_end_logits` semantics, takes base-match + SHM-reliability inputs, exposes the log-partition reader, and is proven to match the full soft-DP coords/score (parity within fp tolerance) — the reference the fused Triton kernel will validate against.

**Architecture:** A band-mask helper restricts the score matrix to a ±w window around a per-read band center; `SeedExtendAligner` reuses the existing, correct `soft_dp_end_logits` recurrence (`Hm + Ins` + `logcumsumexp` germline-skip — NOT a new affine-D state) on the masked matrix, runs it forward for `end` and reversed for `start`, and returns the log-partition as the final allele reader score. Parity is asserted against `SoftDPAligner` at full band width (no retrain).

**Tech Stack:** PyTorch (`.venv/bin/python`, `PYTHONPATH=src`), pytest, the trained `.private/models/scaled_long.pt`, the FrozenLattice gym.

**Spec:** `docs/superpowers/specs/2026-06-25-structural-seed-and-extend-neural-dp-design.md` (this plan implements build step 3 — the sequential banded exact DP — sequenced first for risk: it is parity-testable WITHOUT a retrain and is the reference for the fused kernel, build step 5).

## Global Constraints

- Run everything with `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`. `alignair` is NOT pip-installed. NEVER bare `python`/`python3`.
- Package under test is `src/alignair` (lowercase), not `src/AlignAIR` (legacy TF).
- Git commit messages: NEVER add Co-Authored-By or any Claude/AI mention (project rule).
- **The DP MUST replicate the CURRENT `soft_dp_end_logits` semantics exactly** — `Hm + Ins` + `logcumsumexp` germline-skip; NO new full-affine (M,I,D) deletion state (spec §4.5, hard rule). Full-affine-D is a separate future experiment, NOT this plan.
- **"Exact" = mathematically equivalent within fp tolerance** (band masking + reductions reorder logsumexp; not bit-identical). Parity tests assert `atol`/`rtol`, not equality.
- **Base-match + SHM-reliability are mandatory DP inputs** (spec rule 4) — wired via the shared `base_match_channel`.
- **The final allele score is the DP log-partition, never MaxSim** (spec rule 1).
- This plan does NOT wire `aligner="seed_extend"` into `DNAlignAIR` (that is the integration/encoder-refactor plan) and does NOT add the band predictor (the band center is an INPUT here, oracle in tests). Keep `nn/soft_dp_aligner.py` unchanged (it is the A/B oracle).
- TDD: failing test first, watch it fail, minimal implementation, watch it pass, commit.

## File Structure

- `src/alignair/nn/seed_extend_aligner.py` — NEW. `band_mask_scores`, `SeedExtendAligner(nn.Module)` (`forward` → start/end logits; `alignment_score` → log-partition reader). Reuses `soft_dp_end_logits`, `_reverse_valid_2d`, `NEG` from `nn/soft_dp_aligner.py` and `base_match_channel` from `nn/base_match.py`.
- `scripts/exp_banded_dp_parity.py` — NEW. On the trained model + lattice: assert banded (oracle band) coords/score match full soft-DP within tolerance per cell.
- Test: `tests/alignair/nn/test_seed_extend_aligner.py`.

---

## Task 1: Band-mask helper

**Files:**
- Create: `src/alignair/nn/seed_extend_aligner.py` (this helper only)
- Test: `tests/alignair/nn/test_seed_extend_aligner.py`

**Interfaces:**
- Produces: `band_mask_scores(M[B,S,Lg], band_center[B], w[int]) -> Tensor[B,S,Lg]` — sets `M[b,i,j] = NEG` where `|j − (band_center[b] + i)| > w`; columns within the band unchanged. `w >= Lg` (or `band_center` covering all) is effectively a no-op band.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/nn/test_seed_extend_aligner.py
import torch
from alignair.nn.seed_extend_aligner import band_mask_scores
from alignair.nn.soft_dp_aligner import NEG


def test_band_mask_keeps_band_drops_rest():
    B, S, Lg = 1, 4, 20
    M = torch.zeros(B, S, Lg)
    center = torch.tensor([5])
    out = band_mask_scores(M, center, w=2)
    # row i=0: keep |j-5|<=2 -> cols 3..7 ; row i=1: center 6 -> cols 4..8
    assert (out[0, 0, 3:8] == 0.0).all()
    assert out[0, 0, 2] <= NEG + 1 and out[0, 0, 8] <= NEG + 1
    assert (out[0, 1, 4:9] == 0.0).all()


def test_full_width_band_is_noop():
    B, S, Lg = 2, 5, 12
    M = torch.randn(B, S, Lg)
    center = torch.zeros(B, dtype=torch.long)
    out = band_mask_scores(M, center, w=Lg)        # band covers everything
    assert torch.equal(out, M)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_seed_extend_aligner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.nn.seed_extend_aligner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/nn/seed_extend_aligner.py
"""Sequential banded exact soft-DP (the "extend" of seed-and-extend). Reuses the existing,
correct soft_dp_end_logits recurrence (Hm + Ins + logcumsumexp germline-skip) on a band-masked
score matrix, so it is the SAME math as the full soft-DP, restricted to a ±w window the seed
places. Emits start/end coordinate posteriors AND the log-partition as the final allele reader
score. This is the reference the fused Triton kernel (build step 5) will validate against."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_dp_aligner import soft_dp_end_logits, _reverse_valid_2d, NEG
from .base_match import base_match_channel


def band_mask_scores(M: torch.Tensor, band_center: torch.Tensor, w: int) -> torch.Tensor:
    """Mask the score matrix to a ±w band: keep column j for read row i iff
    |j - (band_center[b] + i)| <= w; else set to NEG. band_center is the predicted germline
    start offset per read (an INPUT here; oracle in tests, the seed head at deployment)."""
    B, S, Lg = M.shape
    i = torch.arange(S, device=M.device)[None, :, None]            # (1,S,1)
    j = torch.arange(Lg, device=M.device)[None, None, :]           # (1,1,Lg)
    center = band_center.view(-1, 1, 1) + i                        # (B,S,1)
    in_band = (j - center).abs() <= w
    return M.masked_fill(~in_band, NEG)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_seed_extend_aligner.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/seed_extend_aligner.py tests/alignair/nn/test_seed_extend_aligner.py
git commit -m "Add band-mask helper for the seed-extend banded DP"
```

---

## Task 2: SeedExtendAligner forward (banded start/end) + full-band parity

**Files:**
- Modify: `src/alignair/nn/seed_extend_aligner.py`
- Test: `tests/alignair/nn/test_seed_extend_aligner.py`

**Interfaces:**
- Consumes: `band_mask_scores`, `soft_dp_end_logits`, `_reverse_valid_2d`, `base_match_channel`.
- Produces: `SeedExtendAligner(d_model, match_floor=1.0)` with `forward(seg_reps, seg_mask, germ_reps, germ_mask, band_center, w, seg_tok=None, germ_tok=None, seg_reliability=None) -> (start_logits[B,Lg], end_logits[B,Lg])`. With base-match params copied from a `SoftDPAligner`, and tokens=None + full band, output equals `SoftDPAligner.forward` within fp tolerance.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_seed_extend_aligner.py
from alignair.nn.seed_extend_aligner import SeedExtendAligner
from alignair.nn.soft_dp_aligner import SoftDPAligner


def _toy(B=2, S=10, Lg=24, d=16):
    torch.manual_seed(0)
    return (torch.randn(B, S, d), torch.ones(B, S, dtype=torch.bool),
            torch.randn(B, Lg, d), torch.ones(B, Lg, dtype=torch.bool))


def test_forward_shapes():
    al = SeedExtendAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    sl, el = al(seg, sm, germ, gm, center, w=8)
    assert sl.shape == (2, 24) and el.shape == (2, 24)


def test_full_band_matches_softdp_within_tol():
    # copy the soft-DP's projection+gap params so the cosine score matrices are identical;
    # at FULL band width + tokens=None (pure cosine) the banded DP must equal SoftDPAligner.
    sd = SoftDPAligner(d_model=16)
    al = SeedExtendAligner(d_model=16)
    al.seg_proj.load_state_dict(sd.seg_proj.state_dict())
    al.germ_proj.load_state_dict(sd.germ_proj.state_dict())
    with torch.no_grad():
        al.log_scale.copy_(sd.log_scale)
        al._gap_open.copy_(sd._gap_open); al._gap_extend.copy_(sd._gap_extend)
        al._del_gap.copy_(sd._del_gap)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    sl_a, el_a = al(seg, sm, germ, gm, center, w=germ.shape[1])    # full band
    sl_s, el_s = sd(seg, sm, germ, gm)
    assert torch.allclose(el_a, el_s, atol=1e-3, rtol=1e-3)
    assert torch.allclose(sl_a, sl_s, atol=1e-3, rtol=1e-3)


def test_base_match_changes_output():
    al = SeedExtendAligner(d_model=16)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    st = torch.randint(1, 5, (2, 10)); gt = torch.randint(1, 5, (2, 24))
    _, el_plain = al(seg, sm, germ, gm, center, w=24)
    _, el_bm = al(seg, sm, germ, gm, center, w=24, seg_tok=st, germ_tok=gt)
    assert not torch.allclose(el_plain, el_bm)                     # base-match is a live input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_seed_extend_aligner.py -k "forward or full_band or base_match" -q`
Expected: FAIL — `cannot import name 'SeedExtendAligner'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/alignair/nn/seed_extend_aligner.py`:

```python
class SeedExtendAligner(nn.Module):
    """Banded exact soft-DP. Same params/score-matrix shape as SoftDPAligner; the only
    difference is the band mask (and base-match/reliability are first-class inputs)."""

    def __init__(self, d_model: int, match_floor: float = 1.0):
        super().__init__()
        self.seg_proj = nn.Linear(d_model, d_model)
        self.germ_proj = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.tensor(1.6))
        self._gap_open = nn.Parameter(torch.tensor(3.0))
        self._gap_extend = nn.Parameter(torch.tensor(2.0))
        self._del_gap = nn.Parameter(torch.tensor(3.0))
        self._match_weight = nn.Parameter(torch.tensor(1.0))
        self.match_floor = float(match_floor)

    def _scores(self, seg_reps, germ_reps, seg_tok, germ_tok, seg_reliability):
        S = F.normalize(self.seg_proj(seg_reps), dim=-1)
        G = F.normalize(self.germ_proj(germ_reps), dim=-1)
        M = self.log_scale.clamp(-2.0, 3.0).exp() * torch.einsum("bid,bjd->bij", S, G)
        return base_match_channel(M, seg_tok, germ_tok, seg_reliability,
                                  self._match_weight, self.match_floor)

    def _gaps(self):
        return (-F.softplus(self._gap_open), -F.softplus(self._gap_extend),
                -F.softplus(self._del_gap))

    def forward(self, seg_reps, seg_mask, germ_reps, germ_mask, band_center, w,
                seg_tok=None, germ_tok=None, seg_reliability=None):
        go, ge, dg = self._gaps()
        M = self._scores(seg_reps, germ_reps, seg_tok, germ_tok, seg_reliability)
        M = band_mask_scores(M, band_center, w)
        end = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
        seg_len = seg_mask.sum(dim=1); germ_len = germ_mask.sum(dim=1)
        Mr = _reverse_valid_2d(M.transpose(1, 2), germ_len).transpose(1, 2)
        Mr = _reverse_valid_2d(Mr, seg_len)
        end_rev = soft_dp_end_logits(Mr, seg_mask, germ_mask, go, ge, dg)
        start = _reverse_valid_2d(end_rev, germ_len)
        return start.masked_fill(~germ_mask, NEG), end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_seed_extend_aligner.py -q`
Expected: PASS (all). The full-band parity test confirms the banded DP equals the soft-DP within fp tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/seed_extend_aligner.py tests/alignair/nn/test_seed_extend_aligner.py
git commit -m "Add SeedExtendAligner banded forward with full-band soft-DP parity"
```

---

## Task 3: Log-partition reader (rule 1)

**Files:**
- Modify: `src/alignair/nn/seed_extend_aligner.py`
- Test: `tests/alignair/nn/test_seed_extend_aligner.py`

**Interfaces:**
- Produces: `SeedExtendAligner.alignment_score(seg_reps, seg_mask, germ_reps, germ_mask, band_center, w, seg_tok=None, germ_tok=None, seg_reliability=None) -> Tensor[B]` — length-normalized log-partition over end positions (the FINAL allele reader; spec rule 1). At full band + matched params it equals `SoftDPAligner.alignment_score` within tolerance.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/nn/test_seed_extend_aligner.py
def test_alignment_score_full_band_matches_softdp():
    sd = SoftDPAligner(d_model=16)
    al = SeedExtendAligner(d_model=16)
    al.seg_proj.load_state_dict(sd.seg_proj.state_dict())
    al.germ_proj.load_state_dict(sd.germ_proj.state_dict())
    with torch.no_grad():
        al.log_scale.copy_(sd.log_scale); al._gap_open.copy_(sd._gap_open)
        al._gap_extend.copy_(sd._gap_extend); al._del_gap.copy_(sd._del_gap)
        al._match_weight.copy_(sd._match_weight)
    seg, sm, germ, gm = _toy()
    center = torch.zeros(2, dtype=torch.long)
    st = torch.randint(1, 5, (2, 10)); gt = torch.randint(1, 5, (2, 24))
    a = al.alignment_score(seg, sm, germ, gm, center, w=24, seg_tok=st, germ_tok=gt)
    s = sd.alignment_score(seg, sm, germ, gm, seg_tok=st, germ_tok=gt)
    assert a.shape == (2,) and torch.allclose(a, s, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_seed_extend_aligner.py -k alignment_score -q`
Expected: FAIL — `SeedExtendAligner has no attribute 'alignment_score'`.

- [ ] **Step 3: Write minimal implementation**

Append to `SeedExtendAligner`:

```python
    def alignment_score(self, seg_reps, seg_mask, germ_reps, germ_mask, band_center, w,
                        seg_tok=None, germ_tok=None, seg_reliability=None):
        """Final allele reader = banded soft-DP log-partition (length-normalized). Rule 1."""
        go, ge, dg = self._gaps()
        M = self._scores(seg_reps, germ_reps, seg_tok, germ_tok, seg_reliability)
        M = band_mask_scores(M, band_center, w)
        end = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
        n_valid = germ_mask.sum(dim=-1).clamp(min=1).to(end.dtype)
        return torch.logsumexp(end, dim=-1) - torch.log(n_valid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_seed_extend_aligner.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/seed_extend_aligner.py tests/alignair/nn/test_seed_extend_aligner.py
git commit -m "Add seed-extend log-partition reader with full-band soft-DP parity"
```

---

## Task 4: Oracle-band coordinate parity on the trained model

**Files:**
- Create: `scripts/exp_banded_dp_parity.py`

**Interfaces:**
- Consumes: `SeedExtendAligner` (params copied from the trained `SoftDPAligner`), the FrozenLattice gym, the trained model.

- [ ] **Step 1: Write the parity experiment**

```python
# scripts/exp_banded_dp_parity.py
"""Parity gate for build step 3: on the trained model, the banded SeedExtendAligner (band
centered on the TRUE germline_start, oracle) must produce coordinates that match the full
soft-DP within +-1nt per lattice cell — confirming banding is exact in the new code path
(reproduces the Gate-0 band-sweep result through SeedExtendAligner). No retrain.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_banded_dp_parity.py --n 300 --w 16
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.nn.seed_extend_aligner import SeedExtendAligner
from alignair.nn.germline_aligner import decode_germline_coords
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    sd = model.aligner                                   # the trained SoftDPAligner
    al = SeedExtendAligner(d_model=ck["config"]["d_model"]).to(device).eval()
    al.seg_proj.load_state_dict(sd.seg_proj.state_dict())
    al.germ_proj.load_state_dict(sd.germ_proj.state_dict())
    with torch.no_grad():
        for p in ("log_scale", "_gap_open", "_gap_extend", "_del_gap", "_match_weight"):
            getattr(al, p).copy_(getattr(sd, p))
    rs = ReferenceSet.from_dataconfigs(dc); ref_emb = model.encode_reference(rs)
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}

    print(f"banded(oracle w={a.w}) vs full soft-DP coords | start+end within +-1nt")
    print(f"{'cell':18s} {'agree-frac':>12s}")
    for cname in CELLS:
        cur = type("C", (), {"params": lambda s, p=0.0, c=cells[cname]: dict(lat.cell_params(c)),
                             "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
        loader = DataLoader(AlignAIRGym([dc], rs, n=a.n, seed=0, curriculum=cur),
                            batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
        agree = tot = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"],
                                                           batch["region_labels"], "V")
                seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
                idx = batch["v_primary_idx"]
                gr = ref_emb["V"]["pos_reps"][idx]; gm = ref_emb["V"]["pos_mask"][idx]
                center = batch["v_germline_start"]
                sl_f, el_f = sd(seg_reps, seg_mask, gr, gm)                       # full soft-DP
                sl_b, el_b = al(seg_reps, seg_mask, gr, gm, center, a.w)          # banded (oracle)
                gs_f, ge_f = decode_germline_coords(sl_f, el_f, soft=True)
                gs_b, ge_b = decode_germline_coords(sl_b, el_b, soft=True)
                ok = ((gs_f - gs_b).abs() <= 1) & ((ge_f - ge_b).abs() <= 1)
                agree += int(ok.sum()); tot += ok.numel()
        print(f"{cname:18s} {agree/max(tot,1):12.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the parity experiment**

Run: `PYTHONPATH=src .venv/bin/python scripts/exp_banded_dp_parity.py --n 300 --w 16`
Expected: `agree-frac` ≥ ~0.98 on clean/heavy_shm/indel (the band-sweep showed banding is exact where signal exists). Junction may be lower (the oracle band is correct there, but the full soft-DP itself wanders on its hard reads — agreement reflects soft-DP self-consistency, still expected high). Record the table.

- [ ] **Step 3: Decision + commit**

If agree-frac is high (≥0.97 on the three clean cells), the banded DP is confirmed exact in the new module → proceed (next plan: the fused Triton kernel, validated against THIS sequential reference). If low, debug the band-mask/recurrence parity before any kernel work.

```bash
git add scripts/exp_banded_dp_parity.py
git commit -m "Add oracle-band coordinate parity experiment (banded == full soft-DP)"
```

---

## Notes on scope and what comes next

This plan delivers the **sequential banded exact DP reference** (spec build step 3) — the foundation, parity-proven without a retrain. It deliberately does NOT: wire `aligner="seed_extend"` into `DNAlignAIR`, refactor the encoder, add the band predictor, or write the fused kernel. Those are subsequent gated plans, in order:

1. **(this plan)** Sequential banded exact DP + parity.
2. **Fused banded Triton/CUDA kernel** — validate numerical parity vs THIS sequential reference, then profile the speed win (build step 5; the high-risk/high-value unknown, de-risked against the reference).
3. **Encoder refactor** — shared type-embedded encoder, delete `GermlineEncoder` + `caller="classifier"`, retrain; verify retrieval recall@k + coord parity; **Gate-1 repeat** (co-trained band head — where junction may improve past 0.97).
4. **Band predictor wiring + Gate 2** (predicted region + top-k; multiplicative recall) and **Gate 3** (full A/B vs the soft-DP oracle: competence + coord parity + speed at B=64).

Each is its own spec-faithful, independently-testable increment.
