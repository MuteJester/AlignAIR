# Layer 1: Classical Rescore of Neural Top-k — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether classically re-ranking the trained XAttnAligner's neural top-k V candidates (raw-base alignment via the `align/` package) cracks heavy-SHM / sibling V discrimination — no retrain — to decide if Layer 2 is needed.

**Architecture:** A measurement experiment. Per read: run `XAttnAligner.forward` → neural V top-k pool + (trained) `region_logits`-derived V segment string → classically align the segment to each pooled candidate germline (`call_segment` from the WFA caller, which already does pool→align→pick) → compare the classical pick vs the neural pick vs GenAIRR truth, per FrozenLattice cell. Reuses `align/` + `wfa_caller.call_segment` + the per-cell eval harness.

**Tech Stack:** Python 3.12, PyTorch, GenAIRR, the `align/` package (parasail/WFA), `pytest`. Venv: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`; prefix GPU runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Never bare `python`.
- Reuse, do not re-implement: `XAttnAligner` (`forward(tokens, mask, ref_emb, topk=, seed_m=, cand_chunk=)` returns `out["match"]["V"]["pool_idx"] (B,k)` and `out["region_logits"]`); `decode_boundaries(region_logits, mask, has_d)` (`nn/heads/region.py`) → per-row `{v_start, v_end, ...}`; `canonicalize_sequence(seq, orientation_id)` (`inference/dnalignair_infer.py`); `call_segment(seg, gene, topk_idx, reference_set, seed_prefilter, aligner, m_seed=, allowed=)` (`inference/wfa_caller.py`) returns `SegmentCall` with `.best_idx`/`.best_global_idx`; `get_aligner()`, `SeedPrefilter` (`align/`); `FrozenLattice.standard(0)` + `AlignAIRGym`/`gym_collate`.
- The V segment **string** comes from the TRAINED `region_logits` (via `decode_boundaries`), NOT the untrained boundary head.
- The classical rescore must run over the **neural top-k only** (`m_seed=0` in `call_segment`) so we measure the rescore of the model's own pool, not added seed candidates.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`. Do not touch the matcher/loss/align source (that is Layer 2 — this is a read-only measurement on the trained model).

---

## File Structure

- Create `scripts/exp_xattn_rescore.py` — the per-cell neural-vs-classical-rescore V-accuracy experiment.
- Test: `tests/alignair/inference/test_xattn_rescore.py` — a small unit that the rescore-over-neural-top-k logic picks the matching germline (locks the experiment's core call is wired right).

---

### Task 1: Rescore-over-neural-top-k unit (lock the core call)

**Files:**
- Test: `tests/alignair/inference/test_xattn_rescore.py`

**Interfaces:**
- Consumes: `call_segment`, `get_aligner`, `SeedPrefilter`, `ReferenceSet`. No new production code — this verifies that re-ranking a neural pool with `call_segment(..., m_seed=0)` picks the true germline when it is in the pool (the mechanism Layer 1 relies on).

- [ ] **Step 1: Write the test**

```python
# tests/alignair/inference/test_xattn_rescore.py
import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.align import SeedPrefilter, get_aligner
from alignair.inference.wfa_caller import call_segment


def test_classical_rescore_picks_true_allele_from_neural_pool():
    # given a neural top-k pool that CONTAINS the true allele, classical raw-base rescore
    # (call_segment over the pool, no seed additions) must pick it — even when the true allele
    # is NOT first in the pool (the sibling-rescue Layer 1 is meant to provide).
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    sp, al = SeedPrefilter(rs, k=11), get_aligner()
    vg = rs.gene("V")
    true_idx = 11
    seg = vg.sequences[true_idx]                              # read segment == true germline
    neural_pool = [3, 7, true_idx, 9, 2]                      # true allele present but not first
    call = call_segment(seg, "V", neural_pool, rs, sp, al, m_seed=0)
    assert call.best_global_idx == true_idx
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_xattn_rescore.py -q`
Expected: PASS (`call_segment` raw-base aligns the segment to each pooled germline and picks the exact match). If it FAILS, the rescore call is mis-wired — fix the `call_segment` arguments before building the experiment.

- [ ] **Step 3: Commit**

```bash
git add tests/alignair/inference/test_xattn_rescore.py
git commit -m "test: classical rescore over a neural top-k pool picks the true allele"
```

---

### Task 2: Per-cell neural-vs-rescore experiment + run

**Files:**
- Create: `scripts/exp_xattn_rescore.py`

**Interfaces:**
- Consumes: everything in Global Constraints. Produces a printed table: per cell (clean, heavy_shm_fulllen, indel, junction_boundary), V top-1 set-accuracy for **neural** (`out["match"]["V"]["best_global_idx"]`) vs **rescore** (`call_segment` over the neural pool), plus the neural pool's recall (is the true allele even in the pool — the rescore ceiling).

- [ ] **Step 1: Write the experiment script**

```python
# scripts/exp_xattn_rescore.py
"""Layer-1 de-risk: does classical raw-base rescore of the trained XAttnAligner's neural top-k V
candidates crack heavy-SHM / sibling V discrimination, with NO retrain? Per FrozenLattice cell,
compares V top-1 set-accuracy of the neural matcher vs the classical rescore of its own top-k,
and reports the pool recall (the rescore ceiling).

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src .venv/bin/python \
      scripts/exp_xattn_rescore.py --model .private/models/xattn_igh.pt --n 400 --topk 16
"""
import argparse
import torch
import GenAIRR.data as gdata
from torch.utils.data import DataLoader

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.nn.heads.region import decode_boundaries
from alignair.inference.dnalignair_infer import canonicalize_sequence
from alignair.inference.wfa_caller import call_segment
from alignair.align import SeedPrefilter, get_aligner
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def _loader(dc, rs, cell, lat, n, bs, seed):
    cur = type("C", (), {"params": lambda s, p=0.0: dict(lat.cell_params(cell)),
                         "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
    return DataLoader(AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=cur),
                      batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/xattn_igh.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--topk", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    m = XAttnAligner(DNAlignAIRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    with torch.no_grad():
        ref = m.encode_reference(rs)
    sp, al = SeedPrefilter(rs, k=11), get_aligner()
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}
    print(f"model {a.model} (step {ck.get('step')}) | topk={a.topk} | aligner={type(al).__name__}\n")
    print(f"{'cell':20s} {'neural_V':>9s} {'rescore_V':>10s} {'pool_recall':>12s}")
    for cn in CELLS:
        nh = rh = pr = tot = 0
        with torch.no_grad():
            for b in _loader(dc, rs, cells[cn], lat, a.n, a.bs, 7):
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                o = m(b["tokens"], b["mask"], ref, topk=a.topk, seed_m=0, cand_chunk=2)
                mh = b["v_allele"]
                neural_best = o["match"]["V"]["best_global_idx"]
                pool = o["match"]["V"]["pool_idx"]                       # (B,k) neural top-k
                dec = decode_boundaries(o["region_logits"], b["mask"], has_d=rs.has_d)
                # the canonical read string per row, straight from the model's canon_tokens
                canon = [_detok(o["canon_tokens"][i], b["mask"][i]) for i in range(neural_best.shape[0])]
                for i in range(neural_best.shape[0]):
                    nh += int(mh[i, neural_best[i]] > 0)
                    pr += int((mh[i].index_select(0, pool[i]) > 0).any())
                    vs, ve = dec[i]["v_start"], dec[i]["v_end"]          # region-derived V segment
                    seg = canon[i][vs:ve] if (vs is not None and ve and ve > vs) else ""
                    sc = call_segment(seg, "V", pool[i].cpu().tolist(), rs, sp, al, m_seed=0)
                    rh += int(sc is not None and mh[i, sc.best_global_idx] > 0)
                    tot += 1
        print(f"{cn:20s} {nh/tot:9.3f} {rh/tot:10.3f} {pr/tot:12.3f}")
    print("\nrescore_V > neural_V on heavy_shm => classical rescue works (Layer 1 wins).")
    print("rescore_V ~ pool_recall => rescore is near-optimal given the pool (ceiling = retrieval).")


def _detok(tok_row, mask_row):
    # canonical read string from the model's canon_tokens (already orientation-canonicalized)
    from alignair.data.tokenizer import TOKEN_DICT
    inv = {v: k for k, v in TOKEN_DICT.items()}
    return "".join(inv.get(int(t), "N") for t, ok in zip(tok_row.tolist(), mask_row.tolist()) if ok)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the experiment**

Run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src .venv/bin/python scripts/exp_xattn_rescore.py --model .private/models/xattn_igh.pt --n 400 --topk 16`
Expected: a 4-row table. The decisive read: on `heavy_shm_fulllen`, `rescore_V` vs `neural_V` (~0.32 baseline) and vs `pool_recall`. If the read-string reconstruction path errors (e.g., the gym batch lacks `tokens` for a row), fix `_read_str` to use the actual token key present in the batch (inspect one batch's keys first) — do not fabricate coordinates.

- [ ] **Step 3: Commit**

```bash
git add scripts/exp_xattn_rescore.py
git commit -m "exp: Layer-1 classical rescore of XAttnAligner neural top-k (per-cell V vs neural)"
```

---

## Done criteria

- `tests/alignair/inference/test_xattn_rescore.py` passes.
- `scripts/exp_xattn_rescore.py` prints per-cell `neural_V` vs `rescore_V` vs `pool_recall`, giving the decision number: **does classical rescore lift heavy-SHM-V above the ~0.32 neural baseline, toward `pool_recall`?**

## Decision after Layer 1

- If `rescore_V` ≈ `pool_recall` and clearly > `neural_V` on heavy-SHM with no clean regression →
  Layer 1 is the fix; wire it into `predict_reads_xattn` (replace the matcher's V pick with the
  classical rescore) and re-run the 2M assay. Layer 2 may be unnecessary.
- If `rescore_V` is gated by a low `pool_recall` (retrieval misses the true allele) → the bottleneck
  is retrieval, not the matcher pick → Layer 2 + the gym retrain (embargo, genotype aug, sibling
  pool) become the path.
