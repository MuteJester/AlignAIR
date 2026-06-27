# XAttnAligner Training Script Implementation Plan (LLM-Encoder Aligner — Training Script)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable training script `scripts/train_xattn.py` that trains `XAttnAligner` on the stratified GenAIRR gym with the four-task loss, LLM-style optimization (AdamW + cosine schedule + warmup + bf16 + grad-clip), checkpoint/resume, and a periodic set-aware allele-accuracy readout.

**Architecture:** Wraps `xattn_losses` in a long training loop over `AlignAIRGym` + `StratifiedCurriculum`, modelled on the existing `scripts/train_seed_extend.py`. A smoke test runs a few steps, checkpoints, and reloads to prove the script + checkpoint round-trip work; the actual long run is a manual invocation.

**Tech Stack:** Python 3.12, PyTorch (cuda + bf16), GenAIRR, `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`. Never bare `python`.
- Reuse: `XAttnAligner` (forward + `encode_reference` + seed admission), `xattn_losses` (Plan training), `build_sibling_index`, `AlignAIRGym` + `gym_collate`, `StratifiedCurriculum`.
- Checkpoint format: `torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step}, path)` — matches the existing `train_seed_extend.py` convention.
- bf16 autocast on cuda (`torch.cuda.is_bf16_supported()` is True here); plain float on cpu.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Create `scripts/train_xattn.py` — the training script.
- Create `tests/alignair/training/test_train_xattn_smoke.py` — runs a few steps + checkpoint round-trip.

---

### Task 1: `train_xattn.py` + checkpoint round-trip smoke

**Files:**
- Create: `scripts/train_xattn.py`
- Test: `tests/alignair/training/test_train_xattn_smoke.py`

**Interfaces:**
- Produces: `train_xattn(cfg, dc, steps, batch_size, lr, device, save=None, ckpt_every=2000, eval_n=200, seed_m=4, progress=True) -> model` (importable for the smoke test), plus an argparse `main()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/training/test_train_xattn_smoke.py
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner


def test_train_xattn_runs_and_checkpoints(tmp_path):
    import scripts.train_xattn as T
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64)
    save = tmp_path / "xattn_smoke.pt"
    model = T.train_xattn(cfg, gdata.HUMAN_IGK_OGRDB, steps=5, batch_size=8, lr=1e-3,
                          device="cpu", save=str(save), ckpt_every=5, eval_n=0, progress=False)
    assert save.exists()
    ck = torch.load(str(save), map_location="cpu", weights_only=False)
    m2 = XAttnAligner(DNAlignAIRConfig(**ck["config"]))
    m2.load_state_dict(ck["model"])                       # round-trips into a fresh model
    p1 = dict(model.named_parameters())
    for n, p in m2.named_parameters():
        assert torch.allclose(p, p1[n].cpu(), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_train_xattn_smoke.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.train_xattn'` (or import error).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/train_xattn.py
"""Train XAttnAligner (LLM-encoder aligner) on the stratified GenAIRR gym with the four-task loss
(orientation + region + allele set-NCE + germline-span), AdamW + cosine schedule + bf16, with
checkpoint/resume and a set-aware allele-accuracy readout.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/train_xattn.py --d-model 128 --n-layers 8 \
      --steps 20000 --batch-size 64 --locus igh --save .private/models/xattn_d128.pt
"""
import argparse
import math
import random
import time

import torch
import GenAIRR.data as gdata
from torch.utils.data import DataLoader

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.training.reader import build_sibling_index
from alignair.training.xattn_loss import xattn_losses
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum


def _cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def _allele_accuracy(model, rs, dc, ref_emb, device, n, bs, seed_m):
    gym = AlignAIRGym([dc], rs, n=n, seed=123)
    loader = DataLoader(gym, batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    hit = tot = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["tokens"], batch["mask"], ref_emb, seed_m=seed_m, reference_set=rs)
        best = out["match"]["V"]["best_global_idx"]                      # (B,)
        mh = batch["v_allele"]                                          # (B,K) set
        hit += int((mh.gather(1, best[:, None]).squeeze(1) > 0).sum()); tot += best.shape[0]
    return hit / max(tot, 1)


def train_xattn(cfg, dc, steps, batch_size, lr, device, save=None, ckpt_every=2000,
                eval_n=200, seed_m=4, warmup=200, grad_clip=1.0, progress=True):
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(dc)
    model = XAttnAligner(cfg).to(device).train()
    sib = build_sibling_index(rs)
    gym = AlignAIRGym([dc], rs, n=batch_size * 8, seed=0, curriculum=StratifiedCurriculum())
    loader = DataLoader(gym, batch_size=batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    rng = random.Random(0)
    it = iter(loader)
    t0 = time.perf_counter()
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = _cosine_lr(step, steps, warmup, lr)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        ref_emb = model.encode_reference(rs)                            # re-encode (weights move)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            total, parts = xattn_losses(model, batch, ref_emb, sib, rng)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if progress and (step % 100 == 0 or step == steps - 1):
            el = time.perf_counter() - t0
            print(f"[{step:6d}/{steps}] loss {float(total):.3f} "
                  + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items())
                  + f"  {el:.0f}s", flush=True)
        if save and ((step + 1) % ckpt_every == 0 or step == steps - 1):
            torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step + 1}, save)
    if eval_n:
        model.eval()
        acc = _allele_accuracy(model, rs, dc, model.encode_reference(rs), device, eval_n,
                               batch_size, seed_m)
        print(f"\nset-aware V allele accuracy (n={eval_n}, seed_m={seed_m}): {acc:.3f}", flush=True)
        model.train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--locus", default="igh", choices=["igh", "igk"])
    ap.add_argument("--eval-n", type=int, default=250)
    ap.add_argument("--seed-m", type=int, default=4)
    ap.add_argument("--save", default="")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = {"igh": gdata.HUMAN_IGH_OGRDB, "igk": gdata.HUMAN_IGK_OGRDB}[a.locus]
    cfg = DNAlignAIRConfig(d_model=a.d_model, n_layers=a.n_layers, nhead=a.nhead,
                           dim_feedforward=2 * a.d_model, backbone="shared")
    train_xattn(cfg, dc, a.steps, a.batch_size, a.lr, device, save=a.save or None,
                ckpt_every=a.ckpt_every, eval_n=a.eval_n, seed_m=a.seed_m)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_train_xattn_smoke.py -q`
Expected: PASS (5 steps run on CPU, checkpoint saved and reloaded into a fresh model with matching weights). If `scripts` is not importable as a package, add `conftest.py`/`sys.path` handling — but the repo already imports `scripts.*` in other tests, so it should resolve.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_xattn.py tests/alignair/training/test_train_xattn_smoke.py
git commit -m "scripts: train_xattn.py — XAttnAligner gym training (AdamW+cosine+bf16, ckpt, allele-acc eval)"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/training/test_train_xattn_smoke.py -q` is green.
- `scripts/train_xattn.py` runs a real training loop with cosine LR + bf16 + grad-clip, checkpoints, and prints a set-aware allele accuracy; a manual long run (`--locus igh --steps 20000 --save ...`) is launchable.

## Follow-on plans (not this plan)

1. **Embargo** — hold out an allele fraction from training, include at eval, verify held-out recall ≈ control (the dynamic-genotype gate).
2. **Derivations module** — four heads → full GenAIRR record (vectorized).
3. **Eval/gates** — adapt `LatticeEvaluator` to XAttnAligner outputs; benchmark assay; IgBLAST head-to-head; throughput.
