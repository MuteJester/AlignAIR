# AA Convergence Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Determine, via a checkpointed convergence curve on frozen simulator cohorts, whether continued training of the Stage-1 AA model improves its weaknesses without regressing gates — with a resumable trainer, a unified evaluator, and a FROZEN blinded selection rule.

**Architecture:** (A) make the trainer faithfully resumable (full optimizer/scheduler/RNG state, cosine LR, immutable checkpoints); (B) build a unified evaluator over a hash-fingerprinted frozen cohort computing the full metric set (reusing existing scattered logic + new metrics); (C) a driver that trains to +300k, evaluates each candidate, and applies the frozen §5 selection rule; (D) execution — smoke-validate resume, run, select, recalibrate the winner, rerun Phase-3 descriptively.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, GenAIRR, the existing `alignair.aa` package, pytest.

## Global Constraints
- Spec/predeclaration: `docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md`. The §5 selection rule is FROZEN — implement it exactly; do not re-tune thresholds.
- **Never modify/overwrite** the frozen incumbent: `experiments/aa/strategic/model_stage1_PASS.pt`, `docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json`, `docs/superpowers/artifacts/2026-07-21-aa-phase3/*`.
- Candidate `.pt` files go under `experiments/aa/strategic/` (git-ignored); NEVER commit `.pt` files.
- Selection is BLINDED: the Phase-3 16-antibody cohort is NOT loaded or consulted during training/eval/selection (only in the final §7 descriptive rerun).
- Evaluator cohort seeds must be DISJOINT from training and from calibration's 1000/9000.
- Run python as `PYTHONPATH=src .venv/bin/python ...` (GPU/cuda available). No Co-Authored-By/Claude line in commits.
- Artifacts dir `ART = docs/superpowers/artifacts/2026-07-22-aa-convergence/`.

---

## Milestone A — Resumable trainer

### Task 1: Full-state checkpoint I/O
**Files:** Create `src/alignair/aa/resume.py`; Test `tests/alignair/aa/test_resume.py`.
**Interfaces:** Produces `save_checkpoint(path, *, model, cfg, step, optimizer, scheduler, lr, history, provenance)`; `load_checkpoint(path) -> dict` (keys: model, cfg, step, optimizer, scheduler, rng_numpy, rng_torch, lr, history, provenance; missing keys → None for old-format `{model,cfg,step}`); `capture_rng() -> tuple`; `restore_rng(rng_numpy, rng_torch)`.

- [ ] **Step 1: Write the failing test**
```python
# tests/alignair/aa/test_resume.py
import numpy as np, torch
from alignair.aa.resume import save_checkpoint, load_checkpoint

def _tiny():
    m = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    # take one step so optimizer has non-empty moment state
    loss = m(torch.randn(2, 4)).sum(); loss.backward(); opt.step(); sch.step()
    return m, opt, sch

def test_full_state_roundtrip(tmp_path):
    m, opt, sch = _tiny()
    p = str(tmp_path / "ck.pt")
    save_checkpoint(p, model=m, cfg={"max_seq_length": 192}, step=5, optimizer=opt,
                    scheduler=sch, lr=1e-4, history=[{"step": 5, "total": 1.0}],
                    provenance={"seed": 0})
    ck = load_checkpoint(p)
    assert ck["step"] == 5 and ck["provenance"]["seed"] == 0
    # optimizer moment state present and restorable
    m2 = torch.nn.Linear(4, 3); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
    opt2.load_state_dict(ck["optimizer"])
    assert opt2.state_dict()["state"], "optimizer moment state must be preserved"
    assert ck["scheduler"]["T_max"] == 100

def test_old_format_load_is_graceful(tmp_path):
    p = str(tmp_path / "old.pt")
    torch.save({"model": torch.nn.Linear(4, 3).state_dict(), "cfg": {}, "step": 40000}, p)
    ck = load_checkpoint(p)
    assert ck["step"] == 40000 and ck["optimizer"] is None and ck["scheduler"] is None
```

- [ ] **Step 2: Run to verify fail** → `pytest tests/alignair/aa/test_resume.py -q` FAIL (module missing).
- [ ] **Step 3: Implement**
```python
# src/alignair/aa/resume.py
"""Full-state checkpoint I/O for faithful training resume (optimizer/scheduler/RNG/provenance)."""
from __future__ import annotations
import numpy as np, torch

def capture_rng():
    return (np.random.get_state(), torch.get_rng_state())

def restore_rng(rng_numpy, rng_torch):
    if rng_numpy is not None:
        np.random.set_state(rng_numpy)
    if rng_torch is not None:
        torch.set_rng_state(rng_torch if isinstance(rng_torch, torch.Tensor) else torch.tensor(rng_torch, dtype=torch.uint8))

def save_checkpoint(path, *, model, cfg, step, optimizer, scheduler, lr, history, provenance):
    rng_numpy, rng_torch = capture_rng()
    torch.save({
        "model": model.state_dict(), "cfg": cfg, "step": step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_numpy": rng_numpy, "rng_torch": rng_torch,
        "lr": lr, "history": history, "provenance": provenance,
    }, path)

def load_checkpoint(path) -> dict:
    ck = torch.load(path, map_location="cpu")
    return {k: ck.get(k) for k in
            ("model", "cfg", "step", "optimizer", "scheduler", "rng_numpy", "rng_torch",
             "lr", "history", "provenance")}
```
- [ ] **Step 4: Run to verify pass; commit**
```bash
git add src/alignair/aa/resume.py tests/alignair/aa/test_resume.py
git commit -m "feat(aa): full-state checkpoint I/O for faithful training resume"
```

### Task 2: Resumable `stage1_train` (cosine LR, immutable names, validation hook, plateau)
**Files:** Modify `src/alignair/aa/strategic_trunc.py`; Test `tests/alignair/aa/test_resume_train.py`.
**Interfaces:** `stage1_train(..., *, resume_path=None, base_lr=1e-4, lr_min=1e-5, total_budget=None, val_every=0, val_fn=None, ckpt_dir=None, ckpt_steps=(), plateau_k=0)`. When `resume_path` is a full-state checkpoint, restores optimizer/scheduler/RNG/step and continues; when it is old-format, warm-starts weights + fresh optimizer (documented). Uses `CosineAnnealingLR(opt, T_max=total_budget, eta_min=lr_min)` with `AdamW(lr=base_lr)`. Saves immutable `{ckpt_dir}/model_cont_step{N}.pt` at each `ckpt_steps` via `save_checkpoint`. Calls `val_fn(model, step)` every `val_every` (records result). Detects plateau over `plateau_k` validations (logs; does not auto-stop).

- [ ] **Step 1: Write the failing test** (focused on resume-faithfulness + LR, not full training)
```python
# tests/alignair/aa/test_resume_train.py
import torch
from alignair.aa.resume import load_checkpoint
from alignair.aa.strategic import strategic_setup
from alignair.aa.strategic_trunc import stage1_train

def test_resume_restores_optimizer_and_advances_step(tmp_path):
    model, cfg, cmaps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    d = str(tmp_path)
    # 6 steps, checkpoint at 6
    stage1_train(model, cfg, cmaps, meta, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=12, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(6,))
    ck = load_checkpoint(f"{d}/model_cont_step6.pt")
    assert ck["step"] == 6 and ck["optimizer"] is not None and ck["optimizer"]["state"]
    # resume for 6 more; step advances to 12, optimizer state still present
    model2, cfg2, cmaps2, meta2 = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    stage1_train(model2, cfg2, cmaps2, meta2, steps=6, batch_size=8, base_lr=1e-4, lr_min=1e-5,
                 total_budget=12, seed=0, device="cpu", log_every=0, ckpt_dir=d, ckpt_steps=(12,),
                 resume_path=f"{d}/model_cont_step6.pt")
    ck2 = load_checkpoint(f"{d}/model_cont_step12.pt")
    assert ck2["step"] == 12
```
- [ ] **Step 2: Run to verify fail** → FAIL (new kwargs / behavior not present).
- [ ] **Step 3: Implement.** Modify `stage1_train` to: build `opt = AdamW(model.parameters(), lr=base_lr)` and `sch = CosineAnnealingLR(opt, T_max=total_budget or steps, eta_min=lr_min)`; if `resume_path`, `ck = load_checkpoint(resume_path)`, `model.load_state_dict(ck["model"])`, and if `ck["optimizer"]` is not None restore `opt.load_state_dict`, `sch.load_state_dict`, `restore_rng(ck["rng_numpy"], ck["rng_torch"])`, and set `start_step = ck["step"]` (else warm-start weights only, `start_step` from ck or 0, log the non-faithful hop); loop `for step in range(start_step+1, start_step+steps+1)`; after `opt.step()` call `sch.step()`; save immutable `model_cont_step{step}.pt` via `save_checkpoint(...)` when `step in ckpt_steps`; every `val_every` call `val_fn(model, step)` and append to a `val_history`; plateau-log over `plateau_k`. Keep the old `save_path`/`save_every`/`init_path` params working (back-compat). Provenance dict includes git sha (via `subprocess`), base_model_sha, seed, hyperparams.
- [ ] **Step 4: Run to verify pass; run full AA subset; commit**
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_resume_train.py tests/alignair/aa/test_strategic_trunc.py -q
git add src/alignair/aa/strategic_trunc.py tests/alignair/aa/test_resume_train.py
git commit -m "feat(aa): resumable stage1_train — full-state resume, cosine LR, immutable checkpoints, validation+plateau hooks"
```

---

## Milestone B — Unified evaluator + new metrics

### Task 3: New metrics (macro, frequency bins, NLL, Brier)
**Files:** Create `src/alignair/aa/convergence/__init__.py`, `src/alignair/aa/convergence/metrics.py`; Test `tests/alignair/aa/convergence/test_metrics.py`.
**Interfaces:** `macro_accuracy(y_true, y_pred) -> float` (mean per-true-class recall); `per_frequency_bins(y_true, y_pred, freq, edges=(...)) -> dict` (accuracy per allele-frequency bin); `softmax_nll(probs, y_true) -> float`; `brier_score(probs, y_true) -> float`.

- [ ] **Step 1: Write the failing test**
```python
# tests/alignair/aa/convergence/test_metrics.py
import numpy as np
from alignair.aa.convergence.metrics import macro_accuracy, per_frequency_bins, softmax_nll, brier_score

def test_macro_downweights_common_class():
    # class 0 has 8 examples all correct; class 1 has 2 examples all wrong -> micro 0.8, macro 0.5
    yt = np.array([0]*8 + [1]*2); yp = np.array([0]*8 + [0]*2)
    assert abs(macro_accuracy(yt, yp) - 0.5) < 1e-9

def test_frequency_bins():
    yt = np.array([0,0,1]); yp = np.array([0,1,1]); freq = np.array([0.9,0.9,0.01])
    out = per_frequency_bins(yt, yp, freq, edges=(0.0,0.05,1.0))
    assert out["[0.0,0.05)"]["acc"] == 1.0 and out["[0.05,1.0]"]["acc"] == 0.5

def test_nll_and_brier_perfect_and_uniform():
    probs = np.array([[1.0,0.0],[0.0,1.0]]); yt = np.array([0,1])
    assert softmax_nll(np.clip(probs,1e-12,1), yt) < 1e-6
    assert brier_score(probs, yt) < 1e-9
    unif = np.array([[0.5,0.5],[0.5,0.5]])
    assert abs(brier_score(unif, yt) - 0.5) < 1e-9   # sum (0.5-*)^2 = 0.5 per row
```
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement**
```python
# src/alignair/aa/convergence/metrics.py
"""Convergence-study metrics: macro accuracy, allele-frequency bins, softmax NLL, Brier."""
from __future__ import annotations
import numpy as np

def macro_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    accs = [(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]
    return float(np.mean(accs)) if accs else 0.0

def per_frequency_bins(y_true, y_pred, freq, edges) -> dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); freq = np.asarray(freq)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        m = (freq >= lo) & (freq <= hi if last else freq < hi)
        label = f"[{lo},{hi}]" if last else f"[{lo},{hi})"
        out[label] = {"n": int(m.sum()),
                      "acc": float((y_pred[m] == y_true[m]).mean()) if m.any() else None}
    return out

def softmax_nll(probs, y_true) -> float:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    p = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))

def brier_score(probs, y_true) -> float:
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    oh = np.zeros_like(probs); oh[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
```
- [ ] **Step 4: Run to verify pass; commit**
```bash
git add src/alignair/aa/convergence/ tests/alignair/aa/convergence/
git commit -m "feat(aa-conv): metrics — macro accuracy, allele-frequency bins, softmax NLL, Brier"
```

### Task 4: Unified checkpoint evaluator over a frozen fingerprinted cohort
**Files:** Create `src/alignair/aa/convergence/evaluator.py`; Test `tests/alignair/aa/convergence/test_evaluator.py`.
**Interfaces:** `build_cohort(dc_name, n_per_stratum, seed) -> (rows, fingerprint_sha256)` (frozen simulator cohort with the eval_strategic strata; SHA-256 over the ordered records); `evaluate_checkpoint(model, cfg, class_maps, meta, cohort) -> dict` returning the full metric set: `{clean_v_acc, clean_j_acc, v_macro, j_macro, d_allele_acc, d_allele_macro, d_gene_acc, by_stratum{...acc}, freq_bins{v,j}, boundary_mae{v,j}, nll{v,j,d}, brier{v,j,d}, conformal{coverage, set_mean, set_p95}}`.

- [ ] **Step 1: Write the failing test** — build a TINY cohort (small n, fixed seed), assert `build_cohort` fingerprint is deterministic (same seed → same sha) and `evaluate_checkpoint` returns a dict containing the required top-level keys with values in `[0,1]` for accuracies. (Use the real `strategic_setup` model untrained — values need not be good, only well-formed.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement.** Compose from existing code: strata built as in `scripts/aa/eval_strategic.py` (clean pool via `stream_benchmark(default_igh_spec(n_per_stratum, seed), ...)` + `aa_substitute`/`aa_indel`/`il_ambiguity`/`aa_truncate` augmentations); run `class_probs` (Task-4-of-phase3 util) per gene to get argmax + probs; compute accuracies (micro via `==`, macro via `metrics.macro_accuracy`), `freq_bins` using `class_maps` allele frequencies (from `dc.gene_use_dict` or uniform if unavailable — document choice), boundary MAE via the model's start/end soft-argmax heads vs projected coords, `nll`/`brier` via `metrics`, and per-checkpoint conformal via `alignair.aa.conformal` APS fit on a fixed calibration slice (coverage + `_set_stats`). Fingerprint = `hashlib.sha256` over the ordered `(sequence_aa, v_call, j_call, d_call)` tuples.
- [ ] **Step 4: Run to verify pass; commit**
```bash
git add src/alignair/aa/convergence/evaluator.py tests/alignair/aa/convergence/test_evaluator.py
git commit -m "feat(aa-conv): unified checkpoint evaluator over a hash-fingerprinted frozen cohort"
```

---

## Milestone C — Selection + driver

### Task 5: Frozen selection rule (§5 of the spec)
**Files:** Create `src/alignair/aa/convergence/select.py`; Test `tests/alignair/aa/convergence/test_select.py`.
**Interfaces:** `select_checkpoint(incumbent: dict, candidates: list[dict]) -> dict` returning `{winner: <name or "incumbent">, rationale, per_candidate: [...]}`. Each metrics dict has keys used by the gates/objective. Implements EXACTLY: gates (clean_v>=0.980, clean_j>=0.988, macro not down >0.005, conformal coverage>=0.94); objective = count of weakness metrics improved by threshold (accuracy >=+0.01; set-size >=5% relative reduction) over {d_allele_macro, d_gene_acc, trunc_v_acc, shortread_acc, macro_minus_micro, conf_set_mean, conf_set_p95}; tie-break (d_allele_macro+trunc_v gain, then earliest by `step`); null → "incumbent".

- [ ] **Step 1: Write the failing test** — synthetic incumbent + candidates covering: (a) a candidate that fails a gate → rejected; (b) a gate-passer improving 2 weaknesses → selected over one improving 1; (c) tie broken by earliest step; (d) no candidate improves any weakness → winner "incumbent". (Full assertions per case.)
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the rule exactly per §5 (pure function over metric dicts; no I/O, no model).
- [ ] **Step 4: Run to verify pass; commit**
```bash
git add src/alignair/aa/convergence/select.py tests/alignair/aa/convergence/test_select.py
git commit -m "feat(aa-conv): frozen blinded selection rule (gates + weakness-objective + parsimony + null)"
```

### Task 6: Convergence driver + run script
**Files:** Create `src/alignair/aa/convergence/driver.py`, `scripts/aa/convergence/run_study.py`.
- [ ] **Step 1: Write the driver + script.** `run_study.py --resume experiments/aa/strategic/model_stage1_PASS.pt --budget 300000 --ckpt-steps 25000,75000,150000,300000 --device cuda`: builds the frozen evaluator cohort (fixed seed disjoint from 1000/9000/train) + writes `ART/cohort_fingerprint.json`; evaluates the INCUMBENT first (baseline row); calls `stage1_train(..., resume_path=<incumbent>, total_budget=358000-ish, base_lr=1e-4, lr_min=1e-5, ckpt_dir=experiments/aa/strategic, ckpt_steps=<the 4 absolute step numbers>, val_every=..., val_fn=<evaluator>)`; after training, evaluate each saved `model_cont_step{N}.pt`; write `ART/checkpoints_metrics.json` (incumbent + 4 candidates, full metrics + cohort fingerprint) and `ART/convergence.md` (curves table). Copy §5 into `ART/predeclaration.md`. Do NOT run selection here (kept as a separate, auditable step). Print a summary.
- [ ] **Step 2: Commit the code (no run yet).**
```bash
git add src/alignair/aa/convergence/driver.py scripts/aa/convergence/run_study.py
git commit -m "feat(aa-conv): convergence driver + run_study CLI (train->per-checkpoint eval->metrics table)"
```

---

## Milestone D — Execution (run, select, recalibrate, Phase-3 descriptive)

### Task 7: Smoke-validate faithful resume + evaluator end-to-end (+25k or fewer)
- [ ] **Step 1:** Run a SHORT continuation (e.g. `--budget 25000 --ckpt-steps 5000,25000` or even smaller) to validate: (a) resume from `model_stage1_PASS.pt` runs (warm-start hop logged), (b) a `model_cont_step*.pt` is produced with full optimizer/scheduler state, (c) resuming from THAT continues faithfully (optimizer moments non-empty, step advances), (d) the evaluator produces a well-formed metrics row for the incumbent and a candidate. Confirm clean-V/J on the incumbent row ≈ the known 0.986/0.992 (sanity that the evaluator cohort + metric wiring are correct). If clean-V/J are far off, STOP and diagnose the evaluator before the full run.
- [ ] **Step 2:** Record the smoke result in `ART/smoke_check.md`. Commit that note (not the .pt files).

### Task 8: Run the +300k convergence study
- [ ] **Step 1:** `PYTHONPATH=src .venv/bin/python scripts/aa/convergence/run_study.py --resume experiments/aa/strategic/model_stage1_PASS.pt --budget 300000 --ckpt-steps 25000,75000,150000,300000 --device cuda` (background; long). Produces the 4 candidate checkpoints + `ART/checkpoints_metrics.json` + `ART/convergence.md`.
- [ ] **Step 2:** Commit `ART/checkpoints_metrics.json`, `ART/convergence.md`, `ART/cohort_fingerprint.json`, `ART/predeclaration.md` (NOT .pt files).

### Task 9: Blinded selection → recalibrate winner → Phase-3 descriptive
- [ ] **Step 1:** Run `select_checkpoint` over `checkpoints_metrics.json`; write `ART/selection.md` with the gate/objective arithmetic + the winner (or "incumbent"/null). This is still blinded (no Phase-3 data).
- [ ] **Step 2 (only if a candidate won):** Refit conformal on a FRESH cohort (new seeds, disjoint) for the winner → `ART/calibration_<winner>.json` (reuse `calibrate_stage2.py` logic pointed at the winner .pt; do NOT overwrite the frozen calibration.json).
- [ ] **Step 3 (only if a candidate won):** Rerun `scripts/aa/phase3/evaluate.py` on the winner + its new calibration, writing a NEW `phase3_eval_<winner>.json` (descriptive only). Compare family accuracy to the incumbent's 16/16.
- [ ] **Step 4:** Write `ART/CONVERGENCE_RESULT.md` — the honest conclusion: did longer training help (which weaknesses), which checkpoint won (or null), the recalibrated numbers, and the Phase-3 consistency check. Commit all `ART/*.md` + JSON (NOT .pt).

---

## Self-Review notes (spec coverage)
- §2 resumable trainer (full-state checkpoint, cosine LR, immutable names, validation+plateau, back-compat): Tasks 1, 2.
- §3 unified evaluator + new metrics (macro, freq-bins, NLL/Brier, boundary MAE, conformal, fingerprinted cohort): Tasks 3, 4.
- §4 driver / budget / checkpoint schedule: Task 6, 8.
- §5 FROZEN selection rule (gates+objective+tie-break+null, blinded): Task 5, 9.
- §6 recalibrate winner (fresh cohort, no overwrite): Task 9.
- §7 Phase-3 descriptive rerun (winner only): Task 9.
- §0 immutability (incumbent/calibration/phase3 untouched; no .pt committed): Global Constraints, enforced every task.
