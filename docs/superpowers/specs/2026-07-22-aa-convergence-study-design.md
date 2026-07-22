# AA AlignAIR — Longer-Training Convergence Study (Design Spec + Frozen Predeclaration)

**Date:** 2026-07-22 · **Branch:** `experiment/aminoacid-alignair` · **Status:** design approved; this document
is the **immutable predeclaration** — the selection rule in §5 is frozen and committed BEFORE any training.

## §0 Immutability & isolation (do not touch)
- **Never modify or overwrite** the frozen Phase-3 snapshot: `experiments/aa/strategic/model_stage1_PASS.pt`
  (sha `54eef812…`), `docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json`, and the frozen
  Phase-3 cohort under `docs/superpowers/artifacts/2026-07-21-aa-phase3/`. The study produces a NEW candidate
  under new filenames; the incumbent stays byte-identical.
- All work isolated on `experiment/aminoacid-alignair`.

## §1 Purpose
The Stage-1 model has seen 58k steps (40k baseline + 18k truncation fine-tune) and is near-saturated on clean
V (0.986) / J (0.992). This study asks, with a checkpointed convergence curve, whether **continued** training
improves the known weak areas — D resolution, truncation/short-read, rare classes, calibration sharpness —
**without regressing** the accuracy/coverage gates. Phase 3 **cannot** measure this (the model already scores
16/16 at the coarse family endpoint), so selection is on frozen **simulator** cohorts, blinded to Phase 3.

## §2 Resumable trainer (engineering corrections — prerequisite)
The current `stage1_train`/`_save` (`src/alignair/aa/strategic_trunc.py`) cold-restarts: checkpoints hold only
`{model, cfg, step}`, and a fresh `AdamW(lr=3e-4)` with no scheduler is built each run. Corrections:
- **Checkpoint contents:** `{model, optimizer, scheduler, rng_numpy, rng_torch, step, lr, history, provenance}`
  where `provenance = {git_sha, base_model_sha256, seed, hyperparams, dc_name, created_step}`.
- **Faithful resume** from such a checkpoint restores optimizer/scheduler/RNG/step (not just model weights).
- **LR:** continuation uses **cosine decay from 1e-4 → ~1e-5** over the declared budget (not the original fixed
  3e-4).
- **Validation at fixed step intervals** (calls the §3 evaluator on the frozen cohort) — recorded, not used to
  early-abort silently.
- **Immutable checkpoint filenames** `model_cont_step{N}.pt` (never overwritten).
- **Plateau detection:** log when the primary validation objective stops improving for K consecutive
  validations (informational; does not auto-stop the predeclared budget).
- **Deterministic stream:** the augmentation RNG + gym stream re-seed deterministically from `(seed, step)` so a
  resume is reproducible. (Exact byte-identical stream *position* is not required; optimizer/scheduler/RNG state
  is what makes the continuation mathematically faithful — that is restored.)
- Backward-compatible: loading an OLD `{model,cfg,step}` checkpoint (e.g. `model_stage1_PASS.pt`) as the resume
  base is allowed — it warm-starts weights and begins a fresh optimizer/scheduler (documented as the one
  unavoidable non-faithful hop, since the incumbent lacks optimizer state).

## §3 Unified checkpoint evaluator
One evaluator over a **frozen, hash-fingerprinted** simulator cohort (fixed seeds, **disjoint** from training
and from calibration's 1000/9000 seeds), computing the full metric set (reusing existing code where it exists,
building what is missing):
- V/D/J class & allele accuracy — **micro AND macro** (wire `evaluate.py::_macro_accuracy` + D-macro onto the
  strategic checkpoint).
- **Per-allele-frequency bins** (rare/mid/common) — NEW.
- Strata: clean / AA-subs / AA-indel / I-L / coverage / **truncation** / short-read (reuse eval_strategic
  strata).
- **Boundary MAE** for V/J start/end (reuse `evaluate_frozen`/`eval_gym` logic; wire to strategic).
- **Softmax NLL + Brier** over the class heads — NEW (the existing `predict/calibrate.py` ECE is
  nucleotide/sigmoid; adapt to AA softmax).
- **Conformal coverage + mean/p95 set size** per checkpoint (per-checkpoint APS fit on the fixed calibration
  cohort — this is a selection *proxy*; the final winner gets a clean §6 refit).
- Descriptive CIs (Wilson/bootstrap) throughout; the cohort is SHA-256 fingerprinted and recorded.

## §4 Convergence driver
- Continue from `model_stage1_PASS.pt`. Budget: **+300k steps (total ~358k)**. Save+evaluate candidates at
  **+25k, +75k, +150k, +300k** (immutable names). The incumbent is evaluated by the SAME evaluator as the
  baseline row.
- Output: a per-checkpoint metrics table + convergence curves (JSON + a short markdown).

## §5 FROZEN predeclared selection rule (blinded to Phase 3)
Incumbent = `model_stage1_PASS.pt`. Evaluate every candidate on the §3 frozen cohort. **Phase 3's 16-antibody
cohort is NOT consulted during selection.**
- **GATES (candidate rejected if any fails):**
  - clean V class accuracy ≥ **0.980** (incumbent 0.986)
  - clean J class accuracy ≥ **0.988** (incumbent 0.992)
  - no V or J **macro** accuracy regression > **0.5 pt** vs incumbent
  - per-checkpoint conformal coverage ≥ **0.94** (target 0.95)
- **Objective among gate-passers:** maximize the **count of predeclared weakness metrics improved by ≥ a
  threshold** vs incumbent, over the set {D allele-macro, D gene accuracy, truncation-V accuracy, short-read
  accuracy, rare-class signal (macro−micro gap), calibration mean set-size, calibration p95 set-size}.
  Improvement threshold: ≥ **1.0 pt** for accuracy metrics; ≥ **5%** relative reduction for set-size metrics.
- **Tie-breaks:** (1) larger combined (D allele-macro + truncation-V) gain; (2) **earliest** checkpoint
  (parsimony / lower over-specialization risk).
- **Null result is a valid, honest outcome:** if no gate-passing candidate improves ≥ 1 weakness metric, the
  **incumbent is kept** and the study reports "longer training did not help" — the snapshot is unchanged.

## §6 Recalibrate the winner (only if a candidate is selected)
Refit conformal calibration for the selected checkpoint on a **fresh** calibration cohort (new seeds, disjoint
from the evaluator's), producing a NEW `calibration_<winner>.json` — the frozen Phase-3 `calibration.json`
stays untouched.

## §7 Phase-3 descriptive rerun (winner only)
Only after §5-§6, rerun `scripts/aa/phase3/evaluate.py` on the winner + its new calibration, **descriptively**
(a NEW output file). This never feeds selection; it reports whether the real-domain family anchor holds for the
new model. Family accuracy is already 16/16, so this is a consistency check, not an improvement metric.

## §8 Provenance & artifacts (`docs/superpowers/artifacts/2026-07-22-aa-convergence/`)
`predeclaration.md` (a copy of §5, frozen), `cohort_fingerprint.json` (evaluator cohort SHA + seeds),
`checkpoints_metrics.json` (per-checkpoint full metric table), `convergence.md` (curves + the blinded selection
decision with its gate/objective arithmetic), and — only if a winner is selected — `calibration_<winner>.json`
and `phase3_eval_<winner>.json`. Candidate `.pt` files live under `experiments/aa/strategic/` (gitignored).

## §9 Out of scope
Architecture changes, retraining from scratch, changing the training data distribution, D-segment redesign,
and any use of the Phase-3 therapeutic cohort for selection. The first run is capped at +300k (not 24h/1.4M).
