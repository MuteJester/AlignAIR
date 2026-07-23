# AA Convergence Study — Result

**Date:** 2026-07-22 · **Branch:** `experiment/aminoacid-alignair` · **Verdict: longer training HELPED (non-null).**
Blinded §5 winner = **`cont_step318000`** (the +300k checkpoint). Selection used only frozen simulator cohorts;
the Phase-3 therapeutic cohort was NOT consulted for selection. The frozen incumbent snapshot
(`model_stage1_PASS.pt`, `calibration.json`) is unchanged.

## Run
Continued from the incumbent (58k cumulative steps; the checkpoint's own counter reads 18000 from the last
fine-tune stage) for **+300k steps** (~13.5h, CPU-bound gym stream ~22k steps/h), cosine LR 1e-4→1e-5.
Resume semantics: **full optimizer/scheduler-state resume with deterministic fresh-stream continuation** — the
optimizer, scheduler, and Torch/global RNG state are restored, but the local augmentation generator and the
GenAIRR record stream are *re-seeded* from `(seed, step)` rather than restored to the exact uninterrupted
stream position. This is deterministic and statistically reasonable, **not** byte-equivalent / mathematically
exact trajectory continuation. Note also: the incumbent→continuation hop necessarily started a **fresh
optimizer**, because the incumbent checkpoint (old format) carried no optimizer state. Candidates saved +
evaluated at +25k/+75k/+150k/+300k on a hash-fingerprinted frozen cohort (`b962135f…`, seed 20250722, disjoint
from training seed 0 and calibration seeds 1000/9000).

## Blinded §5 selection (`selection.md`)
All four candidates passed every gate. Objective-count (predeclared weaknesses improved by threshold) climbed
**monotonically with training: 1 → 1 → 2 → 3**. Winner `cont_step318000` improved **3** §5 weaknesses
(D-allele-macro, D-gene, short-read) — the max — so it wins outright.

## What improved (incumbent → +300k winner)
**Gates (held/improved):** clean-V 0.984→**0.990**, clean-J 0.996→**0.997**, V-macro 0.982→**0.991**,
conformal coverage 1.0 (≥0.94). No macro regression.

**§5 weakness metrics (counted):**
- D allele-macro 0.797 → **0.820** (+2.3pt)
- D gene accuracy 0.824 → **0.857** (+3.3pt)
- short-read V 0.259 → **0.290** (+3.1pt)
- (trunc-V 0.889→0.896 and macro−micro gap improved but under the +1.0pt threshold; conformal mean set-size
  1.12→1.08, a 3.9% reduction, under the 5% threshold — real but not "counted".)

**Informative strata NOT in the §5 objective — the known heavy-SHM-V / indel frontier.** Two rise
*consistently at every checkpoint*; the others improve overall but are **noisy**:
- moderate-SHM full-length V 0.906 → 0.915 → 0.935 → 0.946 → **0.955** (+4.9pt) — rises every checkpoint
- V-indel-multi 0.720 → 0.726 → 0.743 → 0.759 → **0.770** (+5.0pt) — rises every checkpoint
- V-indel-1 0.684 → 0.705 → 0.690 → 0.712 → **0.723** — net up, fluctuates
- AA-subs-7% 0.939 → **0.948** — small net gain

The `clean` (clean_full, minimal-SHM) V was already saturated (0.986) and stayed there; the gains are where the
model was weak — moderate/high SHM, multi-indels, D resolution, short reads.

## Trend, NOT convergence — corrected wording
Earlier phrasing ("monotonic", "not plateaued") was too strong. The accurate statement:
> The endpoint trend remains favorable, with particularly **consistent** improvements in moderate-SHM and
> multi-indel V recovery; **several other metrics are noisy** (D-allele-macro 0.797→0.807→0.801→0.811→0.820;
> D-gene 0.824→0.833→0.849→0.833→0.857; short-read 0.259→0.277→0.254→0.287→0.290; truncation-V peaks 0.898 at
> +150k then ends 0.896; V-indel-1 fluctuates), so **plateau status is unresolved**.

The §5 objective-count rose 1→1→2→3, but that is a count of **threshold crossings**, not a convergence
diagnostic — it does not establish that the curve is still climbing or has flattened.

## Evaluator caveat: allele-frequency bins were degenerate in this run
The per-allele-frequency stratification in this run collapsed all V/J examples into the lowest bin: GenAIRR's
`gene_use_dict` samples **genes uniformly** (V 1/59, J 1/6), so the derived frequency had no variation. The
`freq_bins` field is present but carries **no useful stratification** for this study. Macro accuracy remains
the valid rare-class signal here. The fix (an **empirical frozen class-frequency prior** from a frozen stream
sample — class frequencies genuinely vary 0.0013–0.029 despite uniform gene sampling) is applied in the
promotion-confirmation study, not retrofitted to these numbers.

## Recalibration + Phase-3 consistency
- Winner recalibrated on a fresh cohort → `calibration_cont_step318000.json` (frozen `calibration.json`
  untouched). Deployable conformal for the winner.
- Phase-3 **descriptive** rerun on the frozen 16-antibody cohort (winner + new calibration + the same ASC→IMGT
  bridge, `phase3_eval_cont_step318000.json`): **top-class family accuracy 16/16 = 1.000, decision
  `inconclusive`** — identical to the incumbent. The coarse family endpoint cannot see the simulator-domain
  gains (it was already 16/16); this is a consistency check that longer training did not break real-domain
  family behavior.

## Provenance / immutability
Incumbent (`model_stage1_PASS.pt`, `calibration.json`, Phase-3 artifacts) unchanged. Candidate `.pt` files
live under `experiments/aa/strategic/` (git-ignored); only metrics/selection/calibration/provenance JSON+MD are
committed. Cohort fingerprint, predeclaration (`predeclaration.md`), and per-checkpoint metrics
(`checkpoints_metrics.json`, `convergence.md`) are recorded.

## Recommendation (revised)
`cont_step318000` is a credible new candidate: it wins the predeclared §5 selection and improves meaningful
weaknesses without breaking the gates. **Before promotion**, run a cheap confirmation — incumbent vs winner
ONLY, on a larger untouched cohort with new seeds, paired per-read analysis (paired bootstrap CIs, McNemar,
class-stratified bootstrap), corrected empirical frequency bins, all strata, and a separate calibration→test
split — against predeclared promotion criteria (see `promotion_criteria.md`). If it passes, promote as a new
simulator-domain model version with its own calibration.

A further training extension is **optional research, not a blocker**: any second run must use a *new*,
deliberately-named LR schedule (e.g. 3e-5→3e-6) — do NOT resume the completed cosine (already at `T_max`). The
larger near-term product gain is the deployable AA inference API, calibrated-set output, model packaging, and
real-protein testing — not squeezing another uncertain point of simulator accuracy.
