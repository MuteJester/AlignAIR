# AA Convergence Study — Result

**Date:** 2026-07-22 · **Branch:** `experiment/aminoacid-alignair` · **Verdict: longer training HELPED (non-null).**
Blinded §5 winner = **`cont_step318000`** (the +300k checkpoint). Selection used only frozen simulator cohorts;
the Phase-3 therapeutic cohort was NOT consulted for selection. The frozen incumbent snapshot
(`model_stage1_PASS.pt`, `calibration.json`) is unchanged.

## Run
Continued from the incumbent (58k cumulative steps; the checkpoint's own counter reads 18000 from the last
fine-tune stage) for **+300k steps** (~13.5h, CPU-bound gym stream ~22k steps/h), cosine LR 1e-4→1e-5,
faithful-resume trainer. Candidates saved + evaluated at +25k/+75k/+150k/+300k on a hash-fingerprinted frozen
cohort (`b962135f…`, seed 20250722, disjoint from training seed 0 and calibration seeds 1000/9000).

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

**Informative strata NOT in the §5 objective — the known heavy-SHM-V / indel frontier — improved the MOST,
monotonically:**
- moderate-SHM full-length V 0.906 → 0.915 → 0.935 → 0.946 → **0.955** (+4.9pt)
- V-indel-1 0.684 → **0.723**; V-indel-multi 0.720 → **0.770** (+5.0pt)
- AA-subs-7% 0.939 → **0.948**

The `clean` (clean_full, minimal-SHM) V was already saturated (0.986) and stayed there; the gains are exactly
where the model was weak — moderate/high SHM, indels, D resolution, short reads.

## Not saturated at +300k
Every weakness stratum is still rising at the final checkpoint and the objective-count is still climbing
(1→1→2→3). The convergence curve has **not plateaued** — a further extension beyond +300k would likely help
more. This first run answered the question (does longer training help? **yes, monotonically**) and localized
the gains.

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

## Recommendation
`cont_step318000` is a genuine, defensible improvement on the weak areas (D resolution, moderate/high-SHM V,
V-indels, short reads, calibration sharpness) with no regression on the strengths. It is a valid candidate to
supersede the incumbent for simulator-domain use once promoted with its recalibration. Because the curve was
still improving, a follow-up extension (e.g. another +300k–500k) is the natural next experiment if further
gains on the SHM/indel frontier are wanted.
