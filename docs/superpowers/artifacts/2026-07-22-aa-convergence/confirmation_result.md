# Promotion Confirmation — Result

**Date:** 2026-07-23 · **Verdict (per the frozen `promotion_criteria.md`): HOLD.** 4 of 5 criteria pass; the
D-gains criterion (c2) fails — the study's D-allele-macro improvement did **not** reproduce on a fresh cohort.
Incumbent (`model_stage1_PASS.pt`) and calibration remain immutable; nothing was promoted.

Method: incumbent vs winner ONLY, larger untouched cohort (n=2000/stratum, seed 770077), paired per-read
analysis; corrected empirical class-frequency bins; conformal calibrated on seed 990099, tested on seed 112211.

## Criteria outcome
| # | criterion | result |
|---|---|---|
| c1 | gates pass (winner clean-V ≥0.980, clean-J ≥0.988) | **PASS** (0.9906 / 0.9935) |
| c2 | D-allele-macro **and** D-gene gains reproduce | **FAIL** |
| c3 | moderate-SHM & multi-indel V do not regress | **PASS** |
| c4 | no stratum loses >1pt (point + McNemar) | **PASS** (every stratum improved) |
| c5 | winner conformal V coverage ≥0.94 | **PASS** (0.999) |

## Why c2 failed — the D "gain" was noise
- **D-allele-macro:** study cohort 0.797→0.820 (+2.3pt); fresh cohort **0.8006→0.7976 (−0.3pt)**, class-stratified
  bootstrap 95% CI **[−0.060, +0.058]** (straddles 0, very wide). The +2.3pt was cohort-specific — consistent
  with its noisy per-checkpoint trajectory (0.797→0.807→0.801→0.811→0.820). D is the hardest, most
  class-imbalanced target, and its macro has high variance.
- **D-gene:** 0.8491→0.8549 (+0.58pt), paired CI [−0.011, +0.022] — positive but **not significant**.
c2 required both point-positive with at least one significant and neither regressing; D-allele-macro's negative
point estimate fails it.

## What DID reproduce — the V improvements are real, significant, and regression-free
Every V stratum improved; the hard ones are strongly significant (paired 95% CI, exact McNemar p):
| stratum | inc→win | 95% CI | McNemar p |
|---|---|---|---|
| v_indel_multi | 0.729→**0.784** (+5.5pt) | [+0.045,+0.065] | 4.6e-29 |
| short_janchor | 0.271→**0.311** (+3.9pt) | [+0.024,+0.056] | <1e-3 |
| v_indel_1 | 0.660→**0.692** (+3.3pt) | [+0.020,+0.047] | <1e-3 |
| dj_indel | 0.533→**0.556** (+2.3pt) | [+0.005,+0.040] | 0.014 |
| aa_truncate_50_80 | 0.872→**0.890** (+1.8pt) | [+0.005,+0.031] | 0.011 |
| moderate_full | 0.925→**0.938** (+1.3pt) | [−0.003,+0.029] | 0.15 |
| aa_single_indel | 0.976→**0.985** (+0.9pt) | [+0.004,+0.015] | 0.002 |
| clean_full | 0.984→**0.991** (+0.65pt) | [+0.002,+0.012] | 0.012 |
| aa_il / aa_subs | small positive | — | — |

- **No stratum regressed** (every point diff ≥ 0).
- **Calibration held** (V coverage 0.999 both; winner set-size slightly sharper).
- **Corrected frequency bins (empirical prior)** show no rare-class harm: rarest bin flat (0.997→0.997), mid bin
  **improved** (0.979→0.995), common bin improved (0.982→0.986).

## Honest reading
The winner is a **broadly and robustly better V model** — large, significant gains on the SHM/indel/short-read
frontier with zero regressions and maintained calibration. Its apparent **D improvement was not real** (noise).
Per the predeclared rule this is a **HOLD**, because c2 specifically required the D gains to reproduce.

**Decision is the user's**, seeing this evidence — the frozen criteria are not overridden here:
1. **Respect HOLD** — the predeclaration required D reproduction; it failed. Do not promote.
2. **Promote on V-merit** — the V gains (the model's primary task) reproduced strongly with no downside;
   accept that D is unchanged. This relaxes the frozen c2 and must be an explicit, recorded human decision.
Either way, the winner should **not** be described as improving D. If promoted, its version notes should say:
"improves V under SHM/indels/truncation/short reads; D resolution unchanged."
