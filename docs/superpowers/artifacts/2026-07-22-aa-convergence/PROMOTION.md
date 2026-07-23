# AA IGH model — Promotion Record: v2 (`cont_step318000`)

**Date:** 2026-07-23 · **Branch:** `experiment/aminoacid-alignair` · **Decision: PROMOTE on V-merit, with an
explicit, human-authorized exception to predeclared criterion c2.**

## What was promoted
- **New simulator-domain version `aa_igh_v2`** = the +300k convergence winner `cont_step318000`.
  - model file: `experiments/aa/strategic/model_aa_igh_v2.pt` (git-ignored)
  - sha256: `81255006fa88bf42973d64bff56984230be89a986a19a749d763c696d97d06de`
  - calibration: `docs/superpowers/artifacts/2026-07-22-aa-convergence/calibration_cont_step318000.json`
    (V class-map fingerprint `e1a455b8…` — identical class maps to v1)
- **v1 (incumbent) stays immutable:** `model_stage1_PASS.pt` sha
  `54eef812942e7b96f3d49c9ed881e878c06f8978576a9f16345e797c8937dfc6` and its `calibration.json` and the Phase-3
  artifacts are unchanged.

## Version notes (use these verbatim; do NOT overstate)
> **aa_igh_v2 improves V-segment recovery under SHM, indels, truncation, and short reads; D-segment resolution
> is unchanged.** It is not a D improvement.

## Basis for promotion
Blinded §5 selection chose `cont_step318000` (see `selection.md`). The promotion confirmation
(`confirmation_result.md`, `confirmation.json`; incumbent vs winner only, fresh cohort seed 770077, paired
per-read stats) returned **4 of 5 predeclared criteria PASS**:
- c1 gates PASS (clean-V 0.991, clean-J 0.994) · c3 frontier-no-regress PASS · c4 no-stratum-loses-1pt PASS
  (every V stratum improved) · c5 conformal coverage PASS (0.999).
- The V gains reproduced robustly and significantly: **v_indel_multi +5.5pt (McNemar p≈1e-29)**, short-read
  +3.9pt, v_indel_1 +3.3pt, dj_indel +2.3pt, truncation +1.8pt, moderate-SHM +1.3pt, clean +0.65pt; corrected
  empirical frequency bins show no rare-class harm (mid/common bins improved).

## The c2 exception — explicit and recorded
- **c2 (D-allele-macro AND D-gene gains reproduce) FAILED.** The study's D-allele-macro gain (0.797→0.820) did
  **not** reproduce on the fresh cohort (0.8006→0.7976, −0.3pt, class-bootstrap CI [−0.060, +0.058]); D-gene's
  +0.58pt was not significant. The apparent D improvement was **noise**.
- The frozen `promotion_criteria.md` says "any criterion fails → HOLD". The user, seeing the full evidence,
  **explicitly authorized promoting on V-merit** — the model's primary task improved robustly with zero
  regressions and maintained calibration — while accepting that **D is unchanged** and recording this as a
  deliberate relaxation of c2, not a passing result. This exception applies only to this promotion and does not
  amend the predeclaration for any future study.

## Provenance / immutability
Incumbent v1 (`model_stage1_PASS.pt`, `calibration.json`, `2026-07-21-aa-phase3/*`) byte-for-byte unchanged.
Phase-3 result remains descriptive and `inconclusive` (v2 also holds 16/16 family accuracy —
`phase3_eval_cont_step318000.json`). No `.pt` committed. v2's deployable pieces are its model file (sha above)
+ its calibration JSON.

## Next
Per the user's steer, the priority is now the **deployable AA inference product** (API, calibrated-set output,
model packaging, real-protein testing) rather than further simulator training. Any future training extension
must use a new, deliberately-named LR schedule (not the completed cosine, already at `T_max`).
