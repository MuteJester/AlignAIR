# Promotion Confirmation — FROZEN Predeclaration (2026-07-23)

Confirm whether the §5 winner `cont_step318000` should be promoted to supersede the incumbent
`model_stage1_PASS.pt` for **simulator-domain** use. This is a separate, cheap confirmation predeclared BEFORE
it is run. **Only the incumbent and the winner are compared** — the intermediate checkpoints are NOT re-scored.
The Phase-3 therapeutic result stays descriptive/inconclusive and plays no part here.

## Cohorts (all NEW seeds, disjoint from everything used so far: train 0, study-eval 20250722, study-calib
1000/9000, freq-prior 555)
- **Confirmation test cohort:** `default_igh_spec(n_per_stratum=2000, seed=770077)`, all strata (clean_full,
  moderate_full, aa_subs_7pct, aa_single_indel, aa_il_ambiguity, aa_truncate_50_80, short_janchor, v_indel_1,
  v_indel_multi, dj_indel), larger than the study's n=1000. Both models predict **per read** on the SAME
  records (paired).
- **Empirical frequency prior (frozen):** V/J class frequencies estimated from a frozen stream sample
  `seed=880088` (≥ 8000 reads) — the corrected frequency binning (uniform `gene_use_dict` is NOT used). Bins:
  rare / mid / common by class frequency (edges chosen from the prior, frozen before scoring).
- **Conformal:** calibrate each model on a **separate** cohort `seed=990099`, then measure coverage/set-size
  on a **separate** test cohort `seed=112211` (disjoint from calibration and from the accuracy cohort).

## Statistics (paired, per read)
- **Paired bootstrap 95% CI** for the winner−incumbent accuracy difference, per stratum and per metric
  (resample reads with replacement; both models scored on the same resample).
- **McNemar** exact test (and odds) on paired top-1 correctness changes (discordant pairs b, c) per stratum.
- **Macro accuracy** with **class-stratified bootstrap** (resample classes) 95% CI for the macro difference.
- Corrected empirical **frequency-bin** accuracy for both models, per bin.

## PROMOTION criteria — ALL must hold (else HOLD, do not promote)
1. **Gates still pass** on the winner: clean_v ≥ 0.980 AND clean_j ≥ 0.988 (clean = clean_full).
2. **D gains reproduce:** winner point estimate > incumbent for BOTH `d_allele_macro` and `d_gene_acc`, AND at
   least one of the two has a paired 95% CI lower bound > 0 (statistically significant gain), AND neither has a
   CI upper bound < 0 (no significant regression).
3. **No regression on the consistent-frontier strata:** for `moderate_full` V and `v_indel_multi` V, winner ≥
   incumbent − 0.01 (point) AND paired 95% CI lower bound > −0.01.
4. **No important stratum loses > 1 pt:** for EVERY stratum, winner_acc ≥ incumbent_acc − 0.01 (point); and no
   stratum shows a McNemar-significant (p<0.05) regression larger than 1 pt.
5. **Conformal coverage valid:** on the separate test cohort after separate calibration, the winner's V
   coverage ≥ 0.94 (target 0.95).

If all five hold → **PROMOTE** `cont_step318000` as a new simulator-domain model version + its own calibration
(the incumbent snapshot stays immutable). If any fails → **HOLD** and report which.

## Non-goals
No intermediate-checkpoint rescoring; no training; no use of the Phase-3 cohort for this decision; no change to
the frozen incumbent, its calibration, or the Phase-3 artifacts.
