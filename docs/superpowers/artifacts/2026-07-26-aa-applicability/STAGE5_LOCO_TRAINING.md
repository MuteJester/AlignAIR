# Stage 5 (a) — LOCO training result (burned-dev only; NO eligibility judgment)

**16/16 models OK** · loco manifest `7d73bce97001` · mask `9101c1735bbe` · device cuda · budget {"wall_s": 740.4, "gpu_h": 0.2056, "n_runs": 28}

Candidates use frozen Stage-4 seeds + unchanged hyperparameters. Each fold restricts the frozen partition (no reshuffle); g excluded from train/dev/mining/norm/source-balance/early-stop; normalization + source vocab recomputed per fold. Blind features per fold. This is a training diagnostic — candidate eligibility and C1–C4 comparison remain Stage 6.

## Per candidate × fold

| model | fold | status | dev_loss | app_AUROC | type_acc | complete_acc | elig_src_frac | n_train | n_excl | stop | wall_s | vram_MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1.s2.igk_igl | igk_igl | ok | 0.8194 | 0.9674 | 0.8725 | 0.9694 | 0.4857 | 729 | 159 | 200 | 9.2 | 344.1 |
| C1.s2.igsf | igsf | ok | 0.6721 | 0.9649 | 0.9 | 0.9796 | 0.512 | 719 | 178 | 100 | 7.6 | 327.5 |
| C1.s2.tcr | tcr | ok | 0.5008 | 0.978 | 0.9416 | 0.9796 | 0.4722 | 793 | 90 | 250 | 9.8 | 352.7 |
| C1.s2.vhh_nonhuman | vhh_nonhuman | ok | 0.4145 | 0.9793 | 0.9518 | 0.9694 | 0.4062 | 807 | 64 | 500 | 14.6 | 373.8 |
| C2.s1.igk_igl | igk_igl | ok | 0.5078 | 0.9864 | 0.906 | 0.9694 | 0.4857 | 729 | 159 | 150 | 8.1 | 344.2 |
| C2.s1.igsf | igsf | ok | 0.5934 | 0.9808 | 0.9214 | 0.9796 | 0.512 | 719 | 178 | 400 | 13.2 | 327.5 |
| C2.s1.tcr | tcr | ok | 0.5145 | 0.9917 | 0.9286 | 0.9796 | 0.4722 | 793 | 90 | 350 | 12.5 | 352.8 |
| C2.s1.vhh_nonhuman | vhh_nonhuman | ok | 0.2905 | 0.9953 | 0.9819 | 0.9796 | 0.4062 | 807 | 64 | 350 | 12.0 | 373.8 |
| C3.s1.igk_igl | igk_igl | ok | 0.5779 | 0.9901 | 0.8792 | 0.9796 | 0.4857 | 729 | 159 | 300 | 11.7 | 344.2 |
| C3.s1.igsf | igsf | ok | 0.5221 | 0.9825 | 0.9214 | 0.9796 | 0.512 | 719 | 178 | 250 | 12.2 | 327.5 |
| C3.s1.tcr | tcr | ok | 0.5487 | 0.9803 | 0.9286 | 0.9694 | 0.4722 | 793 | 90 | 250 | 10.6 | 352.8 |
| C3.s1.vhh_nonhuman | vhh_nonhuman | ok | 0.2470 | 0.9978 | 0.9639 | 0.9796 | 0.4062 | 807 | 64 | 450 | 14.7 | 373.8 |
| C4.s1.igk_igl | igk_igl | ok | 3.7995 | 0.9929 | 0.8926 | 1.0 | 0.4857 | 729 | 159 | 400 | 15.0 | 344.2 |
| C4.s1.igsf | igsf | ok | 3.6148 | 0.9714 | 0.9071 | 0.9898 | 0.512 | 719 | 178 | 800 | 25.3 | 327.6 |
| C4.s1.tcr | tcr | ok | 3.8785 | 0.9598 | 0.8831 | 0.949 | 0.4722 | 793 | 90 | 400 | 15.2 | 352.8 |
| C4.s1.vhh_nonhuman | vhh_nonhuman | ok | 3.4571 | 0.9965 | 0.9096 | 0.9796 | 0.4062 | 807 | 64 | 200 | 11.2 | 373.9 |

## Cross-fold stability per candidate (dev_loss / AUROC spread over its 4 folds)

| candidate | folds_ok | dev_loss mean±sd | AUROC mean±sd |
|---|---|---|---|
| C1 | 4/4 | 0.6017±0.1562 | 0.9724±0.0063 |
| C2 | 4/4 | 0.4766±0.1125 | 0.9886±0.0055 |
| C3 | 4/4 | 0.4739±0.1325 | 0.9877±0.0069 |
| C4 | 4/4 | 3.6875±0.1639 | 0.9802±0.0152 |

## Scope audit
- files opened: ['.private/aa-gate3-hydration/dev_cohort.jsonl', '.private/aa-applicability/profiles/*', 'stage4 manifests (config)']
- NOT opened: ['fresh_selection (*.normalized.json fresh cohort)', 'loco_challenge subsets', 'far_ood pools', '.private/aa-gate3-hydration/locked_test.jsonl (SEAL)']
- No eligibility judgment made. Next: Task 5.3 + fresh one-shot sequence (separate review).
