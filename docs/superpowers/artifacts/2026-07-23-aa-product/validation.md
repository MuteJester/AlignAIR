# AA inference product-path validation (Task 9)

Model: `alignair-aa-igh-human@2.0.0` &middot; device: `cuda` &middot; cohort seed: `20260201` &middot; n_reads: 45

**ALL PASS: True**

| Check | Result | Summary |
|---|---|---|
| `golden_cohort` | PASS | 15 full / 15 trunc / 15 short reads; deterministic rebuild: True |
| `parity` | PASS | 135 (read x gene) checks over 45 reads x 3 genes (V, J, D); 0 mismatches |
| `coordinates` | PASS | bounds_ok=True junction_ok=True voiding_contract={'voided_ok': True, 'clipping_ok': True} |
| `groups` | PASS | pooled/supplied labels + warning correct; supplied 'full' V-set == q_mondrian['full']: True (1/45 reads differ from the pooled set) |
| `phase3_descriptive` | PASS | 16/16 (1.000) |
| `tamper_rejection` | PASS | tamper raises=True; wrong-model raises=True |

## Phase-3 16-antibody descriptive family accuracy

16/16 (1.000) over 16/16 scoreable gold rows (0 missing hydration). DESCRIPTIVE ONLY — no gate/promotion decision; Phase-3's frozen decision remains `inconclusive` regardless (see `../2026-07-21-aa-phase3/PHASE3.md`).

Full detail (including per-check provenance) in `validation.json`.
