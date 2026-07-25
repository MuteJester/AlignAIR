# Gate-3 Phase-0 Feasibility — Real GO/STOP

**Decision: GO**

Zero-error 95% upper-bound floors: <=5% -> 59, <=10% -> 29 locked-test examples. Powers the LOCKED TEST after lineage dedup + fit/val/test allocation; fit/val are additional.

| group | effective (dedup) | counts→floor | descriptive | required | test | val | fit | both loci | powered |
|---|---|---|---|---|---|---|---|---|---|
| ighk_igl | 320 | 320 | 0 | 118 | 160 | 160 | 0 | True | YES |
| tcr | 242 | 230 | 12 | 118 | 115 | 115 | 0 | True | YES |
| vhh_nonhuman | 159 | 159 | 0 | 58 | 79 | 79 | 0 | True | YES |
| igsf | 357 | 357 | 0 | 58 | 178 | 178 | 0 | True | YES |
| structural | 70 | 70 | 0 | 58 | 35 | 35 | 0 | True | YES |
| positive_real | 370 | 370 | 0 | 116 | 92 | 92 | 185 | True | YES |
| positive_sim | 200 | 200 | 0 | 116 | 50 | 50 | 100 | True | YES |

All gated groups meet their locked-test floor after lineage deduplication. Combined IGK/IGL and combined TCR have both constituent loci present; gamma/delta TCR are present as descriptive strata (excluded from the TCR floor); structural negatives are counted by parent lineage. **GO** -> Task 3 (cohort freeze) unblocks.

## Data-quality caveats (Task-3 fixes — NOT feasibility blockers)

- pos_igh_structure: median length 221 aa suggests constant-domain (CH1) contamination — trim to the VH domain (e.g. ANARCI/IMGT) before Task 3
- murine_vh: median length 217 aa suggests constant-domain (CH1) contamination — trim to the VH domain (e.g. ANARCI/IMGT) before Task 3
- trg: 100% invalid sequences (broken emitter) — fix before Task 3
- trd: 100% invalid sequences (broken emitter) — fix before Task 3
- cdr3_only: 100% invalid sequences (broken emitter) — fix before Task 3
- j_only: 100% invalid sequences (broken emitter) — fix before Task 3

These do not reverse the GO: each affected group stays powered without the affected category (positive_real clears its floor on repertoire+therapeutic positives alone; structural keeps its 70 parent lineages via the other transforms). They MUST be fixed during Task 3 cohort assembly.
