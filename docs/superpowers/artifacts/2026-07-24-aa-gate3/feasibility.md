# Gate-3 Phase-0 Feasibility — POST-CLEANUP

**Decision: GO** (after the mandatory pre-freeze cleanup gate).

Cleaned cohort has **zero invalid sequences**. Floors (zero-error 95% upper bound): <=5% -> 59, <=10% -> 29 locked-test examples.

| group | effective (dedup) | counts→floor | required | powered |
|---|---|---|---|---|
| igk_igl | 320 | 320 | 118 | YES |
| tcr | 230 | 230 | 118 | YES |
| vhh_nonhuman | 138 | 138 | 58 | YES |
| igsf | 357 | 357 | 58 | YES |
| structural | 70 | 70 | 58 | YES |
| positive_real | 361 | 361 | 116 | YES |
| positive_sim | 200 | 200 | 116 | YES |

## Cleanup applied (procedures + drops)

- **Trim** (`motif_fr4_junction_v1`): SAbDab Fab-heavy -> VH domain. pos_igh_structure 42/50, murine_vh 34/50, vhh 44/49 (no-anchor / out-of-range dropped, not guessed).
- **Rebuild** (`motif_cys_wgxg_v1`): cdr3_only + j_only rebuilt from 57/70 parent VHs, PARENT lineage key preserved (transforms don't inflate effective N).
- **gamma/delta (trg/trd): deferred**, descriptive + NON-BLOCKING — ADC V-domain extraction was empty; a networked IMGT TRGV/TRDV fetch is the follow-up. The combined-TCR gate is carried by TRA/TRB and remains powered.

Provenance: `cleanup.json` (counts + procedures), `trim_provenance.json` (per-sequence orig/trimmed hashes + boundaries). Sequences remain in the git-ignored store.
