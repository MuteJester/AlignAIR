# AA Applicability Model — Phase 0 Feasibility: GO

**Verdict: GO.** All three Phase‑0 hard gates pass. Unlike Gate 3 (FAILED) and Phase 4 (STOP), the
applicability‑model project clears its feasibility‑first gate: the tooling exists and is permissive, the
sealed locked test is already powered for every §10 gate, and the fresh validation **supply census** —
including the one category that looked like a STOP risk (VHH) — is well‑powered. Actual post-hydration,
post-disjointness allocation remains the binding Stage‑1 gate. One downstream item (profile‑bank germline
**redistribution** for a public bundle) remains open and must be resolved before a production profile bank
is used to train or select candidates. Contract intact; the Gate‑3 locked test was **not** opened.
Companion artifacts: `tooling_feasibility.json`, `locked_category_map.json`.

## 1. Tooling + licensing — GREEN
| Tool | Version | License | Deterministic smoke |
|---|---|---|---|
| `pyhmmer` (profile‑HMM engine) | 0.12.1 | MIT AND BSD‑3 AND BSD‑2 | build_msa→search bit‑identical over 2 runs (VH hit 81.991, light‑chain filtered); `hits_sha16=49dfb373c9e4d20d` |
| `anarcii` (numbering comparator) | 2.0.8 | BSD‑3‑Clause (Oxford, 2025) | numbers trastuzumab VH → chain `H` |

`pyhmmer` statically vendors HMMER (no system HMMER on PATH needed). `anarcii` is LM‑based (pulls torch,
downloads weights on first use); it is a **comparator/teacher**, not auto‑admitted to production (spec §3.1).
Both licenses are permissive → no *tooling* redistribution blocker. Wheel‑hash pinning is a plan task.

## 2. Locked‑seal power + category‑map re‑audit — PASS
From committed public metadata (`split_fingerprint.json`), counts‑only, **seal not opened**. All eight §10
release gates map to a present, floored frozen category → `STOP_LOCKED_CATEGORY_MAP_MISMATCH` **not**
triggered. Detail in `locked_category_map.json`. Recorded scope boundary: the seal contains **no** generic
non‑antibody or de‑novo‑unsupported category, so far‑OOD/open‑set energy is descriptive‑only (not certified),
consistent with spec §10/§15.

## 3. Fresh validation‑cohort census — GO
Requirement per subset (spec §3.3 guardband): **74** for 5% gates, **36** for 10% gates, across **two**
disjoint fresh subsets (selection + blind‑LOCO challenge), each carrying **≥19 VHH**.

### 3.1 VHH — the pivot — GO (rich headroom)
Live pull of SAbDab `all-single-domain-summary` (2026‑07‑26; 4,637 chain rows; CC‑BY 4.0):

| Quantity | Value |
|---|---:|
| single‑domain‑heavy (SD‑H) rows | 4,481 |
| camelid SD‑H rows | 3,552 |
| distinct PDB (camelid SD‑H) | 1,814 |
| **distinct HEAVY_ID (identical‑heavy cluster = lineage‑ceiling proxy)** | **1,087** |
| already used by Gate‑3 (capped at 50/cat; emitted 49) | 49 |
| **seal‑disjoint VHH lineages available** | **~1,038** |
| fresh need (2 subsets × 19) | 38 |
| headroom | **~27×** |

Genus mix within the source: *lama glama* 2,031 · *vicugna pacos* 1,008 · *camelus dromedarius* 217 ·
*c. bactrianus* 84 · *camelidae* 155. Gate‑3 selection was **capped at 50/category**, so it never exhausted
the pool — the earlier "44 used" was a cap, not a ceiling.

**VHH caveat (honest, recorded):** SAbDab is the *only* clean‑license VHH source (AVIDa‑hIL6 dropped:
CC BY‑NC + single donor). Fresh VHH can be lineage/structure‑disjoint and **temporally** disjoint
(post‑deposition‑date holdout) and **genus**‑diverse, but not sourced from a genuinely independent second
database. This weakens only the *strength of the VHH unseen‑category generalization claim*, not the count;
it does not change the GO. To be surfaced in the model card's limitations.

### 3.2 Other categories — abundant (not constraining)
Established sources (Gate‑3 recon + manifest), all far above the 74/36 per‑subset guardband floors:
- **IGK/IGL** — OAS human light (millions of reads; light‑chain raw units already cached at 24 MB).
- **TCR (TRA/TRB)** — VDJServer/AIRR‑ADC (TRA 244 repertoires, TRB 2,848); TRG/TRD descriptive (scarce).
- **IgSF V‑set** — UniProt IPR013106 reviewed non‑IG/TR (~735 cross‑species).
- **Positives (real)** — OAS human heavy (millions) + SAbDab human VH structures.
- **Positives (sim)** — GenAIRR HUMAN_IGH_OGRDB (unlimited; held out from model training).
- **Structural** — deterministic transforms of positives (unlimited; parent‑lineage counted).

All must be hydrated **disjoint** from the burned‑dev cohort, the 16 Phase‑3 therapeutics, and the locked
test (by sequence hash / lineage / donor / study‑source) — a plan task, not a supply risk.

## 4. Open (not a census blocker; binding before candidate training)
- **Profile‑bank germline redistribution (§3.1 hard gate):** the reference used to *build* the profiles
  (e.g. IMGT terms are restrictive). IMGT-derived banks may be built only as local research comparators.
  The bank used by C0–C4 must have a resolved production path before candidate training: redistributable
  reference inputs / explicit permission / an exact, reproducible user-side hydration product path / STOP
  the production-profile candidate. A validated bank cannot be swapped after selection or after the
  one-shot test.

## Decision
**GO to the implementation plan.** Feasibility‑first supply/tooling gate cleared; VHH — the sole apparent
supply risk — has ~27× headroom. Recorded caveats: VHH single‑source; far‑OOD non‑certifiable by the seal;
actual fresh allocation remains Stage‑1 work; profile-germline redistribution must be resolved before the
production bank enters candidate training.
