# Amino-Acid AlignAIR — Phase 3: Real-Protein Validation Result

**Date:** 2026-07-21 · **Branch:** `experiment/aminoacid-alignair` · **Decision: `inconclusive`** (never
`supported` — Phase 3 cannot emit that verdict by design; see §1 of the spec). No retraining, no
recalibration. The frozen Stage-1 simulator snapshot
(`docs/superpowers/artifacts/2026-07-20-aa-strategic/SNAPSHOT.md`) is unmodified.

This document reports the result of running that frozen model, unchanged, on 16 real curated therapeutic
antibody heavy chains — a genuine out-of-domain protein cohort (humanized/engineered, not GenAIRR-simulated
SHM). Full design rationale: `docs/superpowers/specs/2026-07-21-aa-phase3-realprotein-design.md`.

---

## Headline: a family design-ancestry accuracy anchor, not a germline ground-truth result

**Top-class IMGT V-family accuracy = 16/16 = 1.000, exact one-sided 95% CI [0.829, 1.0]**, on 16 real
therapeutic antibodies drawn from **16 independent lineages** (one representative VH per lineage, the
binding statistical unit; source: `phase3_eval.json` → `gold.binding`).

This is a **family design-ancestry** label, not literal germline ground truth. The label is the
**source-declared humanization design template** (e.g. a patent stating "human VH subgroup III consensus
framework"), read from the originating patent/paper and mapped to an IMGT V-family through a frozen, cited
Kabat-subgroup → IMGT-family crosswalk (`alias_crosswalk.tsv`) — never by aligning the therapeutic sequence
itself. It answers "did the model recover the family the antibody was designed from," not "did the model
recover the real germline gene that produced this sequence" — for engineered/humanized antibodies the latter
question frequently has no alignment-independent answer at all (see §"What Phase 3 cannot conclude" below).

The strongest defensible statement this result supports:

> The frozen simulator-trained AA model correctly assigned the IMGT V family to all 16 real therapeutic
> antibodies under genuine domain shift, but with n=16 this cannot certify the 95% guarantee; it falsifies
> severe transfer failure and provides a real-domain family design-ancestry accuracy anchor.

## Decision: `inconclusive` — not `supported`, and why

Machine-readable decision (`phase3_eval.json` → `decision`): **`inconclusive`**.

Per the predeclared §1 decision rule (α = 0.05, exact one-sided binomial/Clopper–Pearson, one representative
VH per lineage): `failed` if the Gold family-set-coverage upper bound < 0.95; `inconclusive` otherwise;
`supported` is **structurally unreachable** in Phase 3 and is never emitted. Family-set coverage here is
1.000 with a 95% CI of **[0.829, 1.0]** (`phase3_eval.json` → `gold.binding.family_set_coverage_ci`) — the
upper bound clears 0.95 (no `failed`), but the lower bound (0.829) does not reach 0.95 (no `support`), so the
outcome is `inconclusive`, exactly as predeclared.

**Why this is the *expected* outcome, not a shortfall:** with **zero** observed errors, an exact one-sided
95% lower bound cannot reach 0.95 until **n ≥ 59** independent observations (`0.05^(1/n) ≥ 0.95 ⇒ n ≥ 59`).
The Gold cohort has **n = 16** independent lineages — far below that floor. No perfect-accuracy cohort at this
scale could ever cross into `supported`; `inconclusive` was the only reachable non-`failed` outcome by
construction, which is why the informative content of Phase 3 is the **point-accuracy anchor**, not a
coverage guarantee.

## What Phase 3 can and cannot conclude

Phase 3 **cannot validate the original simulator-domain 95% allele/class coverage guarantee** established at
the Stage-1 snapshot. The blinded acquisition spike (`feasibility_report.md`) established that
**alignment-independent, per-sequence allele/class truth is not available at scale in this therapeutic
cohort** — assigning a *specific germline gene or allele* to an observed, engineered antibody sequence is
itself an alignment (via ANARCI or IMGT closest-match), so no allele/class-level ground-truth label exists
independent of the model's own task. The only alignment-independent label available at scale is the coarser
**design-ancestry family** used above. Consequently Phase 3 can:

- **falsify** transfer at family granularity (it did not — family coverage held at 1.000, 95% CI lower bound
  0.829),
- report the real-domain **family design-ancestry accuracy anchor** above,
- report **concordance** with an established aligner (not applicable here — see next section),

but it **cannot certify** that the simulator-domain 95%-coverage allele/class guarantee transfers to real
protein. That guarantee remains simulator-domain only.

## Concordance tier (secondary — explicitly not ground truth)

`phase3_eval.json` → `concordance`: **n_rows = 0** — no concordance-tier (IMGT/ANARCI exact-gene) rows were
curated in this cohort. Had this tier been populated, it would be labelled **"alignment-derived, not ground
truth"** (tool agreement under domain shift, per §2 of the spec) and would never feed the headline decision.
Its absence here does not affect the Gold family-tier result or the `inconclusive` decision — it is simply an
empty, honestly-reported secondary block.

## ASC → IMGT bridge and family coverage of the cohort

The frozen model predicts in **OGRDB ASC (allele/sequence-class) space**, not directly in IMGT gene/family
names. Family labels for scoring were obtained by mapping the model's ASC-space top classes to IMGT families
through **GenAIRR's own `asc_tables` IMGT-allele provenance** — a **germline-name bridge**, not a read-level
sequence alignment (`phase3_eval.json` → `asc_bridge`, source: `GenAIRR HUMAN_IGH_OGRDB (OGRDB V8) asc_tables`).

The Gold cohort's family distribution (`phase3_eval.json` → `family_distribution`, `sources.json` →
`gold_cohort.family_distribution`):

| Family | n |
|---|---|
| IGHV3 | 11 |
| IGHV1 | 4 |
| IGHV4 | 1 |

**Stated limitation:** the cohort is **IGHV3-dominated** (11/16), and only **3 of the 8** catalogue V
families (`IGHV1`–`IGHV8`) are represented at all. This reflects which humanization consensus frameworks are
documented in the therapeutic-antibody literature, not a deliberate sampling choice, and it means the 1.000
accuracy anchor is **not evidence about IGHV2/5/6/7/8 transfer** — those families are simply untested here.

**Other Gold-tier numbers** (`phase3_eval.json` → `gold`):
- OOV rate: 0.0 (no truth family fell outside the frozen model catalogue).
- End-to-end accuracy = in-catalog accuracy = 1.000 (OOV rate is 0, so the two coincide).
- Descriptive (all 16 deduplicated VHs, lineage-cluster bootstrap, non-binding): family accuracy 1.000,
  bootstrap 95% CI [1.0, 1.0].
- Predicted-set sizes (mean/median/p95): families 1.0/1.0/1.0; genes 2.875/2.0/6.5; classes 3.4375/3.0/7.75;
  nucleotide alleles 5.5625/4.5/12.0 — i.e. the model's *family*-level top prediction was a clean singleton in
  every case, while its underlying class/allele sets remain appropriately larger (family is the coarsest,
  easiest rung of the coarsening ladder — see spec §3).

## Truncation stress test (separate, synthetic — not part of the primary real-domain result)

A **separately labelled synthetic stress test on the same 16 real VH sequences**: deterministic crops via
`aa_truncate`, **frozen seed (seed=0)**, using the exact original Mondrian coverage-bin definitions and
per-bin frozen quantiles from `calibration.json` (`truncation_stress.json`):

| Bin | n | keep range | frozen quantile | family coverage |
|---|---|---|---|---|
| `full` | 16 | (uncropped) | 0.9353 | 1.000 |
| `trunc` | 16 | [0.6, 0.9] | 0.9331 | 1.000 |
| `short` | 16 | [0.3, 0.6] | 0.9384 | 1.000 |

Family coverage is 1.000 in all three bins. This is reported **honestly, not as a strong result**: at
family resolution and n=16 per bin, there is very little statistical power to detect degradation under
truncation even if it were present — a bin would need to accumulate several misses before coverage moved
appreciably below 1.000. This stress test can rule out gross, catastrophic breakage under partial coverage on
real protein; it cannot certify robustness at the allele/class level the way the simulator-domain calibration
did (thousands of held-out examples per bin — see `calibration.json`, n≈2,500–5,000 per V bin).

## Licensing / attribution

- **Thera-SAbDab:** redistribution terms are **not established** — no explicit permissive redistribution
  license was located for Thera-SAbDab downloads. Accordingly, **raw VH sequences were never committed to
  Git.** The cohort is reproducible via a **manifest + hydration** path (`manifest.tsv`,
  `hydrated_sequences.tsv` is git-ignored; only `sha256(vh_aa)` per row and the overall `cohort_sha256` are
  committed). Source hydration file: Thera-SAbDab PL132 (Feb 2025), local path git-ignored,
  `pl132_sha256 = cc9402788297ed7b6ab3d676daa21953a6d28ebbd9cc78ec7248b19f74d49d6d`. Citation: Dunbar et al.,
  *Nucleic Acids Res.* 2020 (https://academic.oup.com/nar/article/48/D1/D383/5573951).
- **IMGT:** **CC BY 4.0**, terms recorded 2026-07-21, attribution required. Citation:
  https://www.imgt.org/about/termsofuse.php. (Used for the Kabat-subgroup → IMGT-family crosswalk citations,
  not for any sequence redistribution.)

## Provenance

| Field | Value | Source |
|---|---|---|
| Model | `experiments/aa/strategic/model_stage1_PASS.pt` (Stage-1, gate-passed) | `phase3_eval.json` → `provenance.model` |
| Model SHA-256 | `54eef812942e7b96f3d49c9ed881e878c06f8978576a9f16345e797c8937dfc6` | `phase3_eval.json` → `provenance.model_sha256` |
| Class-map fingerprint (V db_hash) | `e1a455b87f614f0d` | `phase3_eval.json` → `provenance.class_map_v_db_hash` |
| Calibration source | `docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json` | `phase3_eval.json` → `provenance.calibration` |
| Bundle SHA-256 (curation artifacts) | `ad9e8edaaede7a8fdceada5c2ff7221c45989d1a233e3df08540603205f52ebb` | `phase3_eval.json` → `bundle_sha256`, `bundle.manifest.json` |
| Cohort SHA-256 (frozen before inference) | `ba1790fe11decc27834ec017521d2e40a6d3b77c884bf38b5c4eb9abf2b396a7` | `phase3_eval.json` → `cohort_sha256`, `bundle.manifest.json` |
| Gold lineages | 16 (= n rows; one lineage per row, all `lineage_representative=true`) | `phase3_eval.json` → `n_gold`, `sources.json` → `gold_cohort` |
| Candidates reviewed | 42 total → 16 admitted to Gold, 26 disposed to audit (of which 21 lack a citable independent design-ancestry label, and 5 are flagged as derivative/same-lineage duplicates — e.g. trastuzumab emtansine/deruxtecan, ranibizumab, motavizumab, ravulizumab) | `candidate_inventory.tsv` |

α = 0.05 throughout; exact one-sided binomial (Clopper–Pearson); binding statistical unit = one frozen
representative VH per `lineage_id`; descriptive statistics use a lineage-cluster bootstrap (never binding).

## Explicitly out of scope for Phase 3

D-segment validation (undefined junctions on real protein), retraining, and real-domain recalibration
(conformal recalibration needs hundreds of real-domain rows; 16 Gold rows cannot recalibrate — deferred to a
future phase with sufficient real N). All work stays isolated on `experiment/aminoacid-alignair`; the
Stage-1 simulator snapshot is unmodified.
