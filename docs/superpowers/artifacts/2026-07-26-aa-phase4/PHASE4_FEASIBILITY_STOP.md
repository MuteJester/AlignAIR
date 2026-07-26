# Phase 4 — Feasibility Gate Result: STOP (real-domain formal coverage guarantee is unpowerable)

**Verdict: STOP at the Phase-0 feasibility gate.** The powered exchangeable real-domain conformal-coverage
test cannot be assembled from the available concordance-truth-able human IGH data. The contract was **not**
weakened (no goalpost movement), the Gate-3 locked test was never opened, and the tracked worktree is clean.
Detailed recon artifacts are private/git-ignored (`.private/aa-phase4/recon/phase4_recon.{md,json}`); this
file records the committable counts + decision.

## The binding arithmetic
- **Exact power** (`H0: coverage ≤ 0.93` vs expected 0.95, one-sided α=0.05, power 0.80): **882 donor-lineage
  units**, binding on the exchangeable locked test (one deterministically sampled lineage per donor; the
  same units validate both V and J).
- **Maximum harmonized donor pool: ~1,232** (OAS 811 donors / 62 studies / 13,265 human-heavy units + AIRR
  ADC 584 donors / 17 of 18 nodes, minus 521 exact run-accession overlaps; upper bound — no cross-repo
  sequence-hash audit).
- **Donor-first concordance sample:** 86/160 attempted donors yielded a productive, complete, translatable
  IGH VDJ record (across 40 studies), run through **both** IgBLASTn and nt AlignAIR.

| Concordance endpoint | rate |
|---|---:|
| V gene | 0.919 |
| V allele | 0.860 |
| J gene | 0.953 |
| J allele | 0.919 |
| **Joint V+J gene** | **0.884** |
| Joint V+J allele | 0.802 |

- **Projected usable donors ≈ 585** (eligibility × joint-gene concordance) — a **297-donor shortfall vs the
  882 locked-test requirement alone.**
- **Even the unrealistically generous scenario** (all 584 ADC donors eligible + concordant) projects **~891**
  — only **9** beyond the locked test, leaving **nothing** for donor-disjoint calibration, trigger, and
  source-held-out partitions.

## Why this cannot be rescued without weakening the contract
- The **donor pool (~1,232) is the ceiling** — even 100% concordance yield can't reach `882 (test) + calibration
  + trigger + source-held-out` under donor-disjoint splitting. Improving concordance yield alone does not fix it.
- Relaxing the coverage margin / power target, or dropping the two-method independence, would be **post-hoc
  goalpost movement** — explicitly rejected.

## Provenance
- IgBLASTn 1.22.0, `immcantation/suite:4.5.0` (digest-pinned), bundled IMGT human IG reference.
- nt AlignAIR `alignair-igh-human@1.0.0` (SHA-256 in the private JSON); ASC calls expanded via GenAIRR
  `HUMAN_IGH_OGRDB.asc_tables`. Real "agreement" is method concordance, **not biological truth**.

## Consequence
The AA product's calibrated sets remain **simulator-domain only** (as Gates 1–2 already state) — a **formal
real-domain coverage guarantee is not attainable** with current data. Two honest, contract-preserving paths
remain (owner decision): **(A) accept the STOP** — no real-domain guarantee is asserted; or **(B) a clearly-
labelled DESCRIPTIVE real-domain audit** on the ~585 usable donors (real-domain concordance/coverage/
set-size/reliability with confidence intervals but **no** formal non-inferiority claim). What is NOT
available is a powered formal guarantee.
