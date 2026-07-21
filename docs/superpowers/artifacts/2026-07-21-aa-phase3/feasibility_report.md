# Phase 3 — Blinded Acquisition Spike: Feasibility Report

**Date:** 2026-07-21 · **Branch:** `experiment/aminoacid-alignair` · **No model inference was run.**
Purpose: decide, before writing the implementation plan, whether the headline endpoint (a source-declared,
alignment-independent **Gold** cohort of ≥15 independent lineages with **exact-gene** human IGHV labels) is
achievable. Spec: `docs/superpowers/specs/2026-07-21-aa-phase3-realprotein-design.md`.

## Provisional verdict: **NO-GO for the strict-Gold headline as specified.** Decision required.

The blocker is not sequences or licensing — it is that the **Gold label definition and the therapeutic-antibody
domain are structurally mismatched**. Details below.

---

## Q1 — Sequences: acquirable? redistributable?
- **Acquirable: yes, with a caveat.** Thera-SAbDab publishes a single downloadable sequence+metadata file
  (per the paper, 461 unique molecules as of 2019; ~"as of 20.12.2024" snapshots referenced in the 2025
  literature). The site was rebuilt ("SAbDab2", a JS SPA); the historical static CSV path
  (`.../static/downloads/TheraSAbDab_SeqStruc_OnlineDownload.csv`) now returns the SPA shell, so the current
  download endpoint must be captured from the live UI during implementation. Not a blocker.
- **Redistributable: not established.** No explicit permissive redistribution license was located for
  Thera-SAbDab downloads. Conservative default per spec → **manifest + acquisition script**, do not commit raw
  sequences. IMGT data are **CC BY 4.0** (record terms date + attribution).
- **Critical structural fact:** Thera-SAbDab carries **no germline V/D/J field** (columns are INN, format,
  isotype, light-chain category, highest clinical stage, developmental status, + sequences). So germline
  truth is *entirely* a literature/IMGT problem — Thera-SAbDab supplies sequences only.

## Q2/Q3 — How many lineages have source-declared **exact-gene** / **exact-allele** human IGHV labels?
**Estimated: far below the 15-lineage floor; plausibly near zero at the strict definition.** Two independent,
structural reasons:

1. **Humanization templates are usually consensus/subgroup, not a specific germline gene.** The dominant
   humanization method (CDR grafting onto a human *consensus* framework) declares a **VH subgroup consensus**
   (e.g. "VH subgroup III consensus") — resolution `subgroup_only`, which the spec bars from the Gold gene
   endpoint. Worked examples: **bevacizumab** (humanized from murine A.4.6.1 onto human consensus frameworks),
   the Genentech consensus-VH-III lineage generally.
2. **Enumerable "germline of origin" for therapeutics is alignment-derived.** Every resource that assigns a
   *specific* IGHV gene to therapeutics does so by **alignment to the closest human germline** (ANARCI;
   IMGT closest-match; the 481-therapeutic compilation in Prihoda et al. 2021). That is exactly the Tier-2
   (alignment-informed) category the design forbids from Gold. Fully-human antibodies (transgenic-mouse /
   phage) have **no** declared germline at all — only a closest match.

Net: the intersection {therapeutic antibody} ∩ {source-declared} ∩ {exact-gene} ∩ {alignment-independent} is
essentially empty in the enumerable literature. A handful of modern "germline-humanized" antibodies name a
specific acceptor germline in their patent, but (a) they are scarce, (b) the name is typically still obtained
by alignment, and (c) reaching **15 independent lineages** this way is not credible.

## Q4/Q5/Q6 — audit categories, alias mapping, floor
Not enumerated in detail because Q2/Q3 already fail the gate. The expected category breakdown, by construction:
- `subgroup_only` (consensus-framework humanizations) — the largest documentable group, **Gold-ineligible**.
- alignment-derived exact labels (ANARCI/IMGT) — **Tier-2 only**, not Gold.
- fully-human (no declared germline) — unresolved for Gold.
- exact-gene source-declared — the target — **too few for a 15-lineage independent cohort.**

## Statistical corollary (independent of the above)
Even had Gold been populated, with **zero** errors an exact one-sided 95% lower bound needs ≈ **59**
independent observations to reach 0.95. A ~15-lineage Gold set could only ever return `inconclusive` or
`failed` on the guarantee-relevant endpoint — never `supported`. So strict-Gold-only cannot validate transfer
regardless; it can only *falsify severe failure*.

## Sources
- Thera-SAbDab (Dunbar/Raybould et al., *Nucleic Acids Res.* 2020): https://academic.oup.com/nar/article/48/D1/D383/5573951
- IMGT terms of use (CC BY 4.0): https://www.imgt.org/about/termsofuse.php
- Consensus-framework humanization (VH subgroup III consensus; bevacizumab): patent/framework descriptions
  surfaced via search (e.g. US patents on humanized antibodies; bevacizumab humanization from A.4.6.1).
- Alignment-derived therapeutic germline mapping (ANARCI / IMGT closest-match; 481-therapeutic compilation):
  Prihoda et al. 2021, *Bioinformatics* https://academic.oup.com/bioinformatics/article/37/22/4041/6295884 ;
  "Data-driven analyses of human antibody variable domain germlines" (*mAbs* 2025).

## Options for the feasibility decision (for the user)
1. **Redefine Gold to `subgroup_only` truth** — score the model at **subgroup/family** resolution against
   source-declared consensus templates (a legitimate, source-declared, alignment-independent label that
   *does* exist at scale). Headline becomes "family-level transfer," honestly weaker than gene-level.
2. **Accept alignment-derived labels as the headline** (IMGT/mAb-DB or ANARCI), explicitly reframed as a
   **tool-concordance** study, not ground truth — the option previously declined.
3. **Change the data source** — a held-out, independently-annotated **repertoire** (e.g. an AIRR dataset with
   germline truth established by orthogonal means) rather than therapeutics. Gets N and defensible labels, at
   the cost of leaving the "curated therapeutic" framing.
4. **Stop Phase 3 at the frozen simulator milestone** — conclude that a *defensible* real-domain validation is
   not currently reachable, and record that as the honest end state.
