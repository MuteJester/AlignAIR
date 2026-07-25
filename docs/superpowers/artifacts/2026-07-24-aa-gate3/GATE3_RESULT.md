# Gate 3 — Final Result: FAILED for the current product candidate

**Verdict: Gate 3 fails.** The AlignAIR-AA applicability gate **cannot be built on the frozen model's
predeclared representation**, so the AA product does **not** clear Gate 3 and must not be published with an
applicability guarantee. This is the honest Option-2 outcome — the contract was never weakened and the
locked test was never opened.

## What was established (independently reproduced, byte-identical)
- **Stage-1** high-specificity structural prefilter: 0% hard-reject on IGH positives; ~3.5% on non-IGH
  complete-domain positives (short TCR reads); decisive defects caught, ambiguous deferred.
- **Stage-2** validation selection: for BOTH endpoints and BOTH predeclared scorers,
  `STOP_NO_FEASIBLE_THRESHOLD`. The `vj_penultimate_v1` representation gives AUC 0.55–0.75 and, at any
  operating point with a ≤10% false-reject upper bound, a **66–100% false-accept upper bound** — one to two
  orders of magnitude above the 5–10% gates, on well-powered validation groups.
- **Mechanism:** the V+J penultimate is the class-head bottleneck ("which IGH V/J gene, assuming IGH"); it
  does not encode heavy-vs-light/TCR membership or complete-vs-fragment structure strongly enough to gate.

## Decisions (owner, 2026-07-24)
1. **Record Gate 3 as FAILED for the current product candidate.**
2. **Do not publicly ship the bare predictor** — an unguarded predictor recreates the exact community-trust
   risk Gate 3 was designed to prevent (deceptively confident IGH annotations on out-of-domain input).
3. **Keep the locked test sealed** (never opened; the seal + unlock audit remain intact).
4. **Three reporting/code corrections applied** (none change the STOP): prefilter validation now audits the
   non-IGH complete-domain positives too; diagnostic AUC counts ties as 0.5; the unused `-inf`-hitting
   quantile was removed.

## Go-forward: a separately-predeclared applicability-MODEL project
This Gate-3 run is frozen and failed; a working gate is NEW capability, predeclared afresh:
- **Reuse the now-burned Gate-3 development cohort for TRAINING** (it can no longer serve as an unbiased
  selection set).
- **Acquire a fresh lineage/source-independent validation cohort** for architecture + threshold selection.
- **Train explicit completeness and locus/applicability heads from the spatial backbone features** (not the
  class-head bottleneck), with supervised negatives.
- **Evaluate with category/species LOCO.**
- **Freeze the architecture and thresholds before opening the existing sealed locked test exactly once.**

## Framing
This is not a failure of the research program — it is the gate working. Gate 3 prevented an unguarded model
from becoming a deceptively confident AIRR product, which is precisely what a trustworthy release gate is
for. The Gate-3 machinery (record-bound frozen cohort, sealed loader, prefilter, endpoint roles, embedding
extractor, scorer candidates, confidence-bounded validation harness) is sound and reusable for the new
applicability-model project.
