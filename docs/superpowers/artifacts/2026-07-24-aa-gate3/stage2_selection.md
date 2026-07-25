# Gate 3 Stage-2 — Validation Selection: STOP (Gate 3 does not pass)

**Decision: `STOP_NO_FEASIBLE_THRESHOLD`.** Selection ran on VALIDATION only; the locked test was never
loaded. No threshold, for either endpoint, on either predeclared scorer, meets the confidence-bounded
constraints — so per the predeclared rule the process stops before the locked test, and **Gate 3 fails on
the honest fallback (Option 2)**. No contract was weakened (Option 3 was rejected up front).

## What was tested
- **Representation:** the predeclared `vj_penultimate_v1` (input to the V and J gene branches' class
  heads, concatenated), extracted from the FROZEN model without changing predictions.
- **Scorers:** both predeclared positive-only candidates — shrinkage (Ledoit-Wolf) Mahalanobis and kNN.
- **Endpoints (endpoint-specific fit roles):** completeness (fit on complete V-domains of any locus
  except IgSF; negatives = parent-aggregated structural transforms) and membership (fit on human IGH
  only; negatives = IGK/IGL, TCR, VHH/non-human, IgSF). Accept = Stage-1-defer AND Stage-2-score ≥ τ.

## Result — the representation cannot separate either question

| Endpoint / group | n (val) | best AUC | FAR upper-bound at FRR≤10% | required |
|---|---:|---:|---:|---:|
| completeness / structural (parent-agg) | 35 | 0.70 | **0.66** | ≤0.10 |
| membership / IGK-IGL | 159 | 0.68 | **1.00** | ≤0.05 |
| membership / TCR | 90 | 0.75 | **0.95** | ≤0.05 |
| membership / IgSF | 178 | 0.72 | **0.99** | ≤0.10 |
| membership / VHH+non-human | 64 | 0.61 | **0.99** | ≤0.10 |

At any operating point with a ≤10% false-reject upper bound, the false-accept upper bound is 66–100% —
one to two orders of magnitude above the 5–10% gates. The groups are well-powered (59+/29+), so this is
genuine inseparability, not a small-sample artifact.

## Why (mechanistic)
The V+J penultimate features are the **class-head inputs** — trained to answer "which IGH V/J gene",
*assuming the input is IGH*. They therefore map any V-domain (light chain, TCR, VHH, IgSF) into the same
"some IGH V/J" space and do not encode heavy-vs-light/TCR **membership**, nor complete-vs-fragment
**structure**, strongly enough to gate on. The frozen model was never trained for out-of-distribution or
locus discrimination, and its representation does not incidentally provide it.

## Conclusion (narrowed)
> **The predeclared `vj_penultimate_v1` representation, with positive-only kNN or shrinkage-Mahalanobis
> scoring, cannot support the applicability gate.**

Under the Gate-3 contract this is a genuine FAIL for the current product candidate: the AA product must
**not** be published with an applicability guarantee built on this representation.

**Scope of the claim.** This does **not** prove that *every* internal representation of the frozen model
is useless. The V+J penultimate is the class-head bottleneck, which may have discarded locus/completeness
information that earlier spatial-convolutional backbone features still retain. Testing those would be a
**new, post-result experiment** — a separately predeclared applicability-model project, not a rescue of
this Gate-3 run (which is frozen and failed). The locked test stays sealed.

**Independently reproduced** (byte-identical): score direction, threshold search, one-sided
Clopper–Pearson bounds, lineage-disjoint fit/validation roles, parent-lineage-aggregated structural
negatives, and both scorers' FAR bounds all verified.
