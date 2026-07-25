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

## Conclusion
**The predeclared hybrid cannot pass the Trust-Gate contract on the frozen model.** Under the Gate-3
contract this is a genuine FAIL: the AA product must **not** be published with an applicability guarantee
built on this representation. A working applicability gate would require NEW capability outside this
predeclaration (e.g. a trained OOD/locus head, or a different representation) — to be proposed and
predeclared afresh, not fished for post-hoc against this validation set. The scientific question posed —
*can the frozen model representation distinguish complete-domain structure and human-IGH membership
without sacrificing valid divergent VHs?* — is answered **no**.
