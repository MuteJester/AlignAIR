# AlignAIR-AA — Gate 3: Applicability & Negative-Controls Design Spec

> **AIRR Utility & Trust Gate, Gate 3 of 6.** Gate 3 introduces a **new product-decision layer**: an
> input-**applicability gate** that decides whether an input is an in-domain human IGH VH protein *before*
> any V/D/J call is trusted, plus the adversarial cohort and the **confidence-bounded** validation protocol
> that proves it works. This is the gate that closes the core risk — *precise-looking IGH annotations on
> inputs the model was never validated to interpret*.

**Status:** ✅ **APPROVED (2026-07-24)** — amended per review · implementation plan follows · Builds on
[Gate 1 contract](2026-07-24-aa-trust-gate-gate1-product-contract.md) and
[Gate 2 audit](2026-07-24-aa-trust-gate-gate2-semantics-audit.md).

## 0. Settled product decisions (owner, 2026-07-24)

Fixed inputs to this spec; the rest of the design serves them.

1. **Whole-input applicability gate first.** A single verdict for the entire input (*is this an in-domain
   human VH at all?*) is the primary safety gate. Per-segment evidence may layer on in a later version.
2. **Refuse calls + reason state.** When the input is not `supported`, the product returns an applicability
   verdict with a reason and **withholds** the V/D/J calls.
3. **Minimize false-accepts.** Wrongly claiming support on out-of-domain input is the worst error; the
   detector is tuned to minimize false-accepts, accepting more false-rejects.
4. **Require a pre-trimmed VH.** Signal-peptide+VH and full heavy chains (VH+constant) are **unsupported**;
   VH extraction is external preprocessing, out of scope for v1.

## 1. What Gate 3 decides, and does not

**Decides:** whether an arbitrary in-alphabet ≤192-aa protein is a **complete human IGH VH domain in N→C
order** the model may annotate — and if not, why not.

**Does not:** extract a VH from a larger chain (decision 4); validate fragment/truncation support (a
separate experimental track, §10); resolve per-segment trust (decision 1); change the V/D/J model itself.

## 2. Output contract — discriminated result union

Gate 3 is where the schema **is** allowed to grow (Gate 1/2 deferred any applicability field to the gate
that validates it). The result becomes a **discriminated union keyed on `applicability.state`**, and this
union is **schema-validated so withheld calls cannot leak** (amendment 5):

```
# state == "supported"  → calls PRESENT
{
  sequence_id, input_length,
  applicability: { state: "supported", score, provenance },
  segments: { V, D, J },        # REQUIRED when supported
  inter_vj_aa, provenance, warnings
}

# state != "supported"  → calls ABSENT (schema forbids `segments`/`inter_vj_aa`)
{
  sequence_id, input_length,
  applicability: {
    state: "insufficient_evidence" | "unsupported_locus" | "unknown_ood",
    reason: str,                # human-readable, specific
    score: float,               # detector score (audit only; NOT a calibrated probability)
    detail: { completeness: {...}, membership: {...} },   # per-stage evidence (§4)
    provenance: { ...§11 }
  },
  provenance, warnings
  # NO segments, NO inter_vj_aa
}
```

- **State semantics** (assigned by the two-stage detector, §4):
  - `unsupported_locus` — a **positively identified** different locus/molecule (IGK/IGL, TCR, non-antibody).
    Only emitted when a locus-specific signal fires. **If the reason cannot be reliably identified, return
    `unknown_ood`; never invent a locus explanation** (amendment 5).
  - `insufficient_evidence` — plausibly antibody-related but not a complete VH domain (too short, fragment,
    no V/J anchors, leader/constant contamination, reversed order).
  - `unknown_ood` — far from the in-domain manifold with no confident locus call (the honest catch-all).
- The verdict is emitted for any **format-valid** sequence regardless of `on_invalid`; format-invalid
  inputs still follow the Gate-2 §A path (raise, or `ErrorResult` under `skip`).
- **Ship condition:** the `applicability` field ships **only if** the detector passes the §7
  confidence-bounded thresholds. There is **no unvalidated fallback** (§9).

## 3. Why model softmax / conformal set-size are PROHIBITED as the detector

Both are **forbidden** as the applicability signal:

- **Softmax score** is uncalibrated and routinely **confidently wrong** on OOD — a TCR or light-chain V
  domain (same Ig fold) can produce a high-scoring, small-set V "call." It measures nothing about in-domain
  membership.
- **Conformal set size** is calibrated **only under the simulator distribution**; on OOD the
  exchangeability assumption fails, so a *small* set is not evidence of support. A widened set is
  simulator-domain ambiguity, not OOD detection (Gate 2 §E).

The detector must measure **distance from the training domain**, not the model's confidence in a
(possibly nonsensical) answer.

## 4. Detector architecture — two explicit stages (amendment 7)

Completeness detection and domain-membership detection are **separated**, so reasons are clear and a single
opaque score never conflates "human light chain" with "partial heavy chain."

### Stage 1 — Completeness / structure gate (rule-based, interpretable)
Checks the **domain contract** independent of germline identity:
- a **complete** variable domain is present (domain-fraction / length within bounds);
- **N→C** order (the model has no orientation head — reversed input must be caught here);
- required **V and J framework anchors / evidence** are present;
- **no leader (signal-peptide) or constant-domain contamination**.

Outcome: pass → Stage 2; else `insufficient_evidence` with a **specific** reason (which check failed).
Reversed VH, CDR3-only, V-only/J-only, leader+VH, and VH+constant are rejected **here**.

### Stage 2 — Domain-membership score (human IGH VH vs near-OOD)
Only runs on inputs that pass Stage 1. Scores membership in the **human IGH VH** manifold:
- **Primary (recommended first candidate): positive-only embedding distance** — distance of the input's
  model **encoder embedding** to the in-domain human-VH manifold (class-conditional Mahalanobis, kNN, or a
  fitted density). Positive-only means it is fit on **in-domain human VH alone** and needs no negative to be
  seen — the safest generalization posture.
- **Optional: supervised negative classifier** — admitted **only if** it earns inclusion by passing the
  leave-one-category-out generalization tests (§6). A classifier that only rejects negatives it trained on
  is a memorizer, not a detector.

Outcome: in-manifold → `supported`; a fired locus-specific signal → `unsupported_locus`; otherwise
`unknown_ood`.

**Final verdict = `supported` iff Stage 1 AND Stage 2 pass.** Whatever the composition, every score is
**thresholded on a held-out split** (§7).

## 5. Cohort composition — independent labels (amendments 4 & 6)

The cohort must be **dominated by hard, VH-like** inputs; *easy random proteins alone would produce a
misleadingly strong detector* and are a small control slice only.

**Independent labelling requirements (amendment 6).** "Complete human VH" is defined **authoritatively and
independently of the detector** — motif rules must not create both the ground truth and the prediction.
Every member requires:
- **source-declared** locus / species / domain;
- **N→C** sequence;
- **complete variable-domain boundaries established independently** (annotation source, not the Stage-1
  rules);
- **lineage deduplication**;
- **broad IGHV-family coverage** for positives (not IGHV3-dominated);
- **real positives reported separately from simulator positives**.

**Positives — expected `supported`:**
- Human IGH **complete VH** from GenAIRR simulation, **held out from model training** (simulator positives).
- **Real** human VH, **disjoint from the 16-antibody Phase-3 cohort**, multi-family (real positives).
- **Complete human-framework VH designs** (engineered but with a defensible complete human IGH framework) —
  a **positive challenge cohort** (amendment 4).
- **Humanized VH with a human framework** — positive, **reported separately** as a challenge slice.

**Near-OOD hard negatives — expected reject (same fold / very VH-like):**
- Human **IGK** and **IGL** light-chain V domains (hardest — same Ig fold).
- Human **TCR** V domains — TRA / TRB / TRG / TRD.
- **VHH / camelid** single-domain antibodies.
- **Non-human VH** (murine, rabbit), and **chimeric antibodies whose variable domain is murine** →
  non-human negative (amendment 4).
- Unrelated **immunoglobulin-superfamily (IgSF)** V-like domains.

**`unknown_ood`-expected challenge:**
- **De novo antibody-like proteins without defensible human-IGH membership** → expected `unknown_ood`
  (amendment 4 — *not* a positive just because they are "antibody-like," *not* a specific locus).

**Structural / coverage negatives — expected reject (Stage 1):**
- **Reversed** VH (C→N) · **CDR3-only** peptide · **V-only** and **J-only** fragments ·
  **leader (signal-peptide) + VH** · **VH + constant** domain(s).

**Easy negatives (small control slice):** unrelated non-antibody proteins.

> **Applicability follows the sequence/domain contract, not whether a sequence was engineered.** A complete
> human-framework design is a positive; a murine-variable chimera is a non-human negative; a de novo
> antibody-like protein is `unknown_ood`.

## 6. Leakage, lineage & unseen-category generalization (amendment 3)

- **Detector fit ≠ model training data** where feasible; **never** reuse the 16-antibody Phase-3 cohort for
  detector fitting or threshold selection.
- **Leave-one-category-out (LOCO) generalization** for any **learned** component, across **all** critical
  groups — **IGK/IGL, TCR, VHH/non-human VH, IgSF**. Fit with one critical category **entirely excluded**,
  test on it; **repeat for every group and report every fold.** **Failure on any one critical category
  prevents the learned detector from shipping.** ("Reject at least one held-out category" is too weak — a
  detector could reject random enzymes yet accept every light chain.)
- **Motif rules:** a category is **not** "unseen" if knowledge of that category influenced rule design.
  Rule-based reasons must be justified from the domain contract, not reverse-engineered from the negatives.
- **Family / lineage / species holdout** for positives and the murine/VHH negatives, so `supported` is not
  memorization of specific sequences.
- Thresholds chosen on a **validation split**; reported **once** on a locked **test split** (no peeking).

## 7. Metrics & confidence-bounded release thresholds (amendment 2)

**Gate on one-sided exact 95% upper confidence bounds, NOT point estimates.** A measured FAR of 0/20 is
**not** evidence the true FAR is < 5%.

- **False-Accept Rate (FAR)** — fraction of OOD/negative inputs wrongly judged `supported`
  (`unknown_ood`-expected de-novo counts as a false-accept if judged `supported`). Reported **overall and
  per category**.
- **False-Reject Rate (FRR)** — fraction of genuine complete human VH positives wrongly withheld, reported
  **separately for real and simulator positives** (and for the design/humanized challenge slices).

**Release gates (all on the one-sided exact 95% upper bound):**
| Metric | 95% upper-bound gate |
|---|---|
| Overall near-OOD FAR | ≤ **5%** |
| IGK/IGL FAR | ≤ **5%** |
| TCR FAR | ≤ **5%** |
| VHH / non-human VH FAR | ≤ **10%** |
| IgSF FAR | ≤ **10%** |
| FRR (real positives) | ≤ **10%** |
| FRR (simulator positives) | ≤ **10%** |

**Minimum sample sizes (predeclared).** With **zero observed errors**, an exact one-sided 95% upper bound
below **5%** needs ≈ **59** independent examples; below **10%** needs ≈ **29**. **Predeclare the minimum
effective sample size per category *after lineage deduplication*** — an under-powered category cannot pass
its gate and its shortfall must be reported, not waived. If errors occur, the required N rises accordingly.

**Reason-state validation (amendment 5).** Binary supported-vs-withheld is **not** sufficient. Add a
**reason-state confusion matrix** over `{supported, insufficient_evidence, unsupported_locus,
unknown_ood}` and a **precision requirement on `unsupported_locus`** (a specific locus claim must be right).
Where reason precision is not met, the detector must **fall back to `unknown_ood`**.

## 8. Validation protocol (summary)

1. Assemble + **freeze** the cohort (§5) with independent provenance and expected verdicts/reasons.
2. **Feasibility check first:** verify each category meets its predeclared minimum effective N after
   lineage dedup (§7). **If real-positive or near-OOD counts cannot be assembled, STOP and report
   feasibility — do not weaken the gate** (planning decision).
3. Split: **fit / validation / test**, with the §6 LOCO / lineage / species holdouts baked in.
4. Fit Stage-1 rules (from the domain contract) and Stage-2 membership (positive-only) on the fit split.
5. Select operating thresholds on validation, prioritizing FAR upper bounds (decision 3).
6. **Lock**, then evaluate **once** on test: per-category FAR upper bounds, FRR (real vs sim), LOCO folds,
   and the reason-state confusion matrix.
7. **Decision** (§9), recorded as a frozen Gate-3 artifact (like the Phase-3 / convergence artifacts).

## 9. Valid outcomes — no unvalidated escape hatch (amendment 1)

A "conservative fallback" is **still a detector** and cannot be trusted merely because it rejects more
inputs — it must pass the **same** FAR validation. The only valid outcomes are:

1. **Graded detector passes** §7 → ship the `applicability` field with all four states.
2. **A stricter rule-based detector passes** §7 (same confidence-bounded FAR gates) → ship that.
3. **No detector passes** → **Gate 3 FAILS; the product is not released.**

**No HF/registry publication until Gate 3 passes** (and Gate 4 comparative utility after it).

## 10. Non-goals

- Not VH extraction from larger chains (decision 4).
- Not per-segment trust scoring (decision 1; a later version).
- Not fragment-support validation (separate experimental track).
- Not a general antibody-vs-world classifier beyond the human-IGH-VH applicability decision.

## 11. Detector provenance & integrity (amendment 8)

The detector becomes **part of the trusted model bundle** and receives the **same safe-load + integrity
treatment** as weights/calibration (sha256-verified section, **pickle-free**). Every verdict carries
detector provenance in `applicability.provenance`:

- `detector_id`, `detector_version`;
- `detector_artifact_sha256`;
- `threshold` and `score_direction` (which side of the threshold is `supported`);
- `ruleset_version` (Stage-1 rules);
- `validation_artifact_sha256` (the frozen Gate-3 validation record the thresholds came from).

A verdict whose detector provenance does not verify against the loaded bundle is a hard load error, exactly
like a calibration/model mismatch (Gate-2 binding contract).

## Next step — implementation plan

On approval, proceed to **writing-plans**. Per the planning decision, the plan **begins with cohort
provenance and sample-size feasibility before any detector construction**; if the required independent real
positives and near-OOD counts cannot be assembled, it **stops and reports feasibility** rather than
weakening the statistical gate. Recommended build order: cohort provenance + feasibility → Stage-1
completeness rules (from the domain contract) → Stage-2 positive-only embedding distance → discriminated-union
schema + withheld-call validation → reason-state + LOCO + confidence-bound evaluation → detector provenance
/ safe-load → frozen Gate-3 artifact and pass/fail decision. The supervised negative classifier stays
optional and must earn inclusion via §6.
