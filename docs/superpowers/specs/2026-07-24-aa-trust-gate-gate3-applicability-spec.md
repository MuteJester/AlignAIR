# AlignAIR-AA — Gate 3: Applicability & Negative-Controls Design Spec

> **AIRR Utility & Trust Gate, Gate 3 of 6.** Gate 3 introduces a **new product-decision layer**: an
> input-**applicability gate** that decides whether an input is an in-domain human IGH VH protein *before*
> any V/D/J call is trusted, plus the adversarial cohort and validation protocol that prove it works. This
> is the gate that closes the core risk — *precise-looking IGH annotations on inputs the model was never
> validated to interpret*.

**Status:** DRAFT for review (design spec; implementation plan follows on approval) · **Date:** 2026-07-24
· Builds on [Gate 1 contract](2026-07-24-aa-trust-gate-gate1-product-contract.md) and
[Gate 2 audit](2026-07-24-aa-trust-gate-gate2-semantics-audit.md).

## 0. Settled product decisions (owner, 2026-07-24)

These four decisions are fixed inputs to this spec; the rest of the design serves them.

1. **Whole-input applicability gate first.** A single verdict for the entire input (*is this an in-domain
   human VH at all?*) is the primary safety gate. Per-segment evidence may layer on in a later version.
2. **Refuse calls + reason state.** When the input is not `supported`, the product returns an applicability
   verdict with a reason and **withholds** the V/D/J calls — it does not emit numbers the user shouldn't
   trust.
3. **Minimize false-accepts.** Wrongly claiming support on an out-of-domain input is the worst error.
   The detector is tuned to minimize false-accepts, accepting more false-rejects (in-domain inputs routed
   to manual review).
4. **Require a pre-trimmed VH.** Signal-peptide+VH and full heavy chains (VH+constant) are **unsupported**;
   VH extraction is external preprocessing, out of scope for v1 (its own validated problem).

## 1. What Gate 3 decides, and does not

**Decides:** whether an arbitrary in-alphabet ≤192-aa protein is a **complete human IGH VH domain in N→C
order** that the model may annotate — and if not, why not.

**Does not:** extract a VH from a larger chain (decision 4); validate fragment/truncation support (that is
a separate experimental question — see §9); resolve per-segment trust (whole-input first, decision 1);
change the V/D/J model itself.

## 2. Output contract — the applicability verdict

Gate 3 is where the result schema **is** allowed to grow (Gate 1/2 deferred any applicability field to the
gate that validates it). A new **top-level** object:

```
applicability: {
  state: "supported" | "insufficient_evidence" | "unsupported_locus" | "unknown_ood",
  reason: str,               # human-readable, specific
  score: float,              # detector score (for auditing; NOT a calibrated probability)
  detail: { ... }            # optional per-check evidence (motif/anchor/distance signals)
}
```

- **`supported`** → V/D/J calls are emitted as today.
- **any other state** → **V/D/J calls are withheld** (decision 2). The result still carries
  `sequence_id`, `input_length`, `applicability`, `provenance`, `warnings`.
- **State semantics:**
  - `unsupported_locus` — confidently a different locus/molecule (IGK/IGL, TCR, non-antibody).
  - `insufficient_evidence` — plausibly antibody-related but lacking a complete VH domain (too short,
    fragment, no V/J anchors, leader/constant present).
  - `unknown_ood` — far from the in-domain manifold with no confident locus call (the catch-all).
- The verdict is emitted **regardless of `on_invalid`** (it is not an input-format error; it is a
  domain judgment on a *valid* sequence). Format-invalid inputs still follow the Gate-2 §A path.

The `applicability` field ships **only if** the detector passes §7 release thresholds. If it does not, the
honest fallback is a **conservative gate that only passes complete human VH and refuses everything else**
(§9), which is an acceptable Gate-3 outcome.

## 3. Why model softmax / conformal set-size are PROHIBITED as the detector

Both are **forbidden** as the applicability signal, for reasons already established:

- **Softmax score** is uncalibrated and routinely **confidently wrong** on OOD — a TCR or light-chain V
  domain (same Ig fold) can produce a high-scoring, small-set V "call." It measures nothing about
  in-domain membership.
- **Conformal set size** is calibrated **only under the simulator distribution**. On genuinely OOD input
  the exchangeability assumption fails, so a *small* set is not evidence of support — it can be small and
  wrong. A widened set is simulator-domain ambiguity, not OOD detection (Gate 2 §E).

The detector must use a signal that actually measures **distance from the training domain**, not the
model's own confidence in its (possibly nonsensical) answer.

## 4. Detector design — options and recommendation

The spec evaluates a **hybrid** detector; the final composition is chosen empirically on the cohort (§6).

| Option | What it is | Strengths | Risks |
|---|---|---|---|
| **A. Embedding-distance OOD** | Score distance of the input's model **encoder embedding** to the in-domain human-VH manifold (class-conditional Mahalanobis, kNN, or a fitted density) | Uses the existing model; no reliance on having seen a specific OOD type; principled for near-OOD | Needs a representative in-domain fit set; sensitive to representation choice |
| **B. Learned applicability classifier** | Supervised classifier (binary or multi-state) on positives + labeled negatives | Directly optimizes the decision; can hit the reason states | **Leakage/overfit to seen negatives** — a detector that only rejects negative *types it trained on* is untrustworthy (§6 holdout) |
| **C. Motif / anchor / structure checks** | Rule checks: VH framework motifs, conserved Cys pattern, V/J anchor presence, domain-length/fraction | Interpretable; strong for `unsupported_locus` and `insufficient_evidence`; cheap | Brittle alone; can be evaded by engineered sequences |

**Recommended primary (to validate):** **A + C hybrid** — motif/anchor/length rules produce the obvious
`unsupported_locus` / `insufficient_evidence` rejects and the interpretable reason; embedding-distance
handles subtle near-OOD and the `unknown_ood` catch-all. **B is admitted only if** it is validated to
generalize to **held-out OOD categories it never trained on** (§6) — otherwise it is a memorizer, not a
detector.

**Non-negotiable:** whatever the composition, the score must be **thresholded on a held-out split** (§7),
and the detector must be demonstrated to reject **at least one OOD category excluded from its fitting**.

## 5. Cohort composition

The evaluation cohort must be dominated by **hard, VH-like** inputs. *Easy random proteins alone would
produce a misleadingly strong detector* and are a small control slice only.

**Positives (must be judged `supported`):**
- Human IGH **complete VH** proteins from GenAIRR simulation **held out from model training**.
- **Real** human VH proteins, curated and **disjoint from the 16-antibody Phase-3 cohort** (no leakage),
  spanning multiple V families (not IGHV3-dominated).

**Near-OOD hard negatives (the crux — same fold / very VH-like; must be rejected):**
- Human **IGK** and **IGL** light-chain V domains (same Ig fold — hardest).
- **VHH / camelid** single-domain antibodies.
- **Murine / chimeric** VH (right locus, wrong species).
- Human **TCR** V domains — TRA / TRB / TRG / TRD.
- Unrelated **immunoglobulin-superfamily** V-like domains.
- **PLM-designed / heavily engineered** antibodies (synthetic, near-distribution).

**Structural / coverage negatives (must be rejected under the "complete VH, N→C" contract):**
- **Reversed** VH (C→N order) — tests the no-orientation-head assumption directly.
- **CDR3-only** peptide.
- **V-only** and **J-only** fragments.
- **Leader (signal peptide) + VH**, and **VH + constant** domain(s).

**Easy negatives (small control slice):** unrelated non-antibody proteins (enzymes, globins, etc.).

Each cohort member carries provenance (source, locus, species, construction) and its expected verdict.

## 6. Leakage & lineage controls

- **Detector fit ≠ model training data** where feasible; **never** reuse the 16-antibody Phase-3 cohort
  for detector fitting or threshold selection (blinded-selection principle).
- **Held-out OOD categories:** fit any density/classifier on in-domain (+ optionally *some* negative
  types), then **test on negative categories excluded from fitting** (e.g. fit without TCR; test on TCR).
  A detector is only trustworthy if it rejects OOD types it never saw.
- **Family / lineage holdout** for real positives: evaluate on V families / lineages not used in fitting,
  so `supported` is not memorization of specific sequences.
- **Species holdout** for the murine/chimeric and VHH negatives.
- Threshold chosen on a **validation split**, reported **once** on a locked **test split** (no peeking).

## 7. Metrics & release thresholds

- **Primary — False-Accept Rate (FAR):** fraction of OOD/negative inputs wrongly judged `supported`.
  Reported **overall and per negative category** (IGK/IGL, TCR, VHH especially).
- **Secondary — False-Reject Rate (FRR):** fraction of genuine complete human VH positives wrongly
  rejected.
- **Generalization:** FAR on **held-out** OOD categories (§6) reported separately — the headline number.

**Proposed release gate (final numbers set empirically on the validation split, then confirmed once on
test):**
- Overall near-OOD **FAR ≤ 5%**, and **per-category FAR ≤ 10%** for the hardest categories (IGK/IGL, TCR,
  VHH), **at FRR ≤ 10%** on complete in-domain human VH; **and**
- non-trivial rejection of **≥ 1 held-out OOD category** not used in fitting.

If these cannot be met at any operating point, the detector is **not** shipped as a graded gate; the
fallback conservative gate (§9) is used instead. Minimizing FAR dominates every tuning choice (decision 3).

## 8. Validation protocol (summary)

1. Assemble + freeze the cohort (§5) with provenance and expected verdicts.
2. Split: **fit / validation / test**, with the §6 category/lineage/species holdouts baked into the split.
3. Fit the detector components on fit-split in-domain (+ permitted negatives).
4. Select the operating threshold on validation, prioritizing FAR (decision 3).
5. **Lock**, then evaluate **once** on test; report FAR (overall + per-category + held-out), FRR.
6. Decision: **ship graded gate** (thresholds met) or **conservative fallback** (§9). Record verdict as a
   frozen Gate-3 artifact, like the Phase-3 / convergence artifacts.

## 9. Acceptable outcomes (explicit)

Gate 3 is **allowed to conclude that only complete human VH is safely supportable.** Broad fragment /
truncation support is an **experimental question, not a required outcome**. Two acceptable end-states:

- **Graded gate** — the hybrid detector meets §7; the `applicability` field ships with all four states.
- **Conservative fallback** — if §7 is not met, ship a gate that passes only inputs it can confidently
  place as complete human VH and refuses (or `unknown_ood`s) everything else. Lower coverage, but the
  trust contract holds: **no confident annotation of out-of-domain input.**

Either way, **no HF/registry publication until Gate 3 passes** (and Gate 4 comparative utility after it).

## 10. Non-goals

- Not VH extraction from larger chains (decision 4).
- Not per-segment trust scoring (decision 1; a later version).
- Not fragment-support validation (separate experimental track).
- Not a general antibody-vs-world classifier beyond the human-IGH-VH applicability decision.

## Next step

On approval of this spec, proceed to **writing-plans** for the implementation: cohort assembly (with
provenance + expected verdicts), detector components (A+C, optional B under holdout), the split/threshold
protocol, the `applicability` output wiring behind a feature flag, and the frozen Gate-3 validation
artifact. Cohort building starts only after the spec is approved.
