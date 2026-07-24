# Gate 3 Applicability Detector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build and validate a whole-input **applicability gate** that decides whether an input is a
complete human IGH VH protein before any V/D/J call is trusted, refusing calls with a reason state when it
is not — meeting confidence-bounded false-accept thresholds, or concluding Gate 3 fails.

**Architecture:** Two-stage detector — (1) a rule-based **completeness/structure** gate, (2) a positive-only
**domain-membership** score over the model's encoder embedding — behind a feature flag, emitting a
**discriminated-union** result (supported → calls present; otherwise → calls withheld). Packaged as a
pickle-free, sha256-verified detector artifact inside the model bundle, and validated on a frozen adversarial
cohort with one-sided exact 95% upper-bound FAR/FRR gates and leave-one-category-out generalization.

**Tech Stack:** Python 3.12, PyTorch (existing AA model encoder), NumPy/SciPy (Mahalanobis/kNN, exact
binomial bounds), the existing `alignair.model_file.container` (safetensors + verified sections), pytest.

**Spec:** [Gate 3 design spec](../specs/2026-07-24-aa-trust-gate-gate3-applicability-spec.md). This plan
implements it exactly; do not re-decide settled items.

## Global Constraints

- **Feasibility gate first (HARD STOP).** Phase 0 must confirm each category meets its predeclared minimum
  effective N *after lineage deduplication* before ANY detector is built. If real-positive or near-OOD
  counts cannot be assembled, **STOP and report feasibility — never weaken the statistical gate.**
- **Confidence bounds, not point estimates.** All FAR/FRR gates are **one-sided exact 95% upper bounds**
  (Clopper–Pearson). Zero-error N: ≈ **59** for a < 5% bound, ≈ **29** for a < 10% bound.
- **Release gates (95% upper bound):** overall near-OOD FAR ≤ 5%; IGK/IGL ≤ 5%; TCR ≤ 5%; VHH/non-human VH
  ≤ 10%; IgSF ≤ 10%; FRR ≤ 10% reported **separately for real vs simulator** positives.
- **FAR is the priority metric** (minimize false-accepts); every threshold choice favors the FAR bound.
- **No unvalidated fallback.** Valid outcomes: graded detector passes / stricter rule-based detector passes
  the same FAR gates / **Gate 3 fails and the product is not released.**
- **Softmax and conformal set-size are PROHIBITED** as the applicability signal.
- **Independent labels:** cohort ground truth (locus/species/domain/boundaries) comes from declared sources,
  **never** from the Stage-1 rules. Motif rules may not treat a category as "unseen" if it shaped their
  design.
- **LOCO generalization** for any learned component across **all** critical groups (IGK/IGL, TCR,
  VHH/non-human, IgSF); failure on any one blocks shipping that component.
- **Discriminated union, schema-enforced:** a non-`supported` result MUST NOT carry `segments`/`inter_vj_aa`.
- **Detector is part of the trusted bundle:** pickle-free, sha256-verified section; verdict carries detector
  provenance (id/version/artifact-hash/threshold/score-direction/ruleset-version/validation-artifact-hash);
  provenance mismatch is a hard load error.
- **No raw sequences committed to git.** Cohort is a provenance manifest + hydration (like the Thera-SAbDab
  PL132 pattern); hydrated FASTA is git-ignored.
- **No `.pt`/`.alignair`/detector-artifact binaries committed.** Nothing frozen/incumbent is overwritten.
- **No Claude/Co-Authored-By line in commits** (project rule).

## File Structure

- `src/alignair/aa/product/applicability/` — new package:
  - `schema.py` — `Applicability`, verdict states, discriminated-union assembly + validation.
  - `completeness.py` — Stage-1 rule gate (domain-fraction, N→C, V/J anchors, no leader/constant).
  - `membership.py` — Stage-2 positive-only embedding-distance scorer (fit + score + persist).
  - `embedding.py` — encoder-embedding extraction from the AA model.
  - `detector.py` — two-stage orchestration → `Applicability` verdict.
  - `provenance.py` — detector-artifact packaging + safe-load binding.
- `src/alignair/aa/product/predictor.py` — integrate the gate behind a flag; withhold calls when not supported.
- `scripts/aa/gate3/` — cohort manifest, feasibility, threshold-selection, and locked-test harnesses.
- `docs/superpowers/artifacts/2026-07-24-aa-gate3/` — feasibility report, cohort fingerprint, validation record.
- `tests/alignair/aa/product/applicability/` — tests per component.

---

## Phase 0 — Feasibility (HARD GATE: stop here if it fails)

### Task 1: Cohort provenance manifest + sourcing

**Files:**
- Create: `scripts/aa/gate3/cohort_manifest.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/cohort_manifest.md`
- Test: `tests/alignair/aa/product/applicability/test_cohort_manifest.py`

**Interfaces:**
- Produces: `CohortEntry {id, category, expected_state, expected_reason, source, locus, species, domain,
  order, boundary_source, lineage_key}` and a `load_manifest() -> list[CohortEntry]`.

- [ ] **Step 1:** Enumerate every category from spec §5 with its **expected verdict/reason**, declared
  source, and independent boundary source (positives: sim held-out, real multi-family disjoint from the
  16-antibody set, human-framework designs, humanized; near-OOD: IGK/IGL, TCR, VHH, non-human/murine-variable
  chimera, IgSF; `unknown_ood`: de-novo antibody-like; structural: reversed/CDR3-only/V-only/J-only/leader+VH/
  VH+constant; easy control: random proteins). Write the manifest schema + a documented sourcing entry per
  category (dataset, accession/query, license, redistribution status) to `cohort_manifest.md`.
- [ ] **Step 2:** Write the failing test: `load_manifest()` returns entries covering **all** spec-§5
  categories, each with a non-null `expected_state`, `source`, and `boundary_source`, and no raw sequence in
  the manifest (provenance only).
- [ ] **Step 3:** Run it — expect FAIL (module missing).
- [ ] **Step 4:** Implement `cohort_manifest.py` (manifest as data + `load_manifest`). Raw sequences live in
  a git-ignored hydration path, never in the manifest.
- [ ] **Step 5:** Run the test — expect PASS.
- [ ] **Step 6:** Commit (`scripts/aa/gate3/cohort_manifest.py`, manifest doc, test).

### Task 2: Sample-size feasibility assessment (STOP/GO)

**Files:**
- Create: `scripts/aa/gate3/feasibility.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/feasibility.md`
- Test: `tests/alignair/aa/product/applicability/test_feasibility.py`

**Interfaces:**
- Consumes: `load_manifest`, a `lineage_key` per entry.
- Produces: `feasibility_report(hydrated) -> {category: {available, effective_after_dedup, min_required,
  powered: bool}}` and an overall `GO`/`STOP` decision. `min_required` = 59 for a 5%-gate category, 29 for a
  10%-gate category (zero-error basis, spec §7).

- [ ] **Step 1:** Write the failing test: for a fixture cohort, `feasibility_report` deduplicates by
  `lineage_key`, computes effective N per category, compares to `min_required` (59 for IGK/IGL, TCR, overall
  near-OOD, real positives; 29 for VHH/non-human, IgSF), and returns `STOP` when any 5%-gate category is
  under 59 after dedup.
- [ ] **Step 2:** Run it — expect FAIL.
- [ ] **Step 3:** Implement `feasibility.py`: lineage dedup, per-category effective counts, exact
  zero-error min-N table, GO/STOP verdict; write `feasibility.md`.
- [ ] **Step 4:** Run the test — expect PASS.
- [ ] **Step 5:** Run the **real** feasibility assessment on the hydrated cohort; write the report.
- [ ] **Step 6:** **DECISION GATE.** If the verdict is `STOP` for any critical category, **halt the plan and
  report** which categories are under-powered and by how much. Do **not** proceed to Phase 1. If `GO`, commit
  the report and continue.

---

## Phase 1 — Cohort freeze (only if Phase 0 = GO)

### Task 3: Assemble, split, and freeze the cohort

**Files:**
- Create: `scripts/aa/gate3/build_cohort.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/cohort_fingerprint.json`
- Test: `tests/alignair/aa/product/applicability/test_cohort_split.py`

**Interfaces:**
- Produces: `build_splits(seed) -> {fit, validation, test}` of `CohortEntry` with LOCO/lineage/species
  holdouts encoded; a `cohort_fingerprint.json` (per-split per-category counts + hashes).

- [ ] **Step 1:** Failing test: splits are disjoint by `lineage_key`; each critical negative category has a
  **held-out** presence in test that is absent from fit (LOCO precondition); real and simulator positives are
  tagged separately; the 16-antibody Phase-3 ids appear in **no** split.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `build_cohort.py` (deterministic split honoring the holdouts; emit fingerprint).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Freeze the cohort fingerprint artifact; commit (script, fingerprint, test).

---

## Phase 2 — Stage 1: completeness / structure gate

### Task 4: Rule-based completeness gate

**Files:**
- Create: `src/alignair/aa/product/applicability/completeness.py`
- Test: `tests/alignair/aa/product/applicability/test_completeness.py`

**Interfaces:**
- Produces: `check_completeness(sequence) -> CompletenessResult {passed: bool, reason: str|None,
  detail: dict, ruleset_version: str}`. Reasons are specific (`too_short`, `reversed_order`,
  `missing_v_anchor`, `missing_j_anchor`, `leader_or_constant_contamination`, `low_domain_fraction`).

- [ ] **Step 1:** Failing tests from the domain contract (not reverse-engineered from negatives): a complete
  in-frame VH passes; a reversed VH → `reversed_order`; a CDR3-only peptide → `too_short`/`missing_*_anchor`;
  a leader+VH and a VH+constant → `leader_or_constant_contamination`; a V-only fragment → `missing_j_anchor`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `completeness.py` with interpretable checks + `ruleset_version`. Document each
  rule's domain-contract justification (Global Constraint: not derived from the cohort negatives).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

---

## Phase 3 — Stage 2: positive-only membership score

### Task 5: Encoder-embedding extraction

**Files:**
- Create: `src/alignair/aa/product/applicability/embedding.py`
- Test: `tests/alignair/aa/product/applicability/test_embedding.py`

**Interfaces:**
- Consumes: the loaded `AAPredictor` (model + tokenizer).
- Produces: `embed(predictor, sequences) -> np.ndarray [n, d]` (pooled encoder representation; one forward
  pass, no grad).

- [ ] **Step 1:** Failing test: `embed` returns a fixed-width `[n, d]` float array for a batch of VH reads,
  deterministic across calls.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `embedding.py` (reuse the predictor's tokenizer + model encoder; mean-pool over
  valid positions).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

### Task 6: Positive-only membership scorer

**Files:**
- Create: `src/alignair/aa/product/applicability/membership.py`
- Test: `tests/alignair/aa/product/applicability/test_membership.py`

**Interfaces:**
- Consumes: `embed`.
- Produces: `MembershipScorer.fit(in_domain_embeddings)`, `.score(embeddings) -> np.ndarray` (higher =
  more in-domain), `.to_bytes()/.from_bytes()` (pickle-free). Fit on **in-domain human VH only**.

- [ ] **Step 1:** Failing test: fit on in-domain fixture embeddings; in-domain holdout scores strictly higher
  than injected far-OOD embeddings; `from_bytes(to_bytes())` round-trips scores exactly; `to_bytes` contains
  no pickle opcodes.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `membership.py` (class-conditional Mahalanobis or kNN density; NumPy arrays
  serialized via safetensors/`np.save`-to-buffer, no pickle).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

*(Optional supervised negative classifier is deferred: it may be added later ONLY if it passes the Task-11
LOCO gate. Not built now — positive-only is the first candidate.)*

---

## Phase 4 — Output contract & integration

### Task 7: Discriminated-union verdict schema + validation

**Files:**
- Create: `src/alignair/aa/product/applicability/schema.py`
- Test: `tests/alignair/aa/product/applicability/test_applicability_schema.py`

**Interfaces:**
- Produces: `Applicability {state, reason, score, detail, provenance}`; `assemble_result(applicability,
  segments_or_none)` that **enforces** the union — `supported` requires `segments`; any other state
  **forbids** `segments`/`inter_vj_aa` (raises on violation).

- [ ] **Step 1:** Failing test: assembling a `supported` verdict without segments raises; assembling a
  non-`supported` verdict **with** segments raises; a valid non-supported result contains no `segments`/
  `inter_vj_aa` key at all.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `schema.py` (the union + validation).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

### Task 8: Two-stage detector + predictor integration (feature-flagged)

**Files:**
- Create: `src/alignair/aa/product/applicability/detector.py`
- Modify: `src/alignair/aa/product/predictor.py`
- Test: `tests/alignair/aa/product/applicability/test_detector.py`, `.../test_predict_withholds_calls.py`

**Interfaces:**
- Consumes: `check_completeness`, `MembershipScorer`, `embed`, `assemble_result`.
- Produces: `applicability_verdict(predictor, sequence) -> Applicability`; `predict(..., applicability=True)`
  returns the discriminated union. Stage-1 fail → `insufficient_evidence`; Stage-2 out-of-manifold with a
  fired locus signal → `unsupported_locus`, else → `unknown_ood`; **reason unresolved → `unknown_ood`**
  (never invent a locus).

- [ ] **Step 1:** Failing tests: a complete human VH → `supported` with segments; a reversed VH →
  `insufficient_evidence`, **no segments**; a far-OOD protein → `unknown_ood`, no segments; the flag default
  (off) preserves today's behavior exactly.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `detector.py` + wire into `predict` behind `applicability=False` default.
- [ ] **Step 4:** Run — expect PASS; run the full AA product suite to confirm no regression when the flag is off.
- [ ] **Step 5:** Commit.

---

## Phase 5 — Detector provenance & safe load

### Task 9: Detector artifact packaging + binding

**Files:**
- Create: `src/alignair/aa/product/applicability/provenance.py`
- Modify: `src/alignair/aa/product/bundle.py`, `loader.py`
- Test: `tests/alignair/aa/product/applicability/test_provenance.py`

**Interfaces:**
- Produces: a detector bundle **section** (pickle-free, sha256-verified) carrying the scorer bytes +
  ruleset_version + threshold + score_direction; `applicability.provenance` populated with
  `detector_id/version/artifact_sha256/threshold/score_direction/ruleset_version/validation_artifact_sha256`;
  load-time verification that the verdict's detector provenance matches the loaded section.

- [ ] **Step 1:** Failing tests: the detector section round-trips with checksum verification; a tampered
  detector section fails load (hard error, like calibration mismatch); every emitted verdict carries complete
  provenance.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `provenance.py` + bundle/loader wiring (reuse `container` verified-section path).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Commit.

---

## Phase 6 — Validation (the pass/fail decision)

### Task 10: Threshold selection on validation split

**Files:**
- Create: `scripts/aa/gate3/select_threshold.py`
- Test: `tests/alignair/aa/product/applicability/test_threshold_selection.py`

**Interfaces:**
- Consumes: frozen splits, `applicability_verdict`.
- Produces: an operating threshold minimizing the **FAR 95% upper bound** on the validation split subject to
  FRR-bound feasibility; records `score_direction`.

- [ ] **Step 1:** Failing test: given fixture validation scores, the selector picks the threshold with the
  lowest FAR upper bound among those whose FRR upper bound ≤ 10%; ties broken toward lower FAR.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `select_threshold.py` (Clopper–Pearson upper bound; FAR-priority selection).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Select the real threshold on the validation split (no test peeking); record it. Commit.

### Task 11: Locked test evaluation + Gate-3 decision (frozen artifact)

**Files:**
- Create: `scripts/aa/gate3/evaluate.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/validation.md` + `.json`
- Test: `tests/alignair/aa/product/applicability/test_evaluate.py`

**Interfaces:**
- Consumes: locked threshold, frozen test split.
- Produces: per-category FAR **upper bounds**, FRR (real vs sim), **LOCO fold** results, the
  **reason-state confusion matrix** + `unsupported_locus` precision, and the §9 **PASS/FAIL** decision.

- [ ] **Step 1:** Failing test: `evaluate` computes exact 95% upper bounds per category; returns `PASS` only
  if **every** gate is met (overall/IGK-IGL/TCR ≤ 5%, VHH-non-human/IgSF ≤ 10%, FRR real & sim ≤ 10%) **and**
  every critical LOCO fold passes **and** `unsupported_locus` precision holds (else reasons fall back to
  `unknown_ood`); otherwise `FAIL`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `evaluate.py` (bounds, LOCO aggregation, confusion matrix, decision).
- [ ] **Step 4:** Run — expect PASS (harness correctness).
- [ ] **Step 5:** Run the **locked, one-shot** test evaluation; write the frozen `validation.md`/`.json`.
- [ ] **Step 6:** **GATE-3 DECISION.** If `PASS`, the `applicability` field is cleared to ship (behind the
  flag → default-on in a follow-up). If `FAIL`, record the failure; **Gate 3 does not pass and the product is
  not released** — do not weaken any threshold. Commit the frozen artifact.

---

## Self-review notes (per writing-plans)

- **Feasibility precedes construction:** Phase 0 gates the entire plan; Tasks 3–11 are contingent on GO.
- **Every statistical gate is a 95% upper bound** with predeclared min-N (Global Constraints), matching spec
  §7 — no point-estimate acceptance anywhere.
- **Withheld-call safety** is enforced structurally (Task 7 union validation) and behaviorally (Task 8).
- **LOCO + reason-state precision** are release-blocking in Task 11, matching spec §6/§7 amendments.
- **No fallback shortcut:** Task 11 Step 6 encodes "FAIL → not released," matching spec §9.
