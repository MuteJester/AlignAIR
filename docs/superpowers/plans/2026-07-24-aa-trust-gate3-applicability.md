# Gate 3 Applicability Detector — Implementation Plan (rev. 2, review-amended)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build and validate a whole-input **applicability gate** that decides whether an input is a
complete human IGH VH protein before any V/D/J call is trusted, refusing calls with a reason state when it
is not — meeting confidence-bounded false-accept thresholds on a **powered locked test split**, or
concluding Gate 3 fails.

**Architecture:** Two-stage detector — (1) a rule-based **completeness/structure** gate, (2) a positive-only
**domain-membership** score over the model's **penultimate V/J branch features** — behind a feature flag,
emitting a **discriminated-union** result (supported → calls present; otherwise → calls withheld). The v1
detector emits **three** states (`supported`, `insufficient_evidence`, `unknown_ood`); `unsupported_locus`
is **reserved** until a validated reason-attribution component exists. Packaged — **after** locked PASS — as
a pickle-free, sha256-verified detector artifact in the model bundle.

**Tech Stack:** Python 3.12, PyTorch (AA model penultimate branch features), NumPy/SciPy (kNN / shrinkage
Mahalanobis, Clopper–Pearson exact bounds), existing `alignair.model_file.container` (safetensors + verified
sections), pytest.

**Spec:** [Gate 3 design spec](../specs/2026-07-24-aa-trust-gate-gate3-applicability-spec.md). Where this
plan is more specific than the spec (states emitted in v1, powering the locked test, category-held-out vs
LOCO), the plan's amended text governs implementation.

## Global Constraints

- **Feasibility gate first (HARD STOP).** Phase 0 must confirm counts **on the LOCKED TEST split** after
  lineage dedup **and** allocation to fit/validation/test — not on the total cohort. If the powered locked
  test cannot be assembled, **STOP and report feasibility — never weaken the gate.**
- **Powered locked test (zero-error floors):** a **≤ 5%** FAR 95% upper bound needs ≈ **59** independent
  locked-test negatives in that group; a **≤ 10%** bound needs ≈ **29**; FRR **≤ 10%** needs ≈ **29**
  locked-test positives. **Observed errors raise the requirement** (recompute the exact Clopper–Pearson N).
  Fit and validation examples are **additional** to these locked-test floors.
- **Confidence bounds, not point estimates.** All FAR/FRR gates are **one-sided exact 95% upper bounds**.
- **Combined vs locus-specific claims:** the ≤ 5% gates apply to **combined IGK/IGL** and **combined TCR**
  (both loci / all chains **present**, none absent). IGK and IGL — and TRA/TRB/TRG/TRD — are reported as
  **descriptive strata**. A formal ≤ 5% per-locus claim requires that locus to be **independently powered**.
- **Release gates (95% upper bound):** combined near-OOD FAR ≤ 5%; combined IGK/IGL ≤ 5%; combined TCR ≤ 5%;
  VHH/non-human VH ≤ 10%; IgSF ≤ 10%; FRR ≤ 10% reported **separately for real vs simulator** positives.
- **FAR is the priority metric.**
- **v1 emits THREE states.** Structural/completeness failure → `insufficient_evidence`; membership rejection
  → `unknown_ood`. **`unsupported_locus` is NOT emitted** until an independent, separately-validated reason
  classifier exists (a membership score cannot say *why* something is out-of-domain).
- **Representation + scorer are selected on VALIDATION** (penultimate V/J branch features vs a predeclared
  concatenation; kNN vs shrinkage Mahalanobis) under a **predeclared tie-break**. **The locked test never
  selects the detector.**
- **FRR-threshold stop:** if **no** validation threshold satisfies the FRR bound, **STOP before opening the
  locked test.**
- **Non-circular packaging:** `frozen cohort → scorer/rules/threshold → validation artifact → final bundle
  containing the validation-artifact hash`. Final packaging happens **after** locked PASS.
- **Softmax and conformal set-size are PROHIBITED** as the applicability signal.
- **Independent labels:** cohort ground truth (locus/species/domain/boundaries) comes from declared database
  sources, **never** from the Stage-1 rules. Motif rules may not treat a category as "unseen" if it shaped
  their design.
- **Positive-only detector uses CATEGORY-HELD-OUT reporting, not LOCO retraining** (it never trains on
  negatives, so every OOD category is already unseen). True LOCO retraining applies only if a supervised
  negative classifier is later introduced.
- **Discriminated union, schema-enforced:** a non-`supported` result MUST NOT carry `segments`/`inter_vj_aa`.
- **No raw sequences committed to git** (provenance manifest + git-ignored hydration). **No
  `.pt`/`.alignair`/detector binaries committed.** Nothing frozen/incumbent overwritten. **No Claude/
  Co-Authored-By line in commits.**

## Recommended data sources (see the spec's cohort in §5)

| Cohort | Primary sources | Use / caveats |
|---|---|---|
| Human IGH positives | OAS (opig.stats.ox.ac.uk/webapps/oas, CC BY 4.0), AIRR Data Commons (ADC) API | Multi-donor productive heavy chains. **Dedup by donor, study, and clonal lineage** — not exact sequence. |
| Human IGK/IGL | OAS + ADC; OGRDB human germline sets as **controls** | Real rearrangements for powered eval; germline alleles are controls, not 59 independent lineages. |
| Human TCR | ADC (powered cohort); STCRDab (structural, smaller N); IMGT refs (controls) | ADC supplies the powered set. |
| Murine VH | OAS mouse repertoires; SAbDab (structural challenge) | OAS = scale; SAbDab = resolved challenge examples. |
| VHH | AVIDa-hIL6 (**CC BY-NC 4.0**, may carry leaders/tags), SAbDab nanobody entries | Hydration-only, dedup unique VHH lineages, **record licensing**. |
| Chimeric / humanized | Thera-SAbDab (non-Phase-3), primary sources | Human-framework designs → **supported-positive** challenge; genuinely chimeric (murine variable) → OOD. **Keep the Phase-3 16 excluded.** |
| IgSF V-set domains | InterPro IPR013106 + API | Reviewed UniProt proteins with V-set coords, **excluding IG/TR loci**; cluster by family + identity before counting effective N. |
| Structural negatives | Deterministic transforms of independently-sourced complete-VH parents | Reversed, CDR3-only, V-only/J-only, leader+VH, VH+constant. **Effective N = number of parent lineages**, not derived transforms. |
| De-novo / PLM antibodies | Challenge set only (descriptive, expected `unknown_ood`) | **Not a Phase-0 hard blocker.** |

OAS/ADC locus + completeness labels are database annotations independent of the detector — sufficient here
because the endpoint is "is this a complete human IGH VH input?", not allele accuracy.

---

## Phase 0 — Feasibility (HARD GATE: stop here if it fails)

### Task 1: Cohort provenance manifest + hydration

**Files:**
- Create: `scripts/aa/gate3/cohort_manifest.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/cohort_manifest.md`
- Create: `scripts/aa/gate3/hydrate.py` (writes to a git-ignored path)
- Test: `tests/alignair/aa/product/applicability/test_cohort_manifest.py`

**Interfaces:**
- Produces: `CohortEntry {id, category, group, expected_state, source, locus, species, domain, order,
  boundary_source, license, lineage_key}` and `load_manifest() -> list[CohortEntry]`; `hydrate(manifest)`
  materializing sequences into the git-ignored hydration dir with recorded provenance + license.

- [ ] **Step 1:** Enumerate every spec-§5 category with its **expected state** (positives/human-framework
  designs/humanized → `supported`; IGK/IGL, TCR, VHH, non-human/murine-variable chimera, IgSF → reject;
  de-novo → `unknown_ood`; structural → `insufficient_evidence`), the declared **source** + **license** +
  **independent boundary source** (from the data-sources table), and the **combined group** it powers
  (`ighk_igl`, `tcr`, `vhh_nonhuman`, `igsf`). Write `cohort_manifest.md`.
- [ ] **Step 2:** Failing test: `load_manifest()` covers all categories; each entry has non-null
  `expected_state`, `source`, `license`, `boundary_source`, `group`, `lineage_key`; **no raw sequence** in
  the manifest.
- [ ] **Step 3:** Run — expect FAIL.
- [ ] **Step 4:** Implement `cohort_manifest.py` + `hydrate.py` (hydration writes git-ignored; add the path
  to `.gitignore`).
- [ ] **Step 5:** Run — expect PASS. Commit (scripts, manifest doc, test, `.gitignore`).

### Task 2: Split-aware feasibility assessment (STOP/GO)

**Files:**
- Create: `scripts/aa/gate3/feasibility.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/feasibility.md`
- Test: `tests/alignair/aa/product/applicability/test_feasibility.py`

**Interfaces:**
- Consumes: `load_manifest`, hydrated sequences, `lineage_key`.
- Produces: `feasibility_report(hydrated) -> {group: {effective_after_dedup, locked_test_alloc,
  min_required, powered: bool, both_loci_present: bool}}` + overall `GO`/`STOP`. `min_required` is the
  **locked-test** floor (59 for 5%-gated groups, 29 for 10%-gated; positives 29), computed after dedup and
  after reserving fit + validation portions.

- [ ] **Step 1:** Failing test (fixture cohort): dedup by `lineage_key`; allocate fit/val/test; verify the
  **locked-test** portion of each group meets its floor; a group whose *test* portion is < floor (even if
  the *total* is ≥ floor) returns `STOP`; a combined group missing one locus (e.g. IGL absent) returns
  `STOP`; structural-negative effective N counts **parent lineages**, not transforms.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `feasibility.py` (lineage dedup → split allocation → locked-test floor check →
  both-loci-present check → exact zero-error min-N table; GO/STOP).
- [ ] **Step 4:** Run — expect PASS.
- [ ] **Step 5:** Run the **real** assessment on the hydrated cohort; write `feasibility.md`.
- [ ] **Step 6:** **DECISION GATE.** If any critical group is `STOP` (under-powered locked test, or a
  missing locus), **halt and report** — do **not** proceed to Phase 1. De-novo/PLM under-power is **not** a
  blocker (descriptive only). If `GO` for the critical groups, commit and continue.

---

## Phase 1 — Cohort freeze (only if Phase 0 = GO)

### Task 3: Assemble, allocate, and freeze the cohort

**Files:**
- Create: `scripts/aa/gate3/build_cohort.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/cohort_fingerprint.json`
- Test: `tests/alignair/aa/product/applicability/test_cohort_split.py`

**Interfaces:**
- Produces: `build_splits(seed) -> {fit, validation, test}` with **locked-test floors satisfied per group**,
  combined groups keeping both loci present, real vs simulator positives tagged, category-held-out subsets
  marked; a `cohort_fingerprint.json` (per-split per-group counts + hashes).

- [ ] **Step 1:** Failing test: splits disjoint by `lineage_key`; each critical group's **test** count ≥ its
  floor; both loci present in each combined group's test split; real and simulator positives tagged
  separately; **the 16 Phase-3 ids appear in no split.**
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `build_cohort.py`.
- [ ] **Step 4:** Run — expect PASS. Freeze the fingerprint; commit.

---

## Phase 2 — Stage 1: completeness / structure gate

### Task 4: Rule-based completeness gate

**Files:**
- Create: `src/alignair/aa/product/applicability/completeness.py`
- Test: `tests/alignair/aa/product/applicability/test_completeness.py`

**Interfaces:**
- Produces: `check_completeness(sequence) -> CompletenessResult {passed, reason|None, detail,
  ruleset_version}`; reasons: `too_short`, `reversed_order`, `missing_v_anchor`, `missing_j_anchor`,
  `leader_or_constant_contamination`, `low_domain_fraction`.

- [ ] **Step 1:** Failing tests from the **domain contract** (not reverse-engineered from negatives):
  complete VH passes; reversed VH → `reversed_order`; CDR3-only → `too_short`/`missing_*_anchor`; leader+VH
  and VH+constant → `leader_or_constant_contamination`; V-only → `missing_j_anchor`.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `completeness.py` with `ruleset_version`; document each rule's domain-contract
  justification.
- [ ] **Step 4:** Run — expect PASS. Commit.

---

## Phase 3 — Stage 2: positive-only membership score

### Task 5: Penultimate V/J branch-feature extraction

**Files:**
- Create: `src/alignair/aa/product/applicability/embedding.py`
- Modify: `src/alignair/aa/model.py` (add a `penultimate(...)` branch method or expose the pre-`cls_head`
  activation via a hook — **no behavior change** to existing outputs)
- Test: `tests/alignair/aa/product/applicability/test_embedding.py`

**Interfaces:**
- Produces: `branch_features(predictor, sequences, branches=("V","J")) -> np.ndarray [n, d]` — the
  **penultimate** `silu(cls_mid(cls_tower(emb*mask)))` per branch, **predeclared concatenation** of V and J,
  pooled over the segmentation-masked region. **Not** the raw `TokenAndPositionEmbedding` (which would
  measure composition/position, not a learned VH representation).

- [ ] **Step 1:** Failing test: `branch_features` returns a fixed-width `[n, d]` array, deterministic; a
  V-branch-only vs V+J concatenation differ in width as declared; existing `model.forward` outputs are
  byte-identical before/after the hook/method addition.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement the penultimate extraction (reuse the branch's `cls_tower`/`cls_mid`; hook or
  method) + pooling over the masked segment region.
- [ ] **Step 4:** Run — expect PASS. Commit.

### Task 6: Positive-only membership scorer + representation/scorer selection

**Files:**
- Create: `src/alignair/aa/product/applicability/membership.py`, `scripts/aa/gate3/select_representation.py`
- Test: `tests/alignair/aa/product/applicability/test_membership.py`

**Interfaces:**
- Produces: `MembershipScorer.fit(in_domain_features)`, `.score(features) -> np.ndarray` (higher = more
  in-domain), `.to_bytes()/.from_bytes()` (pickle-free). Fit on **in-domain human VH only**.
  `select_representation(validation)` picks representation (V vs V+J concat) × scorer (kNN vs shrinkage
  Mahalanobis) by **validation** FAR/FRR under a **predeclared tie-break** (favor lower FAR upper bound, then
  simpler scorer).

- [ ] **Step 1:** Failing tests: fit on in-domain fixture features; in-domain holdout scores strictly higher
  than injected far-OOD; `from_bytes(to_bytes())` round-trips exactly; `to_bytes` contains **no pickle
  opcodes**; `select_representation` is deterministic and uses **only validation data** (assert it never
  reads the test split).
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `membership.py` (kNN + shrinkage Mahalanobis; NumPy via safetensors/buffer, no
  pickle) and `select_representation.py`.
- [ ] **Step 4:** Run — expect PASS. Commit.

---

## Phase 4 — Output contract & integration

### Task 7: Discriminated-union verdict schema (3 states emitted)

**Files:**
- Create: `src/alignair/aa/product/applicability/schema.py`
- Test: `tests/alignair/aa/product/applicability/test_applicability_schema.py`

**Interfaces:**
- Produces: `Applicability {state, reason, score, detail, provenance}` with emitted states ∈
  `{supported, insufficient_evidence, unknown_ood}` (`unsupported_locus` defined but **reserved** — asserting
  it is emitted raises `NotImplementedError` in v1); `assemble_result(applicability, segments_or_none)`
  enforcing the union — `supported` requires `segments`; any other state **forbids** `segments`/
  `inter_vj_aa`.

- [ ] **Step 1:** Failing tests: `supported` without segments raises; non-`supported` **with** segments
  raises; emitting `unsupported_locus` in v1 raises `NotImplementedError`; a valid non-supported result has
  no `segments`/`inter_vj_aa` key.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `schema.py`.
- [ ] **Step 4:** Run — expect PASS. Commit.

### Task 8: Two-stage detector + predictor integration (feature-flagged)

**Files:**
- Create: `src/alignair/aa/product/applicability/detector.py`
- Modify: `src/alignair/aa/product/predictor.py`
- Test: `.../test_detector.py`, `.../test_predict_withholds_calls.py`

**Interfaces:**
- Produces: `applicability_verdict(predictor, sequence) -> Applicability`; `predict(..., applicability=True)`
  returns the discriminated union. Stage-1 fail → `insufficient_evidence` (specific reason); Stage-1 pass +
  Stage-2 reject → **`unknown_ood`** (never `unsupported_locus` in v1); both pass → `supported`.

- [ ] **Step 1:** Failing tests: complete VH → `supported` + segments; reversed VH →
  `insufficient_evidence`, **no segments**; far-OOD → `unknown_ood`, no segments; flag default (off)
  preserves today's behavior exactly.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `detector.py` + wire into `predict` behind `applicability=False` default.
- [ ] **Step 4:** Run — expect PASS; run the full AA product suite (flag off → no regression). Commit.

---

## Phase 5 — Validation (the pass/fail decision)

### Task 9: Threshold selection on validation (with FRR-stop)

**Files:**
- Create: `scripts/aa/gate3/select_threshold.py`
- Test: `tests/alignair/aa/product/applicability/test_threshold_selection.py`

**Interfaces:**
- Consumes: frozen splits, `applicability_verdict`, the selected representation/scorer.
- Produces: an operating threshold minimizing the **FAR 95% upper bound** on **validation** subject to the
  FRR bound; records `score_direction`. **If no threshold satisfies FRR ≤ 10% (upper bound) on validation →
  STOP before the locked test** (Global Constraint).

- [ ] **Step 1:** Failing test: selects the lowest-FAR-upper-bound threshold among those with FRR upper bound
  ≤ 10%; returns a **STOP** sentinel when no threshold meets FRR; never reads the test split.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `select_threshold.py` (Clopper–Pearson; FAR-priority; FRR-stop).
- [ ] **Step 4:** Run — expect PASS. Select the real threshold on validation; if STOP, **halt and report**.
  Commit.

### Task 10: Locked test evaluation + Gate-3 decision (frozen artifact)

**Files:**
- Create: `scripts/aa/gate3/evaluate.py`, `docs/superpowers/artifacts/2026-07-24-aa-gate3/validation.md` + `.json`
- Test: `tests/alignair/aa/product/applicability/test_evaluate.py`

**Interfaces:**
- Consumes: locked threshold, frozen **test** split.
- Produces: per-group FAR **upper bounds** (combined IGK/IGL, combined TCR, VHH/non-human, IgSF) with IGK/IGL
  and TRA/B/G/D as **descriptive strata**; FRR (real vs sim); **category-held-out** FAR per critical group
  (positive-only → held-out by construction, reported, **not** LOCO-retrain); a **3-state** reason-state
  confusion matrix; the §9 **PASS/FAIL** decision.

- [ ] **Step 1:** Failing test: computes exact 95% upper bounds per group; returns `PASS` only if **every**
  gate is met (combined near-OOD/IGK-IGL/TCR ≤ 5%, VHH-non-human/IgSF ≤ 10%, FRR real & sim ≤ 10%) **and**
  every critical group's category-held-out FAR bound holds; otherwise `FAIL`; per-locus strata are reported
  but not gated.
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `evaluate.py`.
- [ ] **Step 4:** Run — expect PASS (harness correctness).
- [ ] **Step 5:** Run the **locked, one-shot** test eval; write frozen `validation.md`/`.json`.
- [ ] **Step 6:** **GATE-3 DECISION.** `PASS` → proceed to packaging (Task 11). `FAIL` → record it; **Gate 3
  does not pass, product not released**; do not weaken any threshold. Commit the artifact.

---

## Phase 6 — Packaging AFTER locked PASS (non-circular)

### Task 11: Final detector artifact + provenance + safe-load

**Files:**
- Create: `src/alignair/aa/product/applicability/provenance.py`
- Modify: `src/alignair/aa/product/bundle.py`, `loader.py`
- Test: `tests/alignair/aa/product/applicability/test_provenance.py`

**Interfaces:**
- Produces (only after Task 10 = PASS): a detector bundle **section** (pickle-free, sha256-verified) carrying
  scorer bytes + `ruleset_version` + `threshold` + `score_direction` + **`validation_artifact_sha256`** (the
  frozen Task-10 record); `applicability.provenance` populated with
  `detector_id/version/artifact_sha256/threshold/score_direction/ruleset_version/validation_artifact_sha256`;
  load-time verification that a verdict's provenance matches the loaded section (mismatch = hard load error).

- [ ] **Step 1:** Failing tests: the detector section round-trips with checksum verification; a tampered
  section fails load (hard error); the packaged `validation_artifact_sha256` equals the frozen Task-10
  artifact hash; every emitted verdict carries complete provenance. *(Early pickle-free serialization was
  already tested in Task 6; this is the final, validation-bound packaging.)*
- [ ] **Step 2:** Run — expect FAIL.
- [ ] **Step 3:** Implement `provenance.py` + bundle/loader wiring (reuse `container` verified-section path).
- [ ] **Step 4:** Run — expect PASS. Commit.

---

## Self-review notes (per writing-plans)

- **Powered locked test, not total cohort** (amendment 1): Global Constraints + Tasks 2/3 gate the **test**
  allocation after dedup+split; fit/val are additional.
- **Combined vs locus claims** (amendment 2): ≤ 5% gates on combined IGK/IGL and combined TCR; per-locus
  strata descriptive (Tasks 2/10).
- **No vacuous LOCO** (amendment 3): positive-only → **category-held-out reporting** (Task 10); LOCO-retrain
  reserved for an optional supervised classifier.
- **Learned representation** (amendment 4): penultimate V/J branch features, not pooled token embedding;
  representation+scorer chosen on **validation** (Tasks 5/6).
- **No invented locus** (amendment 5): v1 emits 3 states; `unsupported_locus` reserved (Tasks 7/8).
- **Non-circular packaging** (amendment 6): final packaging is Phase 6, after locked PASS; early serialization
  tested in Task 6.
- **FRR-threshold stop** (added rule): Task 9 halts before the locked test if no threshold meets FRR.
- **Feasibility precedes construction:** Phase 0 gates the plan; Tasks 3–11 contingent on GO.
