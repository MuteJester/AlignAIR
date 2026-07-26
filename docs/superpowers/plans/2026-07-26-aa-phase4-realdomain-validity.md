# Phase 4 — Real-Domain Result Validity & Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Prove (or refute), with a predeclared confidence-bounded contract, that AA AlignAIR v2's V/D/J
results and calibrated sets can be trusted on manually-admitted complete human IGH amino-acid sequences —
across a construction-truth Gold tier (simulator) and a real nucleotide→AA concordance tier, never pooled.

**Architecture:** Feasibility gate first (real cohort must be *powered* for a coverage non-inferiority
test, else STOP). Then: freeze/hash v2 + generate Gold; acquire real, translate, establish nt-consensus
concordance, leakage-split into trigger/calibration/test with source-balanced (A) and source-held-out (B)
cohorts; evaluate both tiers with Clopper–Pearson (binary) + lineage-cluster bootstrap (aggregate); make the
V-or-J recalibration decision; open the locked test once; comparative baselines without circularity; emit
the field-evidence matrix.

**Tech Stack:** Python 3.12, existing `alignair` AA product, GenAIRR, IgBLAST (nt + protein), nucleotide
AlignAIR, parasail (protein local align), NumPy/SciPy (Clopper–Pearson, bootstrap, exact power), the
`alignair.aa.conformal` module.

**Spec:** [Phase-4 spec](../specs/2026-07-26-aa-phase4-realdomain-validity-spec.md). Where this plan is more
specific, the spec's approved text governs; do not re-decide settled items.

## Global Constraints

- **Feasibility is a HARD STOP.** Phase 0 must produce all §11 outputs; if the real cohort is underpowered
  for the H0:coverage≤0.93 vs ~0.95 test (per V and J, per binding stratum, across calibration+each test),
  **STOP and report — never substitute the "≈59" shortcut.**
- **Real concordance is NOT truth.** Real metrics are `top_call_concordance`, `reference_call_inclusion`,
  `boundary_concordance` — never "accuracy"/"true coverage". Report the concordance rate over ALL eligible.
- **Three frozen real partitions, no reuse:** trigger/selection · calibration · **test (opened once)**.
- **Two endpoints:** (A) source-balanced donor/lineage-disjoint = primary exchangeable conformal audit;
  (B) source-held-out = transfer stress, DESCRIPTIVE only (no conformal guarantee).
- **Recalibration trigger:** V **OR** J calibrated-set coverage 95% LB < 0.93, on the trigger partition, for
  the **default `coverage_group=None` (pooled)** behavior; `full` reported separately.
- **Stats:** binary endpoints → one-sided exact Clopper–Pearson; aggregate/continuous (macro, NLL, Brier,
  MAE, percentiles, risk–coverage) → **lineage/donor-cluster bootstrap** (resample the cluster, not the seq).
- **Gene/allele success = inclusion in the TOP CLASS's one-to-many ASC→IMGT expansion** — never an arbitrary
  representative. **`softmax_score` NLL/Brier/reliability on the COMPLETE class softmax vector, class
  resolution only**; never renamed "probability".
- **Gold is simulator-domain / weak independence**, scoped to (1) procedure/regression, (2) set + expansion
  mechanics, (3) D construction-truth. Gold V/J accuracy is a labelled regression result, never a real claim.
- **D stays experimental**; observable D residues computed from CONSTRUCTION truth, never predicted boundaries.
- **v2 (weights + `calibration.json`) sha256-pinned BEFORE acquisition.** Nothing tunes the model on any real
  partition. Locked test opened exactly once.
- **Data policy:** raw sequences git-ignored; commit only hashes/counts/fingerprints/metrics. No clone/donor/
  therapeutic-lineage/near-identical-VH crosses any two partitions. No Claude/Co-Authored-By in commits.

## File Structure

- `src/alignair/aa/phase4/` — `eligibility.py` (real-record eligibility), `concordance.py` (nt-consensus),
  `projection.py` (nt→AA coordinate projection), `power.py` (exact prospective power + CP + bootstrap),
  `metrics.py` (concordance/inclusion/set-size/reliability/selective-risk/macro), `partitions.py` (leakage
  split + fingerprints).
- `scripts/aa/phase4/` — recon/feasibility, hydrate, split-freeze, evaluate, calibrate-decide, compare,
  matrix.
- `.private/aa-phase4/` — git-ignored hydration + partition stores.
- `docs/superpowers/artifacts/2026-07-26-aa-phase4/` — feasibility, fingerprints, evaluation, calibration
  decision, comparative, PHASE4_RESULT (all counts/hashes/metrics, no sequences).
- `tests/alignair/aa/phase4/`.

---

## Phase 0 — Feasibility (HARD GATE; STOP if underpowered)

### Task 1: Real-record eligibility + source recon manifest
**Files:** create `src/alignair/aa/phase4/eligibility.py`, `scripts/aa/phase4/recon.py`,
`docs/.../aa-phase4/sources.md`; test `tests/alignair/aa/phase4/test_eligibility.py`.
**Interfaces:** produces `is_eligible(record) -> bool` (productive, `complete_vdj`, human `IGH`, valid
translate-able VDJ) and a documented source manifest (OAS units + AIRR ADC nodes, donor/study metadata,
licenses).
- [ ] Step 1: Enumerate OAS + AIRR ADC sources with donor/study metadata; write `sources.md`.
- [ ] Step 2: Failing test: `is_eligible` accepts a productive complete human IGH record, rejects
  non-productive / incomplete_vdj / non-IGH / untranslatable.
- [ ] Step 3: Run → FAIL. Step 4: Implement `eligibility.py`. Step 5: Run → PASS. Step 6: Commit.

### Task 2: Concordance pipeline (nt consensus) + eligible/concordance rates
**Files:** create `src/alignair/aa/phase4/concordance.py`, `scripts/aa/phase4/concordance_stats.py`;
test `.../test_concordance.py`.
**Interfaces:** `consensus(record) -> {agree: bool, call, resolution}` from IgBLASTn + nucleotide AlignAIR
on the source DNA; `concordance_stats(records)` → eligible-before-filtering, concordant/discordant counts.
- [ ] Step 1: Failing test: two nt calls agreeing → `agree=True` at the stated resolution; disagreeing →
  `agree=False`; the stats report the denominator (ALL eligible) and the concordance rate.
- [ ] Step 2: Run → FAIL. Step 3: Implement (invoke IgBLASTn + nt AlignAIR; compare at gene resolution,
  allele where both supply it). Step 4: Run → PASS. Step 5: Commit.

### Task 3: Exact power + STOP/GO
**Files:** create `src/alignair/aa/phase4/power.py`, `scripts/aa/phase4/feasibility.py`,
`docs/.../aa-phase4/feasibility.{json,md}`; test `.../test_power.py`.
**Interfaces:** `min_n_coverage_noninferiority(margin_lb=0.93, expected=0.95, alpha=0.05, power=0.8) -> int`
(exact, NOT the zero-failure shortcut); `feasibility_report(...)` → per §11 outputs + `GO`/`STOP`.
- [ ] Step 1: Failing test: `min_n_coverage_noninferiority` returns **hundreds** (not 59); zero-error
  bound and coverage-non-inferiority power are computed by different functions and differ.
- [ ] Step 2: Run → FAIL. Step 3: Implement exact power (binomial power for H0 coverage≤0.93 vs 0.95) +
  the feasibility report (eligible-before-filter, concordance/discordance rate, independent-lineage counts
  per trigger/calibration/test, exact power per V and J and binding strata, distinct-source counts for A+B).
- [ ] Step 4: Run → PASS. Step 5: Run the REAL feasibility on hydrated recon.
- [ ] Step 6: **DECISION GATE.** If any §11 output fails (underpowered lineages, too few sources for A+B,
  concordance collapses the cohort), **STOP and report.** Do not proceed. If `GO`, commit and continue.

---

## Phase 1 — Freeze v2 + Gold (only after GO)

### Task 4: Pin v2 + generate frozen Gold
**Files:** create `scripts/aa/phase4/freeze_v2.py`, `scripts/aa/phase4/gen_gold.py`,
`docs/.../aa-phase4/v2_pin.json`, `docs/.../aa-phase4/gold_fingerprint.json`; test `.../test_gold.py`.
- [ ] Step 1: Failing test: `v2_pin.json` records the weights + `calibration.json` sha256; Gold records
  carry construction-truth V/D/J + boundaries + SHM/indel/truncation strata; Gold fingerprint reproducible.
- [ ] Step 2: Run → FAIL. Step 3: Implement pin + GenAIRR held-out Gold generation (git-ignored store,
  committed fingerprint). Step 4: Run → PASS. Step 5: Commit.

---

## Phase 2 — Real assembly + coordinate projection + leakage split

### Task 5: nt→AA coordinate projection (deterministic, tested first)
**Files:** create `src/alignair/aa/phase4/projection.py`; test `.../test_projection.py`.
**Interfaces:** `project_nt_to_aa(nt_start_1based_closed, nt_end, frame, cds_start) -> (aa_start_0based,
aa_end_halfopen)` handling reading-frame offset + non-codon-aligned boundaries by a stated rule.
- [ ] Step 1: Failing tests: a codon-aligned boundary projects exactly; a mid-codon boundary maps to the
  defined AA index; 1-based-closed→0-based-half-open is exact on worked examples.
- [ ] Step 2: Run → FAIL. Step 3: Implement. Step 4: Run → PASS. Step 5: Commit.

### Task 6: Hydrate real, translate, concordance-filter
**Files:** create `scripts/aa/phase4/hydrate.py`; test `.../test_hydrate_smoke.py`.
- [ ] Step 1: Failing smoke test: hydration yields AA + retained DNA + nt-consensus label + `R-concordant`/
  `R-discordant` tag; all eligibility + concordance provenance recorded.
- [ ] Step 2: Run → FAIL. Step 3: Implement (git-ignored store). Step 4: Run → PASS. Step 5: Commit.

### Task 7: Leakage-controlled partition freeze (trigger/calibration/test; A + B)
**Files:** create `src/alignair/aa/phase4/partitions.py`, `scripts/aa/phase4/split_freeze.py`,
`docs/.../aa-phase4/partition_fingerprint.json`; test `.../test_partitions.py`.
**Interfaces:** `build_partitions(records, seed)` → trigger / calibration / test, with a **source-balanced
donor/lineage-disjoint** test (A) and a **source-held-out** test (B); enforce no clone/donor/lineage/
near-identical-VH crosses partitions.
- [ ] Step 1: Failing test: partitions disjoint by donor/study/clonal-lineage/near-identical-VH; (A) is
  source-balanced, (B) holds out whole sources; fingerprint = counts + lineage-set hashes (no sequences).
- [ ] Step 2: Run → FAIL. Step 3: Implement + assert disjointness. Step 4: Run → PASS. Freeze fingerprint;
  commit.

---

## Phase 3 — Evaluation (both tiers; test NOT opened yet — uses Gold + non-test real)

### Task 8: Per-head metrics (class/gene/allele, coverage/inclusion, sets, reliability, selective, macro)
**Files:** create `src/alignair/aa/phase4/metrics.py`, `scripts/aa/phase4/evaluate_heads.py`; test
`.../test_metrics.py`.
**Interfaces:** `top_call(kind, ...)` (Gold accuracy / Real concordance); `set_coverage` (Gold true-class /
Real reference-inclusion); `expansion_inclusion`; `set_sizes`; `reliability(full_softmax, class_truth)`
(NLL/Brier + reliability bins, **class resolution**); `selective_risk`; `macro`. CP for binary, cluster
bootstrap for aggregate.
- [ ] Step 1: Failing tests: gene/allele success requires inclusion in the top-class expansion (a
  multi-gene class passes iff truth ∈ its genes); reliability uses the full softmax vector at class
  resolution; binary → CP, macro/NLL → cluster bootstrap; Real metrics carry `concordance`/`inclusion` names.
- [ ] Step 2: Run → FAIL. Step 3: Implement. Step 4: Run → PASS. Step 5: Run on Gold + non-test real
  (trigger/calibration partitions); write metrics artifact. Commit.

### Task 9: Boundary metrics (AA-residue, stratified)
**Files:** create `scripts/aa/phase4/evaluate_boundaries.py`; test `.../test_boundaries.py`.
- [ ] Step 1: Failing test: `boundary_concordance` (Real, via projection) and `boundary_accuracy` (Gold)
  give AA-residue MAE/p95 with cluster bootstrap, stratified by truncation/SHM/indel/family; `inter_vj_aa`
  checked only as the V-end→J-start substring (no junction claim).
- [ ] Step 2: Run → FAIL. Step 3: Implement. Step 4: Run → PASS. Commit.

### Task 10: D on Gold (construction-truth; stays experimental)
**Files:** create `scripts/aa/phase4/evaluate_d.py`; test `.../test_d.py`.
- [ ] Step 1: Failing test: D allele/gene evaluated ONLY on Gold; observable-residue strata computed from
  construction truth (not predicted boundaries); D-set coverage/width reported; artifact labels D experimental.
- [ ] Step 2: Run → FAIL. Step 3: Implement. Step 4: Run → PASS. Commit.

---

## Phase 4 — Calibration decision + the one-shot locked test

### Task 11: Trigger-partition calibration decision (V OR J; pooled default + full)
**Files:** create `scripts/aa/phase4/calibration_decide.py`; test `.../test_calibration_decide.py`.
- [ ] Step 1: Failing test: on the trigger partition, evaluate the UNCHANGED simulator calibration for V
  and J at `coverage_group=None` (default) and `full` separately; decision = recalibrate iff V or J coverage
  95% LB < 0.93; if triggered, fit real-domain conformal quantiles on the CALIBRATION partition only.
- [ ] Step 2: Run → FAIL. Step 3: Implement (reuse `alignair.aa.conformal`). Step 4: Run → PASS; write the
  decision artifact. Commit.

### Task 12: Open the locked test ONCE — both calibrations, endpoints A + B
**Files:** create `scripts/aa/phase4/locked_eval.py`, `docs/.../aa-phase4/locked_eval.{json,md}`; test
`.../test_locked_eval.py`.
- [ ] Step 1: Failing test: the harness refuses to run twice (one-shot audit); evaluates simulator +
  (if fitted) real calibration side by side; endpoint (A) source-balanced = the primary conformal coverage
  (with 95% LB vs the contract); endpoint (B) source-held-out = descriptive transfer only (no guarantee text).
- [ ] Step 2: Run → FAIL. Step 3: Implement one-shot seal + eval. Step 4: Run the single locked evaluation;
  write frozen artifact. Commit.

---

## Phase 5 — Comparative (no circularity) + release matrix

### Task 13: Comparative baselines
**Files:** create `scripts/aa/phase4/compare.py`; test `.../test_compare.py`.
- [ ] Step 1: Failing test: genuine accuracy comparison (AA v2 vs protein local-align vs IgBLAST protein)
  runs on **Gold only**; method-to-method agreement runs on **UNFILTERED eligible** real; AA-vs-consensus
  concordance runs on **R-concordant only**; no reference generator is scored on its own selected truth.
- [ ] Step 2: Run → FAIL. Step 3: Implement (parasail protein SW vs the identical ASC→IMGT reference;
  IgBLAST protein; nt references for concordance). Step 4: Run → PASS. Commit.

### Task 14: Field-evidence matrix + frozen Phase-4 result
**Files:** create `scripts/aa/phase4/matrix.py`, `docs/.../aa-phase4/PHASE4_RESULT.md`; test `.../test_matrix.py`.
- [ ] Step 1: Failing test: the matrix maps each output (V/J top, calibrated sets, D, coordinates,
  softmax_score, inter_vj_aa) to its assembled evidence with the release verdict per row; Real rows use
  concordance/inclusion language; D row = experimental; softmax row = reliability (not probability).
- [ ] Step 2: Run → FAIL. Step 3: Implement + write PHASE4_RESULT.md. Step 4: Run → PASS. Commit.

---

## Self-review (per writing-plans)
- **Feasibility precedes everything:** Phase 0 gates Tasks 4–14; the ≈59 shortcut is explicitly banned; the
  five §11 outputs are the STOP criteria.
- **No calibration-on-test reuse:** trigger/calibration/test are separate; the locked test is one-shot (Task 12).
- **No circular comparison / no truth-generator-vs-own-truth** (Task 13); Real language is concordance/inclusion.
- **Stats correct:** CP for binary, cluster bootstrap for aggregate; gene/allele = top-class-expansion
  inclusion; softmax reliability at class resolution.
- **Gold scoped** to procedure/mechanics/D; D experimental; observable residues from construction truth.
