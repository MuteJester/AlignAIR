# AA Applicability Model Implementation Plan

> **Status:** APPROVED FOR EXECUTION · **Reviewed:** 2026-07-26
> **Execution:** work task-by-task with an isolated implementer and independent review for reversible
> code tasks; keep hydration, training, freeze, and one-shot unlock under primary-agent supervision.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify a separately-trained applicability guard that decides, before annotation, whether an input is a complete human IGH VH amino-acid domain — passing the original Gate-3 release contract on a fresh cohort and the still-sealed Gate-3 one-shot locked test, with the frozen v2 annotator untouched.

**Architecture:** A hybrid guard = competitive profile-HMM evidence bank (pyhmmer) + a dedicated contextual encoder (completeness / domain-type / applicability / open-set-energy heads) fused before the heads, with a conjunctive accept rule and a carried-forward frozen structural prefilter as first authority. Five predeclared candidates C0–C4 plus per-category LOCO folds are trained on the burned Gate-3 development cohort, selected under a winner's-curse guardband on a fresh cohort, frozen+hashed, and evaluated exactly once on the sealed test. Every stage has an explicit GO/STOP.

**Tech Stack:** Python 3.12; `pyhmmer` 0.12.1 (vendored HMMER, profile bank); `anarcii` 2.0.8 (numbering comparator/teacher only); `torch` 2.11 (+cu128); `safetensors`; GenAIRR (`HUMAN_IGH_OGRDB`) for simulator positives; the existing `src/alignair/aa/product/applicability/` Gate-3 machinery (canonical/cohort/feasibility/roles/scorers/completeness/embedding, reused); the frozen `alignair-aa-igh-human@2.0.0` annotator.

## Global Constraints

Every task's requirements implicitly include this section. Values are binding and copied from the approved spec `docs/superpowers/specs/2026-07-26-aa-applicability-model-design.md`.

- **Frozen annotator:** AA v2 V/D/J weights, calibration, class maps, reference bridge, tokenizer, and outputs are byte-for-byte immutable. The guard is a **separate** model + bundle section. A guard failure or absence can **never** fall through to unguarded public annotation.
- **Binding release gates (one-sided exact 95% upper bounds):** Overall near-OOD FAR ≤5%; IGK/IGL FAR ≤5%; TCR FAR ≤5%; VHH/non-human VH FAR ≤10%; IgSF FAR ≤10%; Structural FAR (parent-lineage aggregated) ≤10%; FRR real complete human IGH ≤10%; FRR simulator complete human IGH ≤10%. Zero-error release floors: **59** (5% gate), **29** (10% gate).
- **Pre-unlock winner's-curse guardband:** selection AND blind-LOCO upper bounds must be ≤80% of the release limit (≤4% for a 5% gate; ≤8% for a 10% gate). Zero-error guardband planning floors: **74** (5% gate), **36** (10% gate). Applies to selection and every blind-LOCO fold; neither may compensate for the other (no averaging/pooling).
- **Cohort disjointness:** fresh selection and blind-LOCO cohorts are disjoint and **never pooled**. Both are disjoint from the burned Gate-3 dev cohort, the 16 Phase-3 therapeutics (INN + VH hash), and the sealed locked test by sequence hash / lineage / donor / study-source. Locked disjointness is established without touching the seal: derive a conservative denylist from the complete pre-split Gate-3 `cleaned_cohort.jsonl`, first reproducing the committed global canonical digest. Because that pre-split corpus contains fit, validation, and locked records, excluding all of it excludes the locked records without revealing their assignment. **Global sequence deduplication precedes allocation:** all hydrated fresh records are deduplicated by sequence sha256 across every source (and against the denylist) BEFORE selection/blind-LOCO allocation, so an identical sequence appearing in two sources cannot double-count toward a floor or straddle both subsets.
- **Far-OOD is descriptive-only:** generic non-antibody, de-novo unsupported, and arbitrary far-OOD cohorts are scored and reported but **cannot** tune thresholds, determine candidate eligibility, or break selection ties.
- **VHH (AMENDED 2026-07-26, genus-only):** VHH uses a **genus holdout** — selection = *L. glama*, blind-LOCO challenge = non-*L. glama* camelids (vicugna / camelus). Deposition date is only a soft ordering preference within the challenge; a **hard temporal cutoff is NOT used** because the pre-split Gate-3 VHH span the latest depositions, leaving no post-cutoff window (Stage-1 empirical finding: 537 *L. glama* / 501 non-*L. glama* seal-disjoint distinct-lineage VHH available; 1,038 total). The **single-source (SAbDab) limitation is retained and stated** in the model card — genus/temporal holdout within one database, not an independent second source. Fallback to deterministic sha-rank halving only if a genus side cannot reach 19, and the mode is recorded. **All 19 sealed VHH lineages** are retained in the final locked evaluation (no attrition/cherry-picking). VHH is binding only through the combined `vhh_nonhuman` group; VHH-specific FAR is descriptive.
- **Combined-locus balancing:** each combined 5%-gate group is balanced across its constituent loci so no single locus dominates the gate — target **37 IGK + 37 IGL** in each 74-record IGK/IGL subset, and **37 TRA + 37 TRB** in each 74-record TCR subset (per-locus floor = ⌈guardband_floor / #loci⌉ = 37). TRG/TRD remain descriptive (unfloored).
- **Profile source (resolve BEFORE production-bank construction and candidate training):** use a redistribution-compatible, versioned source from the start; IMGT-derived profiles are **local research comparators only** until redistribution is explicitly cleared. User-side hydration is a **distinct exact-bank product path**: it must deterministically reproduce the same pinned bank used for training/selection, bind its checksum at load, and pass usability/offline-after-hydration/integrity tests. No post-validation bank substitution is allowed. If no production-admissible path exists, **STOP in Stage 2**.
- **Sealed locked test:** opened exactly once, after architecture, all main + LOCO weights, decision logic, category map, and every threshold are frozen and hashed. A complete **mock one-shot** against a synthetic sealed fixture must pass first. No rerun with changed code/model/thresholds.
- **Freeze + hash:** preprocessing, profile construction, all models (main + LOCO), thresholds, decision graph, category map, evaluation code, exact guarded result schema/API path, final bundle, and repository source-tree commit.
- **Budget:** compute/time budget is accounted explicitly for **every** main and LOCO candidate (5 main + 4 learned-LOCO folds per learned candidate), not just the five main models.
- **Packaging/product path:** pickle-free (`safetensors`, `allow_pickle=False`, non-pickle profile format); the exact guarded schema/API path and clean-environment load are proven **before** final freeze and unlock. Rejected inputs are tested to make V/D/J and `inter_vj_aa` structurally unrepresentable. Only documentation, release enablement, and publication happen after a PASS—no inference behavior changes.
- **Seed policy:** architecture/loss hyperparameters and a single production seed are chosen deterministically on burned development data before fresh validation. C1–C4 each receive at least three burned-development seed replicates to measure instability; the predeclared dev-only rule chooses the production seed, and the corresponding seed is used for that candidate’s LOCO folds. Fresh validation never selects a seed.
- **Repo hygiene:** never commit `.pt` / `.alignair` / detector binaries or raw sequences — these live git-ignored under `.private/aa-applicability/`. Artifacts committed under `docs/superpowers/artifacts/2026-07-26-aa-applicability/` are counts/hashes/provenance only. No Claude / Co-Authored-By line in commits.
- **Every stage ends with an explicit `GO` / `STOP` record** written to the artifacts directory.

## File Structure

New production guard package (distinct from the reused Gate-3 dev machinery in `product/applicability/`):

- `src/alignair/aa/guard/__init__.py` — public surface (`load_guard`, `Guard`, `GuardResult`).
- `src/alignair/aa/guard/profiles.py` — paired V/J (or validated full-domain) profile-bank build/load/search (pyhmmer), redistribution manifest, deterministic checksums.
- `src/alignair/aa/guard/features.py` — fixed HMM feature-vector extraction from search results.
- `src/alignair/aa/guard/tokenizer.py` — 23-token AA alphabet + learned `[CLS]`.
- `src/alignair/aa/guard/encoder.py` — `ApplicabilityEncoder` (4-block pre-norm transformer, rotary, padding-aware).
- `src/alignair/aa/guard/heads.py` — completeness / domain-type / applicability / energy heads.
- `src/alignair/aa/guard/fusion.py` — sequence-rep ⊕ normalized HMM features, stored normalization stats.
- `src/alignair/aa/guard/model.py` — candidate assembly C0–C4 + forward.
- `src/alignair/aa/guard/losses.py` — the six loss terms.
- `src/alignair/aa/guard/decision.py` — conjunctive decision graph, reason states, prefilter-as-first-authority.
- `src/alignair/aa/guard/thresholds.py` — threshold grid, exact bounds, guardband selection.
- `src/alignair/aa/guard/freeze.py` — freeze+hash manifest over all frozen artifacts.
- `src/alignair/aa/guard/bundle.py` — pickle-free packaging + safe loader + integrity hard-fails.
- Reused unchanged: `src/alignair/aa/product/applicability/{canonical,cohort,cohort_loader,completeness,feasibility,roles,scorers,embedding}.py`.
- Existing product path modified before unlock: `src/alignair/aa/product/{schema,predictor,loader,bundle,cli}.py`.
- Scripts: `scripts/aa/applicability/{hydrate_fresh.py,audit_disjoint.py,build_profiles.py,train_candidate.py,run_loco.py,select.py,freeze_all.py,mock_oneshot.py,unlock.py,package.py,budget.py}`.
- Tests: `tests/alignair/aa/guard/test_*.py`.
- Private data (git-ignored): `.private/aa-applicability/{fresh_hydration,profiles,checkpoints,mock_seal}/`.
- Committed artifacts: `docs/superpowers/artifacts/2026-07-26-aa-applicability/*.json|md` (counts/hashes/provenance only; `PHASE0_FEASIBILITY.md`, `tooling_feasibility.json`, `locked_category_map.json` already present).

---

## Stage 1 — Fresh-cohort hydration + disjointness audit

**Stage GO/STOP:** GO iff every binding group's fresh **selection** and **blind-LOCO** subsets independently meet the guardband floors (74 for 5% gates, 36 for 10% gates), each carries ≥19 VHH inside `vhh_nonhuman`, and the disjointness audit is clean. Else `STOP_FRESH_COHORT_UNPOWERED`.

### Task 1.1: Fresh-cohort manifest with VHH genus/temporal holdout fields

**Files:**
- Create: `src/alignair/aa/guard/fresh_manifest.py`
- Test: `tests/alignair/aa/guard/test_fresh_manifest.py`

**Interfaces:**
- Produces: `FRESH_MANIFEST: list[FreshCategorySpec]` where `FreshCategorySpec` extends the Gate-3
  `CategorySpec` with `role_split: str` (`"selection" | "loco_challenge"` allocation policy marker),
  `vhh_genus_holdout: dict|None`, and `vhh_temporal_cutoff_policy: str|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/guard/test_fresh_manifest.py
from alignair.aa.guard.fresh_manifest import FRESH_MANIFEST, FreshCategorySpec

def test_manifest_covers_all_binding_groups():
    groups = {c.group for c in FRESH_MANIFEST}
    assert {"igk_igl", "tcr", "vhh_nonhuman", "igsf", "structural",
            "positive_real", "positive_sim"} <= groups

def test_vhh_declares_genus_and_temporal_holdout():
    vhh = [c for c in FRESH_MANIFEST if c.key == "vhh"]
    assert vhh, "vhh category present"
    v = vhh[0]
    assert v.vhh_genus_holdout is not None and "selection" in v.vhh_genus_holdout
    assert "pre-split Gate-3 VHH" in v.vhh_temporal_cutoff_policy
    assert "SAbDab" in v.source and "single-source" in v.note.lower()

def test_all_specs_are_frozen_dataclasses():
    for c in FRESH_MANIFEST:
        assert isinstance(c, FreshCategorySpec)
        assert c.order == "N->C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alignair/aa/guard/test_fresh_manifest.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

Create `FreshCategorySpec` as a frozen dataclass subclassing the Gate-3 `CategorySpec` semantics (copy the
fields, add `role_split`, `vhh_genus_holdout`, `vhh_temporal_cutoff_policy`). Populate `FRESH_MANIFEST`
from the Gate-3 `MANIFEST` sources, but:
- VHH: `source="SAbDab all-single-domain-summary (camelid)"`,
  `vhh_genus_holdout={"selection": ["lama glama"], "loco_challenge": ["vicugna pacos", "camelus
  dromedarius", "camelus bactrianus"]}`, and
  `vhh_temporal_cutoff_policy="latest deposition date among the verified pre-split Gate-3 VHH records"`.
  Selection uses seal-disjoint *L. glama* records at or before that date; blind-LOCO challenge uses
  seal-disjoint non-*L. glama* records deposited after it. The cutoff is derived from burned source
  metadata before model scoring, frozen in the allocation manifest, and never adjusted for performance.
  If either side has fewer than 19 lineages, Stage 1 stops.
- All others reuse the Gate-3 access endpoints (OAS light, VDJServer TRA/TRB, UniProt IPR013106, OAS heavy, GenAIRR, structural transforms).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alignair/aa/guard/test_fresh_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/fresh_manifest.py tests/alignair/aa/guard/test_fresh_manifest.py
git commit -m "feat(aa-guard): fresh-cohort manifest with VHH genus/temporal holdout"
```

### Task 1.2: Disjointness auditor

**Files:**
- Create: `src/alignair/aa/guard/disjoint.py`
- Test: `tests/alignair/aa/guard/test_disjoint.py`

**Interfaces:**
- Produces: `build_gate3_global_denylist(cleaned_cohort, split_fingerprint) -> Gate3Denylist`. It deterministically replays the public Phase-3 exclusion, sequence deduplication, and split assignment needed to reproduce `record_digest.global`; it refuses to return a denylist unless that digest matches the committed fingerprint. The denylist conservatively contains every pre-split sequence hash, lineage, donor, and study.
- Produces: `audit_disjoint(fresh_records, *, gate3_denylist, phase3) -> DisjointReport` with `.clean: bool`, `.overlaps: dict[str, list[str]]` keyed by `{"gate3_seqhash","gate3_lineage","phase3_inn","phase3_vh","donor","study"}`.
- **The implementation never reads `locked_test.jsonl`, calls `unlock_locked_test`, or emits individual locked hashes.**

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/guard/test_disjoint.py
import pytest
from alignair.aa.guard.disjoint import (
    Gate3DigestMismatch, audit_disjoint, build_gate3_global_denylist,
)

def _rec(id_, seq, lineage, donor="d", study="s"):
    return {"id": id_, "aa": seq, "lineage_key": lineage, "donor": donor, "study": study}

def test_flags_sequence_hash_overlap_with_gate3_presplit_denylist():
    import hashlib
    seq = "QVQLVQSGAEVKK"
    h = hashlib.sha256(seq.encode()).hexdigest()
    rep = audit_disjoint([_rec("a", seq, "L1")],
                         gate3_denylist={"sequence_hashes": {h}, "lineages": set(),
                                           "donors": set(), "studies": set()},
                         phase3=({}, {}))
    assert not rep.clean and rep.overlaps["gate3_seqhash"] == ["a"]

def test_clean_when_no_overlap():
    rep = audit_disjoint([_rec("a", "AAAA", "L1")],
                         gate3_denylist={"sequence_hashes": set(), "lineages": set(),
                                           "donors": set(), "studies": set()},
                         phase3=({}, {}))
    assert rep.clean and all(not v for v in rep.overlaps.values())

def test_denylist_requires_global_digest_match(tmp_path):
    # synthetic pre-split corpus + fingerprint fixture
    with pytest.raises(Gate3DigestMismatch):
        build_gate3_global_denylist(tmp_path / "cleaned.jsonl",
                                   {"record_digest": {"global": "wrong"}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alignair/aa/guard/test_disjoint.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

Implement `build_gate3_global_denylist` from the **pre-split** cleaned Gate-3 corpus. Reproduce the committed
global canonical digest using the public freeze procedure before accepting it; then collect a conservative
denylist from all pre-split records. Implement `audit_disjoint`: sha256 each fresh `aa`, compare against that
denylist; match `lineage_key`, donor, and study; match lowercased lineage against Phase-3 INNs and
`sha256(aa)` against Phase-3 VH hashes. Return a frozen `DisjointReport`. Add a source-level test proving
the auditor contains no path or call capable of reading `locked_test.jsonl`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alignair/aa/guard/test_disjoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/disjoint.py tests/alignair/aa/guard/test_disjoint.py
git commit -m "feat(aa-guard): fresh-vs-burned/phase3/locked disjointness auditor"
```

### Task 1.3: Selection vs blind-LOCO allocator (never pooled, floored)

**Files:**
- Create: `src/alignair/aa/guard/allocate.py`
- Test: `tests/alignair/aa/guard/test_allocate.py`

**Interfaces:**
- Consumes: fresh records with `group`, `lineage_key`, `category`, `heavy_species` (VHH), `deposition_date` (VHH).
- Produces: `allocate_fresh(records) -> Allocation` with `.selection`, `.loco_challenge` (disjoint lineage sets per group), `.floors_ok: dict[str,bool]`, `.vhh_min_ok: dict[str,bool]`, and `.pooled_overlap: set` (must be empty).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/guard/test_allocate.py
from alignair.aa.guard.allocate import allocate_fresh, GUARDBAND_FLOOR

def test_selection_and_loco_never_share_a_lineage():
    recs = [{"group": "igk_igl", "category": "igk", "lineage_key": f"L{i}"} for i in range(200)]
    a = allocate_fresh(recs)
    assert not (a.selection["igk_igl"] & a.loco_challenge["igk_igl"])
    assert a.pooled_overlap == set()

def test_guardband_floor_values():
    assert GUARDBAND_FLOOR["gate_5pct"] == 74 and GUARDBAND_FLOOR["gate_10pct"] == 36

def test_vhh_min_19_enforced_in_each_subset():
    recs = ([{"group": "vhh_nonhuman", "category": "vhh", "lineage_key": f"V{i}",
              "heavy_species": "lama glama", "deposition_date": "2020-01-01"} for i in range(50)]
            + [{"group": "vhh_nonhuman", "category": "murine_vh", "lineage_key": f"M{i}"} for i in range(40)])
    a = allocate_fresh(recs)
    assert a.vhh_min_ok["selection"] and a.vhh_min_ok["loco_challenge"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/alignair/aa/guard/test_allocate.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

`GUARDBAND_FLOOR = {"gate_5pct": 74, "gate_10pct": 36}`. Deterministically split each group's lineages by
`sha256(lineage_key)` rank into `selection` / `loco_challenge` halves (never overlapping). For VHH apply
the frozen Task-1.1 rule conjunctively: selection = *L. glama* at/before the cutoff; challenge =
non-*L. glama* after the cutoff, each requiring ≥19. Compute `floors_ok[group]` against the gate's
guardband floor. Assert `pooled_overlap` empty.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/alignair/aa/guard/test_allocate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/allocate.py tests/alignair/aa/guard/test_allocate.py
git commit -m "feat(aa-guard): never-pooled selection/LOCO allocator with guardband floors + VHH mins"
```

### Task 1.4: Hydration driver + Stage-1 GO/STOP gate

**Files:**
- Create: `scripts/aa/applicability/hydrate_fresh.py`, `scripts/aa/applicability/audit_disjoint.py`
- Test: `tests/alignair/aa/guard/test_hydrate_gate.py` (gate logic on a synthetic fixture; no network)

**Interfaces:**
- Produces: `.private/aa-applicability/fresh_hydration/{fresh_cohort.jsonl, allocation.json}` and committed `docs/superpowers/artifacts/2026-07-26-aa-applicability/fresh_cohort_census.json` (counts only) + `STAGE1_RESULT.md` with `GO`/`STOP`.
- Consumes Task 1.1–1.3.

- [ ] **Step 1: Write the failing test** — feed a synthetic allocation to `stage1_decision(allocation, disjoint_report)`; assert `GO` only when all floors_ok, all vhh_min_ok, and disjoint clean; `STOP_FRESH_COHORT_UNPOWERED` otherwise.

```python
# tests/alignair/aa/guard/test_hydrate_gate.py
from scripts.aa.applicability.hydrate_fresh import stage1_decision

def test_go_requires_floors_vhh_and_disjoint():
    ok = dict(floors_ok={"igk_igl": True, "tcr": True, "vhh_nonhuman": True, "igsf": True,
                         "structural": True, "positive_real": True, "positive_sim": True},
              vhh_min_ok={"selection": True, "loco_challenge": True})
    assert stage1_decision(ok, clean=True) == "GO"
    ok["floors_ok"]["tcr"] = False
    assert stage1_decision(ok, clean=True).startswith("STOP")
    ok["floors_ok"]["tcr"] = True
    assert stage1_decision(ok, clean=False).startswith("STOP")
```

- [ ] **Step 2: Run** — FAIL (module not found).
- [ ] **Step 3: Implement** `stage1_decision` (pure) plus the hydration driver that: pulls each source disjoint from the verified full pre-split Gate-3 denylist and Phase 3 (using cached `.private/aa-gate3-hydration/*` where present + live pulls), applies Task 1.2 audit and Task 1.3 allocation, writes the private cohort + committed counts + `STAGE1_RESULT.md`. The locked file remains untouched.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**

```bash
git add scripts/aa/applicability/hydrate_fresh.py scripts/aa/applicability/audit_disjoint.py tests/alignair/aa/guard/test_hydrate_gate.py
git commit -m "feat(aa-guard): fresh hydration driver + Stage-1 GO/STOP gate"
```

- [ ] **Step 6: Execute hydration and record the gate** — run `hydrate_fresh.py`; if `STOP`, halt the plan and report. Commit only `docs/.../fresh_cohort_census.json` + `STAGE1_RESULT.md`.

---

## Stage 2 — Profile bank + HMM baseline (C0)

**Stage GO/STOP:** GO iff an exact production profile path is resolved **before bank construction and
candidate training**, the bank builds deterministically, and C0 produces valid decisions on burned
development fixtures. The path is either a redistribution-compatible bank or a fully specified user-side
hydration flow that deterministically reproduces the pinned bank. IMGT-only remains
`comparator_only` and cannot produce Stage-2 GO. Otherwise `STOP_NO_PUBLISHABLE_PROFILE_SOURCE`.

### Task 2.1: Redistribution-compatible profile source selection + manifest

**Files:**
- Create: `src/alignair/aa/guard/profiles.py`,
  `docs/superpowers/artifacts/2026-07-26-aa-applicability/profile_sources.json`,
  `requirements/aa-guard-build.lock`
- Test: `tests/alignair/aa/guard/test_profile_sources.py`

**Interfaces:**
- Produces: `PROFILE_SOURCES: dict[str, ProfileSource]` (per V/J or full-domain profile: `name`, `url`,
  `version`, exact input hashes, license, `redistributable: bool`, `role:
  "production"|"comparator_only"`, and optional deterministic hydration recipe).
  `resolve_production_bank() -> ProductionBankResolution` returns either
  `redistributable_bundle` or `exact_user_hydration`; it raises `NoPublishableProfileSource` if any
  required profile has neither.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/guard/test_profile_sources.py
import pytest
from pathlib import Path
from alignair.aa.guard.profiles import (
    PROFILE_SOURCES, NoPublishableProfileSource, resolve_production_bank,
    synthetic_hydration_resolution,
)

RECEPTOR_GROUPS = {
    "human_igh", "human_igk", "human_igl", "tra", "trb", "trg", "trd", "vhh", "murine_vh",
}

def test_all_required_groups_have_a_source():
    assert RECEPTOR_GROUPS | {"igsf_vset"} <= set(PROFILE_SOURCES)
    for group in RECEPTOR_GROUPS:
        assert {"v", "j"} <= set(PROFILE_SOURCES[group].regions)

def test_imgt_is_comparator_only_until_cleared():
    for s in PROFILE_SOURCES.values():
        if "imgt" in s.name.lower() and not s.redistributable:
            assert s.role == "comparator_only"

def test_resolution_rejects_unresolved_nonredistributable_source():
    resolution = resolve_production_bank()
    assert resolution.mode in {"redistributable_bundle", "exact_user_hydration"}
    assert resolution.all_required_profiles_resolved

def test_user_hydration_binds_exact_inputs_and_output():
    r = synthetic_hydration_resolution()
    assert r.input_sha256 and r.bank_sha256 and r.hydration_script_sha256

def test_build_tool_lock_is_version_and_hash_pinned():
    text = Path("requirements/aa-guard-build.lock").read_text()
    assert "pyhmmer==0.12.1" in text and "anarcii==2.0.8" in text
    assert "--hash=sha256:" in text
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** Register candidate sources with explicit, reviewed
license/redistribution evidence—never infer permission from database popularity. Register IMGT as
`comparator_only, redistributable=False` until explicitly cleared. A user-hydration resolution must pin
the acquisition URL/release, input hashes, hydration script hash, exact output-bank hash, first-run network
behavior, cached/offline behavior, and failure semantics. `resolve_production_bank` raises if any required
profile is unresolved. Record the adjudication; do not silently substitute a different bank later.
Generate a platform-aware, hash-pinned build/comparator dependency lock covering `pyhmmer==0.12.1`,
`anarcii==2.0.8`, and their transitive dependencies; record the exact bundled HMMER version reported at
runtime. Production runtime metadata includes `pyhmmer` only when the selected product path requires it;
ANARCII remains a comparator/teacher dependency.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/profiles.py requirements/aa-guard-build.lock docs/superpowers/artifacts/2026-07-26-aa-applicability/profile_sources.json tests/alignair/aa/guard/test_profile_sources.py
git commit -m "feat(aa-guard): profile-source registry (redistribution-gated; IMGT comparator-only)"
```

### Task 2.2: Deterministic profile builder

**Files:**
- Modify: `src/alignair/aa/guard/profiles.py`
- Test: `tests/alignair/aa/guard/test_profile_build.py`

**Interfaces:**
- Produces: `build_profile(group, region, aligned_fasta_bytes) -> (HMM, sha256_hex)` and
  `build_bank(resolution) -> ProfileBank` with `.checksum()` deterministic; bank persisted via pyhmmer
  `HMMFile` write (non-pickle).
- Receptor groups contain independently versioned **V and J profiles**. Completeness uses their N→C
  ordering and combined query coverage. A V-only hit can support locus evidence but cannot establish a
  complete domain. IgSF uses a validated full-domain profile. No undocumented V–J concatenation is
  allowed.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/guard/test_profile_build.py
from alignair.aa.guard.profiles import build_bank_fixture, build_profile

MSA = b">a\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n>b\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n>c\nEVQLVQSGAEVKKPGASVKVSCKASGYTL\n"

def test_build_is_deterministic():
    _, h1 = build_profile("human_igh", "v", MSA)
    _, h2 = build_profile("human_igh", "v", MSA)
    assert h1 == h2 and len(h1) == 64

def test_v_only_bank_cannot_claim_complete_domain():
    bank = build_bank_fixture(regions={"human_igh": {"v": MSA}})
    assert not bank.supports_complete_domain("human_igh")
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** using `pyhmmer.easel` (parse FASTA → digitized MSA) and
`pyhmmer.plan7.Builder(alphabet).build_msa(msa, Background(alphabet))`; serialize the HMM to bytes with
`HMM.write` and sha256 those bytes for the checksum. Fix RNG-free builder settings for determinism
(verified in Phase 0). Build receptor V and J profiles separately and store their biological region in the
manifest; never infer full-domain coverage from V alone.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/profiles.py tests/alignair/aa/guard/test_profile_build.py
git commit -m "feat(aa-guard): deterministic pyhmmer profile builder + bank checksum"
```

### Task 2.3: Competitive search → fixed HMM feature vector

**Files:**
- Create: `src/alignair/aa/guard/features.py`
- Test: `tests/alignair/aa/guard/test_features.py`

**Interfaces:**
- Produces: `hmm_features(bank, seq) -> HmmFeatures` with fields exactly per spec §5: best V and J
  profile/species; per-group/per-region bitscore and E-value; human-IGH score; best-alternative score;
  margin; separate V/J aligned-query/profile fractions and boundaries; `v_before_j`; inter-profile gap;
  combined query coverage; unaligned N/C; significant-hit counts; coverage/order flags.
  `to_vector() -> np.ndarray` has fixed length + `FEATURE_NAMES`.

- [ ] **Step 1: Write the failing test** — build paired V/J fixtures for human IGH and IGL; assert a
complete VH yields positive human-IGH margin, coherent `v_before_j`, deterministic fixed-length features,
and combined coverage; assert V-only input never sets complete-domain coverage.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** using `pyhmmer.plan7.Pipeline(...).search_hmm(hmm, targets)` per profile;
assemble paired V/J evidence and the fixed vector; `bias_filter=False` for determinism.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/features.py tests/alignair/aa/guard/test_features.py
git commit -m "feat(aa-guard): competitive HMM search -> fixed feature vector"
```

### Task 2.4: C0 frozen HMM-only decision rules

**Files:**
- Create: `src/alignair/aa/guard/c0_rules.py`
- Test: `tests/alignair/aa/guard/test_c0_rules.py`

**Interfaces:**
- Produces: `c0_decide(features, thresholds) -> GuardVerdict(state, reason, audit_score)`. `state ∈ {supported, insufficient_evidence, unsupported_locus, unknown_ood}`. Rules frozen on burned-dev before fresh validation. `unsupported_locus` only when a named alternative wins by a validated margin AND its reason-precision gate passes; else `unknown_ood`.

- [ ] **Step 1: Write the failing test** — assert: high human-IGH margin ⇒ `supported`; a clearly-winning IGK profile with margin above threshold ⇒ `unsupported_locus:igk` only if precision-gated flag true, else `unknown_ood`; low coverage ⇒ `insufficient_evidence`.
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** the frozen rule ladder. Burned development data fixes the ladder, score
directions, and finite threshold grid; the fresh selection subset chooses one predeclared grid point in
Stage 6. Never invent or hand-tune a rule after viewing fresh data.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/guard/c0_rules.py tests/alignair/aa/guard/test_c0_rules.py
git commit -m "feat(aa-guard): C0 frozen HMM-only decision rules"
```

- [ ] **Step 6: Resolve and build the exact production bank + record Stage-2 gate** — run
`build_profiles.py`; write `STAGE2_RESULT.md` with
`GO_PRODUCTION_REDISTRIBUTABLE` / `GO_EXACT_USER_HYDRATION` /
`STOP_NO_PUBLISHABLE_PROFILE_SOURCE`. Comparator-only banks are reported separately and cannot feed
C0–C4. Profiles go to `.private/aa-applicability/profiles/`; only checksums + manifest are committed.

---

## Stage 3 — Contextual encoder + heads

**Stage GO/STOP:** GO iff encoder+heads instantiate, are deterministic under fixed seed, are padding-invariant, and fusion preserves raw HMM provenance. Else fix before proceeding.

### Task 3.1: Tokenizer (23-token + CLS)

**Files:** Create `src/alignair/aa/guard/tokenizer.py`; Test `tests/alignair/aa/guard/test_tokenizer.py`.

**Interfaces:** `AaTokenizer(max_len=192)` → `.encode(seq) -> (ids: LongTensor[L+1], mask: BoolTensor[L+1])` with a leading `[CLS]`; `.vocab_size == 24` (23 residues + CLS); pad/truncate to `max_len+1`.

- [ ] **Step 1: Failing test** — CLS at position 0; length ≤ max_len+1; unknown residue → `X` token; mask marks pads.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): 23-token AA tokenizer with CLS`.

### Task 3.2: ApplicabilityEncoder

**Files:** Create `src/alignair/aa/guard/encoder.py`; Test `tests/alignair/aa/guard/test_encoder.py`.

**Interfaces:** `ApplicabilityEncoder(width=128, depth=4, heads=4, ff=512, dropout=0.10)`; `.forward(ids, mask) -> (cls_vec[B,128], mean_vec[B,128])` (masked mean); pre-norm blocks, rotary positions, padding-aware attention.

- [ ] **Step 1: Failing test** — output shapes; deterministic under `torch.manual_seed` in eval mode; **padding invariance**: appending pad tokens does not change `cls_vec`/`mean_vec` beyond atol=1e-5.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement standard pre-norm transformer with rotary + key-padding mask; masked mean over valid positions. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): padding-invariant contextual encoder`.

### Task 3.3: Heads

**Files:** Create `src/alignair/aa/guard/heads.py`; Test `tests/alignair/aa/guard/test_heads.py`.

**Interfaces:** `GuardHeads(in_dim)` with `completeness` (5-way), `domain_type` (8-way), `applicability` (binary logit), `energy(logits) -> free_energy = -logsumexp(recognized_type_logits)`. Input = concat(cls, mean, [fused HMM]).

- [ ] **Step 1: Failing test** — logit shapes (5/8/1); energy is `-logsumexp` of the recognized-type logits and decreases as a known-type logit grows.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): completeness/type/applicability/energy heads`.

### Task 3.4: HMM fusion with preserved provenance

**Files:** Create `src/alignair/aa/guard/fusion.py`; Test `tests/alignair/aa/guard/test_fusion.py`.

**Interfaces:** `HmmFusion(feature_names, stats)` → `.normalize(vec)` (train-fit mean/std, stored); `fuse(seq_rep, hmm_vec) -> (fused_tensor, raw_hmm_vec_unchanged)`. Raw HMM vector is returned alongside and never overwritten.

- [ ] **Step 1: Failing test** — normalization uses stored stats; `fuse` returns the **unchanged** raw vector for provenance; fused width = seq_rep + hmm_vec.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): HMM fusion preserving raw provenance`.

---

## Stage 4 — Training C1–C4 + OE/energy + budget

**Stage GO/STOP:** GO iff C1–C4 main candidates train to a documented dev-only stopping point, their
predeclared seed-stability audit completes, one production seed per candidate is frozen by the dev-only
rule, and the full main+LOCO compute budget is recorded. Stage 5 separately gates LOCO completion. Else
`STOP_TRAINING_NONCONVERGENCE` (with evidence).

### Task 4.1: Loss module (six terms)

**Files:** Create `src/alignair/aa/guard/losses.py`; Test `tests/alignair/aa/guard/test_losses.py`.

**Interfaces:** `guard_loss(outputs, targets, weights) -> (total, parts)` implementing `L_type + λ_complete·L_complete + λ_app·L_applicability + λ_oe·L_energy_margin + λ_contrast·L_supcon + λ_source·L_source_adv`. `L_energy_margin` pushes recognized in-domain energy below `m_in`, auxiliary outliers above `m_out`. `L_source_adv` via gradient-reversal.

- [ ] **Step 1: Failing test** — each term computes finite, non-negative where expected; energy-margin penalizes an in-domain sample with energy > `m_in`; source-adversarial reverses gradient sign (check grad on a source-logit param).
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement (GRL as autograd Function). **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): six-term guard loss with energy margin + GRL`.

### Task 4.2: Training-distribution builder

**Files:** Create `src/alignair/aa/guard/train_data.py`; Test `tests/alignair/aa/guard/test_train_data.py`.

**Interfaces:** `build_training_stream(dev_cohort, oe_pool, *, seed) -> IterableDataset` yielding balanced batches of positives / known-hard-negatives / structural / OE, with source-batch balancing and length/composition matching; OE is a minority and cannot dominate.

- [ ] **Step 1: Failing test** — over N batches, OE fraction ≤ configured cap; every biological negative group represented; structural samples carry `parent_lineage`; no fresh/locked ids present (assert against a passed exclusion set).
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): balanced training stream (positives/neg/structural/OE)`.

### Task 4.3: Hard-negative mining (training-only)

**Files:** Create `src/alignair/aa/guard/mining.py`; Test `tests/alignair/aa/guard/test_mining.py`.

**Interfaces:** `mine_hard_negatives(model, aux_pool, *, forbidden_ids) -> list[record]` selecting false-accepts/smallest-margin, deduped by lineage; **raises** if any `aux_pool` id ∈ `forbidden_ids` (fresh ∪ locked).

- [ ] **Step 1: Failing test** — mining raises when the pool intersects fresh/locked; otherwise returns lineage-deduped hardest items.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): training-only hard-negative mining with fresh/locked guard`.

### Task 4.4: Candidate trainer + budget accounting

**Files:** Create `scripts/aa/applicability/train_candidate.py`, `scripts/aa/applicability/budget.py`; Test `tests/alignair/aa/guard/test_budget.py`.

**Interfaces:** `train_candidate(cid, seed) -> CheckpointRef` (C0 rule-fit; C1 supervised; C2 +fusion; C3
+OE/energy; C4 +supcon+source-adv). `choose_production_seed(dev_results, frozen_rule) -> int` may inspect
burned-development metrics only. `budget.record(model_id, wall_s, gpu_h, params)` accumulates into
`docs/.../budget.json`; `budget.plan(candidates, seed_replicates, loco_folds) -> Estimate` covers the four
learned main candidates × at least three dev seeds, the chosen main checkpoints, every learned LOCO
model, C0, and C0 profile-withheld evaluations.

- [ ] **Step 1: Failing test** — `budget.plan` counts every seed replicate and every main/LOCO model;
`choose_production_seed` is deterministic and raises if passed any fresh-validation field; `record`/`total`
round-trip via JSON.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement trainer (AdamW, cosine schedule, fixed seeds,
early-stop on internal dev metric, checkpoint to `.private/`), dev-only seed rule, and budget accounting.
**Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): candidate trainer + full main+LOCO budget accounting`.

- [ ] **Step 6: Train C1–C4 seed replicates, freeze one seed per candidate, record Stage-4 gate** — run
training; write `STAGE4_RESULT.md` (per-seed stability, dev-only selection rule, chosen seed, convergence,
budget). Checkpoints stay in `.private/`. No fresh cohort is loaded.

---

## Stage 5 — LOCO + blind-LOCO

**Stage GO/STOP:** GO iff every learned candidate has all four LOCO folds trained with the fold category fully withheld from training/mining/HP/threshold/profile-rule/source-balancing, and blind-LOCO masks the fold's profile from the feature vector and decision graph. Else fix leakage.

### Task 5.1: LOCO fold trainer with leakage guard

**Files:** Create `scripts/aa/applicability/run_loco.py`, `src/alignair/aa/guard/loco.py`; Test `tests/alignair/aa/guard/test_loco.py`.

**Interfaces:** `loco_exclude(cohort, group_g) -> filtered` removing every `g` record and `g`-source proxy; `assert_no_g_leak(train_ids, threshold_ids, g_ids)` raises on any intersection.

- [ ] **Step 1: Failing test** — after `loco_exclude("igk_igl")`, no igk/igl record remains in train or threshold sets; `assert_no_g_leak` raises if one is injected.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): LOCO fold exclusion + leakage assertion`.

### Task 5.2: Blind-LOCO profile masking

**Files:** Modify `src/alignair/aa/guard/features.py`, `src/alignair/aa/guard/loco.py`; Test `tests/alignair/aa/guard/test_blind_loco.py`.

**Interfaces:** `hmm_features(bank, seq, mask_groups={g}) -> HmmFeatures` with group `g`'s profile removed from the competitive set **and** from any `g`-naming decision path; `blind` result carries `masked_groups`.

- [ ] **Step 1: Failing test** — masking `vhh` removes the vhh profile from the vector and prevents any `unsupported_locus:vhh` reason; the reference-covered vector still contains it.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): blind-LOCO profile masking (feature + decision graph)`.

### Task 5.3: Dual reference-covered vs blind-LOCO evaluation

**Files:** Modify `scripts/aa/applicability/run_loco.py`; Test `tests/alignair/aa/guard/test_dual_eval.py`.

**Interfaces:** `eval_fold(candidate, g, challenge_subset) -> {"reference_covered": Metrics, "blind": Metrics}`. Eligibility later uses **blind** only.

- [ ] **Step 1: Failing test** — returns both results; `blind` never uses `g`'s profile; the challenge subset for `g` is the Stage-1 `loco_challenge` allocation (disjoint from selection).
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): reference-covered + blind-LOCO dual evaluation`.

- [ ] **Step 6: Train+eval all LOCO folds, record Stage-5 gate** — run; write `STAGE5_RESULT.md`.

---

## Stage 6 — Selection under the guardband

**Stage GO/STOP:** GO iff ≥1 candidate is eligible (all main + blind-LOCO gates pass the guardband, reason precision passes or demotes, invariance/runtime/license pass). Else `STOP_NO_FEASIBLE_APPLICABILITY_MODEL` — do **not** open the locked test.

### Task 6.1: Exact bounds + threshold-grid selection

**Files:** Create `src/alignair/aa/guard/thresholds.py`; Test `tests/alignair/aa/guard/test_thresholds.py`.

**Interfaces:** `clopper_pearson_upper(k, n, alpha=0.05) -> float`; `select_thresholds(candidate, selection_subset, grid) -> Thresholds` choosing the frozen operating point; `passes_guardband(metrics) -> bool` requiring FAR upper ≤4%/≤8% and FRR upper ≤8%.

- [ ] **Step 1: Failing test** — `clopper_pearson_upper(0, 74) ≤ 0.04` and `(0, 59) ≤ 0.05`; `clopper_pearson_upper(0, 36) ≤ 0.08`; `passes_guardband` rejects a point whose bound exceeds the guardband.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement exact bound (`scipy.stats.beta.ppf`) + grid search. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): exact bounds + guardband threshold selection`.

### Task 6.2: Eligibility + far-OOD descriptive-only + deterministic tie-break

**Files:** Create `scripts/aa/applicability/select.py`; Test `tests/alignair/aa/guard/test_select.py`.

**Interfaces:** `eligible(candidate) -> bool` (main guardband ∧ every blind-LOCO fold guardband ∧ reason-precision-or-demote ∧ invariance ∧ runtime/license); `pick(eligibles) -> cid` by (1) smallest worst-group FAR ratio, (2) smallest max real/sim FRR, (3) smaller artifact, (4) C0→C4. `far_ood_is_descriptive(inputs) -> None` **raises** if a far-OOD metric is passed into selection/threshold/tie-break.

- [ ] **Step 1: Failing test** — a candidate failing one blind-LOCO fold is ineligible; feeding a far-OOD metric into `pick` raises; tie-break order deterministic; no eligible ⇒ `STOP_NO_FEASIBLE_APPLICABILITY_MODEL`.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): candidate eligibility, far-OOD firewall, deterministic selection`.

- [ ] **Step 6: Run selection, record Stage-6 gate** — write `STAGE6_RESULT.md` (selected cid + thresholds + why; or STOP).

---

## Stage 7 — Final freeze + one-shot locked test

**Stage GO/STOP:** GO to unlock iff the pre-unlock product path and packaging work in Stage 8 Tasks
8.1–8.3 is complete, the production profile path is reverified byte-identical to Stage 2, invariance
passes, the category-map audit passes, the **final** source tree + bundle + artifacts are frozen, and the
mock one-shot passes. Execution order is binding:
`8.1–8.3 → 7.5 → 7.2 → 7.3 → 7.1 → 7.4 → 7.6`. No code or artifact mutation is permitted after 7.1
except the mock audit/result and the real immutable locked result.

### Task 7.1: Freeze + hash manifest

**Files:** Create `src/alignair/aa/guard/freeze.py`, `scripts/aa/applicability/freeze_all.py`; Test `tests/alignair/aa/guard/test_freeze.py`.

**Interfaces:** `freeze_manifest(paths, frozen_code_commit) -> dict` sha256 over **preprocessing config, profile bank, all
main+LOCO weights, thresholds, decision graph+version, category map, evaluation script, guarded public
schema/API/CLI code, safe loader, final bundle, validation artifacts, and repository source-tree commit**;
`verify(manifest) -> bool`.

- [ ] **Step 1: Failing test** — manifest includes every required artifact key and the immutable
`frozen_code_commit`; `verify` fails if any hashed file changes by one byte, a frozen path differs from
that commit, or the current commit is not a descendant. Manifest/result-only commits are allowed; source
changes are not.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement a two-commit protocol: first commit every production
code/artifact and require a clean worktree; generate the manifest against that commit; then commit only
the manifest. Avoid a self-referential “manifest hashes the commit containing itself” design.
**Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): freeze+hash manifest over all frozen artifacts`.

### Task 7.2: Frozen-annotator invariance

**Files:** Create `tests/alignair/aa/guard/test_invariance.py`, `src/alignair/aa/guard/decision.py` (integration point).

**Interfaces:** `assert_v2_invariant(fixtures)` — for each accepted fixture, guarded pipeline's v2 V/D/J logits, coords, calibrated classes, expanded calls, `softmax_score`, `inter_vj_aa` are byte-identical to direct unguarded v2; guard batching/order does not change annotation; loading the guard does not mutate v2.

- [ ] **Step 1: Failing test** — run v2 directly vs through the guard on the regression fixtures; assert byte-identical; assert a guard load leaves v2 checksums unchanged.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement decision-graph integration that calls v2 only on `supported`, detached. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `test(aa-guard): frozen v2 annotator invariance`.

### Task 7.3: Category-map re-audit vs selected candidate

**Files:** Modify `scripts/aa/applicability/freeze_all.py`; Test `tests/alignair/aa/guard/test_catmap_reaudit.py`.

**Interfaces:** `reaudit_category_map(selected_candidate, locked_category_map.json) -> Report`; raises `STOP_LOCKED_CATEGORY_MAP_MISMATCH` if any gate the candidate defines lacks a present, floored locked category, or if a §10 gate is unmapped.

- [ ] **Step 1: Failing test** — passes for the current §10 gate set + `locked_category_map.json`; raises when a synthetic candidate adds an unmapped gate.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement (read the committed counts-only map). **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): pre-unlock category-map re-audit`.

### Task 7.4: Mock one-shot against a synthetic sealed fixture

**Files:** Create `scripts/aa/applicability/mock_oneshot.py`; Test `tests/alignair/aa/guard/test_mock_oneshot.py`.

**Interfaces:** builds `.private/aa-applicability/mock_seal/locked_test.jsonl` (synthetic, labeled, same
schema + loader as the real seal), runs the **entire frozen public product and unlock pipeline** against it
(bundle safe-load → public guarded API → structural withholding checks → hash verify → selected + all
LOCO evaluation → per-category counts/bounds/reason-matrix → immutable result), and asserts the pipeline
runs end-to-end and is **idempotent-blocked** on a second attempt.

- [ ] **Step 1: Failing test** — mock one-shot produces a complete result object with every §10 gate + every LOCO fold populated; a second invocation refuses (one-shot lock engaged).
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement using the real `cohort_loader` one-shot loader semantics against the synthetic file. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): mock one-shot dry-run on synthetic sealed fixture`.

### Task 7.5: Production profile-path identity re-verification (BEFORE final freeze)

**Files:** Create `scripts/aa/applicability/resolve_profile_path.py`; Test `tests/alignair/aa/guard/test_profile_path_gate.py`.

**Interfaces:** `reverify_production_profile_path(selected_candidate, stage2_resolution) -> str` ∈
`{"production_redistributable", "exact_user_hydration"}` or raises
`STOP_PROFILE_PATH_CHANGED`. This task cannot resolve, replace, or rebuild an unresolved bank: it proves
the selected candidate still uses the exact bank, input hashes, build script, license decision, and
hydration mode admitted in Stage 2.

- [ ] **Step 1: Failing test** — a selected candidate whose bank checksum or source mode differs from the
Stage-2 resolution raises `STOP_PROFILE_PATH_CHANGED`; an exact match returns its frozen mode.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement using `profile_sources.json`, Stage-2 result, bank
manifest, and hydration validation report. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `test(aa-guard): pre-freeze production profile identity re-verification`.

### Task 7.6: One-shot unlock (once)

**Files:** Create `scripts/aa/applicability/unlock.py`.

- [ ] **Step 1: Pre-unlock checklist (manual gate)** — confirm the binding execution order was followed;
`verify(freeze_manifest)` true; invariance green; category-map re-audit clean; mock one-shot green;
profile-path identity reverified; exact public result/API path + bundle + clean-env load proven; no frozen
file changed after Task 7.1; no prior unlock audit exists.
- [ ] **Step 2: Execute the single unlock** — run `unlock.py`: verify all hashes → unlock the existing Gate-3 seal with an audit reason → evaluate the selected candidate + all required LOCO folds in one run (all 19 sealed VHH retained) → emit per-category counts, point estimates, exact 95% bounds, reason matrix, provenance → write the immutable `docs/.../LOCKED_TEST_RESULT.md` (`PASS`/`FAIL`).
- [ ] **Step 3: Record outcome** — commit `LOCKED_TEST_RESULT.md` only. No code/model/threshold changes afterward.

---

## Stage 8 — Pre-unlock product path + packaging; publication after PASS

**Stage GO/STOP:** Tasks 8.1–8.3 run **before Task 7.1 final freeze**. They implement the exact product result
union, attach the guard to the AA bundle, safe-load it, and exercise the public API/CLI in a clean
environment. Task 8.4 performs only documentation/release enablement/publication after a Stage-7 `PASS`;
it may not change schema, inference, withholding, model, profile, thresholds, or loading behavior.

### Task 8.1: Pickle-free bundle section

**Files:** Create `src/alignair/aa/guard/bundle.py`; Test `tests/alignair/aa/guard/test_bundle.py`.

**Interfaces:** `write_guard_bundle(dst, *, weights, tokenizer, profile_bank, profile_manifest, norm_stats,
thresholds, decision_graph, hashes)` attaches the guard to the existing AA product container using
`safetensors` weights, non-pickle profile format, non-object NumPy arrays written with
`allow_pickle=False`, and JSON manifests. `read_guard_bundle(path) -> Guard`.

- [ ] **Step 1: Failing test** — round-trip; enforce an archive-member/format allowlist; monkeypatch
`pickle.load`, `pickle.loads`, and `torch.load` to raise during bundle loading and prove none is called;
loading reconstructs identical weights, profile bank, and thresholds. Do not scan arbitrary binary bytes
for “pickle opcodes.”
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): pickle-free guard bundle section`.

### Task 8.2: Safe loader + integrity hard-fails

**Files:** Modify `src/alignair/aa/guard/bundle.py`; Test `tests/alignair/aa/guard/test_bundle_integrity.py`.

**Interfaces:** `read_guard_bundle` hard-fails on: missing guard for a production AA bundle; checksum mismatch; profile/reference mismatch; threshold/model mismatch; unsupported schema; missing runtime; validation-binding mismatch.

- [ ] **Step 1: Failing test** — each corruption raises a specific error; a production AA bundle **without**
a guard refuses to load for public inference. The explicitly unsafe research loader remains separate,
requires its warning flag, and cannot be reached by the public default.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): safe loader with integrity hard-fails`.

### Task 8.3: Clean-environment load proof (before unlock)

**Files:** Create `scripts/aa/applicability/package.py`; Modify
`src/alignair/aa/product/{schema,predictor,loader,bundle,cli}.py`; Test
`tests/alignair/aa/guard/{test_product_union,test_clean_env_load}.py`.

**Interfaces:** extends the native JSON result as a discriminated union: `supported` carries guard evidence
plus annotation, while every rejection state carries guard evidence and structurally forbids V/D/J and
`inter_vj_aa`. Builds the bundle, loads it in a clean subprocess with runtime dependencies only, and runs
the **public** `predict_aa` and `predict-aa` paths on supported and rejected fixtures.

- [ ] **Step 1: Failing tests** — public API and CLI withhold annotation fields on every rejection state;
missing/corrupt guard fails closed; supported output is byte-identical to direct v2 annotation plus the
guard envelope; subprocess load+predict succeeds with runtime dependencies only; monkeypatched pickle
and `torch.load` remain unused.
- [ ] **Step 2: Run** — FAIL. **Step 3:** implement. **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `feat(aa-guard): clean-environment load proof`.

### Task 8.4: Publication (only after Stage-7 PASS)

- [ ] **Step 1: Manual gate** — proceed only if `LOCKED_TEST_RESULT.md` == `PASS`.
- [ ] **Step 2:** make no inference-code changes. Enable/release the already-frozen guarded product path;
write documentation and model-card limitations (VHH single-source; far-OOD non-certified;
simulator-domain conformal; supported-input contract), prepare registry metadata, and run final
distribution smoke tests against the frozen bundle.
- [ ] **Step 3: Finish intentionally** — run the full frozen-bundle release gates, commit only approved
documentation/release metadata, verify no frozen inference path changed, and do not push/publish without
the owner-authorized release action.

---

## Self-Review

**Spec coverage:** §0 frozen annotator → Global Constraints + Task 7.2. §1 product contract/reason states → Tasks 2.4, 6.2, 8.3. §2 hybrid rationale → Stages 2–3. §3 Phase-0 → already done (`PHASE0_FEASIBILITY.md`); §3.3/§3.4 floors + category map → Tasks 1.3, 7.3. §4 leakage control → Tasks 1.2, 4.3, 5.1. §5 paired V/J HMM component → Tasks 2.2–2.4. §6 encoder/heads/fusion → Stage 3. §7 candidates/losses/distributions/mining → Stage 4. §8 validation/LOCO/selection → Stages 5–6. §9 decision rule → Task 7.2 + `decision.py`. §10 release thresholds + scope → Tasks 6.1, 7.3. §11 invariance → Task 7.2. §12 locked-test protocol → Tasks 7.4, 7.6. §13 packaging + exact product path → Stage 8 before final freeze. §14 ablations/report → produced across Stage-5/6/7 result files. §15 non-goals → model card (Task 8.4). §16 sequencing → binding execution order in Stage 7. Profile-source adjustment → Task 2.1 hard gate + Task 7.5 identity re-verification.

**Placeholder scan:** deterministic components carry full test+impl code; ML/data/gate tasks carry full **gate/test** code + concrete procedures (optimizer/schedule/seed/early-stop named). No "TBD"/"add error handling".

**Type consistency:** `GUARDBAND_FLOOR` (74/36) and release floors (59/29) used identically in Tasks 1.3, 6.1. `GuardVerdict`/`GuardResult` state set `{supported, insufficient_evidence, unsupported_locus, unknown_ood}` consistent across 2.4/6.2/7.2. `hmm_features(..., mask_groups)` signature consistent across 2.3/5.2. Far-OOD firewall (`far_ood_is_descriptive`) and `passes_guardband` names stable.

**Ordering safeguard:** Stage 8 Tasks 8.1–8.3 precede Task 7.1 final freeze; Task 7.5 reverifies rather than
changes the Stage-2 profile path; Tasks 7.2/7.3 precede freeze; the mock and real unlock run only against
the frozen public product path; publication (8.4) is documentation/release work only and is gated on PASS.
