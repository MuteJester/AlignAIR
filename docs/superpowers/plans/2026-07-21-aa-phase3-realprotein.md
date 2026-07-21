# AA AlignAIR Phase 3 — Real-Protein Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the frozen simulator-domain AA IGH model on real therapeutic-antibody VH domains, reporting a zero-circularity **family-level design-ancestry** accuracy anchor (headline) plus an explicitly-labelled exact-gene **concordance** picture — never claiming to validate the 95% allele/class guarantee.

**Architecture:** A gated data-curation milestone (feasibility-first, no model inference) produces a frozen, hashed *input bundle* — committed labels/metadata/citations only, with raw VH sequences hydrated into a git-ignored local file. Then small, tested library utilities (hardened inference, frozen-calibration loader, taxonomy) feed a scorer whose only binding test is an exact one-sided binomial on one representative VH per lineage. An eval script verifies the bundle manifest, then emits (git-ignored) per-row predictions and a committed aggregate JSON with a machine-readable `failed`/`inconclusive` decision, plus `PHASE3.md`.

**Tech Stack:** Python, PyTorch (inference only), NumPy, SciPy (`scipy.stats.beta` for Clopper–Pearson), the existing `alignair.aa` package, `pytest`.

## Global Constraints

- Branch `experiment/aminoacid-alignair` only; **never modify the frozen snapshot** (`experiments/aa/strategic/model_stage1_PASS.pt` sha `54eef812…`, `docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json`).
- **No model inference — not even exploratory — until the Milestone-1 feasibility gate passes** (bundle frozen + hashed first).
- **`supported` is never emitted.** The decision is `failed` or `inconclusive` only.
- Gold = **IGHV family**, an **independent design-ancestry label** (engineered ancestry), *not* literal germline ground truth. Exact-gene = **concordance tier only**, never changes the decision, always labelled "alignment-derived, not ground truth".
- **Licensing:** Thera-SAbDab redistribution is not established. **Never commit raw VH sequences or sequence-bearing per-row predictions to Git.** Commit metadata, sequence hashes, labels, citations, and acquisition code; hydrate sequences into a git-ignored local file; hash the fully-joined cohort before inference.
- Binding statistical unit = **one deterministic representative VH per `lineage_id`** (lexicographically smallest `exact_vh_id`). Descriptive statistics span **all deduplicated VHs** with a **lineage-cluster** bootstrap.
- α = 0.05; exact one-sided binomial (Clopper–Pearson). Conformal quantile = **`q_mondrian["full"]`** from the frozen `calibration.json` (not pooled).
- Labels never derived by aligning the therapeutic sequence. Every Gold subgroup→IMGT-family mapping is **individually verified from an authoritative source** and present in the frozen crosswalk — **Kabat subgroup and IMGT family are NOT assumed one-to-one by matching numbers.**
- No Claude co-author line in commits.
- Spec: `docs/superpowers/specs/2026-07-21-aa-phase3-realprotein-design.md`. Artifacts dir: `docs/superpowers/artifacts/2026-07-21-aa-phase3/` (`ART/` below).

---

## Milestone 1 — Data acquisition & family-Gold curation (GATED, no model inference)

Determines feasibility. Produces a frozen, hashed **input bundle** + validators that **enforce** the contract, ending at a stop-and-report gate. Its *content* (antibodies, declared templates, citations, crosswalk entries) is **human-verified from primary sources** — never auto-generated, never alignment-derived, and (per the hybrid decision) **every admitted Gold label and every crosswalk entry is reviewed with the user**.

### Task 1: Label schema + row validator + cohort-level validator

**Files:**
- Create: `src/alignair/aa/phase3/__init__.py`, `src/alignair/aa/phase3/cohort.py`
- Test: `tests/alignair/aa/phase3/__init__.py`, `tests/alignair/aa/phase3/test_cohort.py`

**Interfaces:**
- Produces: `LABEL_FIELDS: list[str]` (committed label columns — **no `vh_aa`**); `@dataclass LabelRow`; `validate_row(d: dict) -> list[str]`; `validate_cohort(rows: list[LabelRow], hydrated: dict[str,str], crosswalk: dict) -> list[str]`; `load_labels(path) -> tuple[list[LabelRow], list[dict]]`; `sha256_seq(vh_aa: str) -> str`; `cohort_sha256(rows, hydrated) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_cohort.py
import hashlib
from alignair.aa.phase3.cohort import (validate_row, validate_cohort, sha256_seq, LabelRow, LABEL_FIELDS)

SEQ = "EVQLVESGGGLVQPGGSLRLSCAAS"
VID = hashlib.sha256(SEQ.encode()).hexdigest()
CROSS = {"VH subgroup III": ("IGHV3", "", "Scaviner 1999 IMGT")}

def _row(**kw):
    d = dict(name="examplemab", arm_id="", exact_vh_id=VID, lineage_id="examplemab",
             lineage_representative="true", chain_class="in_scope", seq_source="therasabdab",
             seq_source_version="2024-12-20", retrieval_date="2026-07-21",
             raw_file_sha256="a"*64, true_v_family="IGHV3", true_v_gene="", true_v_allele="",
             label_resolution="subgroup_only", in_catalog="true", tier="gold",
             label_source="VH subgroup III", label_citation="US1234567 Example 1",
             alias_source="Scaviner 1999 IMGT", evidence_locator="col5 Ex1",
             humanization_status="humanized", true_j="", exclusion_reason="")
    d.update(kw); return d

def test_valid_gold_row_passes_row_check():
    assert validate_row(_row()) == []

def test_row_requires_citation_and_evidence_and_family():
    assert any("citation" in v for v in validate_row(_row(label_citation="")))
    assert any("evidence" in v for v in validate_row(_row(evidence_locator="")))
    assert any("family" in v for v in validate_row(_row(true_v_family="")))

def test_row_rejects_bad_enums_hash_date_boolean():
    assert any("boolean" in v for v in validate_row(_row(lineage_representative="yes")))
    assert any("tier" in v for v in validate_row(_row(tier="silver")))
    assert any("chain_class" in v for v in validate_row(_row(chain_class="weird")))
    assert any("hash" in v for v in validate_row(_row(raw_file_sha256="short")))
    assert any("date" in v for v in validate_row(_row(retrieval_date="2026/07/21")))

def test_cohort_checks_hash_matches_sequence():
    rows = [LabelRow(**_row())]
    assert validate_cohort(rows, {VID: SEQ}, CROSS) == []
    assert any("exact_vh_id" in v for v in validate_cohort(rows, {VID: "DIFFERENT"}, CROSS))

def test_cohort_requires_exactly_one_deterministic_representative():
    a = _row(name="m", lineage_id="L", exact_vh_id="b"*64, lineage_representative="true")
    b = _row(name="m2", lineage_id="L", exact_vh_id="a"*64, lineage_representative="false")
    rows = [LabelRow(**a), LabelRow(**b)]
    hy = {"b"*64: "AAAA", "a"*64: "CCCC"}
    # representative must be the lexicographically smallest exact_vh_id ("a..") — here it's on the wrong row
    assert any("representative" in v for v in validate_cohort(rows, hy, CROSS))

def test_cohort_rejects_conflicting_labels_for_same_vh():
    a = _row(name="x", exact_vh_id=VID, true_v_family="IGHV3")
    b = _row(name="y", exact_vh_id=VID, true_v_family="IGHV1", lineage_id="y")
    assert any("conflict" in v for v in validate_cohort([LabelRow(**a), LabelRow(**b)], {VID: SEQ}, CROSS))

def test_cohort_gold_requires_alias_in_crosswalk():
    r = _row(label_source="Unlisted framework X")
    assert any("crosswalk" in v for v in validate_cohort([LabelRow(**r)], {VID: SEQ}, CROSS))

def test_cohort_concordance_must_be_marked_alignment_derived():
    r = _row(tier="concordance", true_v_gene="IGHV3-23", label_resolution="exact_gene",
             label_source="hand pick", alias_source="")
    assert any("alignment-derived" in v for v in validate_cohort([LabelRow(**r)], {VID: SEQ}, CROSS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_cohort.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase3/cohort.py
"""Frozen Phase-3 label schema + row/cohort validators. No model, no alignment, no raw sequences committed.
vh_aa lives ONLY in a git-ignored hydrated map; committed rows carry exact_vh_id = sha256(vh_aa)."""
from __future__ import annotations
import csv, hashlib, re
from dataclasses import dataclass

LABEL_FIELDS = [
    "name", "arm_id", "exact_vh_id", "lineage_id", "lineage_representative", "chain_class",
    "seq_source", "seq_source_version", "retrieval_date", "raw_file_sha256", "true_v_family",
    "true_v_gene", "true_v_allele", "label_resolution", "in_catalog", "tier", "label_source",
    "label_citation", "alias_source", "evidence_locator", "humanization_status", "true_j",
    "exclusion_reason",
]
_BOOL = {"true", "false"}
_TIER = {"gold", "concordance"}
_CHAIN = {"in_scope", "murine", "chimeric", "vhh", "synthetic", "unresolved"}
_RES = {"exact_allele", "exact_gene", "subgroup_only", "ambiguous", "unresolved"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ALIGN = ("alignment", "anarci", "imgt", "v-quest", "vquest")

@dataclass
class LabelRow:
    name: str; arm_id: str; exact_vh_id: str; lineage_id: str; lineage_representative: str
    chain_class: str; seq_source: str; seq_source_version: str; retrieval_date: str
    raw_file_sha256: str; true_v_family: str; true_v_gene: str; true_v_allele: str
    label_resolution: str; in_catalog: str; tier: str; label_source: str; label_citation: str
    alias_source: str; evidence_locator: str; humanization_status: str; true_j: str
    exclusion_reason: str

def sha256_seq(vh_aa: str) -> str:
    return hashlib.sha256(vh_aa.encode()).hexdigest()

def validate_row(d: dict) -> list[str]:
    v = []
    missing = [f for f in LABEL_FIELDS if f not in d]
    if missing:
        return [f"missing fields: {missing}"]
    if d["lineage_representative"] not in _BOOL: v.append("lineage_representative must be boolean true|false")
    if d["in_catalog"] not in _BOOL: v.append("in_catalog must be boolean true|false")
    if d["tier"] not in _TIER: v.append(f"tier must be {_TIER}")
    if d["chain_class"] not in _CHAIN: v.append(f"chain_class must be in {_CHAIN}")
    if d["label_resolution"] not in _RES: v.append(f"label_resolution invalid")
    if not _HEX64.match(d["exact_vh_id"]): v.append("exact_vh_id must be a 64-char hex hash")
    if not _HEX64.match(d["raw_file_sha256"]): v.append("raw_file_sha256 must be a 64-char hex hash")
    if not _ISO.match(d["retrieval_date"]): v.append("retrieval_date must be ISO YYYY-MM-DD")
    if not d["label_citation"]: v.append("every admitted row needs a label_citation")
    if not d["evidence_locator"]: v.append("every admitted row needs an evidence_locator")
    if d["tier"] == "gold":
        if not d["true_v_family"]: v.append("gold row needs a true_v_family")
        if d["label_resolution"] not in ("subgroup_only", "exact_gene", "exact_allele"):
            v.append("gold design-ancestry label must resolve to a family")
    if d["tier"] == "concordance" and not d["true_v_gene"]:
        v.append("concordance row needs a true_v_gene")
    return v

def validate_cohort(rows: list, hydrated: dict, crosswalk: dict) -> list[str]:
    v = []
    for r in rows:
        v += [f"{r.name}: {e}" for e in validate_row(r.__dict__)]
        seq = hydrated.get(r.exact_vh_id)
        if seq is None:
            v.append(f"{r.name}: no hydrated sequence for exact_vh_id")
        elif sha256_seq(seq) != r.exact_vh_id:
            v.append(f"{r.name}: exact_vh_id != sha256(vh_aa)")
    # one deterministic representative per lineage
    by_lin: dict[str, list] = {}
    for r in rows:
        by_lin.setdefault(r.lineage_id, []).append(r)
    for lin, members in by_lin.items():
        reps = [m for m in members if m.lineage_representative == "true"]
        smallest = min(m.exact_vh_id for m in members)
        if len(reps) != 1:
            v.append(f"lineage {lin}: exactly one representative required, got {len(reps)}")
        elif reps[0].exact_vh_id != smallest:
            v.append(f"lineage {lin}: representative must be the smallest exact_vh_id (deterministic)")
    # no conflicting labels for one VH
    by_vh: dict[str, set] = {}
    for r in rows:
        by_vh.setdefault(r.exact_vh_id, set()).add((r.true_v_family, r.true_v_gene, r.tier))
    for vid, labels in by_vh.items():
        if len({(f, g) for f, g, _ in labels}) > 1:
            v.append(f"vh {vid[:8]}: conflicting labels for one sequence")
    # gold alias in crosswalk; concordance marked alignment-derived
    for r in rows:
        if r.tier == "gold" and r.label_source not in crosswalk:
            v.append(f"{r.name}: gold label_source not in frozen crosswalk")
        if r.tier == "concordance":
            marker = (r.label_source + " " + r.alias_source).lower()
            if not any(k in marker for k in _ALIGN):
                v.append(f"{r.name}: concordance row must be marked alignment-derived (IMGT/ANARCI/…)")
    return v

def load_labels(path: str) -> tuple[list, list]:
    admitted, audit = [], []
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            errs = validate_row(d)
            # excluded/out-of-scope rows are audited, never admitted
            if d.get("exclusion_reason") or d.get("chain_class") not in ("in_scope",):
                audit.append({**d, "_reasons": d.get("exclusion_reason") or "out_of_scope"})
            elif errs:
                audit.append({**d, "_reasons": "; ".join(errs)})
            else:
                admitted.append(LabelRow(**{f: d[f] for f in LABEL_FIELDS}))
    return admitted, audit

def cohort_sha256(rows: list, hydrated: dict) -> str:
    h = hashlib.sha256()
    for line in sorted("\t".join(str(getattr(r, f)) for f in LABEL_FIELDS) +
                       "\t" + hydrated.get(r.exact_vh_id, "") for r in rows):
        h.update(line.encode()); h.update(b"\n")
    return h.hexdigest()
```

- [ ] **Step 4: Run tests; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_cohort.py -q`
Expected: PASS (all).

```bash
git add src/alignair/aa/phase3/ tests/alignair/aa/phase3/
git commit -m "feat(aa-phase3): label schema (no vh_aa) + row & cohort validators (hash/date/enum/rep/conflict/crosswalk)"
```

### Task 2: Acquisition script + git-ignored hydration + individually-cited crosswalk

**Files:**
- Create: `scripts/aa/phase3/acquire.py`
- Create: `ART/alias_crosswalk.tsv` (starts with a header only — entries added per verified source)
- Modify: `.gitignore` (ignore hydrated sequences + per-row predictions)
- Modify: `src/alignair/aa/phase3/cohort.py` (add `load_crosswalk`, `load_hydrated`)
- Test: `tests/alignair/aa/phase3/test_acquire.py`

**Interfaces:**
- Produces: `load_crosswalk(path) -> dict[str, tuple[str,str,str]]` (`alias -> (imgt_family, imgt_gene_or_empty, citation)`); `load_hydrated(path) -> dict[str,str]` (`exact_vh_id -> vh_aa`). `acquire.py` fetches the current Thera-SAbDab file (endpoint captured live), writes `ART/manifest.tsv` (INN + metadata + per-seq sha256, **no sequences**) and the git-ignored `ART/hydrated_sequences.tsv` (`exact_vh_id\tvh_aa`), and updates `ART/sources.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_acquire.py
import pytest
from alignair.aa.phase3.cohort import load_crosswalk

def test_crosswalk_requires_citation(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("alias\timgt_family\timgt_gene\tcitation\nVH subgroup III\tIGHV3\t\tScaviner1999\n")
    cw = load_crosswalk(str(p))
    assert cw["VH subgroup III"] == ("IGHV3", "", "Scaviner1999")

def test_crosswalk_rejects_uncited_row(tmp_path):
    p = tmp_path / "x.tsv"
    p.write_text("alias\timgt_family\timgt_gene\tcitation\nVH subgroup III\tIGHV3\t\t\n")
    with pytest.raises(ValueError):
        load_crosswalk(str(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_acquire.py -q`
Expected: FAIL (`load_crosswalk` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `cohort.py`:

```python
def load_crosswalk(path: str) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            if not d.get("citation"):
                raise ValueError(f"crosswalk row for {d.get('alias')!r} has no citation")
            out[d["alias"]] = (d["imgt_family"], d.get("imgt_gene", ""), d["citation"])
    return out

def load_hydrated(path: str) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            out[d["exact_vh_id"]] = d["vh_aa"]
    return out
```

Create `ART/alias_crosswalk.tsv` with ONLY the header `alias\timgt_family\timgt_gene\tcitation` — **no pre-seeded rows**. Each entry is added later only after the Kabat/consensus-subgroup→IMGT-family equivalence is verified against an authoritative source (e.g. IMGT Scaviner et al.), never assumed by matching numbers.

Write `scripts/aa/phase3/acquire.py`: fetch the live Thera-SAbDab sequence file, compute per-sequence sha256 → `exact_vh_id`, write `ART/manifest.tsv` (INN, format, isotype, LC-category, `exact_vh_id`, per-seq sha256 — **no sequences**) and git-ignored `ART/hydrated_sequences.tsv`; update `ART/sources.json` (endpoint, retrieval date, licensing: redistribution not established → manifest+hydration).

Append to `.gitignore`:
```
docs/superpowers/artifacts/2026-07-21-aa-phase3/hydrated_sequences.tsv
docs/superpowers/artifacts/2026-07-21-aa-phase3/predictions.jsonl
```

- [ ] **Step 4: Run test; verify ignore; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_acquire.py -q && git check-ignore docs/superpowers/artifacts/2026-07-21-aa-phase3/hydrated_sequences.tsv && echo ignored`
Expected: PASS then `ignored`.

```bash
git add scripts/aa/phase3/acquire.py src/alignair/aa/phase3/cohort.py tests/alignair/aa/phase3/test_acquire.py docs/superpowers/artifacts/2026-07-21-aa-phase3/alias_crosswalk.tsv .gitignore
git commit -m "feat(aa-phase3): acquisition (manifest+ignored hydration, no redistribution) + cited crosswalk loader (no blanket seed)"
```

### Task 3: Curate Gold + concordance labels, verify crosswalk with user, freeze bundle — **STOP-AND-REPORT GATE**

**Files (committed):** `ART/gold_labels.tsv`, `ART/concordance_labels.tsv`, `ART/candidate_inventory.tsv`, `ART/inclusion_rules.md`, `ART/bundle.manifest.json`. **Git-ignored:** `ART/hydrated_sequences.tsv`.

- [ ] **Step 1: Curate candidates (human-verified, primary sources; every Gold label + crosswalk entry reviewed with the user).** For each therapeutic with a **source-declared** human IGHV design ancestry, record the `LABEL_FIELDS` (metadata + declared template in `label_source` + `label_resolution` + `evidence_locator` + `label_citation`), resolve the template to an IMGT **family** via a **verified, cited** crosswalk entry (`alias_source`) — never by alignment. Out-of-scope (`chain_class` ≠ `in_scope`), ambiguous, bispecific-unassigned, duplicates → `candidate_inventory.tsv` with a reason. Concordance rows (IMGT/ANARCI exact-gene) curated separately, `label_source`/`alias_source` explicitly naming the aligner.

- [ ] **Step 2: Deduplicate + assign lineages deterministically.** `exact_vh_id = sha256(vh_aa)`; collapse identical VH. Assign `lineage_id` (shared template/parental lineage). Set `lineage_representative="true"` on exactly the member with the lexicographically smallest `exact_vh_id`.

- [ ] **Step 3: Validate the whole cohort (no predictions).**

Run:
```
PYTHONPATH=src .venv/bin/python -c "
from alignair.aa.phase3.cohort import load_labels, load_hydrated, load_crosswalk, validate_cohort, cohort_sha256
A='docs/superpowers/artifacts/2026-07-21-aa-phase3/'
g,ga=load_labels(A+'gold_labels.tsv'); c,ca=load_labels(A+'concordance_labels.tsv')
hy=load_hydrated(A+'hydrated_sequences.tsv'); cw=load_crosswalk(A+'alias_crosswalk.tsv')
errs=validate_cohort(g+c, hy, cw)
assert not errs, errs
lign={r.lineage_id for r in g}
print('gold rows',len(g),'gold lineages',len(lign),'concordance',len(c),'audited',len(ga)+len(ca))"
```
Every admitted row must pass; violations block the gate.

- [ ] **Step 4: Freeze the entire input bundle.** Hash Gold, concordance, crosswalk, source manifest, inclusion rules, and the hydrated join into one manifest:

Run:
```
PYTHONPATH=src .venv/bin/python -c "
import json,hashlib,pathlib
A=pathlib.Path('docs/superpowers/artifacts/2026-07-21-aa-phase3')
files=['gold_labels.tsv','concordance_labels.tsv','alias_crosswalk.tsv','manifest.tsv','inclusion_rules.md','hydrated_sequences.tsv']
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
per={f:h(A/f) for f in files}
overall=hashlib.sha256(''.join(per[f] for f in files).encode()).hexdigest()
(A/'bundle.manifest.json').write_text(json.dumps({'files':per,'bundle_sha256':overall},indent=2))
print('bundle',overall)"
```

- [ ] **Step 5: GATE.** Count **independent lineage representatives** in `gold_labels.tsv` **after dedup + cited alias resolution + deterministic lineage assignment**. If **< 15**: **STOP** — write the shortfall + options into `feasibility_report.md`, report to the user that family-resolution therapeutic *design ancestry* was also unavailable at defensible scale, and do **not** enter Milestone 2. If **≥ 15**: record the count + rules in `inclusion_rules.md` and continue.

- [ ] **Step 6: Commit (labels/metadata/manifest only — never sequences)**

```bash
git add docs/superpowers/artifacts/2026-07-21-aa-phase3/gold_labels.tsv docs/superpowers/artifacts/2026-07-21-aa-phase3/concordance_labels.tsv docs/superpowers/artifacts/2026-07-21-aa-phase3/candidate_inventory.tsv docs/superpowers/artifacts/2026-07-21-aa-phase3/inclusion_rules.md docs/superpowers/artifacts/2026-07-21-aa-phase3/bundle.manifest.json docs/superpowers/artifacts/2026-07-21-aa-phase3/manifest.tsv
git commit -m "data(aa-phase3): frozen input bundle — Gold(family design-ancestry)+concordance labels, cited crosswalk, bundle manifest (gate: N lineages)"
```

---

## Milestone 2 — Reusable inference, calibration, and taxonomy utilities (TDD)

### Task 4: Hardened AA-head inference utility (shared with the snapshot)

**Files:** Create `src/alignair/aa/infer.py`; Modify `scripts/aa/calibrate_stage2.py`; Test `tests/alignair/aa/test_infer.py`.

**Interfaces:** Produces `class_probs(model, seqs, cfg, device, key, batch=256) -> np.ndarray` — softmax `(len(seqs), n_classes)` for `key ∈ {v_class_logits, j_class_logits, d_class_logits}`. **Raises `ValueError` on empty input, on any sequence longer than `cfg.max_seq_length`, or on non-standard-AA characters — never silently truncates.**

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_infer.py
import numpy as np, pytest
from alignair.aa.infer import class_probs
from alignair.aa.strategic import strategic_setup

def _setup():
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model.eval(); return model, cfg, class_maps

def test_shape_and_normalisation():
    model, cfg, cmaps = _setup()
    p = class_probs(model, ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "QVQLQESGPGLVKPSETLSLTCTVSGGSIS"],
                    cfg, "cpu", "v_class_logits", batch=8)
    assert p.shape == (2, cmaps["V"].n_classes()) and np.allclose(p.sum(1), 1.0, atol=1e-5)

def test_empty_input_raises():
    model, cfg, _ = _setup()
    with pytest.raises(ValueError):
        class_probs(model, [], cfg, "cpu", "v_class_logits")

def test_oversized_sequence_rejected_not_truncated():
    model, cfg, _ = _setup()
    with pytest.raises(ValueError):
        class_probs(model, ["A" * (cfg.max_seq_length + 1)], cfg, "cpu", "v_class_logits")

def test_invalid_alphabet_rejected():
    model, cfg, _ = _setup()
    with pytest.raises(ValueError):
        class_probs(model, ["EVQLZZZ123"], cfg, "cpu", "v_class_logits")
```

- [ ] **Step 2: Run to verify fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_infer.py -q`  → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/infer.py
"""Single hardened AA-head inference path, shared by Stage-2 calibration and Phase-3 eval."""
from __future__ import annotations
import numpy as np, torch
from .tokenizer import AAProteinTokenizer

_AA = set("ACDEFGHIKLMNPQRSTVWY")

def class_probs(model, seqs, cfg, device, key, batch: int = 256) -> np.ndarray:
    if not seqs:
        raise ValueError("class_probs: empty sequence list")
    L = cfg.max_seq_length
    bad_len = [i for i, s in enumerate(seqs) if len(s) > L]
    if bad_len:
        raise ValueError(f"class_probs: sequences exceed max_seq_length {L}: idx {bad_len[:5]}")
    bad_aa = [i for i, s in enumerate(seqs) if set(s) - _AA]
    if bad_aa:
        raise ValueError(f"class_probs: non-standard AA characters at idx {bad_aa[:5]}")
    tok = AAProteinTokenizer(); out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        toks = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):
            e = tok.encode(s); toks[j, :len(e)] = torch.from_numpy(e)
        with torch.no_grad():
            lg = model({"tokenized_sequence": toks.to(device)})[key].cpu()
        out.append(torch.softmax(lg, -1).numpy())
    return np.concatenate(out, 0)
```

In `scripts/aa/calibrate_stage2.py`: replace the local `_probs` definition with `from alignair.aa.infer import class_probs as _probs` (call sites unchanged).

- [ ] **Step 4: Run tests; verify calibration unchanged; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_infer.py -q`  → PASS.
Run (behaviour-preserving check, temp file, no hard-coded path):
```
OUT=$(mktemp); PYTHONPATH=src .venv/bin/python scripts/aa/calibrate_stage2.py --model experiments/aa/strategic/model_stage1_PASS.pt --device cuda --alpha 0.05 --n 5000 --out "$OUT" && diff <(jq -S '.genes' docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json) <(jq -S '.genes' "$OUT") && echo "calibration unchanged"
```
Expected: `calibration unchanged`.

```bash
git add src/alignair/aa/infer.py scripts/aa/calibrate_stage2.py tests/alignair/aa/test_infer.py
git commit -m "feat(aa): hardened class_probs (reject empty/oversized/invalid) shared with calibrate_stage2"
```

### Task 5: Frozen-calibration loader + NON-circular fixture compatibility test

**Files:** Modify `src/alignair/aa/conformal.py`; Create `tests/alignair/aa/fixtures/conformal_fixture.json`; Test `tests/alignair/aa/test_frozen_conformal.py`.

**Interfaces:** Produces `class FrozenConformal` (`.qhat`, `.predict_set(prob_row)->set`) and `FrozenConformal.from_calibration_json(cal, gene, group="full", lam=0.0, k_reg=1)`.

- [ ] **Step 1: Create the fixture with INDEPENDENTLY hand-computed expected sets (no call to `aps_set`).**

APS set = smallest top-ranked prefix whose cumulative probability reaches `qhat`. Compute by hand and commit:
```json
// tests/alignair/aa/fixtures/conformal_fixture.json
{
  "qhat": 0.6,
  "cases": [
    {"probs": [0.5, 0.3, 0.2], "expected_set": [0, 1]},
    {"probs": [0.7, 0.2, 0.1], "expected_set": [0]},
    {"probs": [0.2, 0.5, 0.3], "expected_set": [1, 2]}
  ]
}
```
(Row 1: sorted desc 0.5,0.3,0.2; cum 0.5,0.8 — reaches 0.6 at the 2nd → classes {0,1}. Row 2: 0.7 ≥ 0.6 → {0}. Row 3: sorted 0.5(idx1),0.3(idx2); cum 0.5,0.8 → {1,2}.) These are derived from the APS definition by hand, independent of the implementation under test.

- [ ] **Step 2: Write the failing test**

```python
# tests/alignair/aa/test_frozen_conformal.py
import json, numpy as np, pathlib
from alignair.aa.conformal import FrozenConformal

FIX = json.loads((pathlib.Path(__file__).parent / "fixtures/conformal_fixture.json").read_text())

def test_predict_set_matches_hand_computed():
    fc = FrozenConformal(qhat=FIX["qhat"])
    for case in FIX["cases"]:
        assert sorted(fc.predict_set(np.array(case["probs"]))) == case["expected_set"]

def test_from_calibration_json_reads_full_group_quantile():
    cal = {"genes": {"V": {"calibration_frozen": {"q_mondrian": {"full": 0.5, "short": 0.6}}}}}
    assert FrozenConformal.from_calibration_json(cal, "V", "full").qhat == 0.5
```

- [ ] **Step 3: Run to verify fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_frozen_conformal.py -q`  → FAIL.

- [ ] **Step 4: Write minimal implementation** — add to `conformal.py`:

```python
class FrozenConformal:
    """Deployable calibrator: a fixed qhat loaded from frozen calibration.json. No re-fitting."""
    def __init__(self, qhat: float, lam: float = 0.0, k_reg: int = 1):
        self.qhat, self.lam, self.k_reg = float(qhat), lam, k_reg
    def predict_set(self, prob_row) -> set:
        return aps_set(prob_row, self.qhat, self.lam, self.k_reg)
    @classmethod
    def from_calibration_json(cls, cal, gene, group="full", lam=0.0, k_reg=1):
        return cls(cal["genes"][gene]["calibration_frozen"]["q_mondrian"][group], lam, k_reg)
```

- [ ] **Step 5: Run test; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_frozen_conformal.py -q`  → PASS.
```bash
git add src/alignair/aa/conformal.py tests/alignair/aa/test_frozen_conformal.py tests/alignair/aa/fixtures/conformal_fixture.json
git commit -m "feat(aa): FrozenConformal loader + non-circular hand-computed fixture test"
```

### Task 6: Gene/family taxonomy + class expansion

**Files:** Create `src/alignair/aa/taxonomy.py`; Test `tests/alignair/aa/test_taxonomy.py`.
**Interfaces:** `gene_of(allele)->str`; `family_of(name)->str`; `class_genes(cid, cmap)->set`; `class_families(cid, cmap)->set`; `catalogue_families(cmap)->set`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_taxonomy.py
import pytest
from alignair.aa.taxonomy import gene_of, family_of, class_families, catalogue_families

class FakeMap:
    class_members = {0: ["IGHV3-23*01", "IGHV3-23*02"], 1: ["IGHV1-69D*01"]}

def test_parsing():
    assert gene_of("IGHV3-23*01") == "IGHV3-23"
    assert family_of("IGHV3-23*01") == "IGHV3"
    assert family_of("IGHV1-69D*01") == "IGHV1"

def test_family_parse_failure_raises():
    with pytest.raises(ValueError):
        family_of("notagene")

def test_class_and_catalogue_families():
    assert class_families(0, FakeMap()) == {"IGHV3"}
    assert catalogue_families(FakeMap()) == {"IGHV3", "IGHV1"}
```

- [ ] **Step 2: Run to verify fail** → `pytest tests/alignair/aa/test_taxonomy.py -q` FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/taxonomy.py
"""IMGT allele -> gene -> family canonicalisation + class expansion (no alignment)."""
from __future__ import annotations
import re
_FAM = re.compile(r"^(IGH[VDJ]\d+)")

def gene_of(allele: str) -> str:
    return allele.split("*")[0]

def family_of(name: str) -> str:
    m = _FAM.match(name)
    if not m:
        raise ValueError(f"cannot parse IMGT family from {name!r}")
    return m.group(1)

def class_genes(cid: int, cmap) -> set:
    return {gene_of(a) for a in cmap.class_members[cid]}

def class_families(cid: int, cmap) -> set:
    return {family_of(a) for a in cmap.class_members[cid]}

def catalogue_families(cmap) -> set:
    fams = set()
    for members in cmap.class_members.values():
        fams |= {family_of(a) for a in members}
    return fams
```

- [ ] **Step 4: Run test; commit** → PASS.
```bash
git add src/alignair/aa/taxonomy.py tests/alignair/aa/test_taxonomy.py
git commit -m "feat(aa): allele->gene->family taxonomy + class/catalogue expansion"
```

---

## Milestone 3 — Scorer (TDD)

### Task 7: Exact one-sided binomial + coverage decision

**Files:** Create `src/alignair/aa/phase3/stats.py`; Test `tests/alignair/aa/phase3/test_stats.py`.
**Interfaces:** `clopper_pearson(k, n, alpha)->(lo,hi)` (one-sided bounds at level 1−α); `coverage_decision(k, n, alpha, allow_support)->str` (never `"supported"` when `allow_support=False`).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_stats.py
from alignair.aa.phase3.stats import clopper_pearson, coverage_decision

def test_bounds_bracket_estimate():
    lo, hi = clopper_pearson(9, 10, 0.05); assert 0 <= lo <= 0.9 <= hi <= 1

def test_perfect_small_n_cannot_support():
    assert coverage_decision(15, 15, 0.05, allow_support=True) == "inconclusive"

def test_clear_failure():
    assert coverage_decision(10, 20, 0.05, allow_support=True) == "failed"

def test_family_endpoint_never_supports():
    assert coverage_decision(60, 60, 0.05, allow_support=False) != "supported"
```

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase3/stats.py
"""Exact one-sided binomial (Clopper-Pearson) bounds + the Phase-3 coverage decision."""
from __future__ import annotations
from scipy.stats import beta

def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))
    return lo, hi

def coverage_decision(k: int, n: int, alpha: float, allow_support: bool) -> str:
    lo, hi = clopper_pearson(k, n, alpha)
    if hi < 0.95:
        return "failed"
    if allow_support and lo >= 0.95:
        return "supported"
    return "inconclusive"
```

- [ ] **Step 4: Run test; commit** → PASS.
```bash
git add src/alignair/aa/phase3/stats.py tests/alignair/aa/phase3/test_stats.py
git commit -m "feat(aa-phase3): exact one-sided binomial bounds + coverage decision (family never supports)"
```

### Task 8: Scorer — full contract (accuracy, coverage, sizes, OOV, in-catalog, cluster bootstrap, concordance)

**Files:** Create `src/alignair/aa/phase3/scorer.py`; Test `tests/alignair/aa/phase3/test_scorer.py`.
**Interfaces:** `@dataclass Prediction(top_class:int, set_classes:set)`; `score_gold(rows, preds, cmap, alpha, seed, n_boot=2000)->dict`; `score_concordance(rows, preds, cmap, alpha)->dict`. Both **require `len(rows)==len(preds)` and non-empty** (raise `ValueError` otherwise). Binding numbers use the one representative per lineage; descriptive stats span all deduplicated rows; the bootstrap resamples **lineage clusters**.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_scorer.py
import pytest
from types import SimpleNamespace
from alignair.aa.phase3.scorer import Prediction, score_gold, score_concordance

class FakeMap:  # class0 -> IGHV3-23 (2 alleles); class1 -> IGHV1-69
    class_members = {0: ["IGHV3-23*01", "IGHV3-23*02"], 1: ["IGHV1-69*01"]}

def _r(name, lin, fam, rep, cat="true", gene=""):
    return SimpleNamespace(name=name, lineage_id=lin, lineage_representative=rep,
                           true_v_family=fam, true_v_gene=gene, in_catalog=cat)

def test_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        score_gold([], [], FakeMap(), 0.05, 0)
    with pytest.raises(ValueError):
        score_gold([_r("a","a","IGHV3","true")], [], FakeMap(), 0.05, 0)

def test_perfect_family_but_small_n_inconclusive_and_sizes_reported():
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "true") for i in range(15)]
    preds = [Prediction(0, {0}) for _ in rows]
    out = score_gold(rows, preds, FakeMap(), 0.05, 0)
    assert out["binding"]["top_class_family_accuracy"] == 1.0
    assert out["binding"]["decision"] in ("inconclusive", "failed") != "supported"
    assert out["binding"]["n_lineages"] == 15
    assert out["set_size"]["families"]["mean"] == 1.0   # class 0 -> single family
    assert out["set_size"]["alleles"]["mean"] == 2.0    # class 0 -> two alleles
    assert "bootstrap_ci_family_accuracy" in out["descriptive"]

def test_family_miss_is_failed_and_oov_counted():
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "true") for i in range(10)]
    rows[0] = _r("m0", "L0", "IGHVX", "true", cat="false")   # OOV truth family
    preds = [Prediction(1, {1}) for _ in rows]              # predict IGHV1 != IGHV3
    out = score_gold(rows, preds, FakeMap(), 0.05, 0)
    assert out["binding"]["decision"] == "failed"
    assert out["oov"]["oov_rate"] > 0.0
    assert "in_catalog_accuracy" in out["oov"]

def test_descriptive_spans_all_deduplicated_rows_not_just_reps():
    rows = [_r("rep","L","IGHV3","true"), _r("mem","L","IGHV3","false")]
    preds = [Prediction(0, {0}), Prediction(1, {1})]
    out = score_gold(rows, preds, FakeMap(), 0.05, 0)
    assert out["binding"]["n_lineages"] == 1               # one representative
    assert out["descriptive"]["n_all_vh"] == 2             # both rows

def test_concordance_reports_agreement_and_interval():
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "true", gene="IGHV3-23") for i in range(12)]
    preds = [Prediction(0, {0}) for _ in rows]
    out = score_concordance(rows, preds, FakeMap(), 0.05)
    assert out["label_kind"].startswith("alignment-derived")
    assert out["top_class_gene_agreement"] == 1.0
    assert len(out["agreement_ci"]) == 2
```

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase3/scorer.py
"""Phase-3 scorer. Binding = one rep/lineage; descriptive = all dedup VHs; bootstrap = lineage clusters.
Gold family endpoint never emits 'supported'. Concordance is alignment-derived, not ground truth."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..taxonomy import class_families, class_genes, catalogue_families
from .stats import clopper_pearson, coverage_decision

@dataclass
class Prediction:
    top_class: int
    set_classes: set

def _check(rows, preds):
    if len(rows) != len(preds):
        raise ValueError(f"rows ({len(rows)}) and preds ({len(preds)}) length mismatch")
    if not rows:
        raise ValueError("empty cohort")

def _is_rep(r):
    return str(r.lineage_representative).lower() == "true"

def _set_family(p, cmap):
    return set().union(*[class_families(c, cmap) for c in p.set_classes]) if p.set_classes else set()

def _set_gene(p, cmap):
    return set().union(*[class_genes(c, cmap) for c in p.set_classes]) if p.set_classes else set()

def _sizes(preds, cmap):
    def stat(vals):
        a = np.asarray(vals, float)
        return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}
    classes = [len(p.set_classes) for p in preds]
    fams = [len(_set_family(p, cmap)) for p in preds]
    genes = [len(_set_gene(p, cmap)) for p in preds]
    alleles = [len(set().union(*[cmap.class_members[c] for c in p.set_classes]) if p.set_classes else set())
               for p in preds]
    return {"classes": stat(classes), "families": stat(fams), "genes": stat(genes), "alleles": stat(alleles)}

def score_gold(rows, preds, cmap, alpha, seed, n_boot: int = 2000) -> dict:
    _check(rows, preds)
    cat_fams = catalogue_families(cmap)
    # binding: representatives only
    ridx = [i for i, r in enumerate(rows) if _is_rep(r)]
    lids = [rows[i].lineage_id for i in ridx]
    top = {rows[i].lineage_id: (rows[i].true_v_family in class_families(preds[i].top_class, cmap)) for i in ridx}
    cov = {rows[i].lineage_id: (rows[i].true_v_family in _set_family(preds[i], cmap)) for i in ridx}
    n = len(ridx); kt = sum(top.values()); kc = sum(cov.values())
    # lineage-cluster bootstrap of the binding top-class accuracy
    rng = np.random.default_rng(seed); uniq = list(dict.fromkeys(lids)); boot = []
    for _ in range(n_boot):
        samp = rng.choice(len(uniq), len(uniq), replace=True)
        boot.append(np.mean([top[uniq[j]] for j in samp]))
    # descriptive: all deduplicated rows
    all_hits = [r.true_v_family in class_families(p.top_class, cmap) for r, p in zip(rows, preds)]
    # oov / in-catalog conditional accuracy
    non_oov = [(r, p) for r, p in zip(rows, preds) if r.true_v_family in cat_fams]
    in_cat_hits = [r.true_v_family in class_families(p.top_class, cmap) for r, p in non_oov]
    return {
        "binding": {
            "n_lineages": n,
            "top_class_family_accuracy": kt / n,
            "top_class_family_ci": list(clopper_pearson(kt, n, alpha)),
            "family_set_coverage": kc / n,
            "family_set_coverage_ci": list(clopper_pearson(kc, n, alpha)),
            "decision": coverage_decision(kc, n, alpha, allow_support=False),
        },
        "descriptive": {
            "n_all_vh": len(rows),
            "top_class_family_accuracy_all": float(np.mean(all_hits)),
            "bootstrap_ci_family_accuracy": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        },
        "oov": {
            "oov_rate": 1 - len(non_oov) / len(rows),
            "in_catalog_accuracy": (float(np.mean(in_cat_hits)) if in_cat_hits else None),
            "end_to_end_accuracy": float(np.mean(all_hits)),
        },
        "set_size": _sizes(preds, cmap),
    }

def score_concordance(rows, preds, cmap, alpha) -> dict:
    _check(rows, preds)
    ridx = [i for i, r in enumerate(rows) if _is_rep(r)]
    agree = [rows[i].true_v_gene in class_genes(preds[i].top_class, cmap) for i in ridx]
    cov = [rows[i].true_v_gene in _set_gene(preds[i], cmap) for i in ridx]
    n = len(ridx); ka = sum(agree); kc = sum(cov)
    return {
        "label_kind": "alignment-derived (IMGT/ANARCI) — concordance, NOT ground truth",
        "n_lineages": n,
        "top_class_gene_agreement": ka / n if n else 0.0,
        "agreement_ci": list(clopper_pearson(ka, n, alpha)) if n else [0.0, 1.0],
        "gene_set_coverage": kc / n if n else 0.0,
        "gene_set_coverage_ci": list(clopper_pearson(kc, n, alpha)) if n else [0.0, 1.0],
        "set_size": _sizes([preds[i] for i in ridx], cmap),
    }
```

- [ ] **Step 4: Run test; commit** → PASS.
```bash
git add src/alignair/aa/phase3/scorer.py tests/alignair/aa/phase3/test_scorer.py
git commit -m "feat(aa-phase3): full scorer — binding/descriptive/OOV/in-catalog/set-sizes/lineage-cluster bootstrap/concordance"
```

---

## Milestone 4 — Evaluation run & report

### Task 9: Eval script — verify bundle manifest, predict, score, emit artifacts

**Files:** Create `scripts/aa/phase3/evaluate.py`; Outputs: git-ignored `ART/predictions.jsonl`, committed `ART/phase3_eval.json`.
**Interfaces:** Consumes `load_labels`, `load_hydrated`, `load_crosswalk`, `validate_cohort`, `cohort_sha256` (T1/T2); `class_probs` (T4); `FrozenConformal` (T5); `catalogue_families` (T6); `score_gold`, `score_concordance`, `Prediction` (T8).

- [ ] **Step 1: Write the eval script.** In order, hard-failing on any mismatch: (1) assert model sha256 == `54eef812…`; (2) assert `class_maps["V"].db_hash` == `e1a455b8…` and == `calibration.json.provenance.class_map_db_hash.V`, and α/bin-defs match provenance; (3) **verify `bundle.manifest.json`** — recompute each file's sha256 and the overall bundle hash, assert equal; (4) `load_labels` + `load_hydrated` + `load_crosswalk`, run `validate_cohort` (assert empty), assert `cohort_sha256(gold+concordance, hydrated)` recorded; (5) build `FrozenConformal.from_calibration_json(cal, "V", "full")`; (6) `class_probs` over hydrated VH (**not** the committed labels); (7) `Prediction(argmax, fc.predict_set(row))`; (8) `score_gold` + `score_concordance`; (9) write git-ignored `predictions.jsonl` **keyed by `exact_vh_id`, no `vh_aa`**, and committed `phase3_eval.json` (decision `failed`/`inconclusive`; family-accuracy anchor; concordance block; licensing/attribution; audit reason counts; bundle + cohort hashes).

- [ ] **Step 2: Run against the frozen bundle**

Run: `PYTHONPATH=src .venv/bin/python scripts/aa/phase3/evaluate.py --device cuda`
Expected: writes `phase3_eval.json`; `decision ∈ {failed, inconclusive}`.

- [ ] **Step 3: Sanity-assert output invariants**

Run: `PYTHONPATH=src .venv/bin/python -c "import json; r=json.load(open('docs/superpowers/artifacts/2026-07-21-aa-phase3/phase3_eval.json')); assert r['gold']['binding']['decision'] in ('failed','inconclusive'); assert r['gold']['binding']['decision']!='supported'; assert 'concordance' in r and r['bundle_sha256']; print('ok', r['gold']['binding']['top_class_family_accuracy'], r['gold']['binding']['decision'])"`
Expected: `ok <acc> <decision>`. Verify predictions are ignored: `git check-ignore docs/superpowers/artifacts/2026-07-21-aa-phase3/predictions.jsonl`.

- [ ] **Step 4: Commit (aggregate only; predictions stay ignored)**

```bash
git add scripts/aa/phase3/evaluate.py docs/superpowers/artifacts/2026-07-21-aa-phase3/phase3_eval.json
git commit -m "feat(aa-phase3): eval runner (bundle-manifest verify + precondition asserts) -> aggregate JSON"
```

### Task 10: Truncation stress test (separate, synthetic, frozen seeds)

**Files:** Create `scripts/aa/phase3/truncation_stress.py`; Output committed `ART/truncation_stress.json`.
**Interfaces:** `aa_truncate` (existing), `class_probs`, `FrozenConformal.from_calibration_json(cal, "V", group)` for `group∈{full,trunc,short}`.

- [ ] **Step 1: Write the stress script.** Deterministically crop each hydrated VH with `aa_truncate` to the exact bin ranges from `calibration.json` (`trunc`=keep∈[0.6,0.9], `short`=keep∈[0.3,0.6)), `np.random.default_rng(seed)`, paired (same VH across bins), each bin scored with its own frozen quantile. Report family-set coverage per bin. Label the artifact a **synthetic stress test on real sequences**, separate from the primary result.

- [ ] **Step 2: Run; sanity-check** → `by_bin` has keys `full/trunc/short`; commit.
```bash
git add scripts/aa/phase3/truncation_stress.py docs/superpowers/artifacts/2026-07-21-aa-phase3/truncation_stress.json
git commit -m "feat(aa-phase3): synthetic truncation stress on real VH (frozen seeds, per-bin quantiles)"
```

### Task 11: PHASE3.md report + full suite green

**Files:** Create `ART/PHASE3.md`.

- [ ] **Step 1: Write `PHASE3.md`** per spec §1: family design-ancestry accuracy anchor as headline; the `failed`/`inconclusive` decision with its exact interval; concordance block labelled "alignment-derived, not ground truth"; truncation stress as a separate synthetic section; OOV + in-catalog accuracy; licensing/attribution; and the explicit statement that Phase 3 **cannot** validate the 95% allele/class guarantee (alignment-independent allele truth was not available at scale in this cohort). Softened language throughout (design ancestry, not germline ground truth).

- [ ] **Step 2: Full AA suite** → `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/ -q -m "not slow"` all pass.

- [ ] **Step 3: Commit**
```bash
git add docs/superpowers/artifacts/2026-07-21-aa-phase3/PHASE3.md
git commit -m "docs(aa-phase3): PHASE3.md — family design-ancestry anchor + concordance + honest scope"
```

---

## Self-Review notes (spec coverage)

- §1 framing, softened claims, two-way decision (never `supported`), one-rep/lineage independence: Tasks 7, 8, 9 + Global Constraints.
- §2 tiers, label schema (no vh_aa), citation/evidence gate, **cohort-level** validation (hash==sha256(seq), ISO dates, enums, one-deterministic-rep, no conflicts, crosswalk membership, concordance-marked), dedup/lineage, licensing (no raw seqs in Git; hydration+ignore), OOV, **bundle** freeze+hash: Tasks 1, 2, 3.
- §2 label-resolution taxonomy + family admission: Tasks 1, 8. Alias crosswalk (cited, **no blanket seed**, Kabat≠IMGT-by-number): Task 2.
- §3 preconditions (model sha, class-map fingerprint, calibration params, **bundle manifest** + cohort hash, `q_mondrian["full"]`, model-independent selection), coarsening ladder, top-class family-set, concordance, set sizes (classes/genes/families/alleles), in-catalog accuracy, OOV, **lineage-cluster** bootstrap, concordance intervals, row/pred alignment + empty-cohort guards: Tasks 6, 8, 9. Truncation stress (separate, frozen seeds, per-bin quantiles): Task 10.
- §4 components (label loader/validators, hardened inference util, FrozenConformal loader + **non-circular** fixture test, taxonomy, scorer, runner, report), artifacts, tests: Tasks 1, 4, 5, 6, 8, 9, 11.
- §5.1 gated family-Gold curation, gate **after dedup+alias+lineage**, no exploratory predictions, stop-and-report at 15 lineages: Task 3.
- Not-in-scope (D, retrain, recalibration): honored.
