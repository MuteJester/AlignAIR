# AA AlignAIR Phase 3 — Real-Protein Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the frozen simulator-domain AA IGH model on real therapeutic-antibody VH domains, reporting a zero-circularity **family-level** accuracy anchor (headline) plus an explicitly-labelled exact-gene **concordance** picture — never claiming to validate the 95% allele/class guarantee.

**Architecture:** A gated data-curation milestone (feasibility-first, no model inference) produces a frozen, hashed cohort of source-declared **family-resolution** Gold labels + an alignment-derived concordance tier. Then small, tested library utilities (inference, frozen-calibration loader, taxonomy) feed a scorer whose only binding test is an exact one-sided binomial on one representative VH per lineage. An eval script emits per-row predictions, an aggregate JSON with a machine-readable `failed`/`inconclusive` decision, and `PHASE3.md`.

**Tech Stack:** Python, PyTorch (inference only), NumPy, SciPy (`scipy.stats.beta` for Clopper–Pearson), the existing `alignair.aa` package, `pytest`.

## Global Constraints

- Branch `experiment/aminoacid-alignair` only; **never modify the frozen snapshot** (`experiments/aa/strategic/model_stage1_PASS.pt` sha `54eef812…`, `docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json`).
- **No model inference until the Milestone-1 feasibility gate passes** (cohort frozen + hashed first).
- **`supported` is never emitted.** The decision is `failed` or `inconclusive` only.
- Gold ground truth = **IGHV family** resolution; exact-gene = **concordance tier only**, never changes the decision, always labelled "alignment-derived, not ground truth".
- Binding statistical unit = **one deterministic representative VH per `lineage_id`** (lexicographically smallest `exact_vh_id`). Everything else is descriptive (cluster bootstrap by `lineage_id`).
- α = 0.05; exact one-sided binomial (Clopper–Pearson).
- Conformal quantile = **`q_mondrian["full"]`** from the frozen `calibration.json` (not pooled).
- Labels never derived by aligning the therapeutic sequence. Kabat/consensus-subgroup → IMGT-family via a **cited** crosswalk.
- No Claude co-author line in commits.
- Spec: `docs/superpowers/specs/2026-07-21-aa-phase3-realprotein-design.md`. Artifacts dir: `docs/superpowers/artifacts/2026-07-21-aa-phase3/`.

---

## Milestone 1 — Data acquisition & family-Gold curation (GATED, no model inference)

This milestone determines feasibility. It produces frozen, hashed data + a validator that **enforces** the contract, and ends at a stop-and-report gate. Its *content* (which antibodies, which templates, which citations) is **human-verified from primary sources** — never auto-generated, never alignment-derived. The validator makes an uncited or contract-violating row impossible to admit.

### Task 1: Cohort schema + validator (the contract, enforced in code)

**Files:**
- Create: `src/alignair/aa/phase3/__init__.py`
- Create: `src/alignair/aa/phase3/cohort.py`
- Test: `tests/alignair/aa/phase3/test_cohort.py`

**Interfaces:**
- Produces: `ROW_FIELDS: list[str]`; `class Row` (dataclass mirroring the spec §2 columns); `validate_row(d: dict) -> list[str]` (returns list of violation strings, empty = valid); `load_cohort(path: str) -> tuple[list[Row], list[dict]]` (returns admitted rows + audit rows); `cohort_sha256(rows: list[Row]) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_cohort.py
import pytest
from alignair.aa.phase3.cohort import validate_row, cohort_sha256, Row

BASE = dict(
    name="examplemab", arm_id="", vh_aa="EVQLVESGGGLVQPGGSLRLSCAAS", exact_vh_id="",
    lineage_id="examplemab", lineage_representative="true", chain_class="in_scope",
    seq_source="therasabdab", seq_source_version="2024-12-20", retrieval_date="2026-07-21",
    raw_file_sha256="deadbeef", true_v_family="IGHV3", true_v_gene="", true_v_allele="",
    label_resolution="subgroup_only", in_catalog="true", tier="gold",
    label_source="Patent US1234567 Example 1", label_citation="US1234567",
    alias_source="Kabat1991->IMGT", evidence_locator="col5 Example 1", humanization_status="humanized",
    true_j="", exclusion_reason="",
)

def test_valid_gold_row_passes():
    assert validate_row(BASE) == []

def test_gold_row_without_citation_is_rejected():
    bad = {**BASE, "label_citation": ""}
    assert any("citation" in v for v in validate_row(bad))

def test_gold_row_without_family_is_rejected():
    bad = {**BASE, "true_v_family": ""}
    assert any("family" in v for v in validate_row(bad))

def test_concordance_tier_requires_alignment_label_note():
    row = {**BASE, "tier": "concordance", "label_source": "ANARCI closest-match",
           "true_v_gene": "IGHV3-23", "label_resolution": "exact_gene"}
    assert validate_row(row) == []

def test_cohort_hash_is_order_independent_and_stable():
    r1 = Row(**BASE); r2 = Row(**{**BASE, "name": "other", "lineage_id": "other"})
    assert cohort_sha256([r1, r2]) == cohort_sha256([r2, r1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_cohort.py -q`
Expected: FAIL (module `alignair.aa.phase3.cohort` not found).

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase3/cohort.py
"""Frozen Phase-3 cohort schema + contract validator. No model, no alignment here."""
from __future__ import annotations
import csv, hashlib
from dataclasses import dataclass

ROW_FIELDS = [
    "name", "arm_id", "vh_aa", "exact_vh_id", "lineage_id", "lineage_representative",
    "chain_class", "seq_source", "seq_source_version", "retrieval_date", "raw_file_sha256",
    "true_v_family", "true_v_gene", "true_v_allele", "label_resolution", "in_catalog",
    "tier", "label_source", "label_citation", "alias_source", "evidence_locator",
    "humanization_status", "true_j", "exclusion_reason",
]
_AA = set("ACDEFGHIKLMNPQRSTVWY")
_RES = {"exact_allele", "exact_gene", "subgroup_only", "ambiguous", "unresolved"}

@dataclass
class Row:
    name: str; arm_id: str; vh_aa: str; exact_vh_id: str; lineage_id: str
    lineage_representative: str; chain_class: str; seq_source: str; seq_source_version: str
    retrieval_date: str; raw_file_sha256: str; true_v_family: str; true_v_gene: str
    true_v_allele: str; label_resolution: str; in_catalog: str; tier: str; label_source: str
    label_citation: str; alias_source: str; evidence_locator: str; humanization_status: str
    true_j: str; exclusion_reason: str

def validate_row(d: dict) -> list[str]:
    v = []
    missing = [f for f in ROW_FIELDS if f not in d]
    if missing:
        return [f"missing fields: {missing}"]
    if d["tier"] not in ("gold", "concordance"):
        v.append(f"tier must be gold|concordance, got {d['tier']!r}")
    if not d["vh_aa"] or set(d["vh_aa"]) - _AA:
        v.append("vh_aa must be a non-empty standard-AA string")
    if d["label_resolution"] not in _RES:
        v.append(f"label_resolution invalid: {d['label_resolution']!r}")
    if not d["label_citation"]:
        v.append("every admitted row needs a label_citation")
    if not d["evidence_locator"]:
        v.append("every admitted row needs an evidence_locator")
    if d["tier"] == "gold":
        if not d["true_v_family"]:
            v.append("gold row needs a true_v_family")
        if d["label_resolution"] not in ("subgroup_only", "exact_gene", "exact_allele"):
            v.append("gold family truth must resolve to a family (subgroup_only/exact_gene/exact_allele)")
    if d["tier"] == "concordance" and not d["true_v_gene"]:
        v.append("concordance row needs a true_v_gene")
    return v

def cohort_sha256(rows: list[Row]) -> str:
    h = hashlib.sha256()
    for line in sorted("\t".join(str(getattr(r, f)) for f in ROW_FIELDS) for r in rows):
        h.update(line.encode()); h.update(b"\n")
    return h.hexdigest()

def load_cohort(path: str) -> tuple[list[Row], list[dict]]:
    admitted, audit = [], []
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh, delimiter="\t"):
            errs = validate_row(d)
            if errs:
                audit.append({**d, "_reasons": "; ".join(errs)})
            else:
                admitted.append(Row(**{f: d[f] for f in ROW_FIELDS}))
    return admitted, audit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_cohort.py -q`
Expected: PASS (5 passed). Add `tests/alignair/aa/phase3/__init__.py` if pytest needs it.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/phase3/ tests/alignair/aa/phase3/
git commit -m "feat(aa-phase3): frozen cohort schema + contract validator (citation/evidence/family enforced)"
```

### Task 2: Thera-SAbDab acquisition script + alias crosswalk (reproducible sequences, cited family map)

**Files:**
- Create: `scripts/aa/phase3/acquire.py`
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/alias_crosswalk.tsv`
- Test: `tests/alignair/aa/phase3/test_alias.py`
- Modify: `src/alignair/aa/phase3/cohort.py` (add `apply_alias`)

**Interfaces:**
- Consumes: `ROW_FIELDS` from Task 1.
- Produces: `apply_alias(name: str, crosswalk: dict) -> tuple[str, str, str]` returning `(imgt_family, imgt_gene_or_empty, citation)`; raises `KeyError` if `name` has no cited mapping. `acquire.py` writes `manifest.tsv` (INN + per-sequence sha256, no raw redistribution) + updates `sources.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_alias.py
import pytest
from alignair.aa.phase3.cohort import apply_alias

CROSS = {"VH subgroup III": ("IGHV3", "", "Kabat1991"),
         "IGHV3-23*01": ("IGHV3", "IGHV3-23", "IMGT")}

def test_subgroup_maps_to_family_without_alignment():
    assert apply_alias("VH subgroup III", CROSS) == ("IGHV3", "", "Kabat1991")

def test_uncited_alias_raises():
    with pytest.raises(KeyError):
        apply_alias("mystery clone 7", CROSS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_alias.py -q`
Expected: FAIL (`apply_alias` not defined).

- [ ] **Step 3: Write minimal implementation**

Add to `src/alignair/aa/phase3/cohort.py`:

```python
def apply_alias(name: str, crosswalk: dict) -> tuple[str, str, str]:
    """Map a template/subgroup name to (imgt_family, imgt_gene_or_empty, citation) via the frozen
    crosswalk ONLY. Never aligns the sequence. KeyError if the name has no cited mapping."""
    return crosswalk[name]
```

Seed `alias_crosswalk.tsv` with the cited Kabat-subgroup→IMGT-family rows (header `alias\timgt_family\timgt_gene\tcitation`), e.g. `VH subgroup I\tIGHV1\t\tKabat 1991 / IMGT Scaviner 1999`, through subgroup VII. Each row carries a real citation; rows are added only from primary sources.

Write `scripts/aa/phase3/acquire.py`: fetch the current Thera-SAbDab sequence file (endpoint captured from the live UI), compute per-sequence sha256, emit `manifest.tsv` (INN, format, isotype, LC-category, per-seq sha256 — **no raw sequences committed**) and update `sources.json` with endpoint, retrieval date, and licensing note (redistribution not established → manifest+script).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_alias.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/aa/phase3/acquire.py src/alignair/aa/phase3/cohort.py tests/alignair/aa/phase3/test_alias.py docs/superpowers/artifacts/2026-07-21-aa-phase3/alias_crosswalk.tsv
git commit -m "feat(aa-phase3): Thera-SAbDab acquisition (manifest+hash, no redistribution) + cited alias crosswalk"
```

### Task 3: Curate family-Gold + concordance cohorts, freeze & hash — **STOP-AND-REPORT GATE**

**Files:**
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/candidate_inventory.tsv`
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/therasabdab_gold.tsv` (or `manifest.tsv` if licensing forbids sequences)
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/therasabdab_concordance.tsv`
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/cohort.sha256`
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/inclusion_rules.md`

- [ ] **Step 1: Curate candidates (human-verified, primary sources only).** For each therapeutic antibody with a **source-declared** human IGHV design template, record: INN, arm (if bispecific), VH AA (via manifest), the declared template, its `label_resolution`, the `evidence_locator` (patent example / paper table), and `label_citation`. Resolve the template to an IMGT **family** via `apply_alias` (cited crosswalk) — never by alignment. Out-of-scope (murine/chimeric/VHH/synthetic/unresolved), ambiguous, bispecific-unassigned, and duplicates go to `candidate_inventory.tsv` with a reason. Concordance-tier rows (IMGT/ANARCI exact-gene) are curated separately and labelled alignment-derived.

- [ ] **Step 2: Deduplicate + assign lineages.** Compute `exact_vh_id` (sha256 of VH AA); collapse identical VH. Assign `lineage_id` (shared template/parental lineage); mark exactly one `lineage_representative=true` per lineage (lexicographically smallest `exact_vh_id`).

- [ ] **Step 3: Validate + freeze.** Run the validator over both TSVs; every admitted row must pass `validate_row` (else it moves to the audit). Compute the cohort hash and write it:

Run: `PYTHONPATH=src .venv/bin/python -c "from alignair.aa.phase3.cohort import load_cohort, cohort_sha256; import pathlib; g,a=load_cohort('docs/superpowers/artifacts/2026-07-21-aa-phase3/therasabdab_gold.tsv'); print('gold admitted', len(g), 'audit', len(a)); print('lineages', len({r.lineage_id for r in g})); pathlib.Path('docs/superpowers/artifacts/2026-07-21-aa-phase3/cohort.sha256').write_text(cohort_sha256(g))"`

- [ ] **Step 4: GATE.** Count **independent lineage representatives** in Gold. If **< 15**, **STOP**: write the shortfall + options into `feasibility_report.md` and report to the user; do **not** proceed to Milestone 2. If **≥ 15**, record the count in `inclusion_rules.md` and continue.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/artifacts/2026-07-21-aa-phase3/
git commit -m "data(aa-phase3): frozen family-Gold + concordance cohorts, candidate inventory, cohort hash (gate: N lineages)"
```

---

## Milestone 2 — Reusable inference, calibration, and taxonomy utilities (TDD)

### Task 4: Extract a tested AA-head inference utility (shared with the snapshot)

**Files:**
- Create: `src/alignair/aa/infer.py`
- Modify: `scripts/aa/calibrate_stage2.py` (replace local `_probs` with a call to the utility)
- Test: `tests/alignair/aa/test_infer.py`

**Interfaces:**
- Produces: `class_probs(model, seqs: list[str], cfg, device: str, key: str, batch: int = 256) -> np.ndarray` — softmax probabilities `(len(seqs), n_classes)` for head `key` (`"v_class_logits"`, `"j_class_logits"`, `"d_class_logits"`), tokenizing with `AAProteinTokenizer` and left-padding/truncating to `cfg.max_seq_length`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_infer.py
import numpy as np, torch
from alignair.aa.infer import class_probs
from alignair.aa.strategic import strategic_setup

def test_class_probs_shape_and_normalisation():
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    model.eval()
    seqs = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "QVQLQESGPGLVKPSETLSLTCTVSGGSIS"]
    p = class_probs(model, seqs, cfg, "cpu", "v_class_logits", batch=8)
    assert p.shape == (2, class_maps["V"].n_classes())
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_infer.py -q`
Expected: FAIL (`alignair.aa.infer` not found).

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/infer.py
"""Single verified AA-head inference path. Used by both Stage-2 calibration and Phase-3 eval."""
from __future__ import annotations
import numpy as np, torch
from .tokenizer import AAProteinTokenizer

def class_probs(model, seqs, cfg, device, key, batch: int = 256) -> np.ndarray:
    tok = AAProteinTokenizer()
    L = cfg.max_seq_length
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        toks = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):
            e = tok.encode(s); n = min(len(e), L)
            toks[j, :n] = torch.from_numpy(e[:n])
        with torch.no_grad():
            lg = model({"tokenized_sequence": toks.to(device)})[key].cpu()
        out.append(torch.softmax(lg, -1).numpy())
    return np.concatenate(out, 0)
```

Then in `scripts/aa/calibrate_stage2.py`: delete the local `_probs` body and replace with `from alignair.aa.infer import class_probs as _probs` (keep the `_probs(...)` call sites unchanged).

- [ ] **Step 4: Run tests to verify pass (util + calibration still green)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_infer.py -q`
Expected: PASS. Then re-run calibration and diff the artifact is unchanged:
Run: `PYTHONPATH=src .venv/bin/python scripts/aa/calibrate_stage2.py --model experiments/aa/strategic/model_stage1_PASS.pt --device cuda --alpha 0.05 --n 5000 --out /tmp/claude-1000/-home-thomas-Desktop-AlignAIR/376842b6-033f-49b5-a34c-02e6621fd496/scratchpad/calib_check.json && diff <(jq -S '.genes' docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json) <(jq -S '.genes' /tmp/claude-1000/-home-thomas-Desktop-AlignAIR/376842b6-033f-49b5-a34c-02e6621fd496/scratchpad/calib_check.json) && echo "calibration unchanged"`
Expected: "calibration unchanged" (refactor is behaviour-preserving).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/infer.py scripts/aa/calibrate_stage2.py tests/alignair/aa/test_infer.py
git commit -m "refactor(aa): extract tested class_probs inference util; calibrate_stage2 reuses it"
```

### Task 5: Frozen-calibration loader + fixture compatibility test

**Files:**
- Modify: `src/alignair/aa/conformal.py` (add `FrozenConformal` + `Conformal.from_frozen`)
- Create: `tests/alignair/aa/fixtures/conformal_fixture.json`
- Test: `tests/alignair/aa/test_frozen_conformal.py`

**Interfaces:**
- Consumes: `aps_set` (existing in `conformal.py`).
- Produces: `class FrozenConformal` with `.qhat: float` and `.predict_set(prob_row) -> set`; `FrozenConformal.from_calibration_json(cal: dict, gene: str, group: str = "full", lam=0.0, k_reg=1)`.

- [ ] **Step 1: Create the fixture (frozen probs + expected sets, no sim cohort).**

```python
# tests/alignair/aa/fixtures/_make_fixture.py  (run once; commit the JSON it prints)
import json, numpy as np
from alignair.aa.conformal import aps_set
qhat = 0.9353423896173368                     # V full-group frozen quantile
rows = [[0.7,0.2,0.1],[0.34,0.33,0.33],[0.05,0.9,0.05]]
fix = {"qhat": qhat, "rows": rows,
       "expected_sets": [sorted(aps_set(np.array(r), qhat)) for r in rows]}
print(json.dumps(fix, indent=2))
```

Save its output to `tests/alignair/aa/fixtures/conformal_fixture.json`.

- [ ] **Step 2: Write the failing test**

```python
# tests/alignair/aa/test_frozen_conformal.py
import json, numpy as np, pathlib
from alignair.aa.conformal import FrozenConformal

FIX = json.loads((pathlib.Path(__file__).parent / "fixtures/conformal_fixture.json").read_text())

def test_loader_reproduces_expected_sets():
    fc = FrozenConformal(qhat=FIX["qhat"])
    for row, exp in zip(FIX["rows"], FIX["expected_sets"]):
        assert sorted(fc.predict_set(np.array(row))) == exp

def test_from_calibration_json_reads_full_group_quantile():
    cal = {"genes": {"V": {"calibration_frozen": {"q_mondrian": {"full": 0.5, "short": 0.6}}}}}
    fc = FrozenConformal.from_calibration_json(cal, "V", "full")
    assert fc.qhat == 0.5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_frozen_conformal.py -q`
Expected: FAIL (`FrozenConformal` not defined).

- [ ] **Step 4: Write minimal implementation**

Add to `src/alignair/aa/conformal.py`:

```python
class FrozenConformal:
    """Deployable calibrator: a fixed qhat loaded from the frozen calibration.json. No re-fitting."""
    def __init__(self, qhat: float, lam: float = 0.0, k_reg: int = 1):
        self.qhat, self.lam, self.k_reg = float(qhat), lam, k_reg
    def predict_set(self, prob_row) -> set:
        return aps_set(prob_row, self.qhat, self.lam, self.k_reg)
    @classmethod
    def from_calibration_json(cls, cal: dict, gene: str, group: str = "full", lam: float = 0.0, k_reg: int = 1):
        q = cal["genes"][gene]["calibration_frozen"]["q_mondrian"][group]
        return cls(qhat=q, lam=lam, k_reg=k_reg)
```

- [ ] **Step 5: Run test to verify it passes; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_frozen_conformal.py -q`
Expected: PASS.

```bash
git add src/alignair/aa/conformal.py tests/alignair/aa/test_frozen_conformal.py tests/alignair/aa/fixtures/conformal_fixture.json
git commit -m "feat(aa): FrozenConformal loader (full-group quantile) + fixture compatibility test"
```

### Task 6: Gene/family taxonomy + class→family expansion

**Files:**
- Create: `src/alignair/aa/taxonomy.py`
- Test: `tests/alignair/aa/test_taxonomy.py`

**Interfaces:**
- Consumes: `EquivalenceClassMap.class_members` (class id → nt-allele names).
- Produces: `gene_of(allele: str) -> str`; `family_of(name: str) -> str`; `class_genes(class_id: int, cmap) -> set[str]`; `class_families(class_id: int, cmap) -> set[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_taxonomy.py
import pytest
from alignair.aa.taxonomy import gene_of, family_of

def test_gene_and_family_parsing():
    assert gene_of("IGHV3-23*01") == "IGHV3-23"
    assert family_of("IGHV3-23*01") == "IGHV3"
    assert family_of("IGHV1-69D*01") == "IGHV1"
    assert family_of("IGHV3-23") == "IGHV3"

def test_family_parse_failure_raises():
    with pytest.raises(ValueError):
        family_of("notagene")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_taxonomy.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/taxonomy.py
"""IMGT allele -> gene -> family canonicalisation and class expansion (no alignment)."""
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

def class_genes(class_id: int, cmap) -> set:
    return {gene_of(a) for a in cmap.class_members[class_id]}

def class_families(class_id: int, cmap) -> set:
    return {family_of(a) for a in cmap.class_members[class_id]}
```

- [ ] **Step 4: Run test to verify it passes; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_taxonomy.py -q`
Expected: PASS.

```bash
git add src/alignair/aa/taxonomy.py tests/alignair/aa/test_taxonomy.py
git commit -m "feat(aa): allele->gene->family taxonomy + class expansion"
```

---

## Milestone 3 — Scorer (TDD)

### Task 7: Exact one-sided binomial + per-endpoint decision

**Files:**
- Create: `src/alignair/aa/phase3/stats.py`
- Test: `tests/alignair/aa/phase3/test_stats.py`

**Interfaces:**
- Produces: `clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]` (one-sided lower & upper bounds at level 1−α); `coverage_decision(k: int, n: int, alpha: float, allow_support: bool) -> str` returning `"failed"`/`"inconclusive"`/`"supported"` (never `"supported"` when `allow_support=False`).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_stats.py
from alignair.aa.phase3.stats import clopper_pearson, coverage_decision

def test_bounds_monotone_and_bracketed():
    lo, hi = clopper_pearson(9, 10, 0.05)
    assert 0.0 <= lo <= 9/10 <= hi <= 1.0

def test_all_correct_small_n_cannot_support():
    # 15/15 successes: upper bound = 1.0, lower bound < 0.95 -> inconclusive (cf. ~59-obs reality)
    assert coverage_decision(15, 15, 0.05, allow_support=True) == "inconclusive"

def test_clear_failure_is_failed():
    # 10/20 covered: upper one-sided bound well below 0.95
    assert coverage_decision(10, 20, 0.05, allow_support=True) == "failed"

def test_family_endpoint_never_supports():
    # 60/60 would support at allow_support=True, but family endpoint forbids it
    assert coverage_decision(60, 60, 0.05, allow_support=False) in ("inconclusive", "failed")
    assert coverage_decision(60, 60, 0.05, allow_support=False) != "supported"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_stats.py -q`
Expected: FAIL.

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

- [ ] **Step 4: Run test to verify it passes; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_stats.py -q`
Expected: PASS.

```bash
git add src/alignair/aa/phase3/stats.py tests/alignair/aa/phase3/test_stats.py
git commit -m "feat(aa-phase3): exact one-sided binomial bounds + coverage decision (family never supports)"
```

### Task 8: Scorer — family accuracy, coverage, concordance, cluster bootstrap, decision

**Files:**
- Create: `src/alignair/aa/phase3/scorer.py`
- Test: `tests/alignair/aa/phase3/test_scorer.py`

**Interfaces:**
- Consumes: `class_families`/`class_genes` (Task 6), `coverage_decision`/`clopper_pearson` (Task 7).
- Produces: `Prediction` dataclass (`top_class:int`, `set_classes:set[int]`); `score_gold(rows, preds, cmap, alpha, seed) -> dict` and `score_concordance(rows, preds, cmap, alpha) -> dict`. `score_gold` returns `{top_class_family_accuracy, family_set_coverage, decision, ci, bootstrap_ci, n_lineages, oov_rate, in_catalog_accuracy}` computed on **one representative VH per lineage** for binding numbers, with a lineage cluster bootstrap for descriptive CIs.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/phase3/test_scorer.py
import numpy as np
from types import SimpleNamespace
from alignair.aa.phase3.scorer import Prediction, score_gold

class FakeMap:                      # class 0 -> IGHV3 alleles; class 1 -> IGHV1 alleles
    class_members = {0: ["IGHV3-23*01", "IGHV3-23*02"], 1: ["IGHV1-69*01"]}

def _row(name, fam, rep=True):
    return SimpleNamespace(name=name, lineage_id=name, lineage_representative=str(rep).lower(),
                           true_v_family=fam, in_catalog="true")

def test_perfect_family_accuracy_but_small_n_is_inconclusive():
    rows = [_row(f"m{i}", "IGHV3") for i in range(15)]
    preds = [Prediction(top_class=0, set_classes={0}) for _ in rows]   # all predict IGHV3
    out = score_gold(rows, preds, FakeMap(), alpha=0.05, seed=0)
    assert out["top_class_family_accuracy"] == 1.0
    assert out["decision"] in ("inconclusive", "failed")
    assert out["decision"] != "supported"
    assert out["n_lineages"] == 15

def test_family_miss_lowers_accuracy():
    rows = [_row(f"m{i}", "IGHV3") for i in range(10)]
    preds = [Prediction(top_class=1, set_classes={1}) for _ in rows]   # predict IGHV1, truth IGHV3
    out = score_gold(rows, preds, FakeMap(), alpha=0.05, seed=0)
    assert out["top_class_family_accuracy"] == 0.0
    assert out["decision"] == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_scorer.py -q`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase3/scorer.py
"""Phase-3 scorer: family-level Gold (binding, one rep/lineage) + concordance; no 'supported' for Gold."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..taxonomy import class_families, class_genes
from .stats import clopper_pearson, coverage_decision

@dataclass
class Prediction:
    top_class: int
    set_classes: set

def _reps(rows):
    return [r for r in rows if str(r.lineage_representative).lower() == "true"]

def score_gold(rows, preds, cmap, alpha: float, seed: int) -> dict:
    idx = [i for i, r in enumerate(rows) if str(r.lineage_representative).lower() == "true"]
    rr = [rows[i] for i in idx]; pp = [preds[i] for i in idx]
    top_hits = [r.true_v_family in class_families(p.top_class, cmap) for r, p in zip(rr, pp)]
    cov_hits = [r.true_v_family in set().union(*[class_families(c, cmap) for c in p.set_classes])
                for r, p in zip(rr, pp)]
    n = len(rr); kt = int(sum(top_hits)); kc = int(sum(cov_hits))
    in_cat = [r for r in rr if str(r.in_catalog).lower() == "true"]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(2000):
        samp = rng.integers(0, n, n)
        boot.append(np.mean([top_hits[i] for i in samp]))
    return {
        "n_lineages": n,
        "top_class_family_accuracy": kt / n if n else 0.0,
        "top_class_family_ci": list(clopper_pearson(kt, n, alpha)),
        "family_set_coverage": kc / n if n else 0.0,
        "decision": coverage_decision(kc, n, alpha, allow_support=False),
        "bootstrap_ci_family_accuracy": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "oov_rate": 1 - len(in_cat) / n if n else 0.0,
    }

def score_concordance(rows, preds, cmap, alpha: float) -> dict:
    idx = [i for i, r in enumerate(rows) if str(r.lineage_representative).lower() == "true"]
    rr = [rows[i] for i in idx]; pp = [preds[i] for i in idx]
    agree = [r.true_v_gene in class_genes(p.top_class, cmap) for r, p in zip(rr, pp)]
    cov = [r.true_v_gene in set().union(*[class_genes(c, cmap) for c in p.set_classes])
           for r, p in zip(rr, pp)]
    n = len(rr)
    return {"label_kind": "alignment-derived (IMGT/ANARCI) — concordance, NOT ground truth",
            "n_lineages": n,
            "top_class_gene_agreement": (sum(agree) / n) if n else 0.0,
            "gene_set_coverage": (sum(cov) / n) if n else 0.0}
```

- [ ] **Step 4: Run test to verify it passes; commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/phase3/test_scorer.py -q`
Expected: PASS.

```bash
git add src/alignair/aa/phase3/scorer.py tests/alignair/aa/phase3/test_scorer.py
git commit -m "feat(aa-phase3): family-Gold + concordance scorer (binding one-rep/lineage, no supported)"
```

---

## Milestone 4 — Evaluation run & report

### Task 9: Eval script — preconditions, predict, score, emit artifacts

**Files:**
- Create: `scripts/aa/phase3/evaluate.py`
- Create (output): `docs/superpowers/artifacts/2026-07-21-aa-phase3/predictions.jsonl`, `phase3_eval.json`

**Interfaces:**
- Consumes: `load_cohort`, `cohort_sha256` (Task 1); `class_probs` (Task 4); `FrozenConformal` (Task 5); `score_gold`, `score_concordance`, `Prediction` (Task 8).

- [ ] **Step 1: Write the eval script.** It MUST, in order: (1) assert model sha256 == `54eef812…`; (2) assert `class_maps["V"].db_hash` == `e1a455b8…` and equals `calibration.json.provenance.class_map_db_hash.V`; (3) load the frozen cohort and assert `cohort_sha256(gold)` == `cohort.sha256`; (4) build `FrozenConformal.from_calibration_json(cal, "V", "full")`; (5) `class_probs` over VH; (6) build `Prediction(top_class=argmax, set_classes=fc.predict_set(row))`; (7) `score_gold` + `score_concordance`; (8) write per-row `predictions.jsonl` and `phase3_eval.json` (decision `failed`/`inconclusive` only, family-accuracy anchor, concordance block, licensing/attribution, audit reason counts, cohort hash). Hard-fail if any precondition mismatches.

- [ ] **Step 2: Run it against the frozen cohort**

Run: `PYTHONPATH=src .venv/bin/python scripts/aa/phase3/evaluate.py --device cuda`
Expected: writes `phase3_eval.json` with a `decision` of `failed` or `inconclusive`, a `top_class_family_accuracy`, and a concordance block; no `supported`.

- [ ] **Step 3: Sanity-assert the invariants of the output**

Run: `PYTHONPATH=src .venv/bin/python -c "import json; r=json.load(open('docs/superpowers/artifacts/2026-07-21-aa-phase3/phase3_eval.json')); assert r['gold']['decision'] in ('failed','inconclusive'); assert 'concordance' in r; assert r['cohort_sha256']; print('ok', r['gold']['top_class_family_accuracy'], r['gold']['decision'])"`
Expected: `ok <acc> <decision>`.

- [ ] **Step 4: Commit**

```bash
git add scripts/aa/phase3/evaluate.py docs/superpowers/artifacts/2026-07-21-aa-phase3/predictions.jsonl docs/superpowers/artifacts/2026-07-21-aa-phase3/phase3_eval.json
git commit -m "feat(aa-phase3): eval runner (precondition asserts + frozen cohort) -> per-row + aggregate artifacts"
```

### Task 10: Truncation stress test (separate, synthetic, frozen seeds)

**Files:**
- Create: `scripts/aa/phase3/truncation_stress.py`
- Create (output): `docs/superpowers/artifacts/2026-07-21-aa-phase3/truncation_stress.json`

**Interfaces:**
- Consumes: `aa_truncate` (existing in `alignair.aa.augment`), `class_probs`, `FrozenConformal.from_calibration_json(cal, "V", group)` for `group in {full,trunc,short}`.

- [ ] **Step 1: Write the stress script.** Deterministically crop each real VH with `aa_truncate` to the **exact** coverage-bin ranges from `calibration.json` (`trunc`=keep∈[0.6,0.9], `short`=keep∈[0.3,0.6)), frozen `np.random.default_rng(seed)`, paired (same VH across bins). For each bin use its own frozen full/trunc/short quantile. Score family-set coverage per bin. Label the artifact clearly as a **synthetic stress test on real sequences**, reported apart from the primary result.

- [ ] **Step 2: Run it; sanity check keys**

Run: `PYTHONPATH=src .venv/bin/python scripts/aa/phase3/truncation_stress.py --device cuda && PYTHONPATH=src .venv/bin/python -c "import json; r=json.load(open('docs/superpowers/artifacts/2026-07-21-aa-phase3/truncation_stress.json')); assert set(r['by_bin'])=={'full','trunc','short'}; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/aa/phase3/truncation_stress.py docs/superpowers/artifacts/2026-07-21-aa-phase3/truncation_stress.json
git commit -m "feat(aa-phase3): synthetic truncation stress test on real VH (frozen seeds, per-bin quantiles)"
```

### Task 11: PHASE3.md report + full suite green

**Files:**
- Create: `docs/superpowers/artifacts/2026-07-21-aa-phase3/PHASE3.md`

- [ ] **Step 1: Write `PHASE3.md`** framed per spec §1: the family-accuracy anchor as headline, the `failed`/`inconclusive` decision with its exact interval, the concordance block labelled "alignment-derived, not ground truth", the truncation stress result as a separate synthetic section, OOV/in-catalog accuracy, licensing/attribution, and the explicit statement that Phase 3 **cannot** validate the 95% allele/class guarantee (no alignment-independent allele truth on real data).

- [ ] **Step 2: Run the full AA suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/ -q -m "not slow"`
Expected: all pass (≥101 prior + the new Phase-3 tests).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/artifacts/2026-07-21-aa-phase3/PHASE3.md
git commit -m "docs(aa-phase3): PHASE3.md — family-accuracy anchor + concordance + honest scope"
```

---

## Self-Review notes (spec coverage)

- §1 framing / three-way→two-way decision (never `supported`): Tasks 7, 8, 9. §1 independence (one rep/lineage): Tasks 8, 9.
- §2 two tiers, schema, citation/evidence gate, audit, dedup/lineage, licensing, OOV, cohort freeze+hash: Tasks 1, 2, 3.
- §2 label-resolution taxonomy + admission (family): Tasks 1, 8. Alias crosswalk (cited, alignment-free, Kabat→family): Task 2.
- §3 preconditions (model sha, class-map fingerprint, calibration params, cohort hash, `q_mondrian["full"]`, model-independent selection): Task 9. Coarsening ladder + top-class family-set + concordance metrics + cluster bootstrap: Tasks 6, 8, 9. Truncation stress (separate, frozen seeds, per-bin quantiles): Task 10.
- §4 components (realdata loader/validator, taxonomy, FrozenConformal loader, extracted inference util, runner, scorer, report), artifacts, tests incl. fixture compatibility test: Tasks 1, 4, 5, 6, 8, 9, 11.
- §5.1 gated family-Gold curation with stop-and-report at 15 lineages: Task 3.
- Not-in-scope (D, retrain, recalibration): honored — no such tasks.
