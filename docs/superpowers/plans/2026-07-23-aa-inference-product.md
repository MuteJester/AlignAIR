# AA Inference Product Implementation Plan (Slices 1→5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`).

**Goal:** Ship a production AA IGH inference product around v2 — a pickle-free `.alignair` bundle, a safe loader, `predict_aa`, calibrated-set output, a CLI, and product-path validation — honoring the P0 contracts in the spec.

**Architecture:** Mirror the DNA `.alignair` container (`src/alignair/model_file/`, safetensors, sha256 manifest, registry). New package `src/alignair/aa/product/`. Reuse `alignair.aa.infer.class_probs`, `alignair.aa.conformal.FrozenConformal`, `alignair.aa.equivalence.EquivalenceClassMap`, `alignair.aa.tokenizer.AAProteinTokenizer`, `alignair.aa.strategic.strategic_setup`.

**Tech Stack:** Python, PyTorch, safetensors, NumPy, the `model_file` container, pytest.

## Global Constraints (from spec 2026-07-23-aa-inference-product-design.md — bind every task)
- **v2 = `alignair-aa-igh-human@2.0.0` is the production default**; v1 (`@1.0.0`) loadable only if explicitly requested. Incumbent files (`model_stage1_PASS.pt`, `calibration.json`, `2026-07-21-aa-phase3/*`) are **read-only, never rewritten**. Candidate `.pt` never committed.
- **Zero pickle** in a distributable bundle: weights = safetensors; all else JSON. Legacy `.pt` refused unless `trust_pickle=True`.
- **ASC→IMGT is one-to-many**: `new_allele → [all originating imgt_allele]`; expansion retains every mapping; never an arbitrary representative. `genes`/`families` are always **plural lists**. For D, `class_id` is a D allele-head index, not a protein-class.
- **Calibration↔model cryptographic binding** verified on load (model SHA, class-map fingerprint, label counts, method/alpha/lambda/k_reg/group-defs/quantiles/n_cal/n_test/supported_distribution).
- **Coordinates** = zero-based half-open AA offsets, rounded+clipped; `start>end` → null+warning; cross-gene overlap flagged not repaired. NOT IMGT numbering.
- **Library API takes sequences/records only** (no path); `*` rejected; overlength (>192) rejected (never silently truncated); `on_invalid` = "error" (fail-fast) | "skip" (structured per-record errors).
- **Calibration group:** `None` → pooled fallback (labeled); supplied → used + supplied-group warning; per-sequence groups allowed; the estimator is OFF for v1.
- Card carries **no `host`/`user`**; manifest hashes over **canonical JSON**; the §8 scope string in every result + card.
- No Co-Authored-By/Claude line in commits. Run python as `PYTHONPATH=src .venv/bin/python ...`.

---

## Slice 1 — Bundle schema + safe loader

### Task 1: One-to-many ASC→IMGT map + class expansion
**Files:** Create `src/alignair/aa/product/__init__.py`, `src/alignair/aa/product/asc.py`; Test `tests/alignair/aa/product/test_asc.py`.
**Interfaces:** `asc_to_imgt_multi(dc) -> dict[str, list[str]]` (new_allele → sorted unique list of ALL its imgt_allele); `class_alleles(class_id, cmap, asc_multi) -> list[str]` (sorted unique IMGT alleles for a class's members); `class_genes(class_id, cmap, asc_multi) -> list[str]`; `class_families(...) -> list[str]`. All retain every mapping.

- [ ] **Step 1: Write the failing test**
```python
# tests/alignair/aa/product/test_asc.py
from alignair.aa.product.asc import asc_to_imgt_multi, class_alleles, class_genes, class_families

class FakeDC:
    asc_tables = {"V": [
        {"new_allele": "IGHVF1-G1*01", "imgt_allele": "IGHV3-23*01"},
        {"new_allele": "IGHVF1-G1*01", "imgt_allele": "IGHV3-23D*01"},  # SAME asc -> two IMGT (one-to-many)
        {"new_allele": "IGHVF1-G2*01", "imgt_allele": "IGHV1-2*02"},
    ]}
class FakeMap:
    class_members = {0: ["IGHVF1-G1*01"], 1: ["IGHVF1-G1*01", "IGHVF1-G2*01"]}

def test_one_to_many_retains_all():
    m = asc_to_imgt_multi(FakeDC())
    assert m["IGHVF1-G1*01"] == ["IGHV3-23*01", "IGHV3-23D*01"]   # BOTH kept, sorted
def test_class_alleles_union_all_members():
    m = asc_to_imgt_multi(FakeDC())
    assert class_alleles(1, FakeMap(), m) == ["IGHV1-2*02", "IGHV3-23*01", "IGHV3-23D*01"]
    assert class_genes(1, FakeMap(), m) == ["IGHV1-2", "IGHV3-23", "IGHV3-23D"]
    assert class_families(0, FakeMap(), m) == ["IGHV3"]
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**
```python
# src/alignair/aa/product/asc.py
"""One-to-many ASC->IMGT mapping + class expansion (retains every mapping; no representative)."""
from __future__ import annotations
import re
_FAM = re.compile(r"^(IGH[VDJ]\d+)")

def asc_to_imgt_multi(dc) -> dict:
    out: dict[str, set] = {}
    for row in dc.asc_tables["V"]:
        out.setdefault(row["new_allele"], set()).add(row["imgt_allele"])
    return {k: sorted(v) for k, v in out.items()}

def _gene(a): return a.split("*")[0]
def _fam(a):
    m = _FAM.match(a)
    if not m: raise ValueError(f"cannot parse family from {a!r}")
    return m.group(1)

def class_alleles(class_id, cmap, asc_multi) -> list:
    out = set()
    for member in cmap.class_members[class_id]:
        out.update(asc_multi.get(member, [member]))
    return sorted(out)

def class_genes(class_id, cmap, asc_multi) -> list:
    return sorted({_gene(a) for a in class_alleles(class_id, cmap, asc_multi)})

def class_families(class_id, cmap, asc_multi) -> list:
    return sorted({_fam(a) for a in class_alleles(class_id, cmap, asc_multi)})
```
- [ ] **Step 4: Run → PASS; commit**
```bash
git add src/alignair/aa/product/ tests/alignair/aa/product/
git commit -m "feat(aa-product): one-to-many ASC->IMGT map + class allele/gene/family expansion (retain all)"
```

### Task 2: AA bundle writer (`save_aa_bundle`) + build the v2 bundle
**Files:** Create `src/alignair/aa/product/bundle.py`; Test `tests/alignair/aa/product/test_bundle.py`.
**Interfaces:** `canonical_json(obj) -> bytes` (sorted keys, `separators=(",",":")`, UTF-8); `save_aa_bundle(path, *, state_dict, cfg, class_maps, d_allele_order, asc_multi, calibration, card_extra) -> dict` (writes the `.alignair` container; returns the manifest). Sections: `weights`(safetensors via `serialize.state_dict_to_bytes`), `config`(canonical json of cfg params), `tokenizer`(alphabet+id map json), `class_maps`(canonical json: `{V: EquivalenceClassMap.to_json, J: ..., D_alleles: [...]}`), `asc_imgt`(canonical json of asc_multi), `calibration`(canonical json). Header card: `artifact_type="alignair-aa-model", bundle_schema_version="alignair.aa.bundle.v1", architecture_version, model_id, model_version, class_counts, max_seq_length, calibration_scope (SPEC §8 string), build_commit (git), reference_provenance{class_map_db_hash{V,J}, class_map_sha256, asc_table_sha256, tokenizer_sha256}, calibration_provenance{method,alpha,n_cal,n_test,supported_distribution}, integrity{weights_sha256,config_sha256,calibration_sha256}`. **No host/user.** All hashes over `canonical_json`. Uses `model_file.container.write_container`.

- [ ] **Step 1: Write the failing test** — build a bundle from a fresh `strategic_setup` model (tiny/untrained is fine) + the real class maps + `asc_to_imgt_multi(dc)` + a minimal calibration dict; assert the file `is_alignair_file`, the card has `artifact_type=="alignair-aa-model"` and no `host`/`user` keys, and `class_map_sha256` is stable across two builds of the same inputs (canonical-JSON determinism). Include the exact assertions.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** per the interface, reusing `container.write_container`, `serialize.state_dict_to_bytes`, `container.sha256_hex`. `class_map_sha256 = sha256(canonical_json({"V": V.to_json(), "J": J.to_json(), "D_alleles": d_allele_order}))`. `build_commit` via `subprocess` (best-effort).
- [ ] **Step 4: Run → PASS. Then build the real v2 bundle** (a script step, not committed weights):
Run: `PYTHONPATH=src .venv/bin/python -c "..."` that loads `model_aa_igh_v2.pt` (model-only), the class maps, `asc_to_imgt_multi(HUMAN_IGH_OGRDB)`, and `calibration_cont_step318000.json`, and writes `experiments/aa/product/alignair-aa-igh-human-2.0.0.alignair` (git-ignored). Print the manifest.
- [ ] **Step 5: Commit** (code + test only; the `.alignair` artifact is git-ignored like `.pt`)
```bash
git add src/alignair/aa/product/bundle.py tests/alignair/aa/product/test_bundle.py
git commit -m "feat(aa-product): AA .alignair bundle writer (safetensors+classmaps+asc+calibration, canonical-JSON manifest, no host/user)"
```

### Task 3: AA safe loader (`load_aa_model` → AAPredictor) + integrity + calibration binding
**Files:** Create `src/alignair/aa/product/loader.py`; Test `tests/alignair/aa/product/test_loader.py`.
**Interfaces:** `@dataclass AAPredictor` (model, cfg, tokenizer, class_maps{V,J}, d_allele_order, asc_multi, conformal{V,J,D FrozenConformal}, card, device). `load_aa_model(path_or_id, *, device="auto", trust_pickle=False) -> AAPredictor`.

- [ ] **Step 1: Write the failing test** — load the bundle built in Task 2; assert model weights load (a forward pass runs), the card + asc_multi + class maps reconstruct, and the FrozenConformal quantiles match the calibration. Then **binding-failure tests**: (a) tamper — flip a byte in a section → load raises; (b) wrong calibration — swap in a calibration whose recorded model_sha256 differs from the bundle weights → load raises with a binding error; (c) label-count mismatch → raises. Full assertions.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** Resolve path (Task-8 adds registry IDs; for now accept a filesystem path + a hard-coded map `alignair-aa-igh-human@2.0.0 -> the v2 bundle path`). `container.read_section` (auto-verifies section checksums) for each section; **recompute** the AA integrity manifest over `canonical_json` and assert equality with the card; **calibration binding**: assert `calibration.provenance.model_sha256 == sha256(weights section)` OR the card's weights_sha256; assert `class_map_db_hash` match; assert per-gene label counts == head dims; assert method/alpha/lambda/k_reg/group-defs/quantiles/n_cal/n_test present. Rebuild `AAAlignAIR` (via `strategic_setup` sized from cfg, then `load_state_dict`), tokenizer, maps, `asc_multi`, and `FrozenConformal.from_calibration_json` per gene. Refuse legacy `.pt` unless `trust_pickle`.
- [ ] **Step 4: Run → PASS; commit**
```bash
git add src/alignair/aa/product/loader.py tests/alignair/aa/product/test_loader.py
git commit -m "feat(aa-product): safe AA loader — section+manifest verify + calibration<->model cryptographic binding; tamper/wrong-model rejected"
```

---

## Slice 2 — Core predict_aa

### Task 4: Input validation & normalization (§9.1)
**Files:** Create `src/alignair/aa/product/validate.py`; Test `tests/alignair/aa/product/test_validate.py`.
**Interfaces:** `normalize_inputs(sequences, on_invalid, max_len) -> tuple[list[Record], list[ErrorResult]]` where `Record={sequence_id, sequence}`. Accepts `str | Iterable[str] | Iterable[dict]`. IDs: record `sequence_id` else `seq_<i>`. Uppercase, strip ends, reject interior whitespace; alphabet = 20 canonical + `X`; **reject `*`**; reject empty; reject len>max_len (never truncate). `on_invalid="error"` raises `ValueError` on first invalid; `"skip"` returns per-record `ErrorResult{sequence_id,error,detail}`.

- [ ] **Step 1: Write the failing test** — cover: a bare str → one record `seq_0`; a list of dicts keeps IDs; empty/`*`/`len>192`/interior-space each invalid; `on_invalid="error"` raises; `on_invalid="skip"` returns errors for the invalid ones and records for the valid ones in order. Full assertions.
- [ ] **Step 2-4: RED → implement `validate.py` → GREEN; commit.**
```bash
git commit -m "feat(aa-product): input validation/normalization (sequences-only, * rejected, no truncation, on_invalid modes)"
```

### Task 5: AAPredictor.predict — calls + coordinates (no calibration yet)
**Files:** Create `src/alignair/aa/product/schema.py`, `src/alignair/aa/product/predictor.py`; Test `tests/alignair/aa/product/test_predictor.py`.
**Interfaces:** `@dataclass AAPrediction` (§6 shape) with `to_dict()`. `AAPredictor.predict(sequences, *, coverage_group=None, on_invalid="error") -> list[AAPrediction | ErrorResult]`. This task fills top_call (class_id, class_name, top_alleles, **plural** genes/families via Task-1 expansion, probability) + coordinates (§5 discretization/clip/`start>end`→null+warning, cross-gene overlap warning) + junction_aa; calibration fields are placeholder/empty until Task 7.

- [ ] **Step 1: Write the failing test** — on the loaded v2 AAPredictor, predict a couple of AA VH strings; assert schema keys present, `genes`/`families` are **lists**, coordinates are ints in `[0,len]` (or null), and the §8 scope string is in `provenance.calibration_scope`. Add a synthetic case forcing `start>end` (monkeypatch or a crafted logits stub) → coordinates null + warning. Assert D segment documents `class_id` as allele-head index.
- [ ] **Step 2-4: RED → implement (reuse `class_probs`, Task-1 expansion, Task-4 validation, §5 coord logic) → GREEN; commit.**
```bash
git commit -m "feat(aa-product): AAPredictor core — plural gene/family calls + hardened AA coordinates + junction (no calibration yet)"
```

---

## Slice 3 — Calibrated-set output

### Task 6: Calibration-group contract (§7)
**Files:** Create `src/alignair/aa/product/calibration_group.py`; Test `tests/alignair/aa/product/test_calibration_group.py`.
**Interfaces:** `resolve_group(coverage_group, sequence_id, n) -> tuple[group, source, warning|None]`. `None` → `("pooled","pooled_fallback",None)`; a str in {full,trunc,short} → `(g,"supplied",SUPPLIED_WARNING)`; a per-sequence map/list → the entry for this sequence (same rules); invalid group → ValueError.

- [ ] **Step 1: Write the failing test** — None→pooled; "full"→supplied+warning; a dict `{ "seqA":"trunc" }` resolves per id; bad value raises. Assertions.
- [ ] **Step 2-4: RED → implement → GREEN; commit.**
```bash
git commit -m "feat(aa-product): calibration-group contract (pooled default, supplied+warning, per-sequence)"
```

### Task 7: Wire calibrated sets into AAPredictor + parity
**Files:** Modify `src/alignair/aa/product/predictor.py`; Test `tests/alignair/aa/product/test_calibrated_sets.py`.
**Interfaces:** predict now fills `calibrated_class_set` (via the gene's `FrozenConformal` at the resolved group's quantile — pooled uses `q_pooled`, group uses `q_mondrian[group]`), `calibrated_allele_set` (Task-1 `class_alleles` over the class set, **all** mappings), `calibrated_gene_set`/`calibrated_family_set`, `set_size{classes,alleles,genes}`, and `applied_calibration_group`.

- [ ] **Step 1: Write the failing test** — **parity**: for a batch, the product's V calibrated_class_set (pooled) equals the research path (`class_probs` + `FrozenConformal(q_pooled)` predict_set) exactly; supplied group "full" uses `q_mondrian["full"]`; allele set is the one-to-many expansion of the class set. Assertions.
- [ ] **Step 2-4: RED → implement → GREEN; commit.**
```bash
git commit -m "feat(aa-product): calibrated-set output (conformal sets + one-to-many allele expansion, group-aware, parity with research path)"
```

---

## Slice 4 — CLI / file interface

### Task 8: Façade `predict_aa`/`load_aa_model` + `predict-aa` CLI + run.json + registry IDs
**Files:** Modify `src/alignair/__init__.py`; Create `src/alignair/aa/product/cli.py`; register the verb in the CLI dispatcher; Test `tests/alignair/aa/product/test_cli.py`.
**Interfaces:** `alignair.predict_aa(sequences, *, model="alignair-aa-igh-human@2.0.0", coverage_group=None, device="auto", on_invalid="error") -> list` (loads via cache, calls AAPredictor). `alignair.load_aa_model(...)`. CLI `alignair predict-aa <input.fasta|-> --model … [--coverage-group …] --out out.jsonl` — **file parsing here** (reuse `read_sequences`), writes JSONL + `<out>.run.json` (model_id/version, weights_sha256, calibration_sha256, class_map_db_hash, alignair_version, calibration_scope, counts, command, timestamp).

- [ ] **Step 1: Write the failing test** — `predict_aa(["EVQL…"])` returns a list of dicts with the schema; CLI on a tiny FASTA writes JSONL (n lines) + a run.json with the required fields. Assertions. (Model resolves to the local v2 bundle path via the Task-3 id map.)
- [ ] **Step 2-4: RED → implement → GREEN; run full AA subset; commit.**
```bash
git commit -m "feat(aa-product): predict_aa/load_aa_model facade + predict-aa CLI (file parsing) + run.json sidecar"
```

---

## Slice 5 — Product-path validation

### Task 9: Validation harness (parity, golden cohort, coords, groups, 16-Ab, tamper/wrong-model)
**Files:** Create `scripts/aa/product/validate_product.py`; outputs `docs/superpowers/artifacts/2026-07-23-aa-product/validation.json` + `.md`.
- [ ] **Step 1: Write the harness.** (1) Frozen simulator golden cohort spanning full/trunc/short (fixed seed, disjoint). (2) **V/D/J call + calibrated-set parity** product-path vs research-path — assert exact agreement. (3) Coordinate/junction checks (bounds, start>end voiding, junction substring). (4) pooled vs supplied-group behavior (quantile + labels + warning). (5) The frozen Phase-3 **16-antibody** descriptive product-path family accuracy (should hold ~16/16; descriptive only). (6) tamper + wrong-calibration/wrong-model rejection (assert loader raises). Write `validation.json` (pass/fail per check) + a short `.md`.
- [ ] **Step 2: Run it** (`--device cuda`); confirm all checks pass. Commit the JSON/MD (NOT any `.alignair`/`.pt`).
```bash
git commit -m "docs(aa-product): product-path validation — parity, golden cohort, coordinates, groups, 16-Ab descriptive, tamper/wrong-model rejection"
```

---

## Self-Review notes (spec coverage)
- §3 artifact (safetensors, one-to-many asc, canonical-JSON manifest, card no host/user): Tasks 1,2.
- §4 loader + calibration binding + tamper/wrong-model: Task 3.
- §5 coordinates: Task 5. §6 schema (plural genes/families, D allele-head, set expansion): Tasks 1,5,7.
- §7 calibration group (pooled default, supplied warning, per-sequence): Tasks 6,7.
- §8 guarantee/scope string in every result + card: Tasks 2,5.
- §9/§9.1 API sequences-only + input validation: Tasks 4,8. §10 CLI file parsing + run.json: Task 8.
- §11 slice order 1→5 + §13 tests: all tasks; Slice-5 expanded validation: Task 9.
- v2 default / incumbent immutable / no pickle / no committed .pt: Global Constraints, every task.
