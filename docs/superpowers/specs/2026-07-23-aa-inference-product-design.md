# Amino-Acid AlignAIR — Inference Product Design Spec

**Date:** 2026-07-23 · **Branch:** `experiment/aminoacid-alignair` · **Status:** design for sign-off. The
**artifact contract (§3), result schema (§6), coordinate contract (§5), calibration-group contract (§7), and
guarantee statement (§8) are P0 — frozen before any code.** Then a loader-to-prediction vertical slice around
v2 is built in the §11 order.

## §0 Purpose, scope, non-goals
Turn the promoted AA IGH model (`aa_igh_v2` = `cont_step318000`) into a **production inference product**, not a
research script wrapped as an API. Mirror the existing DNA distribution architecture: a self-contained,
pickle-free `.alignair` container (`src/alignair/model_file/`), safetensors weights, a SHA-256 integrity
manifest, registry/ID resolution, and a run-provenance sidecar.

**Non-goals:** light-chain / TCR / non-IGH AA; real-domain recalibration (Phase-3 stays descriptive); replacing
the nucleotide AIRR product; IMGT residue numbering; a validated coverage estimator (documented future work,
§7). IGH heavy-chain AA only.

## §1 Model identity & versioning
- **Stable IDs:** `alignair-aa-igh-human@2.0.0` (= v2, promoted) and `alignair-aa-igh-human@1.0.0` (= v1, the
  immutable Stage-1 research artifact). ID format mirrors the registry catalog (`id@version`, semver).
- **v2 is the production default.** v1 is an immutable **research** artifact — loadable **only** when explicitly
  requested (`model="alignair-aa-igh-human@1.0.0"`), never the default, never silently selected.
- The incumbent files (`model_stage1_PASS.pt`, its `calibration.json`, the Phase-3 artifacts) remain
  byte-for-byte immutable; packaging reads them, never rewrites them.

## §2 Reused infrastructure (mirror, do not fork)
The `.alignair` binary container (`model_file/container.py`: magic + JSON header + independently sha256-verified
compressed sections), safetensors serialization (`model_file/serialize.py`), the safe `_rebuild` loader
(`model_file/__init__.py`, refuses pickle unless `trust_pickle`), registry resolution + verified atomic cache
(`registry/cache.py::resolve_model`, `hf://`/path/`id@version`), the `alignair models` CLI, and the
`<out>.run.json` sidecar (`cli/predict.py::_write_run_metadata`). The AA product adds AA-specific save/load,
schema, and a `predict-aa` verb on top of these.

## §3 Bundle artifact contract (P0) — the AA `.alignair` container
A single distributable, **zero-pickle** `.alignair` file per model version. Sections (each carries
compressed+payload SHA-256, verified on every read):
- **weights** — `AAAlignAIR` state_dict as **safetensors** (pickle-free).
- **config** — JSON of the `aa_igh_config` params (`n_v_classes, n_j_classes, n_d_alleles, max_seq_length=192,
  vocab_size=23, num_chain_types=1, embed_dim, …`).
- **tokenizer** — the `AAProteinTokenizer` alphabet + id map (PAD=0, 20 canonical residues 1-20, `X`=21, `*`=22;
  `vocab_size=23`; right-pad), so decoding is reproducible from the bundle alone.
- **class_maps** — `EquivalenceClassMap.to_json()` for **V and J** (`schema, gene, db_hash, allele_class,
  class_members`) + the **D allele order** (ordered list). This is the head↔label contract.
- **asc_imgt** — the ASC→IMGT provenance table (`new_allele → imgt_allele`, from `dc.asc_tables["V"]`), so the
  ASC→IMGT bridge is **bundled**, not re-derived from a GenAIRR version at runtime.
- **calibration** — the model's `calibration.json` (`genes.{V,J,D}.calibration_frozen.{q_pooled, q_mondrian}` +
  `meta` with `target_coverage, bin_defs, supported_distribution, class_map_db_hash`).
- **card (header)** — `model_id, model_version, created, {alignair,genairr,torch}_version, class_counts{v,j,d},
  max_seq_length, calibration_scope (the §8 string), license, citation, provenance{git_commit,host,user}`.

**AA integrity manifest** (the analog of the DNA `allele_order_sha256`, guarding head↔class-map alignment):
`weights_sha256`, `config_sha256`, `class_map_db_hash{V,J}` (the existing 16-hex db_hash), `class_map_sha256`
(over the ordered V/J class list + D allele list), `asc_table_sha256`, `calibration_sha256`,
`tokenizer_sha256`. On load, all are recomputed and checked; a mismatch **hard-fails**. No pickle section exists
in a distributable AA bundle (enforced like `Aligner._assert_distributable`).

## §4 Safe loader
`load_aa_model(model, *, device="auto", trust_pickle=False, registry=None, offline=False) -> AAPredictor`:
resolves `model` (filesystem path / `hf://…` / `alignair-aa-igh-human@2.0.0`) via `resolve_model` + verified
cache; opens the container; verifies every section + the §3 integrity manifest; rebuilds `AAAlignAIR` (weights
from safetensors), the tokenizer, the V/J `EquivalenceClassMap`, the D allele order, the `ImgtView` ASC→IMGT
bridge, and a `FrozenConformal` per gene from the embedded calibration. Returns an **`AAPredictor`** holding all
of it. A legacy raw-pickle `.pt` is **refused unless `trust_pickle=True`** (dev-only path, never for
distribution).

## §5 Coordinate contract (P0)
AA segment coordinates are **zero-based, half-open offsets into the input AA sequence**: `[start, end)`, where
`0 ≤ start ≤ end ≤ len(sequence)`, derived from the model's per-gene start/end soft-argmax heads. They are
**input-relative offsets, NOT IMGT residue numbering.** The ASC→IMGT bridge maps allele **names** only; it does
**not** provide IMGT residue numbering. Every result and the model card state this explicitly.

## §6 Result schema (P0) — distinct from nucleotide AIRR
`predict_aa` returns `list[AAPrediction]`. Each `AAPrediction` (JSON-serializable):
```
sequence_id: str
input_length: int
segments: { V|D|J: {
    top_call: { class_id, class_name, top_alleles: [nt-allele names],
                gene, family, probability },          # gene/family via ASC→IMGT bridge
    calibrated_class_set: [class_id, ...],
    calibrated_allele_set: [nt-allele names, ...],    # class set expanded via frozen class map
    calibrated_gene_set:   [IMGT gene, ...],          # via ASC→IMGT bridge
    calibrated_family_set: [IMGT family, ...],
    set_size: { classes: int, alleles: int },
    coordinates: { start: int, end: int } | null      # zero-based half-open AA offsets (§5)
} }
junction_aa: str | null                                # V-end..J-start substring when derivable
applied_calibration_group: { group: "full"|"trunc"|"short"|"pooled",
                             source: "supplied"|"estimated"|"pooled_fallback" }   # §7
uncertainty: { abstain: bool, reason: str|null }       # e.g. set too large / low top-prob
provenance: { model_id, model_version, weights_sha256, calibration_sha256,
              class_map_db_hash, alignair_version, calibration_scope }   # §8 string
warnings: [str, ...]                                   # e.g. input>max_len, group estimated, OOV
```
**Explicitly NOT present** (and must not be implied): nucleotide coordinates, CIGAR strings, productivity calls,
or complete IMGT numbering. The schema may map compatible *concepts* to AIRR field names in a separate optional
adapter, but the native product output is AA-only and must never present nt-domain fields it does not compute.

## §7 Calibration-group selection contract (P0) — the key product decision
The frozen conformal quantiles are Mondrian `full`/`trunc`/`short` (+ a pooled marginal). An arbitrary user
peptide does **not** reveal its original simulator coverage fraction. **Do NOT silently choose a Mondrian bin
from raw sequence length.** Selection, in priority order:
1. **Explicit** — if the caller passes `coverage_group ∈ {"full","trunc","short"}`, use it;
   `applied_calibration_group.source = "supplied"`.
2. **Validated estimate (gated, OFF by default)** — a documented estimator that derives the coverage class from
   the model's *predicted* V-domain coverage (predicted V start/end vs the germline domain span). This path is
   **disabled until validated** (it must be shown, on held-out simulator data, to select the correct bin
   reliably — a separate validation, out of scope for the first slice). When enabled, `source = "estimated"`
   and a warning is attached.
3. **Pooled fallback (default)** — with `coverage_group=None` and the estimator not yet validated, use the
   **pooled quantile** (`q_pooled`), `source = "pooled_fallback"`, and label the result as carrying only the
   **pooled marginal** guarantee. This is the honest default; it never silently pretends to a group-conditional
   guarantee.
Every result records `applied_calibration_group` (group + source).

## §8 Guarantee statement (P0) — displayed in every result + model card
> **Simulator-domain conformal guarantee.** Under exchangeability with the simulator calibration distribution
> (GenAIRR HUMAN_IGH_OGRDB), the conformal protein-**class** set attains the target coverage (1−α = 0.95) for
> the applied Mondrian group (or the pooled marginal coverage under the pooled fallback, §7). Expanding a
> covered V/J protein-class through the frozen class map deterministically preserves inclusion of that class's
> mapped nucleotide alleles (set-membership is preserved by the fixed class→allele expansion). **This is NOT a
> real-therapeutic-domain 95% guarantee** — the real-protein evaluation (Phase 3) was descriptive and
> `inconclusive`. Coordinates are input-relative AA offsets, not IMGT numbering.

This exact scope string is embedded in the bundle card (`calibration_scope`) and surfaced in
`provenance.calibration_scope` of every result.

## §9 Public API
```python
import alignair
results = alignair.predict_aa(
    sequences,                              # str | Iterable[str] | path to FASTA/txt
    model="alignair-aa-igh-human@2.0.0",   # default = v2; v1 only if explicitly requested
    coverage_group=None,                    # None -> pooled fallback (labeled); or "full"/"trunc"/"short"
    device="auto",
)  # -> list[AAPrediction]  (§6)
```
Also `alignair.load_aa_model(...) -> AAPredictor` (§4) for repeated prediction. Kept **distinct** from the
nucleotide `Aligner.predict` / `predict_sequences` surface.

## §10 CLI / file interface
`alignair predict-aa <input.fasta|-> --model alignair-aa-igh-human@2.0.0 [--coverage-group full|trunc|short]
[--device auto] --out results.jsonl` → writes the §6 schema as JSONL + a `<out>.run.json` sidecar
(model_id/version, `weights_sha256`, `calibration_sha256`, class-map db_hash, alignair version, calibration
scope, counts, command, timestamp). A separate verb from the nt `predict`.

## §11 File structure & vertical-slice build order
New package `src/alignair/aa/product/`:
- `bundle.py` — AA `save_bundle(...)` / `load_bundle(...)` over `model_file.container` (§3/§4). **Slice 1.**
- `predictor.py` — `AAPredictor` + `predict_aa` core: tokenize → `class_probs` (reuse `alignair.aa.infer`) →
  per-gene top call + coordinates (start/end heads) → §6 without calibration. **Slice 2.**
- `calibration_group.py` — the §7 contract (`resolve_group`, pooled default). `schema.py` — `AAPrediction`
  dataclass + JSON. Calibrated-set output wired into the predictor. **Slice 3.**
- `cli.py` — `predict-aa` verb + run.json; façade export `predict_aa`/`load_aa_model` in `alignair/__init__.py`.
  **Slice 4.**
- `validation` — run the product path (load bundle → predict) on the frozen Phase-3 16-Ab cohort, descriptively
  (family accuracy consistency; no promotion, no gate). **Slice 5.**

Each slice is thin and end-to-end: Slice 1 produces + loads the real v2 bundle (proving the artifact contract);
Slice 2 predicts calls+coords from that bundle; Slice 3 adds calibrated sets + the group contract; Slice 4 is
the user-facing API/CLI; Slice 5 validates the product path on real protein.

## §12 P0 contracts (frozen before code)
Artifact (§3), result schema (§6), coordinates (§5), calibration-group (§7), guarantee (§8). Building the API or
packaging independently would risk freezing the wrong result schema / calibration contract / artifact format —
so these are settled here, first.

## §13 Testing
Bundle round-trip (save→load→bit-identical weights + integrity-manifest verification + tamper rejection);
tokenizer/class-map/ASC-table/calibration reconstruct from the bundle alone; predict_aa schema conformance;
calibration-group contract (supplied/pooled paths + labeling); coordinate bounds; the guarantee/scope string
present in every result; Slice-5 product-path family accuracy on the frozen cohort. No frozen incumbent/Phase-3
artifacts modified.
