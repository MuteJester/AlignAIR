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
- **asc_imgt** — the ASC→IMGT provenance table as a **one-to-many** map `new_allele → [all originating
  imgt_allele, …]` (from ALL `dc.asc_tables["V"]` rows for that ASC allele, not the first). The product must
  **retain every mapping** and never pick an arbitrary representative for allele/gene/family sets (this
  supersedes Phase-3's first-pick `ImgtView`, which was acceptable only for family scoring). Bundling asserts
  **every V-class member allele resolves** to ≥1 IMGT allele; an unmapped member hard-fails the build.
- **calibration** — the model's `calibration.json` (`genes.{V,J,D}.calibration_frozen.{q_pooled, q_mondrian}` +
  `meta` with `method, alpha, lambda, k_reg, target_coverage, bin_defs, group definitions, supported_distribution,
  class_map_db_hash, n_cal, n_test`). Missing method/alpha/lambda/k_reg/group-defs/quantiles/sample-counts →
  build fails (§4 binds these to the model).
- **card (header)** — `artifact_type:"alignair-aa-model", bundle_schema_version, architecture_version,
  model_id, model_version, created, {alignair,genairr,torch}_version, class_counts{v,j,d}, max_seq_length,
  calibration_scope (§8 string), license, citation, build_commit (public git commit), reference_provenance
  (class-map + ASC-table fingerprints), calibration_provenance (method/alpha/counts/supported_distribution)`.
  **No `host` / `user` fields** (they leak build-machine / maintainer identity) — those stay in a private,
  uncommitted build log only.

**AA integrity manifest** (the analog of the DNA `allele_order_sha256`, guarding head↔class-map alignment).
All hashes are computed over **canonicalized JSON** (sorted keys, fixed separators, UTF-8) so they are
reproducible: `weights_sha256`, `config_sha256`, `class_map_db_hash{V,J}` (existing 16-hex db_hash),
`class_map_sha256` — over the **complete ordered class-map content** (the full `allele_class` and
`class_members` mappings in canonical order, not a loose "class list") for V and J plus the ordered D allele
list — `asc_table_sha256`, `calibration_sha256`, `tokenizer_sha256`. On load, all are recomputed and checked;
any mismatch **hard-fails**. No pickle section exists in a distributable AA bundle (enforced like
`Aligner._assert_distributable`).

## §4 Safe loader
`load_aa_model(model, *, device="auto", trust_pickle=False, registry=None, offline=False) -> AAPredictor`:
resolves `model` (filesystem path / `hf://…` / `alignair-aa-igh-human@2.0.0`) via `resolve_model` + verified
cache; opens the container; verifies every section + the §3 integrity manifest; rebuilds `AAAlignAIR` (weights
from safetensors), the tokenizer, the V/J `EquivalenceClassMap`, the D allele order, the **one-to-many** ASC→IMGT
map (§3), and a `FrozenConformal` per gene from the embedded calibration. Returns an **`AAPredictor`**.

**Calibration↔model cryptographic binding (hard-fail on any mismatch).** Container integrity prevents
corruption but NOT accidentally packaging valid calibration from the *wrong* model. On load, verify:
- the calibration's recorded **model SHA-256 matches the bundled weights** (`weights_sha256`);
- the calibration's **class-map fingerprint(s) match the bundled maps** (`class_map_db_hash{V,J}`);
- the calibration's per-gene **label counts match the output-head dimensions** (V/J classes, D alleles);
- the calibration carries **`method, alpha, lambda, k_reg`, group definitions, and quantiles** (all present);
- **`n_cal, n_test` and `supported_distribution`** are recorded.
Any failure raises before an `AAPredictor` is returned. A legacy raw-pickle `.pt` is **refused unless
`trust_pickle=True`** (dev-only, never for distribution).

## §5 Coordinate contract (P0)
AA segment coordinates are **zero-based, half-open offsets into the input AA sequence**: `[start, end)`,
`0 ≤ start ≤ end ≤ len(sequence)`, derived from the model's per-gene start/end heads.

**Discretization + clipping (specified, not left to the caller):**
- The continuous soft-argmax start/end values are rounded to the nearest integer (`round`, half-to-even) and
  **clipped to `[0, len(sequence)]`**.
- If, after clipping, `start > end` for a gene, that gene's `coordinates` is set to `null` and a warning
  `"{gene}: incoherent start>end"` is attached (the class call is still returned — only coordinates are voided).
- Cross-gene ordering is **not enforced or repaired**; if `V_end > J_start` (or D outside V..J), coordinates are
  still reported as the model produced them, plus a warning `"segment coordinates overlap/out-of-order"`. The
  product does not silently reconcile coordinates — it reports and flags.
- Coordinates are **input-relative offsets, NOT IMGT residue numbering**; the ASC→IMGT bridge maps allele
  **names** only. Stated in every result and the card.

## §6 Result schema (P0) — distinct from nucleotide AIRR
`predict_aa` returns `list[AAPrediction]`. Each `AAPrediction` (JSON-serializable):
```
sequence_id: str
input_length: int
segments: { V|D|J: {
    top_call: { class_id, class_name,
                top_alleles: [IMGT-allele names, ...],  # top class's members, one-to-many ASC-expanded (ALL)
                genes:    [IMGT gene, ...],             # a protein-class can span >1 gene -> PLURAL
                families: [IMGT family, ...],           # and potentially >1 family -> PLURAL
                softmax_score },                        # raw uncalibrated softmax value (NOT a calibrated prob)
    calibrated_class_set:  [class_id, ...],             # sorted by class_id (stable set view)
    calibrated_classes: [                               # SAME set, ranked by the model's class softmax score
        { class_id, rank,                               # rank 1 = highest score; 1..N, deterministic class-id tie-break
          softmax_score,                                # the class's raw softmax value, VERBATIM (see below)
          alleles:  [IMGT-allele names, ...],           # this ONE class's full one-to-many ASC->IMGT members
          genes:    [IMGT gene, ...],
          families: [IMGT family, ...] },
        ...
    ],
    calibrated_allele_set: [IMGT-allele names, ...],    # class set -> members -> ALL originating IMGT alleles
    calibrated_gene_set:   [IMGT gene, ...],            # every gene any covered class maps to (retain all)
    calibrated_family_set: [IMGT family, ...],
    set_size: { classes: int, alleles: int, genes: int },
    coordinates: { start: int, end: int } | null        # §5
} }
inter_vj_aa: str | null                                 # method-defined V-end..J-start substring; NOT the AIRR junction/CDR3 (no anchor motif)
applied_calibration_group: { group: "full"|"trunc"|"short"|"pooled",
                             source: "supplied"|"pooled_fallback" }    # §7 (per-sequence; "estimated" gated off)
provenance: { model_id, model_version, weights_sha256, calibration_sha256,
              class_map_db_hash, alignair_version, calibration_scope }  # §8 string
warnings: [str, ...]                                    # e.g. input>max_len (rejected), group supplied, OOV, coord flags
```
- **Never singular for gene/family.** A V/J protein-equivalence class can contain alleles from multiple genes,
  and potentially multiple families; `genes`/`families` are always lists. A single "representative" may be shown
  **only** if a field explicitly labels it non-canonical (e.g. `representative_gene_non_canonical`); it is never
  the primary field.
- **D is different:** for the D segment, `class_id` is directly a **D allele-head index** (not a
  protein-equivalence class); its `top_alleles` is that one D allele and `calibrated_allele_set` is the D-allele
  set. This is documented in the schema and the card.
- **Allele/gene set expansion retains EVERY mapping** (one-to-many ASC→IMGT, §3) — no arbitrary representative.
- **Uncertainty in v1 is the calibrated-set size + top score, per gene** — there is **no `abstain` flag**
  and no invented confidence threshold. (A versioned abstention rule may be added in a later version; it is
  deliberately omitted from v1 rather than guessed.)
- **`calibrated_classes` scores are class-level, and only class-level.** For V/J the model predicts protein-
  equivalence classes; `softmax_score` is that class's raw softmax value copied verbatim — never renormalized over
  the set, and never divided among the class's alleles. When one class expands to several IMGT alleles, those
  alleles are **indistinguishable under the bundled ASC/protein representation** and share the single class-level
  score; the model has no evidence for a within-class split, so none is invented. The governing statement:
  > *Scores belong to model classes. All alleles expanded from one class share that class-level evidence;
  > no within-class allele score is identifiable.*
  **D is the one exception:** its head predicts allele indices directly, so each D entry's single allele carries a
  defensible allele-specific `softmax_score`. The flattened `calibrated_allele_set` is retained for convenience and
  is exactly the union of the `alleles` lists across `calibrated_classes` (no allele added or dropped).
- **The field is `softmax_score`, not `probability`, by design.** Split-conformal calibration calibrates the
  *set's* coverage (1−α), **not** the individual softmax value as an empirical probability of correctness. The
  per-class (and per-D-allele) score is an **uncalibrated model output**; consumers that need a coverage guarantee
  must use the calibrated *set* + `set_size`, not the raw score. This holds for D too: its allele-specific score is
  defensible but is not a calibrated allele probability.
- **What the calibrated set is — and is not.** The set is the smallest top-ranked class prefix whose calibrated
  cumulative softmax mass reaches the frozen threshold `qhat` (APS). It always contains at least the argmax class.
  Classes with similar scores often *widen* the set, but APS does **not** detect biological ambiguity or apply any
  exact-tie rule — a wider set is a consequence of the calibrated threshold, not a tie decision. The `rank` field
  in `calibrated_classes` uses an independent deterministic class-id tie-break for presentation only; it is not a
  calibration decision. **Displayed score examples are rounded** (e.g. `0.862 + 0.138 = 1.000` is rounding): the
  raw included softmax mass is strictly below 1 whenever any class remains outside the set.

**Explicitly NOT present** (and must not be implied): nucleotide coordinates, CIGAR strings, productivity calls,
or complete IMGT numbering. The schema may map compatible *concepts* to AIRR field names in a separate optional
adapter, but the native product output is AA-only and must never present nt-domain fields it does not compute.

## §7 Calibration-group selection contract (P0) — the key product decision
The frozen conformal quantiles are Mondrian `full`/`trunc`/`short` (+ a pooled marginal). An arbitrary user
peptide does **not** reveal its original simulator coverage fraction. **Do NOT silently choose a Mondrian bin
from raw sequence length.** Selection, in priority order:
1. **Explicit** — if the caller passes `coverage_group ∈ {"full","trunc","short"}`, use it;
   `source = "supplied"`. A supplied group inherits the group-conditional Mondrian guarantee **only if the label
   accurately represents the calibration protocol's coverage definition** (§8); every supplied-group result
   carries the warning: *"Group supplied by caller; conditional coverage applies only if this label matches the
   documented calibration-group definition and the input is exchangeable with that simulator group."*
2. **Validated estimate (gated, OFF for v1)** — a documented estimator deriving the coverage class from the
   model's *predicted* V-domain coverage. **Disabled until separately validated** on held-out simulator data;
   NOT prioritized for the first release (an unvalidated auto-selector would create a stronger-looking but
   weaker scientific contract).
3. **Pooled fallback (default)** — with no group supplied, use the **pooled quantile** (`q_pooled`),
   `source = "pooled_fallback"`, labeled as carrying only the **pooled marginal** guarantee. The honest default;
   never silently pretends to a group-conditional guarantee.

**Per-sequence groups.** `coverage_group` accepts either a single value applied to the whole batch, or a
per-sequence mapping/list (for mixed batches), so different reads in one call can use different groups. Each
result records its own `applied_calibration_group`.

## §8 Guarantee statement (P0) — displayed in every result + model card
> **Simulator-domain conformal guarantee.** Under exchangeability with the simulator calibration distribution
> (GenAIRR HUMAN_IGH_OGRDB), the conformal protein-**class** set attains the target coverage (1−α = 0.95) for
> the applied Mondrian group — **and, for a caller-supplied group, only if that label accurately represents the
> calibration protocol's coverage definition for the input** — or the pooled marginal coverage under the pooled
> fallback (§7). Expanding a covered V/J protein-class through the frozen class map (and the one-to-many ASC→IMGT
> map) deterministically preserves inclusion of that class's mapped nucleotide alleles (set-membership is
> preserved by the fixed expansion). **This is NOT a real-therapeutic-domain 95% guarantee** — the real-protein
> evaluation (Phase 3) was descriptive and `inconclusive`. Coordinates are input-relative AA offsets, not IMGT
> numbering.

This exact scope string is embedded in the bundle card (`calibration_scope`) and surfaced in
`provenance.calibration_scope` of every result.

## §9 Public API
```python
import alignair
results = alignair.predict_aa(
    sequences,                              # str | Iterable[str] | Iterable[{"sequence_id","sequence"}]
    model="alignair-aa-igh-human@2.0.0",   # default = v2; v1 only if explicitly requested
    coverage_group=None,                    # None -> pooled fallback; "full"/"trunc"/"short"; or per-sequence map
    device="auto",
    on_invalid="error",                     # "error" (default) | "skip" -> structured per-record error result
)  # -> list[AAPrediction]  (§6)
```
The **library API accepts sequences/records only — never file paths** (a bare string is always a *sequence*,
not a filename; the current path/sequence ambiguity is removed). **File parsing belongs to the CLI (§10).** Also
`alignair.load_aa_model(...) -> AAPredictor` (§4) for repeated prediction. Kept **distinct** from the nucleotide
`Aligner.predict` / `predict_sequences` surface.

### §9.1 Input validation & failure behavior (P0)
- **Sequence IDs:** taken from record `sequence_id` if provided; otherwise generated deterministically as
  `seq_<0-based index>`.
- **Normalization:** strip surrounding whitespace; **uppercase**; reject any interior whitespace.
- **Alphabet:** the 20 canonical residues only. `X` is accepted (maps to the model's X token). **`*` (stop) is
  REJECTED** — the model was trained on stop-free inputs; a sequence containing `*` is invalid. Any other
  non-canonical character is invalid.
- **Empty sequence:** invalid.
- **Overlength (> `max_seq_length` = 192):** **never silently truncated.** Rejected as invalid by default;
  truncation happens only under an explicit, caller-set policy (`truncate="none"` default; a future explicit
  policy value would be honored + warned) — v1 rejects.
- **Batch failure mode:** with `on_invalid="error"` (default) the whole call raises on the first invalid record
  (fail-fast, deterministic). With `on_invalid="skip"`, each invalid record yields a **structured per-record
  error** object (`{sequence_id, error, detail}`) in place order, and valid records are still predicted — no
  silent dropping. The mode is explicit; the call never partially/silently discards input.

## §10 CLI / file interface
`alignair predict-aa <input.fasta|-> --model alignair-aa-igh-human@2.0.0 [--coverage-group full|trunc|short]
[--device auto] --out results.jsonl` → writes the §6 schema as JSONL + a `<out>.run.json` sidecar
(model_id/version, `weights_sha256`, `calibration_sha256`, class-map db_hash, alignair version, calibration
scope, counts, command, timestamp). A separate verb from the nt `predict`.

## §11 File structure & vertical-slice build order
New package `src/alignair/aa/product/`:
- `bundle.py` — AA `save_bundle(...)` / `load_bundle(...)` over `model_file.container` (§3/§4). **Slice 1.**
- `predictor.py` — `AAPredictor` + `predict_aa` core: validate/normalize input (§9.1) → tokenize → `class_probs`
  (reuse `alignair.aa.infer`) → per-gene top call (plural `genes`/`families` via the one-to-many ASC map) +
  coordinates (start/end heads, §5) → §6 without calibration. **Slice 2.**
- `calibration_group.py` — the §7 contract (`resolve_group`, pooled default, per-sequence, supplied-warning).
  `schema.py` — `AAPrediction` dataclass + JSON. Calibrated set output (class → members → **all** IMGT alleles)
  wired into the predictor. **Slice 3.**
- `cli.py` — `predict-aa` verb (**file parsing lives here**, §10) + run.json; façade export
  `predict_aa`/`load_aa_model` in `alignair/__init__.py`. **Slice 4.**
- `validation/` — **Slice 5**, expanded beyond family consistency (Phase-3 alone cannot validate D/J sets,
  coordinates, or Mondrian behavior):
  1. the frozen Phase-3 **16-antibody** descriptive product-path check (family accuracy);
  2. a **frozen simulator golden cohort** spanning `full`/`trunc`/`short`;
  3. **V/D/J call + calibrated-set parity** between the product path and the research path (`class_probs` +
     `FrozenConformal`) on the golden cohort — they must agree exactly;
  4. **coordinate + inter_vj** checks (bounds, `start>end` voiding, `inter_vj_aa` substring correctness);
  5. **pooled vs supplied-group** behavior (correct quantile, labels, warnings);
  6. **tamper + wrong-calibration/wrong-model rejection** (mutated section → fail; calibration from a different
     model → §4 binding fail).

Each slice is thin and end-to-end: Slice 1 produces + loads the real v2 bundle (proving the artifact contract);
Slice 2 predicts calls+coords from that bundle; Slice 3 adds calibrated sets + the group contract; Slice 4 is
the user-facing API/CLI; Slice 5 validates the product path on real protein.

## §12 P0 contracts (frozen before code)
Artifact (§3), result schema (§6), coordinates (§5), calibration-group (§7), guarantee (§8). Building the API or
packaging independently would risk freezing the wrong result schema / calibration contract / artifact format —
so these are settled here, first.

## §13 Testing
- **Bundle:** save→load→bit-identical weights; canonical-JSON integrity manifest recomputes + verifies; **tamper
  rejection** (mutated section); tokenizer/class-map/ASC-table/calibration reconstruct from the bundle alone.
- **Calibration↔model binding (§4):** load fails when calibration's model-SHA / class-map fingerprint / label
  counts do not match the bundled weights/maps/heads, and when method/alpha/lambda/k_reg/counts are missing.
- **One-to-many ASC expansion (§3/§6):** every V-class member resolves to ≥1 IMGT allele (build asserts);
  expansion retains ALL mappings; a class spanning >1 gene yields plural `genes`; no arbitrary representative.
- **Input validation (§9.1):** empty rejected; `*` rejected; overlength (>192) rejected (never truncated);
  interior whitespace rejected; ID generation + uppercase normalization; `on_invalid="error"` fail-fast vs
  `"skip"` structured per-record errors.
- **Coordinates (§5):** soft-argmax→rounded int, clip to `[0,len]`, `start>end` → null + warning, cross-gene
  overlap flagged not repaired; `inter_vj_aa` substring correctness.
- **Calibration group (§7):** supplied group uses its quantile + attaches the supplied-group warning; `None` →
  pooled fallback labeled; per-sequence groups in a mixed batch.
- **Provenance/scope:** the §8 scope string appears in every result + the card; card carries no `host`/`user`.
- **Parity:** product path == research path (`class_probs`+`FrozenConformal`) V/D/J calls + calibrated sets on
  the golden cohort.
- **Slice-5 validation** (the six §11 items). No frozen incumbent/Phase-3 artifacts modified.
