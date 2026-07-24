# AlignAIR-AA — Gate 2: Result-Semantics Audit

> **AIRR Utility & Trust Gate, Gate 2 of 6.** Every emitted field, reviewed for: biological definition,
> training target, validation evidence (simulator vs real), AIRR/IMGT naming collision, coordinate
> convention, trustworthy interpretation, and failure behavior. This audit resolves `inter_vj_aa`, D's
> status, the meaning of segment coordinates, and the reference semantics.

**Status:** DRAFT for review · **Date:** 2026-07-24 · **Model:** `alignair-aa-igh-human 2.0.0`.
Companion to [Gate 1 product contract](2026-07-24-aa-trust-gate-gate1-product-contract.md).

## Evidence vocabulary (used throughout)

| Level | Meaning |
|---|---|
| **SIM-strong** | Validated on the frozen simulator distribution with strong numbers. |
| **SIM-moderate** | Validated on simulator, weaker / higher-variance. |
| **SIM-mechanical** | Only the *construction* is checked (e.g. a substring equals a slice); no biological validation. |
| **REAL-descriptive** | Observed on real antibodies but statistically **inconclusive** (n=16; exact one-sided binomial cannot reach a 0.95 lower bound below ~59 obs). |
| **NONE** | No real-domain validation at all. |

**The single most important fact:** real-domain evidence exists for **exactly one thing** — V calls at
**family** granularity, descriptively, on 16 therapeutic antibodies (16/16, inconclusive). **Every other
field's real-domain evidence is NONE.** Simulator numbers do not transfer to real proteins without
evidence, and the calibration guarantee is simulator-domain only.

---

## A. Identity / input fields

### `sequence_id`, `input_length`
- **Definition:** the caller's id (or `seq_<i>`), and the input length in aa.
- **Training target:** n/a (bookkeeping). **Evidence:** n/a.
- **AIRR/IMGT collision:** none. **Coordinate convention:** n/a.
- **Trustworthy interpretation:** exact. `input_length` is the validated aa length (whitespace stripped).
- **Failure behavior (mode-dependent):** with `on_invalid="error"` (**default**) validation **raises
  `ValueError`** on the first invalid record. With `on_invalid="skip"` it returns an **`ErrorResult`**
  (codes: `empty`, `invalid_character`, `interior_whitespace`, `overlength`) in the record's original
  batch position, alongside valid `AAPrediction`s. Nothing is ever silently coerced. Leading/trailing
  whitespace is stripped and input is uppercased before validation; interior whitespace is rejected.

## B. V and J segment — class / allele / gene / family

Applies to `segments.V` and `segments.J`: `top_call.{class_id, class_name, top_alleles, genes, families,
softmax_score}` and the `calibrated_*` fields (audited in §E).

- **Biological definition:** the model predicts a **protein-equivalence class** per gene — a set of
  germline alleles identical under the bundled translated representation (149 for V, 7 for J). **V and J
  are named differently:** **V** classes live in **OGRDB-ASC** space and are expanded to IMGT alleles via
  the frozen **ASC→IMGT bridge** (§G); **J** classes already carry **IMGT-style** allele names and pass
  through **without any ASC mapping**. `top_alleles`/`genes`/`families` are the class's members;
  `class_name` (`v_class_<id>` / `j_class_<id>`) is an internal label, not a biological name.
- **Training target:** the ground-truth germline V/J **protein-equivalence class** of simulated human IGH
  reads (GenAIRR `HUMAN_IGH_OGRDB`).
- **Validation evidence:**
  - V, simulator: **SIM-strong** on clean/full (~0.99) down to **SIM-moderate** under SHM/indel/short
    (moderate-SHM ~0.94; multi-indel/short far lower — see Gate 1 coverage contract).
  - V, real: **REAL-descriptive at family level only** (16/16, inconclusive). **V gene and V allele
    accuracy on real proteins: NONE.**
  - J, simulator: **SIM-strong** on clean (~0.994). J, real: **NONE**.
- **AIRR/IMGT naming collision:** the *field names* (`top_alleles`, `genes`, `families`, `top_call`) do
  **not** reuse AIRR `v_call`/`j_call`; no collision. The *values* are IMGT allele/gene/family names. For
  **V** they are a **provenance expansion of an ASC class** (§G), not an alignment against the current
  IMGT database, and ASC family ≠ IMGT family in general. For **J** they are the IMGT-named class members
  directly (no ASC layer).
- **Coordinate convention:** n/a (class fields).
- **Trustworthy interpretation:** on in-contract human VH, the **V family** is the most defensible unit
  (the only one with any real evidence). Treat the **class** (and its allele expansion) as "alleles
  indistinguishable under the bundled representation," not as a resolved single allele. **Do not** read
  real-domain gene/allele accuracy from simulator numbers.
- **Failure behavior:** always returns the argmax class; there is **no abstain** in v1. Out-of-contract
  input (§Gate 1.5) still gets a confident-looking class — the core risk Gate 3 must close.

## C. D segment — **EXPERIMENTAL**

Applies to `segments.D`.

- **Biological definition:** D is emitted from a **D allele-head index** (33 values into
  `d_allele_order`), *not* a protein-equivalence class. `top_call.class_id` is that index; `class_name`
  is the D allele name; `top_alleles`/`genes`/`families` are single-element (one D allele parsed).
- **Training target:** ground-truth D allele of simulated reads.
- **Validation evidence:** simulator **SIM-moderate at best** (D-allele-macro ~0.80 clean; high variance,
  and the v2 promotion explicitly found D **unchanged** — the apparent gain was noise). **Real: NONE.**
  AA D signal is intrinsically weak (short segment, heavy junctional erosion).
- **AIRR/IMGT collision:** field names don't reuse `d_call`; values are IMGT D allele names.
- **Trustworthy interpretation:** **experimental — do not rely on D.** Documented as experimental in the
  **model card** and this audit (owner decision 2026-07-24: no per-result status field in v1). A consumer
  should treat any D output as a hypothesis with no validated support.
- **Failure behavior:** same as V/J — always argmax, no abstain.

## D. Segment coordinates

Applies to `segments.{V,D,J}.coordinates = {start, end} | null`.

- **Biological definition:** the model's estimate of where the segment begins/ends **in the input aa
  sequence**.
- **Training target:** simulated segment boundaries (soft-argmax start/end heads).
- **Validation evidence — per segment, CLEAN SIMULATOR ONLY** (n=115 surviving clean examples;
  convergence evaluation `boundary_mae`, `docs/.../2026-07-22-aa-convergence/checkpoints_metrics.json`):
  - **V:** start MAE **0.92 aa**, end MAE **0.17 aa**.
  - **J:** start MAE **0.34 aa**, end MAE **0.91 aa**.
  - **D coordinates: UNREPORTED** — the convergence evaluation records **no** D boundary MAE. D
    coordinates are therefore **unvalidated even in simulation** (NOT `SIM-moderate`).
  - **Real-domain: NONE** for all segments. Evidence is confined to the **clean** stratum; boundary error
    under SHM / indel / truncation is **unquantified**.
- **AIRR/IMGT collision — IMPORTANT:** AIRR `v_sequence_start`/`v_sequence_end` (etc.) are **1-based,
  closed, nucleotide** coordinates. These are **0-based, half-open, amino-acid** offsets into the input
  (`start` inclusive, `end` exclusive). Different base, different closure, different molecule. AIRR permits
  uniquely-named **custom** fields; AlignAIR-AA uses its **own** 0-based AA schema and deliberately does
  **not** overload AIRR coordinate field names.
- **Coordinate convention (canonical):** **0-based half-open aa offsets into the input sequence.** Not
  IMGT numbering; not nucleotide; not repaired.
- **Trustworthy interpretation:** approximate boundaries for locating regions, not exact delimitation.
- **Failure behavior:** rounded (half-to-even) and clipped to `[0, len]`; if `start > end` after
  clipping the field is set to **`null`** and a warning is added — never silently repaired. Cross-segment
  overlap (V-end > J-start, or D outside V..J) is **flagged in `warnings`, not corrected**.

## E. Calibrated set fields

Applies to `calibrated_class_set`, `calibrated_classes[]` (`{class_id, rank, softmax_score, alleles,
genes, families}`), `calibrated_allele_set`, `calibrated_gene_set`, `calibrated_family_set`, `set_size`.

- **Biological definition:** a **split-conformal (APS) prediction set** of protein-classes for the
  segment, plus its one-to-many ASC→IMGT expansion and a ranked, score-carrying view. `softmax_score` is
  the class's raw softmax value copied verbatim (see §F).
- **Training target:** the conformal quantile is calibrated so the set covers the true **class** with
  probability ≈ 1−α = 0.95 **under the applied Mondrian group** (`full`/`trunc`/`short`) or the pooled
  marginal fallback.
- **Validation evidence:** **SIM-strong** — coverage ~0.95+ on held-out simulator data (V). **Real
  coverage: NONE.** On real/OOD input the set is descriptive, not a 95% guarantee.
- **AIRR/IMGT collision:** none — no AIRR equivalent; custom fields.
- **Trustworthy interpretation:** the set is the honest uncertainty signal *within the simulator domain*.
  Use `set_size` and the class/allele set to separate **narrower from broader simulator-calibrated
  hypothesis sets**. A **singleton or small set on real/OOD input is NOT established confidence** — the
  guarantee is simulator-domain. **A widened set is simulator-domain ambiguity, not OOD detection**
  (Gate 1 §5). Allele expansion **preserves inclusion** deterministically but adds no allele-level
  resolution.
- **Failure behavior:** always contains ≥ the argmax class. On a supplied coverage group unavailable in
  the calibration, the whole sequence falls back to pooled with a warning (never a half-applied group).

## F. `softmax_score` (in `top_call` and `calibrated_classes[]`)

- **Definition:** the model's **raw, uncalibrated softmax value** for the class (for D, the D-head index).
- **Training target:** cross-entropy over classes; the value is a model output, not a calibrated risk.
- **Validation evidence:** **NONE as a probability of correctness.** Conformal calibrates *set coverage*,
  not individual softmax values.
- **AIRR/IMGT collision:** none. Renamed from `probability` precisely so it is not read as a calibrated
  probability (owner decision 2026-07-23).
- **Trustworthy interpretation:** a relative ranking signal only. For a coverage guarantee use the
  **set**, not the score. Class scores are **not** split among a class's alleles (no within-class allele
  score is identifiable). Displayed score sums are rounded; the raw included mass is < 1 whenever any
  class stays outside the set.
- **Failure behavior:** verbatim from softmax; never renormalized.

## G. Reference semantics — `top_alleles`/`genes`/`families` values and provenance

> **This section is V-specific.** Only **V** is ASC-clustered and needs the ASC→IMGT bridge. **J**
> classes are **not** ASC — their members are IMGT-style J allele names used directly, with no mapping
> layer. Do not read the rest of this section as applying to J.

- **What the model actually learns (V):** OGRDB **ASC** space — it predicts cluster names like
  `IGHVF1-G1*01`, **not** IMGT `IGHV3-23*01`.
- **What the emitted IMGT names are:** the `imgt_allele` **provenance** of each ASC cluster, via the
  frozen `asc_tables` mapping bundled with the model. This is a **provenance mapping, not an alignment
  against the current IMGT database.** One ASC class can map to several IMGT alleles (one-to-many).
- **Versioning risk:** OGRDB human germline sets are **versioned and continue to evolve**
  (https://ogrdb.airr-community.org/germline_sets/Homo%20sapiens). The exact frozen source and mapping
  must stay visible — they are pinned by `provenance.class_map_db_hash` and the bundle's
  `reference_provenance`. A user comparing against a *newer* IMGT/OGRDB release may see legitimate
  differences that are versioning, not errors.
- **Trustworthy interpretation:** treat the IMGT names as "the alleles this frozen ASC class stands for,"
  and always alongside the pinned reference hash — not as a live IMGT lookup.

## H. `inter_vj_aa` — RESOLVED (Finding 3)

- **Definition:** the input residues between the predicted V-end and J-start (`sequence[V_end:J_start]`),
  or `null` when V/J coordinates are missing/incoherent. **No anchor-motif detection is performed.**
- **AIRR/IMGT collision — RESOLVED:** AIRR defines `junction` (`junction_aa`) as CDR3 **plus conserved
  flanking residues**, and distinguishes it from `cdr3`. This field is **neither**. It was renamed from
  `junction_aa` → **`inter_vj_aa`** (commit `9ae3605`) so it cannot be read as an AIRR junction or a CDR3.
- **Validation evidence:** **SIM-mechanical only** (the substring equals the slice). Its biological
  correspondence to CDR3/junction is **NOT** validated. **Real: NONE.**
- **Trustworthy interpretation:** a region locator for manual inspection. Interpreting it as CDR3 or an
  AIRR junction is **unsupported** until anchor-based equivalence is separately validated.

## I. `applied_calibration_group`, `provenance`, `warnings`

- **`applied_calibration_group` = {group, source}:** which Mondrian group (or pooled) the set used, and
  why (`supplied` / `pooled_fallback`). Honest default is **pooled** (no group claimed from raw length).
  Supplied groups carry a warning that conditional coverage holds only if the label matches the
  calibration protocol. **Evidence — keep two claims separate:** the quantile-**selection mechanism** is
  **SIM-mechanical** (verified it picks the right frozen quantile, and falls back to pooled when a
  supplied group is absent); the **empirical set coverage** that results is the simulator evidence,
  reported under §E (SIM-strong for V). Real: **NONE**.
- **`provenance`:** `model_id`, `model_version`, `weights_sha256`, `calibration_sha256`,
  `class_map_db_hash`, `alignair_version`, and the `calibration_scope` string (the simulator-domain
  guarantee statement, surfaced in **every** result). Exact/verifiable. No AIRR collision.
- **`warnings`:** free-text flags (coordinate voiding, cross-segment overlap, supplied-group,
  pooled-fallback, etc.). Advisory; not machine-typed in v1.

---

## Field-evidence summary table

| Field | Simulator evidence | Real-domain evidence | Status |
|---|---|---|---|
| V family (`genes`→family / `families`) | SIM-strong→moderate | **REAL-descriptive (n=16, inconclusive)** | **Provisional real-domain anchor:** 16 therapeutic antibodies, 3 families, IGHV3-dominated; **not certified** for general human IGHV-family assignment |
| V gene (`genes`) | SIM-strong→moderate | **NONE** | Simulator-only |
| V allele (`top_alleles`) | SIM-moderate | **NONE** | Simulator-only |
| J (class/gene/allele) | SIM-strong (clean) | **NONE** | Simulator-only |
| **D (all)** | SIM-moderate (noisy) | **NONE** | **Experimental** |
| **V/J** coordinates | clean-sim only: V start 0.92 / end 0.17 aa; J start 0.34 / end 0.91 aa MAE (n=115) | **NONE** | Limited (clean sim); SHM/indel/trunc unquantified |
| **D** coordinates | **unreported** (no boundary MAE recorded) | **NONE** | **Unvalidated** |
| `inter_vj_aa` | SIM-mechanical | **NONE** | Method-defined; not CDR3/junction |
| Calibrated set coverage | SIM-strong (~0.95) | **NONE** | Simulator-domain guarantee only |
| Calibration-group selection | SIM-mechanical (mechanism) | **NONE** | Verified selection; coverage is the separate SIM evidence above |
| `softmax_score` | n/a (uncalibrated) | **NONE** | Ranking signal only |
| Reference (IMGT names) | V: frozen ASC→IMGT provenance map; J: IMGT-direct | — | Provenance mapping (V), not IMGT alignment |

## Resolutions produced by this audit

1. **`junction_aa` → `inter_vj_aa`** — done (commit `9ae3605`); documented as method-defined, not AIRR.
2. **D is experimental** — documented in the model card and this audit; no per-result status field in v1.
3. **Segment coordinates** — canonical convention fixed: **0-based half-open aa offsets into the input**,
   not IMGT numbering, not nucleotide, not AIRR coordinate fields.
4. **Reference semantics** — the emitted IMGT names are a frozen OGRDB-ASC→IMGT **provenance mapping**,
   pinned by the reference hash; not a live IMGT alignment.
5. **No AIRR-compliance claim** — native JSON; no standard AIRR field name is reused with different
   semantics (owner decision 2026-07-24).

## Follow-on (not this gate)

- **Gate 3:** build the adversarial/OOD cohort and a validated applicability decision (or the product
  refuses out-of-domain claims). This is where the "always returns a confident class" risk is closed.
- **Model card update:** fold the field-evidence table + D-experimental + reference-versioning notes into
  the distributed card (before any beta).
