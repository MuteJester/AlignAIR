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
- **Failure behavior:** invalid inputs return an `ErrorResult` (codes: `empty`, `invalid_character`,
  `interior_whitespace`, `overlength`) instead of an `AAPrediction`; nothing is silently coerced.

## B. V and J segment — class / allele / gene / family

Applies to `segments.V` and `segments.J`: `top_call.{class_id, class_name, top_alleles, genes, families,
softmax_score}` and the `calibrated_*` fields (audited in §E).

- **Biological definition:** the model predicts an **OGRDB-ASC protein-equivalence class** (149 for V, 7
  for J) — a set of germline alleles that are *identical under the bundled translated/ASC representation*.
  `top_alleles`/`genes`/`families` are that class's IMGT members via the frozen ASC→IMGT provenance map
  (§G). `class_name` (`v_class_<id>`) is an internal label, not a biological name.
- **Training target:** the ground-truth germline V/J **protein-equivalence class** of simulated human IGH
  reads (GenAIRR `HUMAN_IGH_OGRDB`).
- **Validation evidence:**
  - V, simulator: **SIM-strong** on clean/full (~0.99) down to **SIM-moderate** under SHM/indel/short
    (moderate-SHM ~0.94; multi-indel/short far lower — see Gate 1 coverage contract).
  - V, real: **REAL-descriptive at family level only** (16/16, inconclusive). **V gene and V allele
    accuracy on real proteins: NONE.**
  - J, simulator: **SIM-strong** on clean (~0.994). J, real: **NONE**.
- **AIRR/IMGT naming collision:** the *field names* (`top_alleles`, `genes`, `families`, `top_call`) do
  **not** reuse AIRR `v_call`/`j_call`; no collision. The *values* are IMGT allele/gene/family names, but
  they are a **provenance expansion of an ASC class**, not an alignment against the current IMGT database
  (§G). ASC family ≠ IMGT family in general.
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
- **Validation evidence:** **SIM-moderate** — soft-argmax boundaries carry ±1–2 residue jitter in
  simulation; **real: NONE**.
- **AIRR/IMGT collision — IMPORTANT:** AIRR `v_sequence_start`/`v_sequence_end` (etc.) are **1-based,
  closed, nucleotide** coordinates. These are **0-based, half-open, amino-acid** offsets into the input
  (`start` inclusive, `end` exclusive). Different base, different closure, different molecule. The field
  names deliberately do **not** reuse AIRR coordinate names.
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
  Use `set_size` and the class/allele set to separate confident from ambiguous calls. **A widened set is
  simulator-domain ambiguity, not OOD detection** (Gate 1 §5). Allele expansion **preserves inclusion**
  deterministically but does not add allele-level resolution.
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

- **What the model actually learns:** OGRDB **ASC** space — it predicts cluster names like
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
  calibration protocol. **Evidence:** SIM-strong (mechanism); real NONE.
- **`provenance`:** `model_id`, `model_version`, `weights_sha256`, `calibration_sha256`,
  `class_map_db_hash`, `alignair_version`, and the `calibration_scope` string (the simulator-domain
  guarantee statement, surfaced in **every** result). Exact/verifiable. No AIRR collision.
- **`warnings`:** free-text flags (coordinate voiding, cross-segment overlap, supplied-group,
  pooled-fallback, etc.). Advisory; not machine-typed in v1.

---

## Field-evidence summary table

| Field | Simulator evidence | Real-domain evidence | Status |
|---|---|---|---|
| V family (`genes`→family / `families`) | SIM-strong→moderate | **REAL-descriptive (n=16, inconclusive)** | Supported (family only) |
| V gene (`genes`) | SIM-strong→moderate | **NONE** | Simulator-only |
| V allele (`top_alleles`) | SIM-moderate | **NONE** | Simulator-only |
| J (class/gene/allele) | SIM-strong (clean) | **NONE** | Simulator-only |
| **D (all)** | SIM-moderate (noisy) | **NONE** | **Experimental** |
| Segment coordinates | SIM-moderate (±1–2 jitter) | **NONE** | Simulator-only |
| `inter_vj_aa` | SIM-mechanical | **NONE** | Method-defined; not CDR3/junction |
| Calibrated set coverage | SIM-strong (~0.95) | **NONE** | Simulator-domain guarantee only |
| `softmax_score` | n/a (uncalibrated) | **NONE** | Ranking signal only |
| Reference (IMGT names) | provenance map (frozen) | — | Provenance mapping, not IMGT alignment |

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
