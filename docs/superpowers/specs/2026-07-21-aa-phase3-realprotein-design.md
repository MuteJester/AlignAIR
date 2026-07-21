# Amino-Acid AlignAIR — Phase 3: Real-Protein Validation on Curated Therapeutic Antibodies

**Date:** 2026-07-21 · **Branch:** `experiment/aminoacid-alignair` · **Status:** design approved, ready for
implementation plan. **Does not modify the frozen research snapshot.**

Relationship to prior work: this evaluates the immutable Stage-1 snapshot
(`docs/superpowers/artifacts/2026-07-20-aa-strategic/SNAPSHOT.md`, model sha256 `54eef812…`,
`calibration.json`) on real therapeutic-antibody protein under distribution shift. No retraining, no
recalibration. The simulator-domain milestone stays frozen and tagged (`aa-igh-sim-milestone-2026-07-21`).

---

## §1 Purpose & framing — a falsification test of transfer

Phase 3 asks one question: does the simulator-frozen AA IGH model — its V-gene predictions **and** its
simulator-domain conformal calibration — hold on **real therapeutic-antibody heavy-chain V domains**, which
are a genuine distribution shift (humanized/engineered/affinity-matured, not GenAIRR SHM)?

The result is reported as **"simulator-domain calibration evaluated under real-domain shift,"** never as a
new real-domain guarantee.

**Predeclared three-way decision** on the primary endpoint (gene-set conformal coverage on Gold; and on
class/allele coverage where allele labels exist). Fix α = 0.05 and use **exact one-sided binomial
(Clopper–Pearson) bounds**, decided before any results are seen:

- **`failed` (transfer failure):** the upper one-sided (1−α) bound for coverage is **< 0.95** — i.e. the
  one-sided binomial test rejects H₀: coverage ≥ 0.95.
- **`supported`:** the lower one-sided (1−α) bound for coverage is **≥ 0.95**.
- **`inconclusive`:** anything between — the interval straddles 0.95.

Rationale for the three-way rule (not a bare threshold): an observed proportion below 0.95 is **not**
automatically failure because finite test samples fluctuate; conversely a small cohort scoring 95–100% does
**not** prove transfer. Every outcome is a publishable, honest result — `failed`/`inconclusive` quantifies
the shift and motivates a later recalibration phase.

**Which endpoint licenses which conclusion** (binds §1 to the coarsening argument in §3): the **gene-set
coverage** endpoint is *falsification-only* — it may yield `failed` or `inconclusive` but **never
`supported`** (coarsening makes gene coverage strictly easier than the class/allele guarantee, so a high
gene-coverage bound cannot establish it). `supported` may be concluded **only** from the **class/allele
coverage** endpoint on the allele-labelled subset. Each endpoint is evaluated by the same exact-binomial
rule above; the overall decision is the conjunction (a `failed` on gene coverage stands regardless of the
class/allele result's sample size).

---

## §2 Data contract — two tiers, sequences and labels separated

Sequences and labels are sourced **independently**. Sequences come from Thera-SAbDab by INN/name; labels
come from the tier-specific provenance below. A row is admitted only if **sequence + label + citation** all
resolve.

### Tiers
- **Tier 1 — Gold (headline).** V gene = the **source-declared human IGHV design ancestry / humanization
  template**, independent of AlignAIR and **not necessarily the closest sequence-aligned germline**. One
  **citation per row** (paper/patent/INN monograph). Tool-independent, gene-level, small N (~tens).
- **Tier 2 — Secondary.** V gene from **IMGT/mAb-DB** expert curation. Larger N, always labeled
  *"expert-curated, alignment-informed → weaker independence."* **Never pooled with Gold**; reported in a
  separate block.

### Inclusion / out-of-scope
Admit only **conventional human/humanized heavy-chain variable domains with a human IGHV label.** The
following are separate **out-of-scope categories** (recorded, counted, excluded — never silently dropped):
murine, chimeric nonhuman-variable, **VHH / single-domain (camelid)**, synthetic scaffolds, and
unresolved/ambiguous chains.

### Row identity & dedup
- **Bispecifics** get a chain/arm identifier (`arm_id`); the INN alone is insufficient.
- **Deduplicate identical VH** sequences (`exact_vh_id` = hash of the VH AA string).
- Record **related therapeutic lineages/templates** (`lineage_id`) so related products don't inflate
  effective sample size. Cluster-level statistics (§3) resample by `lineage_id`.

### Provenance & audit columns (per row)
`name/INN`, `arm_id`, `vh_aa`, `exact_vh_id`, `lineage_id`, `chain_class` (in-scope vs which out-of-scope
category), `seq_source`, `seq_source_version`, `retrieval_date`, `raw_file_sha256`, `true_v_gene`,
`true_v_allele?`, `in_catalog` (present in frozen model catalogue?), `tier` (gold|imgt), `label_source`,
`label_citation`, `evidence_locator` (page/table/patent-example), `humanization_status`, `true_j?`,
`exclusion_reason?`.

### OOV handling
Truth genes **absent from the frozen model catalogue** are flagged **OOV**, not counted as ordinary errors.
Report **both** end-to-end accuracy (OOV counts as miss) **and** conditional in-catalog accuracy (OOV
excluded).

### Licensing / redistribution
- **Thera-SAbDab:** download availability is **not** a redistribution license. Verify terms; if
  redistribution is not clearly permitted, commit a **manifest + acquisition script** (names + hashes +
  fetch logic) instead of the raw sequences. Cite: Thera-SAbDab / SAbDab (Dunbar et al.,
  *Nucleic Acids Res.* 2020, https://academic.oup.com/nar/article/48/D1/D383/5573951).
- **IMGT:** currently **CC BY 4.0** — record the terms version + attribution in the artifact. Cite:
  IMGT terms of use (https://www.imgt.org/about/termsofuse.php).

### Feasibility gate (not an evidence gate)
A **Gold floor of 15** is an *acquisition-feasibility* gate: below it, **stop and report** rather than pad
with guesses. It is **not** evidence that 95% coverage transfers — even 15/15 has a wide exact interval. Any
small-Gold result is labeled **exploratory/inconclusive** regardless of the observed proportion.

---

## §3 Prediction & scoring

### Preconditions (assert before any inference; hard-fail otherwise)
- Model **SHA-256** matches the snapshot (`54eef812…`).
- **Class-map fingerprint** matches (V db_hash `e1a455b8…`).
- **Calibration parameters + group definition** match `calibration.json` provenance (α, bin defs,
  min_group).
- Input passes the **full-VH completeness contract** (a complete heavy variable domain; exact predicate
  defined in the plan — anchors + length floor).
- Use **`q_mondrian["full"]`**, the frozen full-group quantile — **not** the pooled quantile.

### Two semantic adjustments
1. **Gene-level coverage is a *coarsened* test.** The conformal object was calibrated on protein
   classes/alleles; mapping predictions to genes can only make coverage **easier**. Therefore: gene coverage
   **< 0.95 disproves transfer**, but gene coverage **≥ 0.95 does not establish** allele/class calibration.
   Report **exact class/allele coverage wherever allele labels exist** — that is the only endpoint that can
   speak to the original guarantee.
2. **"Top-1 gene" may not be singular.** A top-ranked protein class can expand to alleles from multiple
   genes. Report **top-class gene-set accuracy**: whether the documented gene appears among the genes
   represented by the top-ranked class. **Never pick an arbitrary single gene.**

### Reported per tier
- **OOV rate.**
- **Top-class gene-set accuracy.**
- **Gene-set conformal coverage** (primary decision endpoint; falsification-only per adjustment 1).
- **Class/allele coverage** on the allele-labelled subset (the only endpoint that can *support* the
  original guarantee).
- **Set size** — mean / median / p95 — in **classes, genes, and nucleotide alleles**.
- **Wilson/binomial (exact) intervals** plus a **cluster bootstrap** resampling by unique VH (`exact_vh_id`)
  or therapeutic lineage (`lineage_id`).

### Truncation stress test (separate, synthetic)
A **separately labelled synthetic stress test on real sequences**: deterministic crops via the existing
`aa_truncate`, **frozen seeds**, **paired analysis**, and the **exact original coverage-bin definitions**
(`full`/`trunc`/`short` from `calibration.json`). Reported apart from the real-domain primary result; probes
the `trunc`/`short` Mondrian groups under real partial coverage (truth gene unchanged).

### Central caveat (the most important correction)
> A small gene-labelled cohort **can falsify severe transfer failure**, but **cannot by itself validate** the
> original 95% allele/class guarantee.

---

## §4 Components, artifacts, tests

### Components (small, single-purpose, independently testable)
- **`realdata` loader / validator** — parses both tiers into the row schema; enforces the provenance +
  inclusion/exclusion gate; emits an **inclusion/exclusion audit with reason counts**.
- **Gene-canonicalization + class→gene map** — strips allele suffix to IMGT gene; expands protein-class →
  nt-alleles → genes via the frozen `class_members`.
- **Frozen-calibration loader** — loads `q_mondrian`/`q_pooled` + params from `calibration.json` and
  reconstructs a `predict_set` **through a proper loader**, not by assigning `Conformal` internals ad hoc.
- **Predict-on-real runner** — reuses `_probs` (softmax over the V head); no new inference path.
- **Scorer** — per-tier OOV, top-class gene-set accuracy, gene-set coverage, class/allele coverage, set
  sizes; exact/Wilson intervals + cluster bootstrap; emits the **three-way decision**.
- **Report writer** — per-row predictions **and** aggregate JSON.

### Artifacts (`docs/superpowers/artifacts/2026-07-21-aa-phase3/`)
- `therasabdab_gold.tsv` / `therasabdab_imgt.tsv` — **or** `manifest.tsv` + `acquire.py` if licensing
  forbids redistribution.
- `predictions.jsonl` — **per-row** prediction output (not only aggregate).
- `phase3_eval.json` — aggregates + **machine-readable decision** field (`supported` | `failed` |
  `inconclusive`) + **dataset licensing/attribution metadata** + inclusion/exclusion audit reason counts +
  exact-VH & lineage cluster identifiers.
- `PHASE3.md` — real-domain numbers + interpretation, framed per §1.

### Tests
- Unit: loader/validator (schema + gate + audit counts), gene-canonicalization, class→gene expansion,
  scorer (synthetic rows with known answers, incl. OOV and multi-gene top-class cases), three-way decision
  boundaries.
- **Compatibility test:** the frozen-calibration loader's quantiles **reproduce the frozen Stage-2 sets**
  bit-for-bit on the Stage-2 test cohort.
- The real-data eval is a **script** producing the artifacts above (not a unit test).

### Explicitly NOT in Phase 3
D validation (undefined junctions), retraining, and **real-domain recalibration** (conformal needs hundreds;
tens of Gold rows cannot recalibrate — deferred to a later phase with sufficient real N). All work stays
isolated on `experiment/aminoacid-alignair`.

---

## Predeclarations to freeze before running
α = 0.05; exact one-sided binomial (Clopper–Pearson) decision rule; cluster-bootstrap resampling unit =
`lineage_id`; primary endpoint = gene-set conformal coverage on Gold (falsification-only); guarantee-relevant
endpoint = class/allele coverage on the allele-labelled subset; Gold feasibility floor = 15 (feasibility, not
evidence).

## Known feasibility risks
Gold N may be very small (documented humanization templates are scarce and often subgroup-level); IMGT/mAb-DB
programmatic access/parsing may be limited; Thera-SAbDab redistribution terms may force the manifest+script
path. Each risk has a defined fallback above; none is silently absorbed.
