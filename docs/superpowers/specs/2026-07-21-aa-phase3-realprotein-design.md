# Amino-Acid AlignAIR — Phase 3: Real-Protein Validation on Curated Therapeutic Antibodies

**Date:** 2026-07-21 · **Branch:** `experiment/aminoacid-alignair` · **Status:** design approved; next step is
the **blinded acquisition spike** (§5), then a feasibility decision, then the implementation plan. **Does not
modify the frozen research snapshot.**

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

**Predeclared decision.** Fix α = 0.05 and use **exact one-sided binomial (Clopper–Pearson) bounds**, decided
before any results are seen. Each endpoint below is classified per-endpoint as:

- **`fail`:** the upper one-sided (1−α) bound for coverage is **< 0.95** (rejects H₀: coverage ≥ 0.95).
- **`support`:** the lower one-sided (1−α) bound for coverage is **≥ 0.95**.
- **`inconclusive`:** the interval straddles 0.95.

**Independence requirement.** Clopper–Pearson assumes independent Bernoulli observations, but related
therapeutic lineages are explicitly clustered (§2). Therefore the **binding exact-binomial decision uses
exactly one deterministic representative VH per `lineage_id`** (chosen by a frozen rule — lexicographically
smallest `exact_vh_id` within the lineage). All deduplicated VHs are additionally reported **descriptively**
with the **lineage cluster bootstrap** — descriptive only, never the binding test.

**Endpoints and which conclusion each can license** (binds §1 to the coarsening argument in §3):
- **Gold gene-set coverage** is *falsification-only* — it may `fail` or be `inconclusive` but can **never**
  `support` (coarsening makes gene coverage strictly easier than the class/allele guarantee).
- **In-catalog Gold class/allele coverage** on the allele-labelled subset is the only endpoint that can
  `support` the original guarantee.

**Overall decision truth table** (headline; Tier 2 never changes it):
- **`failed`** if the Gold **gene** endpoint `fail`s **OR** the in-catalog Gold **allele/class** endpoint
  `fail`s.
- **`supported`** only if the Gold **allele/class** endpoint `support`s **AND** the Gold **gene** endpoint
  has **not** failed.
- **`inconclusive`** otherwise.

Rationale (not a bare threshold): an observed proportion below 0.95 is **not** automatically failure because
finite samples fluctuate; a small cohort scoring 95–100% does **not** prove transfer. Every outcome is an
honest result — `failed`/`inconclusive` quantifies the shift and motivates a later recalibration phase.

**Statistical reality to state up front.** With **zero** errors, an exact one-sided 95% lower bound cannot
reach 0.95 until ≈ **59 independent observations** (0.05^(1/n) ≥ 0.95 ⇒ n ≥ 59). So a Gold cohort near the
feasibility floor of 15 lineages can **falsify a severe failure** but can **never** produce `supported`,
regardless of the observed proportion.

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

### Label resolution (frozen taxonomy)
Every candidate label is resolved to exactly one level, recorded in `label_resolution`:
- `exact_allele` — source declares a specific IMGT allele (e.g. IGHV3-23*01).
- `exact_gene` — source declares a specific IMGT gene (e.g. IGHV3-23), no allele.
- `subgroup_only` — source declares only a subgroup/family (e.g. IGHV3).
- `ambiguous` — source lists several incompatible candidates.
- `unresolved` — no defensible source-declared germline.

**Endpoint admission:** only **`exact_gene`** (and `exact_allele`, which implies its gene) enter the **Gold
gene endpoint**; only **in-catalog `exact_allele`** labels enter the **guarantee-relevant class/allele
endpoint**. `subgroup_only` labels may receive a **separate exploratory hierarchical score** but are
**never promoted to Gold gene truth**. `ambiguous`/`unresolved` go to the acquisition audit, not the TSV.

### Alias crosswalk (frozen, cited, alignment-free)
Historical template names (clone IDs, framework/consensus identifiers, old gene names) are mapped to current
IMGT genes through a **frozen, cited crosswalk** — **never by aligning the therapeutic sequence** (that would
destroy the label's independence). The crosswalk file records, per alias, the target IMGT gene/allele and the
authoritative citation for the mapping. An alias with no citable mapping resolves to `unresolved`.

### Candidate inventory vs evaluation rows
Two distinct artifacts. The **candidate inventory** (acquisition audit) holds every candidate — including
`unresolved`, `ambiguous`, out-of-scope, bispecific-unassigned, and duplicates — with a reason code. The
**admitted TSV rows** are only those that satisfy the **full sequence + label + citation** contract. A
candidate is never silently dropped; it is moved to the audit with its reason.

### Cohort freeze before inference (no outcome-informed inclusion)
Acquisition, label resolution, alias mapping, inclusion/exclusion decisions, deduplication, and lineage
assignment are **completed and hashed before any model inference is run.** The frozen cohort's SHA-256 is
recorded; inference asserts the cohort hash matches. This structurally prevents outcome-informed inclusion.

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
`name/INN`, `arm_id`, `vh_aa`, `exact_vh_id`, `lineage_id`, `lineage_representative` (bool; the frozen
one-per-lineage pick used for the binding test), `chain_class` (in-scope vs which out-of-scope category),
`seq_source`, `seq_source_version`, `retrieval_date`, `raw_file_sha256`, `true_v_gene`, `true_v_allele?`,
`label_resolution` (exact_allele|exact_gene|subgroup_only|ambiguous|unresolved), `in_catalog` (present in
frozen model catalogue?), `tier` (gold|imgt), `label_source`, `label_citation`, `alias_source` (crosswalk
citation, if the name was aliased), `evidence_locator` (page/table/patent-example), `humanization_status`,
`true_j?`, `exclusion_reason?`.

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
- **Frozen-cohort SHA-256** matches the hash recorded at cohort freeze (§2) — no inference on an unfrozen or
  edited cohort.
- Use **`q_mondrian["full"]`**, the frozen full-group quantile — **not** the pooled quantile.

**Sequence-completeness selection must be model-independent** (avoid selection through model-adjacent
checks). Admit `full` sequences by **Thera-SAbDab's domain designation** plus **alphabet / length /
source-boundary** validation (valid AA alphabet, length within the VH-domain range, complete per the source's
own domain boundaries) — **not** by any model-derived completeness score. If motif anchors (e.g. a Cys/W
frame) are needed, the predicate is **frozen before inference** and every sequence it excludes is reported in
a **sensitivity audit**, so the selection can be inspected for bias.

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
- **Gene-set conformal coverage** (falsification-only endpoint per adjustment 1; feeds the §1 truth table).
- **Class/allele coverage** on the allele-labelled subset (the only endpoint that can *support* the
  original guarantee).
- **Set size** — mean / median / p95 — in **classes, genes, and nucleotide alleles**.
- **Binding intervals:** exact one-sided binomial (Clopper–Pearson) on the **one-representative-VH-per-lineage**
  cohort — this is the set the truth table in §1 reads. **Descriptive intervals:** all deduplicated VHs with a
  **cluster bootstrap resampling by `lineage_id`** (reported alongside, never binding).

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
- **AA-head inference utility** — a **tested library function** (`alignair.aa`), extracted so it is **not
  imported from an evaluation script**. `scripts/aa/calibrate_stage2.py::_probs` is refactored to call it too,
  so the calibration snapshot and Phase 3 share one verified inference path.
- **Predict-on-real runner** — calls the inference utility; no new inference path.
- **Scorer** — per-tier OOV, top-class gene-set accuracy, gene-set coverage, class/allele coverage, set
  sizes; exact one-sided binomial (binding, one-per-lineage) + cluster bootstrap (descriptive); emits the
  **overall decision truth table** of §1.
- **Report writer** — per-row predictions **and** aggregate JSON.

### Artifacts (`docs/superpowers/artifacts/2026-07-21-aa-phase3/`)
Spike (produced first, no predictions): `candidate_inventory.tsv` (every candidate + reason code),
`inclusion_rules.md`, `alias_crosswalk.tsv` (cited, alignment-free), `sources.json` (versions + SHA-256s +
licensing/attribution), `feasibility_report.md` (the GO/stop decision inputs).
Evaluation (only after a GO): `therasabdab_gold.tsv` / `therasabdab_imgt.tsv` — **or** `manifest.tsv` +
`acquire.py` if licensing forbids redistribution — plus `cohort.sha256` (frozen before inference);
`predictions.jsonl` (**per-row**); `phase3_eval.json` (aggregates + **machine-readable decision**
`supported`|`failed`|`inconclusive` + licensing/attribution + inclusion/exclusion reason counts + exact-VH &
lineage identifiers + the frozen-cohort hash it ran against); `PHASE3.md` (framed per §1).

### Tests
- Unit: loader/validator (schema + gate + audit counts), gene-canonicalization, class→gene expansion,
  scorer (synthetic rows with known answers, incl. OOV and multi-gene top-class cases), the §1 decision
  truth-table boundaries (per-endpoint fail/support/inconclusive and their conjunction).
- **Compatibility test:** uses **small frozen probability/group fixtures with expected prediction sets**
  (checked into the repo) to prove the loader reproduces the Stage-2 conformal sets. It does **not** regenerate
  the whole simulator cohort — that is not guaranteed bit-identical across library/GPU versions.
- The real-data eval is a **script** producing the artifacts above (not a unit test).

### Explicitly NOT in Phase 3
D validation (undefined junctions), retraining, and **real-domain recalibration** (conformal needs hundreds;
tens of Gold rows cannot recalibrate — deferred to a later phase with sufficient real N). All work stays
isolated on `experiment/aminoacid-alignair`.

---

## Predeclarations to freeze before running
α = 0.05; exact one-sided binomial (Clopper–Pearson) decision rule; **binding statistical unit = one frozen
representative VH per `lineage_id`**; cluster-bootstrap resampling unit = `lineage_id` (descriptive only);
gene endpoint = falsification-only; guarantee-relevant endpoint = in-catalog `exact_allele` class/allele
coverage; overall decision = the §1 truth table; Gold feasibility floor = 15 **independent lineage
representatives** (feasibility, not evidence — cf. the ≈59-observation reality in §1). Cohort frozen and
hashed before any inference.

## Sequencing (approved order of work)
**spec patch → blinded acquisition spike → feasibility decision → implementation plan.** No model inference
until after the feasibility decision.

## §5 Blinded acquisition spike (runs first, no model inference)
A standalone, inference-free spike whose only job is to decide whether the headline endpoint is achievable.
It **must not** load the model or produce predictions. It answers:

- Can Thera-SAbDab sequences be acquired **reproducibly**, and may they be **redistributed** (or is the
  manifest+script path required)?
- How many **unique lineages** have **source-declared `exact_gene`** human IGHV labels? How many `exact_allele`?
- How many are `subgroup_only`, `unresolved`, `ambiguous`, OOV, bispecific, duplicated, or out-of-scope?
- Can **every admitted label** be backed by an **evidence locator** + **authoritative alias mapping**?
- Does Gold plausibly clear **15 independent lineage representatives**?

**Deliverables (frozen, hashed, no predictions):** a candidate manifest, inclusion rules, the alias
crosswalk, source versions/hashes, and a **feasibility report**. Licensing recorded per §2 — Thera-SAbDab
download availability is not redistribution permission
(https://academic.oup.com/nar/article/48/D1/D383/5573951); IMGT data are CC BY 4.0, record the terms date +
attribution (https://www.imgt.org/about/termsofuse.php).

**Feasibility decision after the spike:** GO to the implementation plan only if Gold clears the floor with
citable, alias-verified, `exact_gene`+ labels and licensing is settled; otherwise stop and report options
(relax scope, alternative source, or accept an exploratory-only run).

## Known feasibility risks
Gold N may be very small (documented humanization templates are scarce and often subgroup-level); IMGT/mAb-DB
programmatic access/parsing may be limited; Thera-SAbDab redistribution terms may force the manifest+script
path. Each risk has a defined fallback above; none is silently absorbed.
