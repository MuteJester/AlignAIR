# Amino-Acid AlignAIR — Phase 3: Real-Protein Validation on Curated Therapeutic Antibodies

**Date:** 2026-07-21 · **Branch:** `experiment/aminoacid-alignair` · **Status:** design approved; blinded
spike **done** → **Gold redefined to family resolution, exact-gene demoted to a concordance tier** (§1). Next:
family-Gold curation (§5.1), then implementation plan. **Does not modify the frozen research snapshot.**

Relationship to prior work: this evaluates the immutable Stage-1 snapshot
(`docs/superpowers/artifacts/2026-07-20-aa-strategic/SNAPSHOT.md`, model sha256 `54eef812…`,
`calibration.json`) on real therapeutic-antibody protein under distribution shift. No retraining, no
recalibration. The simulator-domain milestone stays frozen and tagged (`aa-igh-sim-milestone-2026-07-21`).

---

## §1 Purpose & framing — a falsification test of transfer

Phase 3 asks: does the simulator-frozen AA IGH model hold on **real therapeutic-antibody heavy-chain V
domains**, a genuine distribution shift (humanized/engineered/affinity-matured, not GenAIRR SHM)? The result
is reported as **"simulator-domain calibration evaluated under real-domain shift,"** never as a new
real-domain guarantee.

**The redefinition that makes Phase 3 defensible (post-spike, 2026-07-21).** The blinded acquisition spike
(`docs/superpowers/artifacts/2026-07-21-aa-phase3/feasibility_report.md`) established that *alignment-independent
per-sequence germline truth does not exist for real antibodies* — assigning a germline to an observed, mutated
sequence **is** the alignment task. The only alignment-independent label is the developer's **declared design
template**, which for therapeutic antibodies is almost always a **subgroup/family consensus** (e.g. "VH
subgroup III"), not a specific gene. Therefore:

- **Gold (ground-truth, headline)** is redefined to **subgroup/family resolution** — the model's V-family
  assignment vs the source-declared template family. Zero circularity; coarse.
- **Concordance (secondary, NOT ground truth)** is the exact-gene comparison vs IMGT/ANARCI closest-match,
  reported explicitly as **tool agreement under shift**, never as validation.

**Consequence stated up front — what Phase 3 can and cannot conclude.** Because no alignment-independent
allele/class truth exists on real data, Phase 3 **cannot `support` the original simulator-domain 95%
allele/class guarantee.** It can (a) *falsify* transfer at family granularity, (b) report a real-domain
**family-accuracy anchor**, and (c) report **concordance** with an established aligner. A positive outcome is
"family-level transfer not falsified + high concordance" — honestly weaker than the simulator-domain claim.

**Predeclared decision (α = 0.05, exact one-sided binomial / Clopper–Pearson, one representative VH per
lineage — see below).** Per endpoint: `fail` if the upper (1−α) bound for coverage < 0.95; `support` if the
lower (1−α) bound ≥ 0.95; else `inconclusive`.

**Independence requirement.** Related therapeutic lineages are clustered (§2), so the binding exact-binomial
test uses **exactly one deterministic representative VH per `lineage_id`** (frozen rule: lexicographically
smallest `exact_vh_id`). All deduplicated VHs are reported **descriptively** with a **lineage cluster
bootstrap** — never binding.

**Endpoints and what each can license:**
- **Gold family-set coverage** is *falsification-only* and *loose* — family coverage is the extreme of the
  §3 coarsening (only ~7 IGHV families), so `support` here is expected and near-meaningless; it may only
  usefully `fail`.
- **Gold top-class family accuracy** (a point accuracy, not a coverage guarantee) is the **primary
  informative ground-truth number** — reported with an exact-binomial CI on the one-per-lineage cohort.
- **Concordance (exact-gene) coverage/agreement** vs IMGT/ANARCI is descriptive only; it can neither
  `support` nor `fail` the ground-truth guarantee (it is not truth).

**Overall decision (machine-readable; concordance never changes it):**
- **`failed`** if Gold **family-set coverage** `fail`s (upper bound < 0.95) — a real transfer breakdown even
  at the coarsest granularity.
- **`inconclusive`** otherwise — the expected outcome given the ≈59-observation floor below; the *informative*
  content is then the family-accuracy anchor + concordance summary, not a guarantee verdict.
- **`supported` is not attainable** in Phase 3 and is never emitted (recorded rationale: no
  alignment-independent allele/class truth on real data).

**Statistical reality.** With zero errors, an exact one-sided 95% lower bound cannot reach 0.95 until ≈ **59
independent observations** (0.05^(1/n) ≥ 0.95 ⇒ n ≥ 59). A Gold cohort near the 15-lineage floor can falsify
a severe failure but can never certify coverage — reinforcing that the family-accuracy anchor, not a coverage
verdict, is the headline.

---

## §2 Data contract — two tiers, sequences and labels separated

Sequences and labels are sourced **independently**. Sequences come from Thera-SAbDab by INN/name; labels
come from the tier-specific provenance below. A row is admitted only if **sequence + label + citation** all
resolve.

### Tiers (redefined post-spike)
- **Tier 1 — Gold (ground truth, headline), FAMILY resolution.** Truth = the **source-declared human IGHV
  design-template subgroup/family** (e.g. "VH subgroup III" → IGHV3), from the originating paper/patent/INN,
  independent of AlignAIR and **not** a sequence alignment. One **citation per row**. This is the only
  alignment-independent label available for therapeutics; it is coarse (family) by necessity.
- **Tier 2 — Concordance (secondary, NOT ground truth), GENE/ALLELE resolution.** Exact-gene labels from
  **IMGT/mAb-DB** and/or **ANARCI closest-match**, always labeled *"alignment-derived → tool concordance, not
  validation."* Reported in a separate block; **never pooled with Gold** and **never changes the headline
  decision**.

### Label resolution (frozen taxonomy)
Every candidate label is resolved to exactly one level, recorded in `label_resolution`:
- `exact_allele` — source declares a specific IMGT allele (e.g. IGHV3-23*01).
- `exact_gene` — source declares a specific IMGT gene (e.g. IGHV3-23), no allele.
- `subgroup_only` — source declares only a subgroup/family (e.g. IGHV3).
- `ambiguous` — source lists several incompatible candidates.
- `unresolved` — no defensible source-declared germline.

**Endpoint admission (redefined post-spike).** The Gold ground-truth endpoint is **family-level**, so a label
qualifies for Gold if it resolves to a **subgroup/family** — i.e. `subgroup_only`, **or** `exact_gene`/
`exact_allele` coarsened to their family (any of these fixes the family). These enter the **Gold family
endpoints** (top-class family accuracy + family-set coverage). Separately, `exact_gene`/`exact_allele` labels
that are **alignment-derived** (IMGT/ANARCI) feed the **concordance tier only** — never the Gold ground-truth
decision. `ambiguous`/`unresolved`/family-unresolvable go to the acquisition audit, not the TSV.

### Alias crosswalk (frozen, cited, alignment-free)
Historical template names (clone IDs, framework/consensus identifiers, old gene names) are mapped to current
IMGT genes/**families** through a **frozen, cited crosswalk** — **never by aligning the therapeutic sequence**
(that would destroy independence). This includes the **Kabat/consensus-subgroup → IMGT-family** mapping
(e.g. "VH subgroup III" → IGHV3), which is an established, citable equivalence — **not** an alignment. The
crosswalk records, per alias, the target IMGT family (and gene/allele where the source is specific) plus the
authoritative citation. An alias with no citable mapping resolves to `unresolved`.

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
A **Gold floor of 15 independent lineages** with source-declared **family** templates is an
*acquisition-feasibility* gate — confirmed during the acquisition step (the first spike ruled out
*exact-gene* Gold; family-level Gold is expected to clear the floor since consensus-framework humanizations
are numerous and documented, but this is **verified, not assumed**, with the same stop-and-report gate). It is
**not** evidence that coverage transfers.

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

### Coarsening ladder (allele → gene → family)
Coverage gets **monotonically easier** as truth is coarsened. The conformal object was calibrated on protein
classes/alleles; mapping to genes is easier; mapping to **families** (Gold ground truth) is easiest of all.
So **family coverage < 0.95 disproves transfer** (strong), but **family coverage ≥ 0.95 establishes nothing**
about the allele/class guarantee. This is why the informative Gold metric is a **point accuracy**, not the
family-coverage guarantee. **"Top prediction" may not be a single gene/family** — a top-ranked protein class
can expand to alleles from multiple genes/families; always score against the **set** of families the top
class represents, **never an arbitrary single pick**.

### Reported metrics
**Gold tier (ground truth, family resolution) — headline:**
- **Top-class family-set accuracy** — the documented template family ∈ families(top-ranked class). *Primary
  informative number.* Exact-binomial CI on the one-representative-VH-per-lineage cohort.
- **Family-set conformal coverage** — reported, but *falsification-only and loose* (feeds the §1 `failed`
  test; can never emit `supported`).
- **OOV rate**, end-to-end **and** in-catalog family accuracy.

**Concordance tier (secondary, gene/allele, NOT ground truth) — clearly labelled:**
- **Top-class gene-set agreement** with IMGT/ANARCI exact-gene labels; **gene-set coverage** of the
  IMGT/ANARCI gene. Descriptive only — cannot support or fail the headline.

**Both tiers:** set size — mean/median/p95 — in **classes, genes, families, and nucleotide alleles**.
**Binding intervals:** exact one-sided binomial on the one-per-lineage cohort (the set §1 reads).
**Descriptive intervals:** all deduplicated VHs with a **cluster bootstrap by `lineage_id`** (never binding).

### Truncation stress test (separate, synthetic)
A **separately labelled synthetic stress test on real sequences**: deterministic crops via the existing
`aa_truncate`, **frozen seeds**, **paired analysis**, and the **exact original coverage-bin definitions**
(`full`/`trunc`/`short` from `calibration.json`). Reported apart from the real-domain primary result; probes
the `trunc`/`short` Mondrian groups under real partial coverage (truth gene unchanged).

### Central caveat (the most important correction)
> A small, **family-resolution** cohort of source-declared templates **can falsify severe transfer failure**
> and provide a real-domain **family-accuracy anchor**, but **cannot validate** the original 95% allele/class
> guarantee — no alignment-independent allele/class truth exists on real data. The exact-gene concordance
> tier shows agreement with an established aligner, not ground truth.

---

## §4 Components, artifacts, tests

### Components (small, single-purpose, independently testable)
- **`realdata` loader / validator** — parses both tiers into the row schema; enforces the provenance +
  inclusion/exclusion gate; emits an **inclusion/exclusion audit with reason counts**.
- **Gene/family-canonicalization + class→gene→family map** — strips allele suffix to IMGT gene and gene to
  IMGT **family**; expands protein-class → nt-alleles → genes → families via the frozen `class_members`.
- **Frozen-calibration loader** — loads `q_mondrian`/`q_pooled` + params from `calibration.json` and
  reconstructs a `predict_set` **through a proper loader**, not by assigning `Conformal` internals ad hoc.
- **AA-head inference utility** — a **tested library function** (`alignair.aa`), extracted so it is **not
  imported from an evaluation script**. `scripts/aa/calibrate_stage2.py::_probs` is refactored to call it too,
  so the calibration snapshot and Phase 3 share one verified inference path.
- **Predict-on-real runner** — calls the inference utility; no new inference path.
- **Scorer** — Gold: top-class **family**-set accuracy + family-set coverage + OOV + in-catalog accuracy;
  Concordance: gene-set agreement + gene-set coverage vs IMGT/ANARCI; set sizes; exact one-sided binomial
  (binding, one-per-lineage) + cluster bootstrap (descriptive); emits the **§1 decision** (`failed` or
  `inconclusive` only — never `supported`).
- **Report writer** — per-row predictions **and** aggregate JSON.

### Artifacts (`docs/superpowers/artifacts/2026-07-21-aa-phase3/`)
Spike (produced first, no predictions): `candidate_inventory.tsv` (every candidate + reason code),
`inclusion_rules.md`, `alias_crosswalk.tsv` (cited, alignment-free), `sources.json` (versions + SHA-256s +
licensing/attribution), `feasibility_report.md` (the GO/stop decision inputs).
Evaluation (only after a GO): `therasabdab_gold.tsv` / `therasabdab_imgt.tsv` — **or** `manifest.tsv` +
`acquire.py` if licensing forbids redistribution — plus `cohort.sha256` (frozen before inference);
`predictions.jsonl` (**per-row**); `phase3_eval.json` (aggregates + **machine-readable decision**
`failed`|`inconclusive` (never `supported`) + Gold family-accuracy anchor + concordance summary +
licensing/attribution + inclusion/exclusion reason counts + exact-VH & lineage identifiers + the
frozen-cohort hash it ran against); `PHASE3.md` (framed per §1).

### Tests
- Unit: loader/validator (schema + gate + audit counts), gene/family-canonicalization, class→gene→family
  expansion, scorer (synthetic rows with known answers, incl. OOV and multi-family top-class cases), the §1
  decision boundaries (family-coverage `fail`/`inconclusive`; assert `supported` is never emitted).
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
**Gold ground truth = family resolution**; primary informative metric = **top-class family-set accuracy**;
family-coverage endpoint = falsification-only (emits `failed`/`inconclusive`, never `supported`); exact-gene
comparison = **concordance tier only** (not ground truth, never changes the decision); Gold feasibility floor
= 15 **independent lineage representatives** (feasibility, not evidence — cf. the ≈59-observation reality in
§1). Cohort frozen and hashed before any inference.

## Sequencing (approved order of work)
**spec patch → blinded acquisition spike → feasibility decision → implementation plan.** No model inference
until after the feasibility decision.

## §5 Blinded acquisition spike — DONE (2026-07-21, no model inference)
Outcome recorded in `docs/superpowers/artifacts/2026-07-21-aa-phase3/{feasibility_report.md,sources.json}`.
Findings: (1) Thera-SAbDab publishes sequences (SPA-served; live endpoint captured at implementation) but
**no germline field** and **no explicit redistribution license** → manifest+script path; IMGT CC BY 4.0.
(2) **Strict exact-gene Gold is infeasible** — humanization templates are consensus/subgroup, and every
enumerable exact-gene germline label is alignment-derived. **Feasibility decision: redefine Gold to family
resolution + demote exact-gene to a concordance tier** (this spec, §1–§3).

### §5.1 Next acquisition step (family-Gold curation, still inference-free)
Before building the evaluation, curate the **family-resolution Gold** cohort and confirm the floor:
- how many **independent lineages** have a **source-declared subgroup/family** template (→ Gold), each with an
  evidence locator + a cited Kabat-subgroup→IMGT-family alias mapping;
- the concordance-tier inventory (IMGT/ANARCI exact-gene labels);
- the acquisition audit (out-of-scope, unresolved, ambiguous, bispecific, duplicates).
Deliverables (frozen, hashed, no predictions): `candidate_inventory.tsv`, `inclusion_rules.md`,
`alias_crosswalk.tsv`, updated `sources.json`. **Stop-and-report** if family-Gold cannot clear 15 independent
lineages with citable templates.

## Known feasibility risks
Gold N may be very small (documented humanization templates are scarce and often subgroup-level); IMGT/mAb-DB
programmatic access/parsing may be limited; Thera-SAbDab redistribution terms may force the manifest+script
path. Each risk has a defined fallback above; none is silently absorbed.
