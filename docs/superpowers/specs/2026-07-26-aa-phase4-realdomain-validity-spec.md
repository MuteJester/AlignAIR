# Phase 4 — Real-Domain Result Validity & Calibration (paired DNA→AA benchmark)

> **Status:** ✅ **APPROVED (2026-07-26)** — amended per review · **Model in scope:** frozen
> `alignair-aa-igh-human 2.0.0` (v2 = `cont_step318000`) + its frozen simulator calibration.
>
> Gate 3 tested whether an **input** can be trusted. Phase 4 tests whether, on **manually-admitted complete
> human IGH amino-acid sequences**, the emitted **V/D/J results and calibrated sets** can be trusted. This
> is **not** training, **not** packaging, and **not** the applicability gate (the separate, later
> trained-applicability-head project). Phase 4 is output validity + calibration only.

## §0 Standing decisions (owner, 2026-07-26)
1. **Gold = GenAIRR simulator, tightly scoped (§9).** Frozen GenAIRR held-out simulation, V/D/J known **by
   construction**. **It is the training-distribution simulator: weak independence.** Its evidentiary role is
   limited to (a) procedure/regression correctness, (b) calibrated-set + class→allele expansion mechanics,
   (c) **D** construction-truth. Gold V/J accuracy may be reported for completeness but **only** as a
   simulator-domain regression result; it **never** strengthens a real-domain release claim, and the tiers
   are **never pooled**.
2. **Real concordance is a SECONDARY CONSENSUS LABEL, not truth (§1).** IgBLASTn + nucleotide-AlignAIR
   agreement carries **incorporation bias** — nucleotide AlignAIR shares reference and architecture ancestry
   with the AA model — and conditioning on agreement selects easier cases. Real-tier metrics are named as
   **concordance**, never "accuracy"/"true-class coverage".
3. **Three frozen real partitions (§6), no reuse:** **trigger/selection**, **calibration**, **untouched
   test** (opened exactly once).
4. **Recalibration trigger:** recalibrate iff **V OR J** calibrated-set coverage 95% lower bound < **0.93**
   (target 0.95 − 2pt), evaluated on the trigger/selection partition, for the **default** product behavior.
5. **Two evaluation endpoints (§6):** a source-balanced, donor/lineage-disjoint test = the **primary
   conformal (exchangeable) audit**; a source-held-out test = **transfer stress test, descriptive only, no
   formal conformal guarantee.**
6. **Discordant real records = a separate lower tier** (descriptive only, never pooled).
7. **v2 (model + simulator calibration) frozen and sha256-pinned BEFORE acquisition.**

## §1 Two evidence tiers (never pooled)

**Tier G — Gold (construction-known, simulator-domain).** Frozen GenAIRR `HUMAN_IGH_OGRDB` held-out
simulation with exact V/D/J alleles, boundary coordinates, and controllable SHM/indel/truncation. Role:
the three §9 purposes only. Labelled simulator-domain / weak independence everywhere.

**Tier R — Real (nucleotide→AA), truth = a secondary consensus, not biology.** Productive, complete human
IGH **nucleotide** rearrangements from multiple donors and studies (OAS + AIRR Data Commons), translated to
amino acid; the DNA is retained for the nucleotide references. AIRR fields (`sequence_aa`, `complete_vdj`,
`locus`, gene calls, alignment coordinates) are **annotations, not experimental truth.** The consensus
label is **agreement between IgBLASTn and nucleotide AlignAIR** at the required resolution — an imperfect,
incorporation-biased reference (§0.2), optionally constrained by donor genotypes and versioned AIRR-C/OGRDB
germline sets. Strata:
- **R-concordant:** the two nt methods agree → the real-domain **concordance** cohort. **The concordance
  rate is itself reported over ALL otherwise-eligible sequences** (conditioning on agreement selects easier
  cases and must be visible).
- **R-discordant:** they disagree → a **separate lower tier**, descriptive only, never pooled.

## §2 What must be validated — per gene head (V, J; D per §4)

All confidence-bounded, per tier, never pooled. **Binary endpoints (coverage, top-call correctness) use
one-sided exact Clopper–Pearson; all aggregate/continuous endpoints (macro accuracy, NLL, Brier, MAE,
percentiles, risk–coverage curves) use lineage/donor-cluster BOOTSTRAP intervals** (the resampling unit is
the independent lineage/donor cluster, not the sequence).

- **Top-call correctness** at three resolutions — protein-equivalence **class**, **gene**, **allele**:
  - Gold: `top_class/gene/allele_accuracy` (construction truth).
  - Real: `top_call_concordance` (agrees with the nt consensus). Not "accuracy".
  - **Gene/allele success = truth/reference INCLUSION in the TOP CLASS's one-to-many ASC→IMGT expansion**
    (a top class may expand to several genes/alleles). **Never select an arbitrary representative.**
- **Calibrated class-set coverage:**
  - Gold: `class_set_coverage` (true class ∈ set).
  - Real: `reference_call_inclusion` (nt-consensus class ∈ set). Not "true-class coverage".
- **Class→nucleotide-allele expansion coverage** (truth/reference allele ∈ the covered classes' expansion).
- **Set sizes:** mean, median, p95 (classes / alleles / genes) — bootstrap CIs.
- **`softmax_score` reliability — at CLASS RESOLUTION only, on the COMPLETE class softmax vector:** NLL,
  Brier, reliability (calibration) plot. Computed on the full per-class softmax, **not** an expanded
  gene/allele call. Reported as an uncalibrated model score; **never renamed "probability".**
- **Selective risk:** error/concordance-loss vs coverage as wide/uncertain sets are progressively withheld
  (risk–coverage curve, bootstrap band).
- **Macro accuracy** (per-family averaged, lineage-cluster bootstrap) so common IGHV families cannot conceal
  rare-family failure.

## §3 Boundaries (coordinates)

- **V/J start and end error in amino-acid residues** — `boundary_concordance` on Real (vs the nt-consensus
  boundaries projected to AA), `boundary_accuracy` on Gold (construction truth). Reported as MAE + p95 with
  bootstrap CIs, stratified by truncation, SHM level, indels, and V family.
- **Explicit nt→AA coordinate projection (deterministic).** AIRR nucleotide coordinates are **1-based
  closed**; AA AlignAIR emits **0-based half-open** AA offsets. The projection must be defined and
  implemented deterministically, handling the **reading-frame offset** and **non-codon-aligned boundaries**
  (a boundary mid-codon maps to a defined AA index by a stated rule). This projection is part of the
  benchmark, spec'd and tested before any boundary metric is computed.
- **`inter_vj_aa` is validated ONLY** as the model-defined `sequence[V_end:J_start]` substring and its
  semantics — **never** as an AIRR junction or CDR3. No AIRR-junction claim is made or tested.

## §4 D — construction-truth only, stays experimental

- **Allele and gene recovery only where construction truth exists → Tier G (Gold) only.** Tier R has no
  reliable D allele/gene truth (short segment, junctional erosion, nt methods disagree) → D allele/gene is
  **not** validated on real data.
- **Evidence-conditioned performance: observable D residues are computed from CONSTRUCTION truth**, never
  from predicted D boundaries. Stratify by observable-residue count.
- **Calibrated D-set coverage and width** (Gold, exact truth).
- **D keeps EXPERIMENTAL status.** Gold is the simulator → Gold D evidence is simulator-domain and cannot
  promote D to real-supported. Promotion requires real construction-truth D evidence, which this phase lacks.

## §5 Leakage controls + independence design

**Independence design (owner, 2026-07-26) — roles are distinct:**
- **Prediction unit:** one **clonal-lineage representative**.
- **Dependence cluster:** the **donor** (sequences within a donor share genotype + V-usage; not independent).
- **Split barrier:** **donor AND source/study** (no donor or source crosses partitions).
- **Binding exact analysis (the coverage non-inferiority endpoint):** **one deterministically sampled
  lineage per donor** — so the exact independent N is a **donor count**, and the SAME donor-lineage units
  validate BOTH V and J (not 882 per head).
- **Real-domain conformal calibration:** the guarantee-producing quantile is fit on **one lineage per
  calibration donor**; additional lineages enter only the secondary cluster-aware analysis.
- **Secondary precision analysis:** additional lineages per donor, reported with **donor-cluster bootstrap**
  intervals; these NEVER inflate the binding independent N.

**Leakage controls:**
- **Freeze v2** (weights + simulator `calibration.json`, both sha256-pinned) **before any acquisition.**
- **Split Real by donor + source/study** (barriers above); **germline family** balanced where practical;
  **near-identical-VH dedup** at a predeclared identity threshold; **duplicate sequences audited across
  repositories** (OAS↔ADC) and donor identity **harmonized** (source accession + subject id).
- No clone, donor, therapeutic lineage, or near-identical VH crosses any two partitions.
- Partition membership committed as **hashes/counts/fingerprints only** (raw sequences git-ignored).

## §6 Calibration data flow (three partitions, no reuse) + the two endpoints

**Partitions** (frozen, leakage-disjoint per §5):
- **Trigger/selection** — evaluate the **unchanged** frozen simulator calibration; decide recalibration by
  §0.4 (V OR J coverage 95% LB < 0.93). No other decision touches this partition.
- **Calibration** — if triggered, fit new real-domain conformal quantiles here (and here only).
- **Test (untouched)** — **opened exactly once**, at the end, to evaluate **both** the simulator calibration
  and (if fitted) the real-domain calibration **side by side.** The model is never tuned on it.

**Product behavior tested (the claim must follow what users get by default):**
- **Default `coverage_group=None` (pooled marginal)** — the primary release claim.
- **Explicit `full` group** — reported separately (does not substitute for the default).

**Two endpoints (exchangeability vs transfer):**
- **(A) Source-balanced, donor/lineage-disjoint test — PRIMARY conformal coverage audit.** Calibration and
  test are drawn to be exchangeable (source-balanced); this is where the formal 1−α coverage contract is
  assessed.
- **(B) Source-held-out test — transfer stress test, DESCRIPTIVE ONLY.** A deliberately unseen-source cohort
  is distribution-shifted by design; conformal exchangeability does not hold, so **no formal conformal
  guarantee** is claimed — coverage is reported as transfer evidence only.

## §7 Comparative evidence (does native AA AlignAIR add utility?) — without circularity

The nt reference generators must **not** be scored against their own selected truth:
- **Genuine accuracy comparisons → Gold construction truth only.** AA AlignAIR v2 vs a simple protein
  local alignment against the identical ASC→IMGT protein reference (e.g. parasail Smith–Waterman) vs IgBLAST
  protein mode, all scored on Gold's exact truth.
- **Method-to-method agreement/disagreement → UNFILTERED eligible real records** (before concordance
  filtering), so the reference generators (IgBLASTn, nt AlignAIR) and AA AlignAIR are compared on the same
  unselected population — not on records their agreement admitted.
- **AA-vs-consensus concordance → R-concordant only.** Reported as concordance (§2 naming), acknowledging the
  consensus is what defined that cohort.

## §8 Release decision — field-level evidence matrix

| Output | Required evidence |
|---|---|
| **V/J top calls** | Gold accuracy (simulator-domain regression) **+ real-cohort top-call CONCORDANCE** (per-resolution, confidence-bounded) |
| **Calibrated V/J sets** | Real-domain **reference-call-inclusion coverage** on the primary exchangeable test (default pooled group) + set-size audit + expansion-inclusion; source-held-out transfer reported descriptively |
| **D calls** | Gold construction-truth only → otherwise **experimental** (stays experimental this phase) |
| **Coordinates** | Paired DNA→AA **boundary concordance** (AA-residue error, stratified) + Gold boundary accuracy |
| **`softmax_score`** | Class-resolution reliability (NLL/Brier/reliability plot on the full softmax vector); **never renamed probability** |
| **`inter_vj_aa`** | Semantic correctness only (V-end→J-start substring); **no AIRR-junction claim** |

## §9 Gold's tightened evidentiary role
Gold serves exactly three purposes: **(1) procedure/regression correctness; (2) calibrated-set + class→
allele expansion mechanics; (3) D construction-truth evaluation.** Gold V/J accuracy may be reported for
completeness, labelled a **simulator-domain regression result**; it must not strengthen the real-domain
release claim, and Gold and Real are never pooled.

## §10 Statistical framing
No study proves an individual prediction correct. Phase 4 proves that, for a **precisely defined
population**, the **set-producing procedure** meets a **predeclared coverage/error contract with quantified
uncertainty**. Binary endpoints: one-sided exact Clopper–Pearson. Aggregate/continuous endpoints:
lineage/donor-cluster bootstrap. Real-tier language is always **concordance/inclusion**, never accuracy.

## §11 Feasibility-first (Phase-4 Phase 0 — a HARD gate)

Gold is generatable at will. The binding question is the Real cohort. **Do NOT use the "≈59 for a ≤5%
zero-failure bound" shortcut** — that bounds an error rate, not a coverage non-inferiority test. Compute
**exact prospective power** for `H0: coverage ≤ 0.93` vs expected ≈ 0.95 (one-sided, α=0.05, power 0.80):
**≈882 donor-lineage units** (fewer if expected coverage is higher: 544 at 0.955, 360 at 0.96).

**Per-partition sizing (NOT 4×882):**
- **Exchangeable locked test (A):** the ~**882 donor-lineage** units are **binding here** — and the SAME
  units validate both V and J (one deterministically sampled lineage per test donor). This is the only
  partition that must satisfy the 882 gate.
- **Trigger/selection:** small; it only decides recalibration. If too small to establish LB ≥ 0.93, the
  conservative outcome is simply "recalibrate" — it can never wrongly APPROVE the simulator calibration.
- **Calibration:** sized for **conformal-quantile stability/efficiency** (one lineage per calibration
  donor), from a **predeclared quantile-stability simulation** — not the coverage hypothesis test.
- **Source-held-out stress (B):** descriptive transfer only; sized for a usable estimate, not the 882 gate.

**Donor-first recon outputs (STOP if any fails; never substitute the shortcut):**
- **source/study/donor census** with OAS↔ADC donor-identity **harmonization** (source accession + subject id)
  and a **cross-repository duplicate-sequence audit**;
- **eligible records before concordance filtering** (productive, complete_vdj, human IGH; denominator);
- **concordance and discordance rates reported separately at V/J GENE and ALLELE resolution** (and how much
  conditioning shrinks/biases the cohort);
- **donors retaining ≥1 concordant lineage** (the real independent-N supply);
- a **donor-DISJOINT allocation** to trigger / calibration / ~882-donor locked test / source-held-out that
  fits the available donor pool, with additional-lineage counts reported **separately** (never inflating N).

## §12 Non-goals
- Not training; not packaging; not the applicability gate.
- Not IMGT numbering/CDR delimitation; `inter_vj_aa` is never an AIRR junction/CDR3.
- `softmax_score` is never renamed/reinterpreted as a calibrated probability.
- Gold (simulator) evidence is never presented as real-domain evidence; tiers never pooled; nt-consensus
  concordance is never called biological accuracy.

## §13 Sequencing
1. This spec (approved).
2. **Writing-plans**, beginning with §11 feasibility (Real recon → eligible/concordance counts → exact power
   → leakage-controlled trigger/calibration/test split with sources for (A) and (B)) BEFORE any evaluation;
   STOP there if underpowered. Then: freeze/hash v2 + Gold generation → the §2–§4 evaluation on both tiers →
   §6 calibration decision + two endpoints → §7 comparative (Gold + unfiltered + concordant) → §8 evidence
   matrix + frozen Phase-4 artifact.
3. After Phase 4, the separate trained-applicability-head project remains the final prerequisite for a
   trustworthy public product.
