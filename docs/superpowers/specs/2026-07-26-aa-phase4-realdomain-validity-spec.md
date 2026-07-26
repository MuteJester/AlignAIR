# Phase 4 — Real-Domain Result Validity & Calibration (paired DNA→AA benchmark)

> **Status:** DRAFT for review · **Date:** 2026-07-26 · **Model in scope:** frozen `alignair-aa-igh-human
> 2.0.0` (v2 = `cont_step318000`) + its frozen simulator calibration.
>
> Gate 3 tested whether an **input** can be trusted. Phase 4 tests whether, on **manually-admitted complete
> human IGH amino-acid sequences**, the emitted **V/D/J results and calibrated sets** can be trusted. This
> is **not** training, **not** packaging, and **not** the applicability gate (that remains the separate,
> later trained-applicability-head project). Phase 4 is output validity + calibration only.

## §0 Standing decisions (owner, 2026-07-26)
1. **Gold = GenAIRR simulator, documented.** The Gold known-truth tier is a frozen GenAIRR held-out
   simulation whose V/D/J alleles are known **by construction**. It validates the result-producing
   *procedure* against **exact** truth — especially D (allele/gene) and allele-resolution, where real truth
   is unobtainable — and the calibrated-set mechanics. **It is the training-distribution simulator: weak
   independence.** Gold provides simulator-domain procedure evidence, NOT real-domain generalization; the
   two tiers are **never pooled**.
2. **Recalibration trigger (predeclared):** recalibrate iff the real-domain V calibrated-set coverage
   one-sided **95% lower bound < 0.93** (the 0.95 target minus a 2-pt tolerance).
3. **Discordant real records = a separate lower tier** (descriptive only, never pooled with the concordant
   real cohort).
4. **v2 (model + simulator calibration) is frozen and hash-pinned BEFORE acquisition.**
5. **Simulator calibration is evaluated on real data UNCHANGED first**, before any real-domain calibration.

## §1 Two evidence tiers (never pooled)

**Tier G — Gold (construction-known, simulator-domain).** Frozen GenAIRR `HUMAN_IGH_OGRDB` held-out
simulation with exact V/D/J alleles, junction/boundary coordinates, and controllable SHM/indel/truncation.
Role: exact-truth validation of (a) the result-producing procedure and calibrated-set behavior, and (b) D
allele/gene recovery where real truth is impossible. **Explicitly labelled simulator-domain / weak
independence** everywhere it appears — a Gold result never counts as real-domain evidence.

**Tier R — Real concordance (secondary-reference truth).** Productive, complete human IGH **nucleotide**
rearrangements from multiple donors and studies (OAS + AIRR Data Commons), translated to amino acid. The
DNA is retained as the source for nucleotide concordance references; the AA is what AA AlignAIR consumes.
Truth is a **secondary reference**, established by **agreement between independent nucleotide methods**
(IgBLASTn + nucleotide AlignAIR) at the required call resolution, and — where possible — constrained by
donor genotypes and versioned AIRR-C/OGRDB germline sets. Strata:
- **R-concordant:** independent nt methods agree → the real-domain evaluation cohort.
- **R-discordant:** they disagree → a **separate lower tier**, reported descriptively, never pooled.

## §2 What must be validated — per gene head (V, J; D per §4)

On both tiers as applicable (Gold for exact truth; Real-concordant for real-domain), all confidence-bounded
(one-sided exact Clopper–Pearson), reported per tier and never pooled:

- **Top-class accuracy** at three resolutions: protein-equivalence **class**, **gene**, **allele**.
- **Calibrated class-set coverage** (does the true class fall in the emitted set at ≥ 1−α?).
- **Class→nucleotide-allele expansion coverage** (does the true allele fall in the one-to-many ASC→IMGT
  expansion of the covered class?).
- **Set sizes:** mean, median, p95 (classes / alleles / genes).
- **`softmax_score` reliability:** NLL, Brier, and reliability (calibration) plots — reported as an
  uncalibrated model score, **never renamed "probability".**
- **Selective risk:** error rate as uncertain / wide-set results are progressively withheld (risk–coverage
  curve).
- **Macro accuracy** (per-family averaged) so common IGHV families cannot conceal rare-family failure.

## §3 Boundaries (coordinates)

- **V/J start and end error in amino-acid residues**, using the paired DNA→AA reference: the nt-annotated
  V/J boundaries (from the concordant nt methods) mapped to AA offsets. Reported as MAE + p95.
- **Stratified** by truncation, SHM level, indels, and V family.
- **`inter_vj_aa` is validated ONLY as the model-defined V-end→J-start substring** — that it equals
  `sequence[V_end:J_start]` and its semantics — **never** as an AIRR junction or CDR3. No AIRR-junction
  claim is made or tested.

## §4 D — construction-truth only, stays experimental

- **Allele and gene recovery only where construction truth exists → Tier G (Gold) only.** On Tier R there
  is no reliable D allele/gene truth (D is short, junctionally eroded, and nt methods disagree), so D
  allele/gene is **not** validated on real data.
- **Evidence-conditioned performance:** stratify by the number of observable D residues.
- **Calibrated D-set coverage and width** (on Gold, where truth is exact).
- **D keeps EXPERIMENTAL status.** Because Gold is the simulator, Gold D evidence is *simulator-domain*; it
  can document simulator D behaviour but **cannot promote D to real-supported.** Promotion to non-
  experimental requires real construction-truth D evidence, which this phase does not have.

## §5 Leakage controls

- **Freeze v2** (model weights + simulator `calibration.json`, both sha256-pinned) **before any acquisition.**
- **Split Real by:** donor · study · clonal lineage · germline family (where practical) · **data source.**
- **No clone, donor, therapeutic lineage, or near-identical VH may cross the calibration and test
  partitions.** Near-identical VH deduplication (identity threshold predeclared).
- **Source-held-out evaluation:** use OAS donor/study metadata so test sources are unseen in any
  calibration/fit partition.
- All partition membership is committed as **hashes/counts/fingerprints only** (raw sequences git-ignored,
  per the established data policy).

## §6 Calibration decision flow

1. **Evaluate the frozen simulator calibration UNCHANGED on Tier R-concordant.** Report V/J coverage
   (with 95% LB), set sizes, class→allele expansion coverage.
2. **If real V coverage 95% LB < 0.93** (§0.2): the simulator calibration undercovers → build a **new
   real-domain calibration** on a **separate real calibration partition** (disjoint from test by every
   §5 leakage key).
3. **Evaluate the new real-domain calibration ONCE on the untouched real test partition.** The model is
   never tuned on the test partition.
4. **Report both** results side by side: simulator-calibration-on-real, and (if triggered) real-
   calibration-on-real-test. The simulator calibration and its scope string remain the product default
   unless a validated real-domain calibration is explicitly promoted in a later phase.

## §7 Comparative evidence (does native AA AlignAIR add utility?)

Run the **same** sequences through, and compare on the metrics above:
- **AA AlignAIR v2** (native).
- **Simple protein local alignment** against the **identical** ASC→IMGT protein reference (e.g. parasail
  Smith–Waterman) — the "translate then align conventionally" baseline.
- **IgBLAST protein mode** where applicable.
- **Nucleotide AlignAIR** and **IgBLASTn** on the source DNA (the concordance references / truth-establishers).

The question this answers: does native AA AlignAIR add real utility (accuracy, calibrated uncertainty,
robustness) **beyond translation followed by conventional alignment**?

## §8 Release decision — field-level evidence matrix

Phase 4 produces this matrix; each output is releasable only against its required evidence:

| Output | Required evidence |
|---|---|
| **V/J top calls** | Gold + real-cohort accuracy (per-resolution, confidence-bounded) |
| **Calibrated V/J sets** | Real-domain coverage + set-size audit (and expansion coverage) |
| **D calls** | Gold construction-truth evidence only → otherwise **experimental** (stays experimental this phase) |
| **Coordinates** | Paired DNA→AA boundary validation (AA-residue error, stratified) |
| **`softmax_score`** | Reliability evidence (NLL/Brier/reliability plot); **never renamed probability** |
| **`inter_vj_aa`** | Semantic correctness only (V-end→J-start substring); **no AIRR-junction claim** |

## §9 Statistical framing

No statistical study can prove an individual prediction is correct. What Phase 4 proves is that, for a
**precisely defined population** (Tier G construction-truth; Tier R-concordant real human IGH), the
**set-producing procedure** achieves a **predeclared error/coverage contract with quantified uncertainty**
(confidence-bounded). Point estimates are never the gate; one-sided 95% bounds are.

## §10 Feasibility-first (Phase-4 Phase 0 — a hard gate, like Gate 3)

Before the study: a **data recon + sample-size feasibility gate.** Gold is generatable at will (GenAIRR).
The binding question is the Real cohort: after donor/study/clonal-lineage/near-identical-VH deduplication
and source-held-out splitting, are there **enough independent R-concordant examples** to power the coverage
/accuracy bounds (predeclared minimum effective N per stratum, e.g. ≈59 for a ≤5% two-sided-ish coverage-
miss bound, scaled to the contract) across a separate **calibration** and **test** partition? If the powered
real cohort cannot be assembled, **STOP and report feasibility** — do not weaken the contract.

## §11 Non-goals

- Not training; not packaging; not the applicability gate (Gate 3 / the separate trained-applicability-head
  project).
- Not IMGT numbering or CDR delimitation; `inter_vj_aa` is never an AIRR junction/CDR3.
- `softmax_score` is never renamed or reinterpreted as a calibrated probability.
- Gold (simulator) evidence is never presented as real-domain evidence, and the two tiers are never pooled.

## §12 Sequencing

1. **This spec** approved.
2. **Phase-4 implementation plan** (writing-plans), beginning with §10 feasibility (Real recon + powered
   split) before any evaluation, then: freeze/hash v2 + Gold generation → Real acquisition + nt-concordance
   truth + leakage-controlled split → the §2–§4 evaluation on both tiers → §6 calibration decision →
   §7 comparative baselines → §8 evidence matrix + frozen Phase-4 result artifact.
3. After Phase 4, the separate **trained-applicability-head project** remains the final prerequisite for a
   trustworthy public product.
