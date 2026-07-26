# AlignAIR-AA — Applicability Model Design

> **Status:** APPROVED FOR IMPLEMENTATION · **Date:** 2026-07-26
> **Project:** successor to failed Gate 3 · **Annotation model:** frozen
> `alignair-aa-igh-human@2.0.0` (`cont_step318000`)
> **Decision:** the AA annotator does not become a public product unless this project passes the original
> Gate-3 release contract on a fresh validation cohort and the still-sealed Gate-3 locked test. A locked
> `PASS` certifies only the near-domain, completeness, structural, and positive-input threat model actually
> represented in that seal; far-OOD/open-set results remain descriptive.

## 0. Purpose and binding context

The current AA AlignAIR annotator is a closed-world human-IGH classifier. Given any format-valid amino-acid
sequence, its V/D/J branches must choose among human IGH classes; a high `softmax_score` or narrow conformal
set therefore cannot establish that the input is human IGH.

The original Gate-3 experiment proved that this is not repairable by post-hoc distance scoring on the
predeclared `vj_penultimate_v1` representation:

- completeness and membership AUC were only 0.55–0.75;
- at any operating point compatible with the false-reject constraint, false-accept upper bounds were
  66–100%;
- the class-head bottlenecks were trained to answer “which IGH gene, assuming IGH,” not “is this a complete
  human IGH VH domain?”

The current model also has no learned shared contextual backbone suitable for this decision. Its shared
input is a 32-dimensional token-plus-position embedding; the learned convolutional towers live inside the
V/D/J branches and collapse their features toward within-IGH gene discrimination.

This project adds **new, separately trained safety capability**:

> A hybrid applicability guard that combines competitive biological profile evidence with a dedicated
> contextual sequence encoder trained explicitly to distinguish a complete human IGH VH domain from
> fragments, other immune-receptor loci, related Ig-fold proteins, and non-human/VHH domains. Outlier
> exposure and energy scoring add defense in depth against more distant proteins, but the existing locked
> test does not certify arbitrary or previously unseen protein distributions.

The guard executes before annotation. The frozen v2 annotator runs only when the guard returns
`supported`.

### 0.1 Standing decisions

1. **The AA v2 annotation model remains byte-for-byte frozen.** Applicability training cannot change its
   V/D/J weights, calibration, class maps, reference bridge, tokenizer, or outputs.
2. **The guard is a separate model and bundle section.** A guard failure cannot silently fall through to
   unguarded annotation.
3. **The original Gate-3 product contract and FAR/FRR thresholds remain binding.** They are not relaxed to
   fit the new model.
4. **The old Gate-3 development cohort is burned for inference and may be reused for training/development.**
   It cannot serve as fresh validation evidence.
5. **A fresh source/species/lineage-independent validation cohort is mandatory.**
6. **The existing Gate-3 locked test remains sealed.** It is opened exactly once, after the architecture,
   all model weights, all LOCO models, decision logic, and every threshold are frozen and hashed.
7. **No public AA product if this project stops or fails.** A bare predictor is not an acceptable fallback.
8. **Phase 4 remains STOPPED.** Passing applicability would not create a real-domain calibration guarantee;
   V/J conformal coverage remains simulator-domain unless separately established later.
9. **The existing high-specificity structural prefilter is carried forward frozen.** It may reject only
   the decisive defects already validated by Gate 3; fresh-validation observations cannot add new hard
   motif rules. Learned completeness is additional evidence, not a license to reintroduce the failed
   mandatory-anchor design.
10. **Certification scope follows the sealed instrument.** The locked test can certify the represented
    human-IGH positives, IGK/IGL, TRA/TRB, combined VHH/non-human VH, IgSF V-set, and parent-aggregated
    structural categories. It cannot certify generic non-antibody proteins, de-novo antibody-like
    negatives, arbitrary future OOD, or the semantic boundary between humanized, human-framework, and
    chimeric constructs.

## 1. Product contract

### 1.1 Supported input

The guard decides whether a format-valid input is:

- one complete **human IGH VH amino-acid domain**;
- in N→C order;
- no longer than the product maximum (192 residues);
- without signal-peptide/leader or constant-domain contamination;
- within the biological and engineering scope defined by the approved Gate-1 product contract.

Complete human-framework and defensibly humanized VH designs are supported challenge categories. Applicability
follows the sequence/domain contract, not whether a sequence was engineered.

The existing seal contains 13 `humanized` and 16 `design_human_framework` positives and 31
`chimeric_murine_variable` negatives. Those labels are inherited exactly as frozen and are not relabelled
in this project. Each category is below a stand-alone confidence floor; their separate results are
descriptive. Only the chimeric category also contributes to the binding combined VHH/non-human gate.

### 1.2 Unsupported or insufficient input

- Human IGK/IGL
- TRA/TRB/TRG/TRD
- Camelid VHH
- Non-human VH, including murine-variable chimeras
- IgSF V-like domains
- De-novo antibody-like proteins without defensible human-IGH membership
- Non-antibody proteins
- Reversed VH
- CDR3-only, V-only, J-only, or other incomplete fragments
- Leader+VH or VH+constant/full-heavy constructs
- Multiple domains or otherwise out-of-contract sequences

### 1.3 Output union

The approved discriminated union remains binding:

```text
supported
  → applicability evidence + V/D/J segments + inter_vj_aa

insufficient_evidence | unsupported_locus | unknown_ood
  → applicability evidence only
  → schema forbids segments and inter_vj_aa
```

Reason semantics:

- `insufficient_evidence`: a high-precision structural/completeness defect was identified.
- `unsupported_locus`: a specific alternative receptor/domain type was positively identified and that
  reason passes its own precision gate.
- `unknown_ood`: unsupported, but the reason is not sufficiently reliable.

When a specific reason is uncertain, the implementation must fall back to `unknown_ood`.

## 2. Why a hybrid guard

No one component is trusted to solve the entire problem.

### 2.1 Competitive profile evidence

Profile hidden Markov models are an established method for recognizing protein-family/domain membership.
ANARCI applies chain- and species-specific profile HMMs to classify and delimit antibody/TCR variable
domains. A competitive profile bank provides:

- explicit human-heavy versus alternative-chain evidence;
- domain alignment range and coverage;
- interpretable best-profile and runner-up scores;
- a biological score margin instead of closed-world IGH softmax confidence.

Profile scores, E-values, and margins are **evidence values, not probabilities of applicability**.

### 2.2 Dedicated contextual representation

A new encoder receives the full residue sequence and is trained on the applicability task itself. It must
learn:

- global N→C order;
- full-domain structure and coverage;
- relationships between framework and junction regions;
- biological domain/locus distinctions;
- features that remain useful under SHM, realistic substitutions/indels, and engineering.

It is not initialized from or attached to the annotator’s class-head bottlenecks. Existing branch spatial
features may be evaluated as a documented development ablation, but they are not the production default.

### 2.3 Outlier exposure and open-set scoring

Known biological negative classes cannot enumerate every possible unsupported protein. The neural guard
therefore receives:

- supervised hard negatives for known, near-domain categories;
- diverse, source-matched auxiliary outliers;
- an energy objective that separates recognized biological-domain inputs from unknown outliers.

Outlier exposure improves robustness to unseen outlier distributions, but it is not assumed to provide
universal rejection. Generalization to the sealed near-domain categories must be demonstrated by
category-LOCO and the locked test. Far-OOD and generic-protein challenges are evaluated on fresh,
source-disjoint cohorts as non-certified defense-in-depth evidence; they cannot enlarge the locked-test
claim.

## 3. Phase 0 — feasibility-first hard gate

No guard model is trained until both tooling and fresh-validation feasibility pass.

### 3.1 Tooling and licensing

The feasibility artifact must record:

- `pyhmmer` exact version, wheel/source hash, HMMER engine version, and deterministic invocation;
- ANARCI and ANARCII exact versions where installable;
- their roles as external comparators/teachers unless separately admitted as a production dependency;
- all software licenses and whether model/profile redistribution is permitted;
- reference source, release/version, acquisition URL, file hashes, and redistribution terms;
- deterministic profile-build scripts and output hashes;
- CPU and CUDA runtime requirements and reproducibility across supported platforms.

**Redistribution is a hard gate.** A profile bank derived from a source whose redistribution terms do not
permit bundling cannot be placed in a public `.alignair` artifact. Valid resolutions are:

1. use versioned, redistributable reference inputs;
2. obtain explicit permission;
3. require user-side hydration under a clearly documented license, if acceptable for product usability;
4. STOP the production-profile candidate.

The production path is resolved before the bank used by C0–C4 is built and before candidate training.
User-side hydration must reproduce the exact pinned bank used for training and validation; it is not
permission to hydrate an arbitrary future release. A bank cannot be substituted after fresh validation or
after the locked test. IMGT-derived banks remain comparator-only until explicitly cleared.

ANARCI/ANARCII availability does not automatically admit either into production. They are first treated as
comparators. The project-owned competitive profile bank is preferred only if its construction, scientific
meaning, runtime, and redistribution are reproducible and clear.

### 3.2 Profile bank scope

The bank must contain independently versioned profiles representing, at minimum:

- human IGHV;
- human IGKV;
- human IGLV;
- TRA, TRB, TRG, and TRD;
- camelid VHH;
- murine and at least one additional non-human VH group;
- representative non-receptor IgSF V-set domains.

For rearranging receptor loci, a V-germline profile alone is not evidence of a complete variable domain.
The production design must use either:

1. separately versioned V- and J-region profiles with an explicit, tested N→C pairing/coverage rule; or
2. a defensibly constructed full-variable-domain profile whose training sequences, junction treatment,
   licensing, and domain-boundary meaning are versioned and validated.

It must never report “complete-domain coverage” from V-profile coverage alone or create undocumented
synthetic V–J concatenations. IgSF profiles may remain full-domain profiles because they have no
rearranged J segment.

Profile construction uses reference/germline data only—never Gate-3 validation or locked-test sequences.
The profile manifest records:

- biological label and species/locus;
- source records;
- alignment/build procedure;
- sequence counts;
- profile checksum;
- score convention and search mode;
- any coverage or calibration parameters.

### 3.3 Fresh validation-cohort feasibility

Fresh validation records must be disjoint from:

- the burned Gate-3 development cohort by sequence hash, lineage, donor, study/source, and known aliases;
- the 16 Phase-3 therapeutics by INN and VH hash;
- the existing Gate-3 locked test by committed record hashes;
- model training and convergence cohorts where identities are available.

Minimum effective fresh-validation counts after lineage deduplication:

| Binding group | Contract floor per subset | Pre-unlock guardband planning floor per subset |
|---|---:|---:|
| IGK+IGL combined, both loci represented | 59 | 74 |
| TCR combined, TRA/TRB represented; TRG/TRD descriptive if underpowered | 59 | 74 |
| VHH/non-human VH combined | 29 | 36 |
| IgSF V-set | 29 | 36 |
| Structural parent lineages | 29 | 36 |
| Real complete human-IGH positives | 29 | 36 |
| Simulator complete human-IGH positives | 29 | 36 |

These are zero-error floors for the original one-sided confidence gates, not promises that the eventual
sample will pass after errors. The guardband planning floors are the zero-error counts required for an
upper bound no greater than 80% of the release limit: 4% for a 5% gate and 8% for a 10% gate. Actual
required counts are recomputed from the observed error budget; the table is not a substitute for exact
power code.

**Each binding selection subset and each disjoint blind-LOCO challenge subset must independently meet its
applicable guardband floor**; for groups without a LOCO fold, an equivalently disjoint challenge subset
fills the second role. The same record cannot power both threshold selection and held-out-category
evaluation. Each of those two fresh subsets must also contain at least 19 lineage-deduplicated VHH
sequences inside the combined VHH/non-human group. This mirrors the sealed VHH sub-count and prevents
fresh candidate selection from being driven only by easier non-human members. VHH remains a descriptive
stand-alone endpoint unless it independently reaches its exact confidence floor. Phase 0 must therefore
assess the post-allocation counts, not only the total hydrated category count. VHH is the highest-priority
supply-risk audit.

### 3.4 Locked-seal power and category-map re-audit

Phase 0 must produce a committed, counts-only `locked_category_map.json` derived from the already public
metadata in `split_fingerprint.json`; it must not load locked sequences. The artifact binds every release
gate to the exact frozen group, component categories, independent-unit definition, count, floor, and
fingerprint:

| Binding group | Locked independent units | Existing floor |
|---|---:|---:|
| IGK+IGL (`igk` 80 + `igl` 79) | 159 | 59 |
| TCR (`tra` 75 + `trb` 15) | 90 | 59 |
| VHH/non-human (`vhh` 19 + murine 15 + chimeric 31) | 65 | 29 |
| IgSF | 179 | 29 |
| Structural, parent-lineage aggregated | 35 | 29 |
| Real complete human IGH | 105 | 29 |
| Simulator complete human IGH | 50 | 29 |

The audit must also record that generic non-antibody proteins and de-novo unsupported antibody-like
proteins are absent from the locked test. All 19 frozen VHH lineages must be evaluated inside the combined
gate; no filtering, attrition, or cherry-picking is permitted. VHH-specific FAR and its exact interval are
reported descriptively because 19 is below the stand-alone floor.

Any mismatch between §10, the candidate’s category mapping, and this artifact is
`STOP_LOCKED_CATEGORY_MAP_MISMATCH` before training or unlock.

Phase 0 reports:

- source/category/species/family/lineage counts;
- exact overlap audit;
- sequence-validity and independent-boundary-label audit;
- license/redistribution status;
- executable tooling smoke results;
- projected selection and LOCO allocations;
- the locked-seal category-map and power re-audit;
- `GO` or `STOP`.

If any critical binding group or production tooling/license requirement fails, record STOP and do not train.

## 4. Data roles and leakage control

### 4.1 Training/development

The burned Gate-3 development cohort may be used for:

- training all neural candidates;
- fitting HMM/neural fusion components;
- hard-negative mining;
- internal cross-validation and hyperparameter selection;
- preliminary ablation.

It must be repartitioned deterministically by source, species, lineage, and parent-transform relationship.
Structural transformations inherit the parent lineage and split.

Additional training data may be acquired, but every source is recorded and must not overlap fresh
validation or locked test.

### 4.2 Fresh validation

Fresh validation has two non-overlapping roles:

1. **Selection/threshold subset:** selects the predeclared candidate and its operating thresholds.
2. **Unseen-category challenge subset:** evaluates LOCO models on categories excluded from their training
   and threshold selection.

For LOCO group `g`, no member of `g`, no source-specific proxy for `g`, and no rule engineered from `g`
may influence:

- that fold’s training;
- its hyperparameter choice;
- its threshold-selection subset.

### 4.3 Locked test

The existing Gate-3 locked test remains a physically separate file behind the audited one-shot loader.
Before unlock, the repository must contain hashes for:

- the selected production candidate;
- every LOCO model;
- tokenizer and profile bank;
- all thresholds;
- decision logic/version;
- validation artifact;
- evaluation script and output schema.

The unlock evaluates the full candidate and every predeclared LOCO fold in one run. No thresholds, models,
profiles, reason rules, or code paths may change afterward. Failure is final for this project version.

The locked test is a certifying instrument only for its frozen categories. Fresh generic-protein,
far-OOD, and de-novo unsupported challenges are never spliced into it and do not acquire certified status
by being evaluated alongside it.

### 4.4 Shortcut prevention

Training and validation must prevent trivial source shortcuts:

- length- and composition-match positives and hard negatives;
- balance source batches;
- remove dataset-specific prefixes, tags, identifiers, and padding artifacts;
- keep identical preprocessing across categories;
- hold out entire sources and species;
- train a post-hoc source probe on the learned representation and report its accuracy;
- include a source-adversarial objective only in the predeclared full candidate (§7), not as a post-hoc
  rescue.

## 5. Competitive profile-HMM component

For a sequence `x`, the HMM component emits a fixed feature vector:

- best V-profile and J-profile type/species (or the validated full-domain profile for a non-rearranging
  family);
- bitscore and E-value for every profile group;
- human-IGH score;
- best-alternative score;
- human-IGH minus best-alternative margin;
- separate V/J aligned-query and aligned-profile fractions;
- V/J domain start/end, ordering, inter-profile gap, combined query coverage, and unaligned N/C lengths;
- number of significant domains/hits;
- coverage/order flags.

The HMM-only applicability score and rule set are frozen from burned-development work before fresh
validation. No profile threshold is hand-adjusted after viewing fresh-validation failures.

HMM-specific reason rules:

- `unsupported_locus` may be emitted only when a named alternative profile wins by a validated margin and
  the reason precision requirement passes;
- incomplete/extra-domain evidence may produce `insufficient_evidence` only when its rule-specific false
  reject bound passes;
- otherwise rejection maps to `unknown_ood`.

## 6. Contextual applicability encoder

### 6.1 Fixed base architecture

The initial neural architecture is fixed before fresh validation:

- AA tokenizer: the existing 23-token alphabet, plus a dedicated learned `[CLS]` token;
- maximum input length: 192 residues;
- contextual width: 128;
- 4 Transformer encoder blocks;
- 4 attention heads;
- feed-forward width: 512;
- pre-norm residual blocks;
- dropout: 0.10;
- relative or rotary positional encoding;
- padding-aware attention;
- representation: concatenated `[CLS]` and masked-mean contextual embeddings.

One small, predeclared hyperparameter grid may be explored only on burned development data. The grid and
winner are committed before fresh validation. A larger protein-language model is not silently introduced
after validation.

ANARCII, ESM-family, or antibody-language-model embeddings may be reported as external development
comparators or teachers. A PLM runtime dependency becomes a separate candidate requiring its own size,
license, latency, reproducibility, and validation decision.

### 6.2 Heads

The contextual representation feeds:

1. **Completeness head**
   - `complete_domain`
   - `fragment_or_truncated`
   - `reversed`
   - `leader_contaminated`
   - `constant_or_multidomain_contaminated`

2. **Biological domain-type head**
   - `human_igh`
   - `human_igk`
   - `human_igl`
   - `tcr`
   - `vhh`
   - `nonhuman_vh`
   - `igsf_vset`
   - `other_known`

3. **Applicability head**
   - binary audit logit for `supported human complete IGH` versus known unsupported inputs.

4. **Open-set energy**
   - derived from the recognized-domain logits;
   - low energy means the guard recognizes a trained biological-domain class;
   - high energy means unknown relative to the guard’s training support.

The binary logit, type logits, completeness logits, and energy are model scores—not calibrated
probabilities of correctness.

Energy is a defense-in-depth component. Its efficacy on source-disjoint generic proteins and other
far-OOD inputs is reported descriptively on fresh data; the existing locked test contains no such
certifying category. Energy may improve rejection and route a rejection to `unknown_ood`, but it cannot
support a release claim that arbitrary unknown proteins are reliably detected.

### 6.3 HMM fusion

Hybrid candidates concatenate the sequence representation and normalized HMM feature vector before the
heads. Normalization statistics are fit on training data only and stored in the artifact.

The neural network cannot overwrite raw HMM provenance. Results retain both raw component evidence and the
combined decision.

## 7. Training objectives and five predeclared candidates

### 7.1 Candidate set

Exactly five candidates enter the comparison:

| ID | Candidate | Components |
|---|---|---|
| C0 | `hmm_only` | Competitive profile evidence + frozen interpretable decision rules |
| C1 | `neural_supervised` | Contextual encoder + completeness/type/applicability supervision |
| C2 | `hybrid_supervised` | Contextual encoder + HMM fusion + supervised heads |
| C3 | `hybrid_oe_energy` | C2 + outlier exposure + energy-margin training |
| C4 | `hybrid_full` | C3 + supervised contrastive representation + source-adversarial loss |

No sixth rescue candidate is added after fresh validation. If all five fail, the project stops before the
locked test.

### 7.2 Losses

For applicable candidates:

```text
L =
  L_type
  + λ_complete · L_complete
  + λ_app · L_applicability
  + λ_oe · L_energy_margin
  + λ_contrast · L_supervised_contrastive
  + λ_source · L_source_adversarial
```

- `L_type`: cross-entropy over known biological domain types.
- `L_complete`: cross-entropy over completeness/contamination states.
- `L_applicability`: binary cross-entropy for the supported contract.
- `L_energy_margin`: recognized training-domain inputs are pushed below an in-domain energy margin;
  auxiliary unknown outliers are pushed above an outlier margin.
- `L_supervised_contrastive`: pulls biologically equivalent domain types together and separates known
  near-domain groups.
- `L_source_adversarial`: gradient-reversal source classifier discourages source/dataset encoding.

Loss weights, margins, temperatures, and the internal development-selection rule are frozen using burned
development data only.

### 7.3 Training distributions

**Supported positives**

- real complete human IGH across families, donors, and sources;
- held-out GenAIRR human IGH;
- complete human-framework/humanized challenge sequences;
- realistic SHM and supported AA substitutions/indels;
- I/L-equivalent encodings where applicable.

**Known hard negatives**

- human IGK/IGL;
- all available TCR loci;
- VHH;
- murine/rabbit/other non-human VH;
- murine-variable chimeras;
- IgSF V-set domains;
- de-novo antibody-like proteins with unsupported membership.

**Structural negatives**

- deterministic N/C crops;
- CDR3-only, V-only, J-only;
- reversed sequence;
- signal peptide + VH;
- VH + constant or multi-domain sequence;
- internal deletions and domain concatenations.

Every derived structural negative keeps the parent lineage key. Effective sample size is parent-lineage
count, never transform count.

**Outlier exposure**

- diverse non-receptor protein domains;
- length- and composition-matched proteins;
- related but unsupported Ig-fold domains;
- synthetic receptor-like corruptions;
- generated sequences outside the supported human-IGH contract.

Outlier-exposure sources are disjoint from fresh validation and locked test. Easy random proteins are a
minority and cannot dominate the loss or metrics.

### 7.4 Hard-negative mining

Mining may occur only inside training/development:

1. score a versioned auxiliary pool;
2. select false accepts or smallest-margin outliers;
3. deduplicate by lineage/sequence;
4. add them to a versioned training manifest;
5. retrain from the documented seed/checkpoint policy.

Fresh validation and locked test are never mined.

## 8. Validation and LOCO

### 8.1 Main validation

For each candidate:

1. fit weights/rules only on training;
2. select its finite, predeclared threshold grid on fresh selection validation;
3. compute per-group false accepts and positive false rejects;
4. compute one-sided exact 95% upper bounds;
5. validate reason-state precision;
6. record latency, memory, artifact size, and platform behavior.

Source-disjoint generic-protein, de-novo unsupported, and other far-OOD cohorts are scored and reported,
but their results are `descriptive_noncertified`: they do not tune thresholds, determine candidate
eligibility, or break selection ties. This prevents a fresh descriptive cohort from becoming an
unacknowledged surrogate locked test.

### 8.2 LOCO models

Train and freeze four additional models for every learned candidate:

- exclude all IGK/IGL;
- exclude all TCR;
- exclude all VHH/non-human VH;
- exclude all IgSF.

For fold `g`:

- no `g` record enters training, mining, hyperparameter selection, threshold selection, profile-rule
  adjustment, or source balancing;
- the `g`-specific competitive profile is masked from the fold’s HMM feature vector and decision graph;
  otherwise the hybrid could appear to generalize only because a reference profile explicitly names the
  supposedly unseen category;
- the held-out fresh challenge subset for `g` is opened only after that fold’s threshold is frozen;
- the fold must satisfy the same group-specific FAR bound;
- failure of any critical fold prevents that candidate from selection.

Profile HMMs are biological reference tools, not trained cohort classifiers. Nevertheless, an HMM-only or
hybrid candidate must separately report whether each held-out group’s profile family was built using
knowledge of that group. Two results are therefore required:

1. **Reference-covered:** production profile bank intact. This measures the intended production path.
2. **Blind LOCO:** group-specific profile masked and all learned exposure to the group removed. This is the
   binding unseen-category generalization result.

For C0, blind LOCO is a profile-withheld HMM test. For learned/hybrid candidates it is a profile-withheld
and training-category-withheld model. A candidate that passes only the reference-covered result is not
eligible.

### 8.3 Candidate selection

A candidate is eligible only if:

- every main fresh-validation FAR/FRR gate passes with the predeclared unlock guardband: the one-sided
  exact 95% upper bound is at most 80% of its release limit (≤4% for a 5% gate; ≤8% for a 10% gate);
- every required blind-LOCO fold passes the same guardband (profile-withheld for C0; profile- and
  training-category-withheld for learned/hybrid candidates);
- selection and blind-LOCO results use the same frozen score direction and decision graph, and each
  independently clears its guardband—averaging, pooling, or allowing one to compensate for the other is
  forbidden;
- reason-state precision passes or specific reasons are demoted to `unknown_ood`;
- artifact/runtime/licensing requirements pass;
- frozen-annotator invariance passes.

The 80% rule is a pre-unlock winner’s-curse safeguard, not a change to the locked release contract.
Phase 0 powers both disjoint validation roles for it; if observed errors make the achieved upper bound
larger than the guardband, the candidate is ineligible regardless of its rank among C0–C4.

Among eligible candidates, select deterministically:

1. smallest worst-group FAR upper-bound ratio (`observed upper bound / allowed bound`);
2. then smallest maximum of real/simulator FRR upper bounds;
3. then smaller runtime artifact;
4. then simpler candidate order: C0 → C1 → C2 → C3 → C4.

If no candidate is eligible: `STOP_NO_FEASIBLE_APPLICABILITY_MODEL`; do not open the locked test.

## 9. Decision rule and result semantics

For hybrid candidates, `supported` is conjunctive:

```text
format valid
AND high-specificity structural prefilter passes
AND HMM human-IGH/domain evidence passes
AND neural completeness passes
AND neural type/applicability evidence supports human IGH
AND open-set energy passes
```

Components not present in a candidate are omitted, never treated as implicit passes.

The frozen structural prefilter is the decisive first authority. A prefilter rejection is final and
cannot be overridden by the learned completeness head. The neural `reversed` class is defense in depth:
when the prefilter defers, the learned guard may still reject, but it may emit
`insufficient_evidence:reversed` only if that reason passes the reason-precision gate; otherwise it emits
`unknown_ood`.

Thresholds and score directions are part of the frozen candidate. The implementation must not substitute
softmax confidence or conformal set size.

Applicability detail contains:

- candidate/detector id and version;
- final state and reason;
- audit score (not a probability);
- HMM best profile, raw score/margin, domain coverage, and profile-bank hash;
- neural completeness/type scores and energy, when present;
- threshold ids and decision-rule version;
- validation-artifact hash;
- warnings.

When state is not `supported`, V/D/J calls and `inter_vj_aa` are structurally absent.

## 10. Binding release thresholds and certification scope

All gates use one-sided exact 95% upper confidence bounds, not point estimates.

| Metric | Required upper bound |
|---|---:|
| Overall near-OOD FAR | ≤5% |
| IGK/IGL FAR | ≤5% |
| TCR FAR | ≤5% |
| VHH/non-human VH FAR | ≤10% |
| IgSF FAR | ≤10% |
| Structural FAR, parent-lineage aggregated | ≤10% |
| FRR on real complete human IGH | ≤10% |
| FRR on simulator complete human IGH | ≤10% |

These binding locked-test gates certify only the frozen near-domain/completeness threat model enumerated
in §3.4. They do **not** certify generic non-antibody rejection, de-novo unsupported designs, arbitrary
far-OOD distributions, or VHH as a stand-alone category. The VHH result is binding only through the
combined `vhh_nonhuman` group, which must include all 19 frozen VHH lineages; VHH-specific FAR and its wide
exact interval are reported descriptively.

Report separately:

- each locus/category, even when the binding gate is combined;
- human-framework and humanized positive challenges;
- de-novo antibody-like challenges;
- structural transform type, parent-lineage aggregated;
- easy unrelated proteins;
- source, species, family, length, mutation burden, and engineering status.

Generic non-antibody, de-novo unsupported, and other far-OOD results are explicitly labelled
`descriptive_noncertified`. Humanized, human-framework, and chimeric results retain their frozen labels
and are reported separately, but their individual categories are not certified.

Reason validation:

- full four-state confusion matrix;
- `unsupported_locus` precision reported overall and by named reason;
- a reason whose one-sided exact 95% **lower** confidence bound is below **90% precision** is disabled and
  mapped to `unknown_ood`;
- `insufficient_evidence` is evaluated by structural category and parent lineage.

## 11. Frozen annotator invariance

The guard must not alter annotation:

- for every accepted regression fixture, v2 V/D/J logits, decoded coordinates, calibrated classes,
  expanded calls, `softmax_score`, and `inter_vj_aa` are byte-identical to direct unguarded v2 execution;
- guard batching/order does not change annotation output;
- loading a guard does not mutate v2 weights or calibration;
- guard failure/absence cannot enable public unguarded inference;
- research-only bypass, if retained, requires an explicitly unsafe API/CLI flag and emits a persistent
  warning. It is not the product default and is excluded from release examples.

## 12. Locked-test protocol

Before unlock:

- selected candidate and all LOCO models saved;
- the exact public result union, withholding behavior, bundle loader, and public API product path are
  implemented and tested; a rejection can no longer expose annotation fields;
- thresholds and decision graph frozen;
- selection and every blind-LOCO challenge independently pass the §8.3 guardband;
- `locked_category_map.json` matches the sealed fingerprint and §10;
- profile/tokenizer/reference artifacts hashed;
- evaluation code committed;
- validation decision committed;
- safe-load test passes;
- no tracked or private audit indicates an earlier unlock.

One invocation:

1. verifies all frozen hashes;
2. unlocks the existing Gate-3 test with a substantive audit reason;
3. evaluates the selected candidate and all required LOCO models;
4. emits per-category counts, point estimates, confidence bounds, reason matrix, and provenance;
5. writes the immutable final result.

No rerun with changed code/model/thresholds is permitted. An infrastructure failure before any prediction
may be retried only if the artifact proves no test output was observed and all candidate hashes remain
identical.

Valid outcomes:

1. `PASS` — all binding near-domain/completeness, LOCO, reason, invariance, integrity, and runtime gates
   represented by the sealed test pass.
2. `FAIL` — any binding locked-test gate fails.
3. `STOP` — feasibility, licensing, fresh validation, or candidate selection fails before unlock.

Only `PASS` permits integration of the applicability field into a public AA product, and only with the
validated-population statement from §10. A `PASS` does not authorize an “arbitrary unknown protein
detection” claim.

## 13. Packaging and integrity

The selected candidate becomes a pickle-free, hash-verified section of the AA `.alignair` candidate bundle
**before** locked-test unlock so the exact safe-load and public product path can be frozen and evaluated.
The bundle is not distributed unless the locked result is `PASS`.

- neural weights in `safetensors`;
- model config and tokenizer manifest;
- profile HMM bank in a non-pickle deterministic format;
- profile/reference/license manifest;
- feature normalization arrays with `allow_pickle=False`;
- thresholds and score directions;
- decision-rule graph/version;
- validation and locked-test artifact hashes;
- software/runtime compatibility metadata.

Load hard-fails on:

- missing guard for a production AA bundle;
- checksum mismatch;
- profile/reference mismatch;
- threshold/model mismatch;
- unsupported schema/version;
- missing required runtime;
- validation binding mismatch.

Every result includes guard provenance sufficient to reproduce the verdict.

## 14. Required ablations and scientific report

Report:

- C0–C4 main and LOCO results;
- HMM-only versus neural-only versus fused contributions;
- effect of OE;
- effect of supervised contrastive and source-adversarial losses;
- known-category versus unseen-category performance;
- source-probe accuracy;
- score distributions and failure examples;
- latency/memory/artifact-size tradeoffs;
- false-accept and false-reject error taxonomy;
- performance on humanized/human-framework positives;
- VHH-specific FAR with its exact interval, explicitly descriptive when underpowered;
- fresh far-OOD/generic-protein energy results explicitly marked non-certified;
- explicit limitations and the exact validated population.

Do not claim universal OOD detection. A passing model establishes only that the frozen guard satisfies the
predeclared near-domain/completeness threat model and confidence-bounded sealed-cohort contract. Open-set
energy evidence outside that cohort is defense in depth, not certification.

## 15. Non-goals

- No changes to AA v2 V/D/J training or calibration.
- No real-domain conformal guarantee.
- No light-chain or TCR annotation.
- No VH extraction from full chains.
- No fragment-support product.
- No IMGT numbering/CDR3 claim.
- No calibrated per-input probability that the verdict is correct.
- No guarantee against arbitrary future OOD distributions.
- No locked-test certification of generic non-antibody or de-novo unsupported-protein rejection.
- No stand-alone certified VHH FAR from the existing 19-sequence VHH subset.

## 16. Sequencing

1. Approve this design.
2. Write the implementation plan.
3. Phase 0 tooling/license/profile/fresh-validation feasibility and counts-only locked-category-map
   re-audit.
4. Freeze training and fresh-validation manifests.
5. Build HMM baseline and contextual candidates on burned development data.
6. Train C0–C4 plus required LOCO models.
7. Select candidate and thresholds on fresh selection plus blind-LOCO validation; STOP unless both
   independently clear the predeclared guardband.
8. Implement and test the exact public result union, withholding behavior, bundle loader, and public API
   product path; then freeze the candidate, product code, decision logic, evaluation script, and safe-load
   artifact.
9. Open the existing Gate-3 locked test once.
10. Record PASS/FAIL.
11. Only after PASS: enable the already-frozen guarded product path for release, complete documentation
    and model-card work, and revisit public distribution. No inference/schema behavior changes occur
    after the one-shot.

## 17. Method references

- Dunbar J, Deane CM. **ANARCI: antigen receptor numbering and receptor classification.**
  *Bioinformatics* (2016). https://pmc.ncbi.nlm.nih.gov/articles/PMC4708101/
- Greenshields-Watson A, Agarwal P, Robinson SA et al. **ANARCII enables alignment-free antigen receptor
  numbering using a generalised language model.** *Communications Biology* (2026).
  https://www.nature.com/articles/s42003-026-10186-z
- Hendrycks D, Mazeika M, Dietterich T. **Deep Anomaly Detection with Outlier Exposure.**
  *ICLR* (2019). https://openreview.net/forum?id=HyxCxhRcY7
- Liu W, Wang X, Owens JD, Li Y. **Energy-based Out-of-distribution Detection.**
  *NeurIPS* (2020).
  https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
- Khosla P et al. **Supervised Contrastive Learning.** *NeurIPS* (2020).
  https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
- Rives A et al. **Biological structure and function emerge from scaling unsupervised learning to
  250 million protein sequences.** *PNAS* (2021).
  https://pubmed.ncbi.nlm.nih.gov/33876751/
