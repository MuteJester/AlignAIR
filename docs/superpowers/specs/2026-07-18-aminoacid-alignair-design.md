# Amino-Acid AlignAIR — Design Spec

**Date:** 2026-07-18
**Branch:** `experiment/aminoacid-alignair` (off `origin/main` @ v3.0.0, in-place, upstream unset)
**Status:** Design approved (with revisions). Phase 0 authorized. Phase 1 gated on Phase 0 results.

---

## 1. Framing

This is **not** an alphabet swap. The convolutional AlignAIR architecture can consume amino acids trivially; that is not the interesting question. The interesting — and load-bearing — question is **how much alignment information survives translation from nucleotides to protein**, and whether a model trained on the surviving signal is biologically honest.

We therefore treat this as an **information-loss and identifiability study first, a model second.** The overriding failure mode to avoid is a model that scores well on learnable simulator labels while making biologically impossible claims (e.g. distinguishing two alleles that translate to identical protein, or reporting a canonical D allele that has no fixed protein form).

**Overall recommendation: GO**, structured as the phased study below. Every neural-training phase is gated behind an information audit that can return NO-GO.

## 2. Use case & scope

- **Input contract:** protein-only reads. Amino-acid antibody sequences with **no underlying nucleotide read available** (protein/mass-spec sequencing, antibody engineering). The model aligns AA reads directly against AA references. No reading frame, codon, orientation, or reverse-complement concept exists at inference.
- **Locus:** IGH (heavy chain) only, to mirror the stable nucleotide IGH model and enable apples-to-apples comparison.
- **Segments:** V, J, and D — but **D is redefined** (see §5.3). V/J are built first; D is a separate feasibility sub-phase (1b).
- **Assumed:** the input protein is an in-frame translation of a productive rearrangement. Stops (`*`) and invalid residues are **explicitly rejected or flagged**, never silently normalized. "Productive" is a contract on the input, not a prediction.

### Non-goals (this study)

- Light chains / TCR (future, only if IGH proves out).
- Full IMGT numbering as a Phase-1 requirement (it is its own deliverable — §5.5).
- Prediction parity with, or replacement of, the nucleotide pipeline. The nucleotide model is an information-rich **upper bound**, not a competitor.
- A native AA mutation simulator. AA-level SHM falls out of translating GenAIRR's nucleotide SHM.

## 3. Central design principle: identifiability, not nucleotide labels

Nucleotide allele identity **does not** map to protein identity. Synonymous substitutions make distinct nucleotide alleles translate to identical protein. No protein model — at any capacity, with any data — can separate them. Training separate outputs for indistinguishable alleles feeds identical inputs conflicting labels.

**Design consequences (mandatory):**

- Build **V/J protein-equivalence classes** by translated coding sequence.
- The model predicts the **equivalence class**, not the nucleotide allele.
- Inference returns **all nucleotide alleles in the predicted class as an ambiguity set**.
- The **theoretical AA identifiability ceiling** is computed *before* any training (Phase 0A) and is a **go/no-go gate**.
- Reporting is **ambiguity-aware**: gene-level, family-level, protein-class, and ambiguity-set accuracy — **never** ordinary nucleotide-allele accuracy, which is unachievable by construction.

**Coverage-aware ceiling (refinement).** Identifiability is not a single number over full-length translations. Two alleles identical across the full translated V but differing only in a region that is frequently trimmed or uncovered are distinguishable on full reads and **indistinguishable on short ones**. Phase 0A's length/coverage audit and the identifiability audit are the *same* computation: the ceiling is reported **conditioned on the read-length/coverage distribution**, not just full-length.

**Equivalence classes are germline-DB-version-specific (refinement).** Classes are computed from the exact GenAIRR dataconfig reference in use and frozen as a **class-map artifact stamped with a DB-version hash**. Inference-time ambiguity-set expansion reads from that frozen artifact. This prevents silent class↔allele drift when the allele DB updates.

## 4. Architecture: task-capability spec (corrects an earlier error)

**Correction:** an earlier draft claimed the core model, loss, and collator were reusable "byte-for-byte." They are not. The model constructs and runs orientation, mutation, indel, and productivity heads unconditionally (`core/model.py`); the loss requires mutation/indel/productivity unconditionally (`core/losses.py:107`); the collator requires those targets (`train/gym/collate.py:90`).

The correct abstraction is a **backward-compatible task-capability spec**, not scattered `if amino_acid` branches:

```python
use_orientation: bool = True
meta_heads: tuple[str, ...] = ("mutation_rate", "indel_count", "productive")
```

This spec is threaded consistently through **model → loss → collator → metrics → checkpoint metadata → validation report → prediction pipeline**. Requirements:

- **Nucleotide defaults are unchanged** and must remain **byte-compatible with existing v3 checkpoints.**
- The task spec is **serialized into the checkpoint metadata**, so a loaded AA model self-describes which heads exist and the pipeline can tell an AA checkpoint from a DNA one without out-of-band knowledge.
- The AA configuration sets `use_orientation=False` and (for the first learning proof) `meta_heads=()`.

### Package layout (Approach A — thin AA subpackage reusing the core)

New subpackage `src/alignair/aa/`. The alphabet-agnostic core (conv towers, gene branches, segmentation soft-argmax, sigmoid allele heads, hierarchical loss) is reused via the task spec — not forked.

| File | Responsibility |
|---|---|
| `aa/tokenizer.py` | AA vocabulary + **right-padding** tokenizer (see §5.4) |
| `aa/reference.py` | AA reference schema (`alphabet="amino_acid"`, `coordinate_unit="residue"`); translated V/J germlines + anchor QC; equivalence-class map |
| `aa/equivalence.py` | Build/serialize protein-equivalence classes + DB-version-stamped class-map artifact + ambiguity-set expansion |
| `aa/config.py` | `aa_igh_config()` factory: AA vocab size, `max_seq_length=192` (provisional), `use_orientation=False`, `meta_heads=()`, retuned kernel schedules |
| `aa/data.py` | GenAIRR→AA adapter: codon-provenance coordinate projection (§5.2), in-frame/productive filter, lineage-aware split keys |
| `aa/pipeline.py` | Slimmed predict: tokenize → model → class selection + ambiguity expansion → AA-space coords → protein alignment → annotation. No orientation, no codon/frame/stop logic. |
| `aa/annotate.py` | AA region projection (V/J framework/CDR via IMGT-gapped projection — §5.5) + `junction_aa` |

## 5. Critical implementation areas

### 5.1 Reference (AA schema, not overloaded nucleotide `GeneReference`)

The current `GeneReference` explicitly stores nucleotide sequences (`reference/reference_set.py:75`). Create a **distinct** AA schema: `alphabet="amino_acid"`, `coordinate_unit="residue"`. It holds translated germline protein, per-allele equivalence-class id, and IMGT-gapped protein (for region projection).

**Anchor QC (refinement).** Any V/J germline whose V-frame translation lacks the conserved Cys-104 (V) / Trp-118 (J) anchor at the expected IMGT position signals a bad frame or pseudogene. Such germlines are **quarantined** from the AA reference, not silently included.

### 5.2 Coordinate projection is NOT `//3`

GenAIRR translates the complete rearrangement in a **junction-derived** reading frame and drops incomplete codons at region boundaries (`_airr_record.py:381`, `_airr_record.py:606`). A codon can straddle V/N, N/D, or D/N boundaries; such a residue belongs cleanly to no single nucleotide segment.

Adapter requirements (**highest-risk area**):

- Preserve half-open `[start, end)` coordinate semantics throughout.
- Derive supervision from **nucleotide provenance per codon**: label a residue's segment only when **all three** of its nucleotides have unambiguous, identical segment provenance.
- **Mask mixed-provenance boundary residues** from the coordinate loss.
- Record start/end **phase residuals** for diagnostics.
- **Recompute the reading frame after any cropping.**
- Tests cover **every boundary-phase combination**, not only frame-zero examples.

### 5.3 D head = experimental source-classification, not germline alignment

"Translate D in the V frame" is wrong. A D allele can appear in any frame, is trimmed on both ends, and its boundary codons incorporate N-region nucleotides — so an isolated D germline has **no single fixed protein form**.

Redefinition:

- Keep the D head, but as a **D-source classification** target trained from translated rearrangement context.
- **No** single canonical protein reference per D allele; **no** claimed conventional AA germline alignment for D.
- Options to evaluate: three frame-conditioned D prototypes, or a learned classifier with no translated D prototype.
- Return **calibrated D ambiguity or abstention** when evidence is insufficient.
- Measure **gene/family recovery separately from allele recovery**, against a **majority-class baseline**.
- Sequencing: **V/J first (Phase 1), D as Phase 1b.** D continues only if it beats simple baselines meaningfully.

### 5.4 Vocabulary & padding (explicit)

- Vocabulary: **PAD + 20 canonical residues + `X` (unknown)**, plus an explicit policy for stop `*` — either a dedicated token **or** explicit rejection of stops at the input boundary. **Stop and unknown are NOT made equivalent.**
- Explicitly define handling of `B, Z, J, U, O`, gaps, lowercase, and otherwise-invalid input (map-to-X vs reject — decided in 0B, documented, tested).
- Padding: **right-padding only**, matching the current trainer. Supporting center + right padding creates two coordinate systems with no near-term research benefit.

### 5.5 Region annotation / IMGT numbering (own deliverable)

FR/CDR boundaries are **not** derivable from the two conserved anchors alone. They require projection through **IMGT-gapped V/J references** or a numbering implementation. Full IMGT numbering is treated as its **own deliverable in Phase 2**, after dedicated validation — not a Phase-1 gate.

### 5.6 Meta-heads / divergence

- **First learning proof omits all meta-heads** (`meta_heads=()`). Prove V/J classification and residue-boundary learning before adding anything.
- **Do not reinterpret nucleotide `mutation_rate` as AA divergence.** The nucleotide target (`train/gym/targets.py:133`) counts synonymous mutations, conflates multi-hit codons, is broken by indels, and ignores per-segment coverage. If a divergence target is later added, compute a **new** `aa_divergence` from an explicit protein alignment over comparable/covered residues — under a new name, never reusing `mutation_rate`.

## 6. Data, splits, evaluation

- **Source:** GenAIRR simulates IGH DNA as today (real SHM, junction trimming, allele sampling) → read `sequence_aa` + project boundaries per §5.2 → filter to in-frame/productive → tokenize. Phase 0B verifies the streaming path exposes `sequence_aa` (confirmed present in GenAIRR's AIRR-record layer, v2.2.0).
- **Leakage control:** split simulated data by **underlying rearrangement / clonal lineage**, not per mutated read — otherwise related descendants cross the train/test boundary. GenAIRR clonal-lineage identity provides the split key.
- **Baselines & upper bound:**
  - **Same-protein-input baseline** (the fair competitor — e.g. `igblastp` / a protein aligner given the identical AA input).
  - **Nucleotide AlignAIR** as an information-rich **upper bound**, explicitly labeled as such — not a fair competitor.
- **Metrics:** ambiguity-aware V/J class accuracy; D gene/family recovery + calibration/abstention vs majority-class; **boundary error in residues**; region-annotation accuracy. Always reported against the coverage-conditioned identifiability ceiling.
- **Real-world validation:** simulator-only evaluation establishes *simulator recovery*, not AIRR utility. **Phase 3 requires a real repertoire with independently defensible labels.**

## 7. Phases

**Phase 0A — information audit (go/no-go gate, no training)**
- Build V/J (and D-source) protein-equivalence classes from the exact dataconfig reference; freeze the DB-version-stamped class map.
- Measure unique V/J/D identifiability — full-length **and** coverage-conditioned on the length audit.
- Audit sequence-length quantiles and unknown/stop frequencies.
- Quantify how often each segment boundary cuts through a codon.
- Measure how many fully owned D residues remain after trimming.
- **Gate:** decide whether enough alleles remain distinguishable to make the model useful.

**Phase 0B — semantic prototype (no neural training)**
- Implement translation + codon-provenance projection (§5.2).
- Validate V/J conserved-anchor residues; apply anchor QC to the reference.
- Fix the vocabulary/invalid-input policy (§5.4).
- Define the D supervision policy (§5.3).
- Verify GenAIRR streaming exposes `sequence_aa`.
- **Deliverable:** a frozen diagnostic report.

**Phase 1 — minimal learning proof**
- Right-padded AA tokenizer; V/J **equivalence-class** heads; `use_orientation=False`, `meta_heads=()`.
- Overfit one batch, then a small held-out (lineage-split) simulation set.
- **Demonstrate boundaries and class probabilities actually learn.**

**Phase 1b — D feasibility**
- Add D-source classification; measure vs gene/family and majority-class baselines; require calibration + abstention results. Continue only on a meaningful win.

**Phase 2 — prediction & annotation**
- Protein-aware alignment (substitution matrix + affine gaps — nucleotide match/mismatch scoring is unsuitable); AA-coordinate output; junction + V/J region projection; full IMGT numbering only after dedicated validation.

**Phase 3 — scientific findings**
- Information ceiling; simulation results; real-data evaluation; same-input protein baseline; nucleotide pipeline as upper bound; explicit **viability / no-go conclusion.**

## 8. Testing strategy

Mirror `tests/alignair/` under `tests/alignair/aa/`:
- Tokenizer round-trip + invalid-input policy.
- Germline translation + anchor QC.
- **Coordinate-projection correctness across all boundary-phase combinations** (highest-risk).
- Equivalence-class construction + ambiguity-set expansion + DB-version stamping.
- Config/model output-shape checks under the AA task spec.
- Task-spec plumbing: nucleotide defaults remain v3-checkpoint-compatible.
- Pipeline smoke; annotation.

## 9. Key risks

1. **Identifiability ceiling too low** → model is uninteresting. Mitigated by the Phase 0A gate *before* training.
2. **Coordinate projection bugs** at codon-straddling boundaries → silently wrong supervision. Mitigated by per-codon provenance masking + exhaustive phase tests.
3. **D is not recoverable in AA space** → mitigated by treating D as a gated feasibility sub-phase with abstention, not a required output.
4. **Simulator-only success** that doesn't transfer → mitigated by the mandatory Phase 3 real-data evaluation.
5. **Task-spec regression** breaking the shipped nucleotide model → mitigated by byte-compatibility tests on v3 checkpoints.

## 10. Decisions (locked)

| Decision | Resolution |
|---|---|
| Branch | `experiment/aminoacid-alignair`, in-place off `origin/main` (v3.0.0), upstream unset |
| Isolation | In-place branch (worktree considered, not chosen) |
| Max length | **192 provisional**; measure length quantiles + truncation in 0A; compare 224 only if meaningful records exceed 192; no jump to 256 without evidence |
| Orientation | Disabled via generic `use_orientation` task flag |
| Productive head | Removed; input contract "assumed in-frame translation"; stops/invalid rejected or flagged |
| D head | Kept as experimental **source-classification** target; V/J first, D in Phase 1b |
| Mutation rate | **Not** reinterpreted; omitted initially; a future `aa_divergence` is a separately computed, separately named target |
| Meta-heads (Phase 1) | None (`meta_heads=()`) |
| Allele target | Protein-equivalence **class**, with nucleotide-allele ambiguity set |

## 11. Open questions (resolve in Phase 0)

- Stop token vs stop rejection — decide in 0B from the stop-frequency audit.
- D representation: three frame-conditioned prototypes vs prototype-free classifier — decide in 0A/1b.
- Real-data source for Phase 3 (which repertoire, what labels count as "independently defensible").

## 15. External review corrections (Codex, 2026-07-20) — VERIFIED

An external architecture review (Codex) challenged three of the "proven" claims in §14 and the
forensics/masking artifacts. I re-verified each empirically; **all three corrections hold.** Recording
them because the original framing overstated what was proven.

1. **The AA-nearest-germline "oracle" is a strong baseline, NOT an information-theoretic ceiling.**
   `nearest_germline` (`aa/forensics.py`) is raw Cys-anchored identity — no allele priors, codon/SHM
   probabilities, gaps, coverage, or full posterior. Proof it is not an upper bound: the trained model
   **beats it on 5.6% of reads** (`forensics.json::model_beats_oracle_frac = 0.056`). Therefore the
   0.90 "ceiling" is the accuracy of an identity baseline, and the **"83% model-deficiency / 17%
   information-loss" split is a heuristic decomposition, not a proof.** (The directional read — that a
   simple baseline recovers much of what the 20k model misses, i.e. real training headroom — still
   stands; the *proof* language does not.) Also: the 0.73-vs-0.90 comparison is on the 2,674 indel-free
   selected reads; the full 4,096-read V class accuracy is **0.625** (`metrics.json`).

2. **The masking analysis masked the WRONG side of a V indel — its 52%/94% numbers are invalid.**
   GenAIRR builds `sequence_aa` in the frame anchored at the V-Cys/junction (`junction_start % 3`). So
   for an indel *before* the Cys, the **anchor-proximal** (3', near-Cys) region stays in-frame and the
   **5' region goes off-frame** — the opposite of `mask_recovery.py`'s "first M-run is the valid
   prefix." Verified on 200 single-V-indel reads: Cys-anchor-proximal identity **0.640** vs 5'-most
   identity **0.218**. (The script also uses `sum(runs)//3` instead of `sum(run//3 for run in runs)`,
   combining partial boundary codons.) **Corrected** (`mask_recovery.py` rewritten anchor-relative +
   complete-codon; `mask_recovery.json` regenerated): of V-indel reads the anchor-proximal in-frame run
   is mean **19** residues; **masking suffices for only 28%** (was wrongly 45%), **65% need full frame
   correction**, 6% truly lost; correctable (all runs) mean **86** → correction recovers ~94%. So
   masking is *weaker* than first reported and frame-correction is the real fix. NB: `maskable` still
   uses true germline+junction (a hidden-truth ceiling); the **AA-only** recovery number comes from the
   no-training Cys-scanning local aligner (de-risk (a)).

3. **The "ambiguity did not emerge" conclusion is confounded by the loss.** BCE target is
   `y*0.9 + 0.05` (`losses.py`), training every one of the 197 negative V alleles toward 0.05; the
   measured non-member mean **0.0566 ≈ 0.05**. The diffuse smear is largely this floor, not evidence
   the network cannot localize ambiguity. The right fix (per review) is **explicit protein-equivalence
   class targets + partial-label likelihood over the compatible set + post-hoc conformal calibration**,
   not normalized sigmoid sums.

**Strategic reframe (endorsed).** Nucleotide frameshift is a self-inflicted DNA-translation artifact;
real protein input has none. Keep the frameshift suite as an adversarial/legacy benchmark; do NOT let
it drive the architecture. The high-value path is: explicit protein-equivalence targets, longer
training, AA-native errors (substitutions/indels, truncation/coverage, I/L equivalence), calibrated
class sets, and real protein validation — building on Phase 1's ~0.95 full-length V class accuracy.

**Verdict on the complex/Fourier superposition idea:** rejected for now (review + our own read). The
abstraction ("marginalize over latent alignment/frame paths") is right, but a log-space forward
algorithm / differentiable alignment computes it more directly with clean probability semantics;
Z₃-phase Fourier is a possible later ablation of the transition kernel, not the architecture.

**Recommended order (endorsed):** (1) correct the oracle/masking claims [done here]; (2) split the
AA-native benchmark from the mis-translated-DNA stress suite; (3) restore explicit protein-equivalence
targets + calibrated class sets; (4) run the no-training Cys-scanning local-alignment de-risk
experiment; (5) add a learned residue-integrity gate only if local evidence proves useful; (6) attempt
the candidate-conditioned nt→protein transducer only if shifted-suffix evidence shows headroom;
(7) route nucleotide-derived cases to nucleotide AlignAIR whenever DNA is available.

## 14. Same-Scheme Experiment — AA input into the real DNA pipeline (2026-07-20)

A different, simpler line than the Phase-1 thin module (user-directed): reuse the **real** nucleotide
training scheme unchanged — same `AlignAIR` model, `hierarchical_loss`, gym curriculum, and **full
nucleotide-allele multi-hot targets (198/33/7)** — and change only the input to amino acids (with
AA-space coordinates, forced by the masked soft-argmax). The one core change was a backward-compatible
`use_orientation=False` flag (RC is undefined for protein and its complement table is nt-only; nt
checkpoints stay byte-identical). Code: `src/alignair/aa/{gym_bridge,train_gym,eval_gym}.py`. Artifact:
`docs/superpowers/artifacts/2026-07-20-aa-same-scheme/`.

**Question:** with single-allele targets and AA input, does synonymous-allele ambiguity **emerge on
its own** (sigmoid heads spreading across the equivalence class), as hypothesised — no equivalence
classes, no reverse-translation layer?

**Run:** 20,000 steps, batch 64, lr 3e-4, RTX 3090 Ti; 4,096-read held-out (junction-intact subset).

**What worked (the pipeline is sound):**
- J allele: nt_top1 **0.942** (7 J alleles are all protein-distinct → no synonymous ambiguity to express)
- Segmentation MAE **0.68–1.64 residues** (V/D/J start+end) — coordinates learn cleanly in AA space
- mutation MAE **0.024**, productive acc **0.999** — the meta heads work

**What did NOT happen — the ambiguity hypothesis is not confirmed at this budget:**
- V: nt_top1 **0.515**, protein-class_top1 **0.625**. Only **22.6%** of V nt-errors are within-class
  (synonymous); ~77% are real cross-class errors.
- Threshold-free diagnostic on the 1,550 ambiguous (multi-member-class) V reads: probability **mass on
  the true synonymous class = 0.071** (members ≈1.3% of alleles get ~7% of mass — elevated ~5× over
  uniform but far from concentrated); per-allele prob on a **non-member 0.057** (≈ members);
  **top-|class| recall 0.63**. The head produces a **diffuse low-confidence smear across all 198
  alleles**, not a spike on the equivalence class. `mean_pred_set_size` < 1 at τ=0.5 (under-confident).

**Verdict: the naive "let ambiguity emerge" approach did NOT produce crisp synonymous-set spreading**
at this budget **and under this loss** — but see §15: the diffuse output is substantially an artifact
of the BCE label-smoothing floor (0.05), so this does NOT prove the network intrinsically cannot
localize ambiguity.

**Load-bearing caveat:** the shipped DNA IGH model trained **~490k steps**; this ran **20k** (24× less).
The classification loss did plateau early (~0.30 by step 3k), so it is not purely step-starved, but the
absolute V numbers are a **floor**, and "ambiguity fundamentally cannot emerge" is **not** a supported
conclusion — only "it has not emerged at 20k steps, and the trend does not suggest it is about to."

**Cleanest next test (proposed, not yet run):** train the identical scheme with **nucleotide** input for
the **same 20k steps** and evaluate identically. Any V/D/J gap at equal budget = pure AA information
loss, with the under-training confound removed. Separately, this result *motivates* explicit ambiguity
modelling (equivalence-class targets, or the reverse-translation/superposition layer) since emergence
did not occur naturally.

## 13. Phase 1 — Learning-Proof Results & Gate Verdict (2026-07-18)

Frozen artifact: `docs/superpowers/artifacts/2026-07-18-aa-phase1/` (`proof.json`, `heldout.jsonl.gz`, `run.log`). 3 seeds × 3,000 steps × batch 64 on an RTX 3090 Ti; one frozen 4,096-read held-out cohort (fingerprint `56be2a9d8efb9c13…`); model 4,212,872 params; torch 2.11.0+cu128 / CUDA 12.8; git `944a1221`.

### Verdict: **PARTIAL** — 5 of 6 hard criteria pass; criterion 3 fails on 1 of 4 boundaries.

Per the predeclared rule, a partial result is **not** rounded up to GO. The failure is diagnosed below and is a **degenerate metric**, not a segmentation failure — but the gate was declared in advance and is not renegotiated after seeing results.

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Capacity (overfit ≥0.90 acc, all MAE <5) | V **1.000**, J **1.000**; MAEs 0.91/0.00/0.01/0.91 | **PASS** |
| 2 | Classification beats prior (paired lift CI>0, ≥0.05) | V **+0.926** CI[+0.918,+0.934]; J **+0.840** CI[+0.827,+0.851] | **PASS** |
| 3 | All 4 boundaries beat constant (CI>0, ≥10%) | v_end **+65.0%**, j_start **+85.1%**, j_end **+74.6%**; **v_start +0.0%** | **FAIL (1/4)** |
| 4 | Prior-learning guard (macro/micro ≥0.5) | V **0.980–0.992**, J **1.000** | **PASS** |
| 5 | Seed consistency (spread ≤0.10) | V acc 0.9495 / 0.9370 / 0.9409, spread **0.0125** | **PASS** |
| 6 | Hygiene | counters balanced (49,382 = 4,096 + 45,286); train ranges disjoint; no NaN; fingerprint present | **PASS** |
| 7 | Strata (diagnostic) | ambiguity 1→0.947, 2→0.945, ≥3→0.965; coverage: single bucket | *diagnostic* |

### Headline result

The model learns V/J protein-equivalence classes **far** beyond the prior: **V 0.949 vs a 0.023 majority baseline** (149 classes) and **J 0.987 vs 0.147**. Macro accuracy tracks micro almost exactly (0.98–1.00 ratio), so this is **not** frequency-prior learning — performance is spread across classes, not concentrated in common ones. Three of four boundaries beat their constant baselines by 62–86%.

### Why `v_start` failed — degenerate target, not a model failure

`v_start` is **exactly constant (0) in all 4,096 held-out reads** (1 distinct value). A constant predictor is therefore perfect by construction (baseline MAE **0.000**), so "beat the constant baseline by ≥10%" is **mathematically unsatisfiable** for a zero-variance target. The criterion is ill-posed here, not failed on merit.

It does expose one **real, minor limitation**: the model scores 0.925 MAE on a constant target, i.e. it cannot emit an exact edge value. The pattern is systematic — the two **extreme** boundaries (`v_start` at 0, `j_end` at n) both sit at ≈0.92 MAE, while the two **interior** boundaries are 0.21 (`v_end`) and 0.46 (`j_start`). This is **soft-argmax edge bias**: the expectation `Σ softmax(logits)·position` at slot 0 can only be ≥0, so any probability leakage biases it upward. Worth addressing in Phase 2 (e.g. an explicit edge-aware decode or a hard-argmax refinement).

### Scope limitation (important, and it explains both anomalies)

Phase-1 training and evaluation used **full-length reads only**. That single fact produces both oddities: every read starts at V residue 0 (⇒ `v_start` has no variance) and every read lands in the `coverage ≥80` stratum (⇒ the coverage stratification is uninformative, n=4096 in one bucket). **Consequently the Phase-0 coverage-conditioned identifiability finding — that short/truncated reads collapse protein classes — is entirely untested by this proof.** The 0.949 V accuracy is measured in the easiest possible regime and should not be read as a general-purpose number.

### Recommended path (for decision, not applied unilaterally)

1. **Preferred:** introduce read-length variation (truncated / fragment reads) in Phase 1b. This simultaneously gives `v_start` real variance (making criterion 3 well-posed), exercises the coverage strata, and tests the Phase-0 short-read identifiability collapse — the scientifically informative fix.
2. **Alternative:** formally declare zero-variance targets out of scope for the beat-the-baseline criterion. Legitimate as a spec correction, but on its own it is goalpost-moving and leaves the short-read question untested.
3. Independently: record the soft-argmax edge bias as a Phase-2 work item.

Data yield held at the Phase-0 estimate: 49,382 records generated → 4,096 usable (8.3%), rejections 42,426 non-productive + 2,860 stop-codon — the stop filter is doing real work, consistent with the Phase-0 finding that "productive" does not imply stop-free.

## 12. Phase 0 — Results & Gate Verdict (2026-07-18)

Frozen report: `docs/superpowers/artifacts/2026-07-18-aa-phase0/report.json` / `.md` (regenerate via `scripts/aa/phase0_report.py`, whose default `--out experiments/aa/phase0` is a gitignored scratch path) — 200,000 simulated `HUMAN_IGH_OGRDB` reads (s5f rate 0.05, seed 0, productive-only), germline DB hash `bf12d1517d0f2a29`. All 198 V and 7 J germlines passed anchor QC (0 quarantined).

**Verdict: GO to Phase 1.**

**Identifiability ceiling (the gate):**
- **V:** 198 alleles collapse to **149 protein-equivalence classes**. 115 alleles (58%) are uniquely identifiable (singleton classes); the rest fall into 34 multi-allele classes — 24 pairs, 6 triples, 3 quads, 1 quintet. **Max ambiguity = 5.** A protein V prediction therefore resolves to a nucleotide-allele set of median size 1 and worst case 5 — a workable ceiling, not a wall.
- **J:** 7/7 alleles are protein-distinct — full identifiability.
- **Coverage-conditioned (V prefix):** degrades gracefully — first 30 residues → 60 classes (0.10 unique), 60 → 118 (0.36), 90 → 137 (0.50), ~full (≥120) → the 149-class ceiling (0.58). N-terminal (FR1-side) fragments collapse many classes; near-full-length reads recover the ceiling. This makes coverage-aware, ambiguity-set reporting mandatory (as designed), not optional.

**Findings that validate / refine the design:**
- **Boundary masking is essential, not an edge case:** the V-end boundary straddles a codon in **81%** of reads and J-start in **68%**. The codon-provenance projection with mixed-provenance masking (§5.2) is load-bearing.
- **"Productive" is a junction-level contract, not a whole-sequence guarantee:** unknown-residue fraction is 0.0, but **0.4% of residues are stops even in productive reads** (stops outside the junction). Resolves the §11 stop-handling question: keep `*` as its own explicit token AND reject/flag stops at the input-contract layer — never fold stop into `X`.
- **Data yield:** only **14.1%** of simulated IGH reads are productive + in-frame with a derivable AA sequence. Simulation is cheap so this is not a blocker, but Phase 1 must over-generate ~7× to hit a target productive count.
- **D is less hopeless than assumed:** reads retain a **mean 5.6 fully-owned D residues**, and only 0.28% have zero owned D residues — more surviving D content than the pessimistic §5.3 estimate. D-as-source-classification (Phase 1b) is worth pursuing; owned-residue content ≠ allele discriminability, so the source-classification framing stands. The prototype vs prototype-free choice remains open for the 1b experiment.
- **Length (see also §13):** AA length p50/p95 = **126/133** residues, **max 146** (corrected 2026-07-18: an earlier revision of this section wrongly reported max=133 by conflating p95 with the maximum). The provisional `max_seq_length=192` still has comfortable headroom; no need to revisit upward.
