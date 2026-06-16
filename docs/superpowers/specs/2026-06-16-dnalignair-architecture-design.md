# DNAlignAIR — Unified DNA Alignment Architecture — Design

**Date:** 2026-06-16
**Status:** Approved (architecture + decomposition); build R1→R4, iterate empirically once trainable.
**Supersedes:** the Phase-1 model (`core/base.py`, `nn/heads.py`, `losses/hierarchical.py`) and the
single/multi-chain split. Reuses/adapts: tokenizer, serialization bundle, trainer skeleton, GenAIRR
synthetic infra.
**Parent:** `docs/superpowers/specs/2026-06-15-alignair-pytorch-migration-design.md`

## 1. Goal & success criteria

Build one unified **`DNAlignAIR`** model that aligns IGH/TCR **DNA** sequences and, with no
post-processing, predicts the full alignment: V/D/J (or V/J) allele calls (multi-label), exact
in-sequence start/end, exact germline start/end, per-position state (germline / SHM-mutation /
sequencing-noise / indel), and auxiliary fields (mutation rate, productivity, indel count). It handles
input orientation internally, supports a genotype restriction, and is trained on a continuous GenAIRR
"gym."

**Near-term success:** the architecture converges and gives good standalone results (low boundary
deviation ≈ 0, high allele-call agreement, calibrated per-position states).
**North star (evaluated later):** beat IgBLAST, MiXCR, and partis on all metrics.

## 2. Decisions (confirmed)

| Topic | Decision |
| --- | --- |
| Model | **One unified `DNAlignAIR`** (no single/multi-chain split). |
| Reference / classification | **Allele-embedding matching**: encode each germline allele → vector; score a segment representation against candidates by similarity; multi-label sigmoid. |
| Multi-dataconfig | Build a **union `ReferenceSet`** from 1..N GenAIRR dataconfigs; the matching candidate set is that union. |
| Genotype | A **candidate-row mask** at inference (subset of alleles). Train on full union + optional random genotype subsets. No retraining per genotype. |
| Segmentation | **Per-position region tagging** (full resolution) for exact in-sequence boundaries; monotonic decode for V→D→J order. |
| Germline coordinates | **Input ↔ matched-germline cross-attention** → exact germline start/end (+ optional full per-position alignment). |
| Backbone | **Conv stem + Transformer** (full per-position resolution). |
| Orientation | **In-model 4-class orientation head → canonicalize → re-encode.** Orientation is an output. |
| Allele calls | **Multi-label** (GenAIRR-indistinguishable allele sets are all positive). |
| Aux outputs | per-position state, mutation rate, productivity, indel count — **all included**. |
| Match training negatives | **Full reference** as candidates each step (references are a few hundred alleles). |
| Input representation | **Right-pad + attention mask** (drop center-padding); coordinates in true sequence space → no post-processing. |
| Training | **GenAIRR "gym"**: online generation, curriculum over corruption, multi-dataconfig sampling; GenAIRR is trusted ground truth. |
| Loss | Composite, **Kendall uncertainty-weighted** across heads. |

## 3. Architecture

### 3.1 Input representation
- Tokens: A/C/G/T/N/pad; IUPAC ambiguity codes → N.
- **Right-pad** to batch max length + an **attention mask**; no center padding. All predicted
  coordinates are in true sequence space (0..len), so nothing needs un-padding downstream.

### 3.2 `ReferenceSet` (reference & genotype subsystem)
- Constructed from 1..N GenAIRR dataconfigs: per-gene **union** allele name list + each allele's
  germline nucleotide sequence (tokenized). Tracks which alleles came from which chain/locus.
- A **`GermlineEncoder`** (shared 1D conv + light attention) maps each allele germline sequence →
  (i) a pooled embedding `e_a ∈ R^d` (for matching), (ii) per-position germline reps `G_a` (for
  cross-attention alignment).
- Allele embeddings `E_g` (per gene V/D/J) are **cached** per dataconfig; recomputed during training
  so gradients flow into the germline encoder; precomputed once at inference.
- **Genotype** = restrict candidate rows of `E_g` to the provided allele subset — a pure inference-time
  mask. Training optionally samples random genotype subsets for robustness.

### 3.3 Backbone (input encoder)
- Light first-pass embedding → **orientation head** (4 classes: identity / reverse-complement /
  complement / reverse) → canonicalize tokens to forward orientation → main encoder. The chosen
  transform is emitted as `orientation`.
- **Conv stem** (local k-mer motifs, full resolution) → **Transformer** encoder (long-range V→D→J
  arrangement) → per-position representations `H ∈ (B, L, d)`. Attention respects the pad mask.

### 3.4 Heads (all derived from `H`, no post-processing)
1. **Region tagging** — per-position softmax over `{pad, pre, V, N1, D, N2, J, post}`. Exact
   in-sequence boundaries = contiguous runs; monotonic decode enforces order. Light chains never emit
   `D`/`N2` (region set adapts to chain via the ReferenceSet / a learned gate).
2. **Per-position state** — per-position softmax over `{germline, SHM-mutation, sequencing-noise,
   insertion, deletion}`. `noise-base count = Σ noise` (cleanly separated from SHM and indels).
3. **Allele matching (multi-label)** — per gene, pool the gene's region positions → segment rep
   `r_g`; `scores = E_g · r_g`; sigmoid → multi-label allele set. Genotype restricts candidates.
4. **Germline coordinates** — each gene's input positions cross-attend to the matched allele's `G_a`
   → exact `germline_start/end` (optionally the full per-position alignment matrix, supervised by
   GenAIRR `germline_position()`; handles indels).
5. **Auxiliary** — `mutation_rate`, `productivity` (frame/junction), `indel_count` (aggregated from
   the per-position state head).

### 3.5 Loss (composite, Kendall uncertainty-weighted)
Per-task terms, each scaled by a learned uncertainty weight:
- orientation CE,
- region per-position CE (class-weighted; padding ignored),
- per-position state CE,
- multi-label allele BCE over the full candidate set (true allele set = positives),
- germline-coordinate loss (soft-target CE on start/end; optional alignment-matrix CE),
- in-sequence boundary sharpness (soft-target CE on segment start/end derived from region runs),
- aux: mutation-rate (regression), productivity (BCE), indel-count (from state).
Optional contrastive/InfoNCE term to sharpen the allele-embedding space.

### 3.6 Training gym
- Build `ReferenceSet` from 1..N dataconfigs.
- Online producer: per step, sample dataconfig(s) and a **corruption profile from a curriculum**
  (clean → progressively harder: SHM rate, 5'/3' end loss, polymerase indels, sequencing errors,
  ambiguous N's, random strand orientation). Stream GenAIRR `Outcome`s and extract **per-position
  ground truth from provenance** (`final_simulation().regions()`, `.germline_position()`, and the
  `events()` trace separating `mutate.s5f` / `corrupt.quality` / `corrupt.indel`), plus the
  multi-allele call sets and the applied orientation.
- Full reference as matching candidates. Feeds the trainer as an `IterableDataset`.
- NOTE: if the provenance trace is awkward to turn into clean per-position masks, add a small
  position-level labels accessor to GenAIRR (locally owned).

## 4. Build decomposition (each: its own spec → plan → build, iterate empirically)

- **R1 — Reference & matching keystone.** `ReferenceSet` (1..N dataconfigs → union alleles + germline
  seqs), `GermlineEncoder`, `AlleleMatchingHead`. *Done when:* on clean germline-derived segments the
  matching head identifies the correct allele(s) (multi-label) with high accuracy; genotype restriction
  works as a candidate mask.
- **R2 — Backbone + orientation + region tagging.** Conv-stem+Transformer encoder, orientation
  head + canonicalize, per-position region head. *Done when:* exact in-sequence boundaries on
  corrupted/oriented inputs with deviation ≈ 0 on clean and small on noisy.
- **R3 — Germline coordinates + per-position state.** Cross-attention aligner → germline start/end;
  per-position state head. *Done when:* germline coords match GenAIRR labels; state head separates
  noise/mutation/indel.
- **R4 — Gym + composite loss + end-to-end training/eval.** Curriculum online generator with
  provenance GT extraction, the full Kendall-weighted loss, full-model training, and an evaluation
  harness (boundary deviation, call agreement, per-position metrics) — later extended to compare vs
  IgBLAST/MiXCR/partis.

## 5. Relationship to existing code

- **Replaced:** `core/base.py`, `core/single_chain.py`, `core/multi_chain.py`, `nn/heads.py`,
  `losses/hierarchical.py`, the single/multi `ModelConfig` chain fields.
- **Reused/adapted:** `data/tokenizer.py` (extend to right-pad + mask), `data/encoders.py` (allele
  vocab → ReferenceSet), `serialization/` (bundle now stores ReferenceSet + germline encoder + the new
  config), `training/` (trainer loop, AMP, callbacks — extend for the gym), the GenAIRR synthetic infra
  (`experiment_presets`, `synthetic`, `genairr`) — extend with provenance extraction and curriculum.
- The current `inference/predict_calls` path is superseded by the no-post-processing model outputs.

## 6. Risks / iterate-on items

- **Cross-attention germline alignment** is the most novel piece; exact germline coords may need
  iteration (start with germline start/end regression, add the full alignment matrix if needed).
- **Per-position state GT** depends on GenAIRR provenance being cleanly extractable; may require a small
  GenAIRR-side labels accessor.
- **Region-tag → boundary exactness**: monotonic decoding and class weighting need tuning to hit
  deviation ≈ 0.
- **Embedding-space collapse**: the matching head may need a contrastive term / temperature tuning.
- **Curriculum schedule** is empirical; start simple (linear ramp) and adjust from loss curves.
- Compute: Transformer at full resolution + germline encoding per step — monitor cost; cache
  aggressively, consider mixed precision.

## 7. Out of scope (for now)

- Protein/amino-acid input (DNA only for now).
- Beating the incumbents is the north star but evaluated only after standalone convergence.
- CLI / hub / reporting (separate Phase-4-style work) and full AIRR-gapped table assembly (thin
  deterministic layer once germline coords are emitted).
