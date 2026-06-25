# Structural Seed-and-Extend Neural DP

**Date:** 2026-06-25
**Status:** design approved (incl. Codex adversarial review), pending spec review → implementation plan
**Sub-project:** B (germline-aligner redesign) — supersedes the "replace the soft-DP with a pointer head" plan

## 1. Problem & context

The differentiable soft-DP germline aligner (`src/alignair/nn/soft_dp_aligner.py`) is the
accuracy gold standard (it marginalizes over alignment paths incl. indels/SHM) but is the
runtime bottleneck: a **sequential `for i in range(S)` recurrence** (S≤576) launching ~184k
tiny logsumexp kernels — **52–94% of runtime, launch-bound not FLOP-bound**.

What we learned this sub-project (all empirical, recorded in memory `aligner-parallel-scan-direction`):
- A fully-parallel single-diagonal "pointer head" (`nn/pointer_aligner.py`) is 3.5–5.8× faster
  but **structurally ~0.1 less accurate on coordinates**, and the gap does NOT close with 6×
  more training. A diagnostic decomposition proved the regression is **coordinates only** — the
  pointer's allele *reader* matches/beats the soft-DP. Root cause: a rigid diagonal cannot
  marginalize over indel/SHM paths.
- **A soft-DP forward is an associative scan** → parallelizable to O(log S) depth (Särkkä
  arXiv:2102.05743; Torch-Struct arXiv:2002.00876). So the exact DP can be made fast without
  approximating it.
- **Band-sweep experiment** (`scripts/exp_band_sweep.py`, on the 8h-trained soft-DP): restricting
  the EXACT DP to an oracle band of width w preserves coord accuracy at even **w=8** (clean 1.000,
  heavy_shm_fulllen 0.980, indel 0.995 = full DP; junction 0.645@w=8 > 0.590@full — a tight band
  regularizes away spurious matches). So O(w·L), w~8–16, is bit-safe **if the band is centered
  correctly**.
- A predicted band via diagonal-argmax on soft-DP reps FAILED (center error ~290nt) — soft-DP reps
  are not organized for diagonal correspondence. **The band predictor must be a trained component.**

This design was hardened by an adversarial Codex review whose corrections are baked in below
(structural band predictor not MaxSim-argmax; DP log-partition is the final reader; fused kernel
before associative scan; top-k+chunked MaxSim memory; fix the coord-DP base-match drop; remove the
classifier path).

## 2. The architecture — Structural Seed-and-Extend Neural DP

**MaxSim is supporting evidence; the exact DP is the geometry and the source of truth.** Selected by
`config.aligner="seed_extend"` (alongside the retained `"softdp"`/`"pointer"`/`"diagonal"`).

```
read (≤576nt) ─┐
               ├─► SHARED type-embedded encoder E ──► read reps R[L,d]
germline refs ─┘   (same E; refs encoded on ingest, CACHED @inference, REFRESHED @train) → G_a[M_a,d]
        │
        ├─ segmentation head on R → V/D/J/N labels                 (segmentation-first; gates all below)
        ├─ pooled-cosine retrieval (R̄ vs cached Ḡ_a) over ALL runtime alleles → top-k   (cheap shortlist)
        ├─ top-k MaxSim grid S_a[i,j]=cos(R_i,G_a[j]) (chunked)     → shortlist refine + BAND FEATURES only
        ├─ STRUCTURAL diagonal-offset band head → P(offset) / top-m bands ±w   (fail-open on low conf.)
        └─ fused banded EXACT soft-DP on top-k (base-match + SHM-reliability inputs):
                 → calibrated start/end coord posteriors
                 AND  log-partition = the FINAL allele rerank score
   + classical parasail raw-nucleotide affine aligner kept as audit / fallback / teacher (novel alleles)
```

The two hard problems are the same object twice — the reader's similarity is the alignment's
scoring evidence — but the **DP**, not MaxSim, defines coordinates and the final call. "Seed" =
the structural band head (cheap, parallel); "extend" = the exact banded DP inside the seed.

## 3. Hard rules (first-class requirements; the design is non-compliant if any is violated)

1. **The final allele score is ALWAYS the exact DP log-partition, never MaxSim.** MaxSim only
   shortlists and supplies band features.
2. **The band predictor is supervised on offset/band recall** — emitting a **distribution over
   offsets or top-m candidate bands**, not a scalar regression.
3. **Low-confidence banding fails OPEN to a wider/full DP**, never to a wrong narrow band.
4. **Base-match and SHM-reliability are MANDATORY DP inputs** (fixes the current coord-DP drop at
   `dnalignair.py:201` try/except fallback; load-bearing for novel alleles).
5. **`GermlineEncoder` and `caller="classifier"` are REMOVED from the compliant path** (the
   classifier memorizes allele identity → violates dynamic-genotype).
6. **`top-k`, `w`, and `top-m` are defaults to VALIDATE, not assumptions to bake in** — every one
   is swept and chosen by measurement.
7. **The old full soft-DP remains the A/B oracle** until frozen-lattice competence AND coordinate
   parity clear bootstrap CIs. The new path ships only when it matches/beats the oracle and is faster.

## 4. Components

### 4.1 Shared type-embedded encoder
One weight-shared tower embeds both read and germline references. `SharedNucleotideEncoder`
already carries read/germline **type embeddings** (`nn/encoder.py:88-121`) — reuse them (a read is
a noisy/mutated reference, not a different modality; no biological reason for separate encoders).
Delete the separate `GermlineEncoder` (`dnalignair.py:85`) and all segment re-encoding
(`dnalignair.py:180-183`, `germline_tf.py:29-30`, `dnalignair_infer.py:299-300`). References are
encoded **on ingest and cached** (dynamic-genotype by construction; novel alleles encoded on the
fly); ref reps are **refreshed during training** so they don't go stale (the model trains, the
cache must follow). Keep a **raw base-match channel** alongside learned reps (novel-allele floor).

### 4.2 Hybrid retrieval (cheap shortlist)
Pooled-cosine of the masked-mean read-segment rep vs cached pooled allele reps, over ALL runtime
alleles → top-k. Cheap (one matmul), preserves dynamic-genotype. `caller="retrieval"` only; the
classifier path is removed.

### 4.3 MaxSim grid (top-k only, chunked — evidence, not geometry)
For the top-k candidates only, compute the token-level grid `S_a[i,j]=cos(R_i,G_a[j])`. Memory: at
B=64, L=576, top-k=16 the grid is ~340M scores (~0.68 GB fp16) — **must be chunked/streamed**;
all-allele grids (~8.4 GB fp16) are infeasible and forbidden. Two consumers: (a) shortlist
refinement (ColBERT MaxSim `Σ_i max_j`), (b) **features for the band head**. It does NOT define
coordinates or the final call (rule 1).

### 4.4 Structural diagonal-offset band head (the "seed")
A supervised head that predicts where the alignment diagonal sits, robust where MaxSim's order-light
argmax fails (heavy-SHM/indel/junction/FR-CDR repeats). Inputs aggregated over offsets `o`: token
cosine along the o-diagonal, **raw base-match** along it, k-mer/minimizer seed hits, and
segment-boundary confidence (a Hough-style vote over offsets). Output: a **distribution `P(offset)`
or top-m bands** (rule 2), each ±w. Trained directly on band recall `P(|center−true_start|≤w)`.
**Fail-open** (rule 3): when the top band's confidence is low, widen w or run the full DP rather
than commit to a narrow wrong band.

### 4.5 Fused banded exact soft-DP (the "extend" — geometry + final reader)
The *same* affine-gap logsumexp recurrence (mathematically equivalent within fp tolerance; tree
reduction reorders logsumexp so not bit-identical), restricted to the seed's ±w band, taking
base-match + SHM-reliability as inputs (rule 4). Emits calibrated start/end coordinate posteriors
**and** its log-partition as the final allele rerank score (rule 1). Execution path, in risk order:
1. **Exact banded *sequential PyTorch* reference** — prove parity vs full soft-DP under oracle bands.
2. **Fused banded Triton/CUDA wavefront kernel** (loop inside one kernel) + custom or recompute
   backward — the production speed win, O(w·L). This is the lowest-risk kernel for S≤576.
3. **(Later, optimization only)** transfer-operator associative scan: the affine DP is associative
   ONLY as a semiring **matrix** composition over the full (M,I,D)×band state (naive scalar
   (logsumexp,+) is NOT associative), inflating work to O(L·w²–w³). Deferred — it is a reward for a
   working fused kernel, not a first step.

### 4.6 Classical parasail fallback / teacher
Keep the raw-nucleotide affine aligner (`io/alignment.py`, already shipped) as an **audit, fallback,
and distillation teacher** for top-k candidates — especially novel alleles and fail-open cases. AIRR
expects explicit alignment coordinates/CIGAR evidence; parasail provides a non-neural ground check.

## 5. Validation — the multiplicative gates (the spec's backbone)

The real failure mode is **multiplicative**: `deployment_quality ≈ retrieval_recall@k ×
band_recall | (true allele ∈ top-k) × fail-open_correctness`. A kernel that looks perfect in
isolation can still regress deployment. So promotion is staged and each gate must clear before the
next investment:

- **Gate 0 — banding math (DONE, green):** oracle-band sweep preserves coord accuracy at w=8
  (`scripts/exp_band_sweep.py`). Recorded.
- **Gate 1 — GEOMETRY gate (cheapest; do FIRST, before any kernel):** freeze the 8h soft-DP, train
  ONLY the structural band head on **true-region / true-allele** segments. Metric: **band recall
  `P(|center−start|≤w)` at w=8,16 per lattice cell** (clean, heavy_shm_fulllen, indel,
  junction_boundary). **If recall misses >0.5–1% at w=16, STOP — do not build the kernel.** This is
  an isolated test of band geometry only.
- **Gate 2 — PIPELINE gate:** repeat with **predicted region + retrieval top-k** (not oracle).
  Report the multiplicative chain: retrieval recall@k, band recall conditioned on true allele ∈
  top-k, and fail-open behavior (does it correctly widen rather than commit wrong?). This catches
  the compounding failure Gate 1 hides.
- **Gate 3 — FINAL gate:** full model with the fused banded DP, trained from scratch, A/B'd on the
  frozen lattice vs the soft-DP oracle. Promote only when **frozen-lattice competence ≥ soft-DP
  (bootstrap-CI lower bound) AND coordinate parity AND faster at B=64**.

## 6. Build order (kernel is the reward for passing Gate 1, not the default next step)

1. **Gate 1 band-head experiment** (frozen soft-DP, true region/allele, band recall by cell). Decision gate.
2. **Encoder refactor:** shared type-embedded encoder; delete `GermlineEncoder` + segment re-encode +
   `caller="classifier"`; preserve raw base-match. Verify retrieval recall@k + coordinate parity vs
   current soft-DP unchanged.
3. **Exact banded sequential PyTorch DP** + base-match/reliability inputs; prove parity vs full soft-DP
   under oracle bands (Gate-0 reproduced in the new code path).
4. **Fused banded Triton/CUDA DP** + backward; validate numerical parity, then profile speed.
5. **Learned band predictor** with top-m / fail-open wired into the pipeline; **Gate 2** (predicted
   region + top-k, multiplicative recall).
6. **Gate 3** full A/B on the frozen lattice (competence + parity + speed).
7. **(Optional, later)** transfer-operator associative scan as a further speedup.

## 7. Constraint compliance
- **Dynamic genotype:** references are runtime inputs, encoded + cached; novel alleles callable via
  the shared encoder + raw base-match channel + parasail fallback; NO allele identity in weights
  (classifier path removed). MaxSim/retrieval/band/DP all operate over runtime-encoded reference reps.
- **Segmentation-first:** pooled retrieval, MaxSim, band head, and DP all operate on the
  segmentation-gated V/D/J segment, not the full read; segmentation runs first and gates them.

## 8. Risks
- **Band-head recall** (Gate 1) is the make-or-break unknown — explicitly gated first.
- **Multiplicative compounding** (Gate 2) — retrieval × band × fail-open; measured before the final build.
- **Triangle of fp parity** — "exact" means within tolerance (tree-reduced logsumexp), not bit-identical;
  parity tests assert a tolerance, with an optional sequential-order mode for strict checks.
- **Triton backward** — custom/recompute backward must be validated against autograd on the sequential
  reference before trusting gradients in training.
- **Memory** — MaxSim must stay top-k + chunked; banded DP at w=16 is ~39 MB (fine). Guard against
  accidentally materializing all-allele grids.
- **Fail-open cost** — too many low-confidence reads falling open to full DP erodes the speed win; track
  the fail-open rate as a first-class metric.

## 9. File map (for the implementation plan)
- `src/alignair/nn/encoder.py` — promote `SharedNucleotideEncoder` (type embeddings) to the single
  encoder; ensure it serves both read and reference.
- `src/alignair/core/dnalignair.py` — delete `GermlineEncoder` instantiation + `caller="classifier"`;
  route reference + segment encoding through the shared encoder; add `aligner="seed_extend"`.
- Create `src/alignair/nn/band_head.py` — the structural diagonal-offset band predictor (P(offset)/top-m,
  fail-open).
- Create `src/alignair/nn/seed_extend_aligner.py` — banded sequential DP reference + fused-kernel wrapper;
  base-match/reliability inputs; log-partition reader.
- Create `src/alignair/nn/banded_dp_triton.py` (or `.cu`) — the fused wavefront kernel + backward.
- `src/alignair/nn/matching.py` — hybrid retrieval (pooled top-k) + top-k chunked MaxSim (features only).
- `src/alignair/losses/dnalignair_loss.py` — band-recall loss term; keep soft-argmax/CDF/consistency coord loss.
- `src/alignair/training/germline_tf.py` — feed base-match/reliability into the new DP; remove re-encode.
- `src/alignair/config/dnalignair_config.py` — `aligner="seed_extend"`, `band_width w`, `band_top_m`,
  retrieval `top_k`, fail-open threshold (all sweepable).
- Experiments: `scripts/exp_band_recall_gate.py` (Gate 1/2), reuse `scripts/exp_band_sweep.py` (Gate 0),
  `scripts/exp_aligner_ablation.py` + frozen-lattice gym (Gate 3).
- Keep: `nn/soft_dp_aligner.py` (A/B oracle), `io/alignment.py` (parasail fallback/teacher),
  `gym/instrument/` (validation).
