# DNAlignAIR Coordinate-Subsystem Redesign — Synthesis

Synthesis of two independent architecture reviews (accuracy lens + speed lens) of the
DNAlignAIR model, 2026-06-22. Both lenses converged on the same core redesign.

## Goal & current state
Win condition: GENERALLY faster AND more accurate than IgBLAST/partis/MiXCR, preserving
(1) dynamic genotype (reference as input, novel alleles callable, nothing allele-specific in
weights) and (2) segmentation-first. Measured today (4400-case/22-stratum/bootstrap-500):
we WIN 21/24 metrics but are **~2.3x SLOWER** (101 vs 238 reads/s); the only real accuracy loss
is `junction_nt_exact` (0.53 vs 0.95), which decomposes to ±1–2nt boundary jitter (J ~1.5x worse
than V).

## Verified grounding (the switches already exist)
- `config.region_decoder`: `"linear"` (current, RegionTagger) | `"query"` (RegionMaskSpanDecoder →
  per-gene start/end **boundary posteriors** with direct NLL supervision). EXISTS, wired into the
  loss, currently OFF (so `out["boundary"]` is None on scaled_long).
- `config.aligner`: `"softdp"` (current, the bottleneck) | `"diagonal"` (legacy cosine corr).
- `config.backbone`: `"conv"` (current) | `"shared"` (RoPE/SDPA/SwiGLU).
- `config.caller`: `"retrieval"` (current, dynamic-genotype-safe) | `"classifier"` (REJECTED — memorizes alleles).
- `training/reader.py::reader_novel_positive` (novel-allele training) and `losses/dnalignair_loss.py` exist.

## The single root cause (both lenses)
Three mechanisms independently answer the same question — "where does this segment sit in the
germline?": (a) region-tagger argmax → in-read coords, (b) soft-DP forward → germline coords,
(c) soft-DP `alignment_score` → allele rerank. They are NOT self-consistent (in-read vs germline
coord error corr only +0.44 V / +0.69 J) → the uncorrelated residual IS the junction jitter. And
the soft-DP's sequential `for i in range(S)` recurrence (`soft_dp_end_logits`) is **66% of
inference** (germline-coord decode 47% + rerank 25%); the neural net is only 18%.
**Fix: collapse to ONE self-consistent, GPU-native coordinate mechanism.**

## Recommended architecture (convergent across both lenses)
1. **GermlinePointerHead** (replaces soft-DP forward for V/J): `q = pooled segment backbone rep`,
   scored by dot-product against the runtime allele's germline per-position reps `G` → start/end
   logits over germline positions. Two matmuls, no recurrence, O(1) in germline length. Same CE
   supervision. Predict start + length for V/J.
2. **Cheap D-only aligner**: D is short (S~20), invertible, can have indels — keep a banded
   diagonal scan (`as_strided` + sum) or the existing soft-DP (already fast at S~20).
3. **Activate `region_decoder="query"` + boundary loss**: sharp, directly-supervised in-read
   boundary posteriors. KEY: when in-read (query decoder) and germline (pointer head) coords both
   derive from the SAME backbone-pooled query, their errors correlate → cancel in the junction
   difference (corr→1). This is the `junction_nt_exact` fix at the architectural level.
4. **Junction reconstruction loss**: directly supervise the predicted junction STRING vs truth →
   forces the two coordinate heads to agree. Direct lever on the metric.
5. **Vectorize the Python loops** (`extract_segment`, `extract_segment_tokens`, `decode_boundaries`)
   and drop the `soft_dp_aligner.py:66` `.clone()`. Free ~10%, correctness-neutral, **NO retrain**.
6. **V reader** (the one disagreement, RESOLVED): score each top-k V candidate at its PREDICTED
   offset by **RAW-nucleotide base match, GPU-vectorized** (`torch.diagonal` at the offset). V is
   offset-only (no indels), so this is the Smith-Waterman math done on GPU — combining the raw-base
   discrimination that hits 0.965 and is novel-allele-safe (accuracy lens) with GPU speed that
   avoids the per-pair CPU bottleneck (speed lens). Decide CPU-SW vs GPU-diagonal by a parasail/edlib
   throughput microbenchmark.
7. **Backbone**: keep the Transformer (both lenses reject TCN — global attention matters for the
   V→D→J context the region head uses; a TCN wouldn't be faster here). Optional later:
   `backbone="shared"` + larger batch for GPU utilization.
8. **Optional last**: germline-denoising / SHM-inversion head (per-position learned reconstruction
   of the ancestral germline base — NOT a hand-coded mutation model) as an extra V-accuracy lever;
   feed the denoised sequence as a second query to retrieval/rerank.

## Property preservation
- Pointer head + rerank score against RUNTIME-encoded germline reps and a floored raw-base channel
  (`match_floor`) ⇒ novel alleles are just new inputs; `reader_novel_positive` training applies;
  `caller` stays `"retrieval"`. → DYNAMIC GENOTYPE preserved.
- All germline computations remain DOWNSTREAM of region segmentation. → SEGMENTATION-FIRST preserved.

## Phased A/B plan (cheap→expensive; reuse `scripts/run_h2h_benchmark.py`, `scripts/profile_inference.py`, `benchmark.cli compare`)
- **Phase 0 (NO retrain):** vectorize loops; drop the DP `.clone()`; raw-nucleotide V rerank at the
  predicted offset (the raw-SW `rescore_alleles` already exists); parasail/edlib microbenchmark.
  A/B vs current + IgBLAST on the frozen 4400-case fixture. Target: V call 0.745→~0.80, throughput up,
  junction unchanged. Validates direction before any retrain.
- **Phase 1 (retrain, warm-start backbone/tagger/matching from scaled_long):** `aligner="pointer"`
  (V/J) + cheap D aligner + `region_decoder="query"` + boundary loss + junction reconstruction loss.
  Target: ≥350 reads/s, `junction_nt_exact` ≥0.75, maintain 21/24, no regression on fragments/D/J/orientation.
- **Phase 2 (optional):** denoising head; `backbone="shared"`; batch scaling.

## A/B variants
- **A** = current (`softdp`/`linear`/`learned`, scaled_long).
- **B (Phase 0)** = current weights + vectorized loops + raw V rerank at predicted offset (NO retrain).
- **C (Phase 1)** = retrained `pointer`/`query` model with junction loss.

## Guardrails (must not regress)
V call ≥0.72; D/J big wins held (≥ current −0.02); all germline MAE ≤ current +1nt; fragment strata;
orientation ≥0.98; overall ≥21/24 metrics.

## Decisive cheapest experiments
- Phase 0: 3-line change to use raw V rerank + run `run_h2h_benchmark.py` on the fixture (one afternoon).
- Parasail/edlib microbench: ~80k short V-pairs CPU time vs the current GPU rerank cost (~30 min) —
  decides CPU-SW vs GPU-diagonal for the reader.
- Phase 1: `boundary_posterior_probe.py` on the retrained checkpoint (V-end exact 0.50→≥0.60 ⇒ junction follows);
  `profile_inference.py` (germline-coords stage 47%→<15% ⇒ speed target met).
