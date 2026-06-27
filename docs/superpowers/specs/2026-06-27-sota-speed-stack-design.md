# DNAlignAIR SOTA Redesign — Design

**Status:** approved design, ready for implementation planning (sub-project 1).
**Date:** 2026-06-27

## Vision (full SOTA target)

DNAlignAIR should be both **SOTA-accurate** on the benchmark assay and **SOTA-fast** (thousands of
reads/s, not tens). Profiling and probes (this session) localized the two problems and proved they
are independent:

- **Speed** is bottlenecked by the **differentiable germline-coordinate DP** (and the per-candidate
  DP reader), *not* the encoder. Measured (d64, RTX 3090Ti): encoder 2559 reads/s; +retrieval
  1336/s; +germline-coord DP decode collapses to 162/s (+5.4 ms/read); full DP-reader predict 77/s.
- **Accuracy** on hard cases is bottlenecked by the **encoder representation** of heavily-mutated
  sequences: heavy-SHM V set-aware retrieval@16 ≈ 0.42 — the true allele set is absent from the
  top-16 ~58% of the time, a ceiling no reader can beat. A learned cross-attention head over the
  frozen encoder neither beat pooled retrieval nor raised this ceiling; a structured aligner (DP
  reader) extracts more (0.27 vs 0.08 on heavy-SHM) but cannot exceed the retrieval ceiling.

The full target architecture: **neural for representation/segmentation/candidate-retrieval,
classical for exact alignment (allele discrimination + coordinates).**

This vision decomposes into two independent sub-projects:

1. **Speed stack** (this spec) — replace the differentiable DP with classical WFA2 alignment over
   neural top-k; remove the soft-DP/seed_extend/band-head subsystem from inference *and* training;
   retrain on segmentation + retrieval losses; accelerate the encoder. Low-risk, ships the 10×+
   speed win, accuracy guardrail = no regression.
2. **Pretrained encoder** (later spec) — raise the heavy-SHM retrieval ceiling (0.42) via an
   LLM-style pretrained nucleotide encoder. The accuracy research bet.

## Scope of this spec: Sub-project 1, the speed stack

Target: **aggressive 10k+ reads/s** (phased). Changes **both** inference and training (clean
end-state, not an inference band-aid): retrain the encoder on the simplified loss set.

### Required properties preserved (from `dnalignair-required-properties` memo)

- **Dynamic genotype/germline**: reference is an INPUT; novel alleles handled; add/remove anytime;
  never memorized. Retrieval-not-classifier. Segmentation-first.
- Verified by the evaluation agent: `encode_reference` re-embeds arbitrary germline strings every
  forward (no retraining); WFA aligns raw germline strings (no learned per-allele parameter). The
  swap requirement holds — see "Swap robustness" below.

## Architecture

```
reads ─► encoder (fp16/compiled) ─► segmentation (read-segment query start/end)
                                 └─► retrieval ─► top-k (K≥32) candidate alleles / gene ──┐
raw germline k-mer/seed prefilter ─► admit divergent candidates (genotype-masked) ────────┤
                                                                                          ▼
                       WFA2 align(read_segment, candidate_germline) over the UNION pool
                          ├─ best score      ─► allele call + score-band equivalence set
                          └─ traceback       ─► germline start/end, CIGAR, trims
                                                                                          ▼
                                                                          assemble AIRR record
```

**Neural/classical split.** Neural: encoder, segmentation (query coords — already near-perfect,
soft-DP-independent via the region/boundary head), retrieval (top-k candidates). Classical: allele
discrimination *within the pool* and all germline coordinates/CIGAR/trims, from one WFA pass.

**Removed** from hot path + training graph: the entire differentiable soft-DP / seed_extend /
band-head reader+coord subsystem.

**Lower-risk than it looks** (agent-confirmed): much is already prototyped —
`io/alignment.py:realign()` already does classical parasail alignment overriding the soft-DP
germline coords; `raw_set_band` already builds score-band equivalence sets for V. WFA2 is a backend
swap of an existing role, generalized to all genes.

## Swap robustness (the load-bearing property)

WFA can only choose among the candidates it is given. The **union pool** (retrieval top-k ∪ k-mer
seed candidates) is what makes "swap to any germline" robust:

- **retrieval top-k** — the encoder's learned ranking; strong for trained/near-neighbor alleles.
- **seed prefilter** — non-learned raw k-mer overlap; admits a *divergent novel* allele into the
  WFA pool even when pooled cosine misranks it.

A new allele can win by being **retrieved or seeded**, so the swap requirement no longer hinges
solely on encoder recall (the weakest link identified by the evaluation agent). Gate: the standing
`embargo_retrain.py` held-out-recall eval must stay ≈ control.

## Components

New `src/alignair/align/` package (classical, non-neural):

| file | responsibility | interface |
|---|---|---|
| `backend.py` | aligner protocol + result type | `align(query, target, mode="sg_dx") -> AlignResult(score, cigar, q_start, q_end, t_start, t_end)` |
| `wfa.py` | WFA2 backend via `pywfa`; query-global, germline-ends-free | implements `backend` |
| `parasail.py` | existing `io/alignment.py:realign()` logic, moved, as fallback | implements `backend` |
| `seed_prefilter.py` | non-learned k-mer index over reference germlines | `SeedPrefilter(ref).candidates(segment, gene, m) -> list[idx]` |
| `batch.py` | multithreaded chunked WFA over (read·gene·candidate) pairs | `align_batch(pairs) -> list[AlignResult]` |

Edited:

| file | change |
|---|---|
| `inference/dnalignair_infer.py` | rewrite `predict_reads` to the union-pool + WFA path; delete the differentiable-DP reader branch and `decode_germline_coords`/`compute_germline_logits` calls; preserve output contract + `genotype` arg |
| `training/gym_trainer.py`, `training/germline_tf.py` | remove `band_offset_loss`, `reader_scores_banded` set-NCE, seed_extend coord losses; KEEP retrieval InfoNCE + segmentation + state + orientation + productivity + junction; ADD a lightweight per-position contrastive term (replaces the reader's lost per-position gradient) |
| `core/dnalignair.py` | `seed_extend` aligner/`band_head` no longer instantiated in the active path (classes retained, unused) |
| `benchmark/evaluation/allele_calibration.py` | re-fit temperature/ε against WFA scores |

New scripts:

| file | purpose |
|---|---|
| `scripts/train_fast.py` | retrain on the simplified loss set; gates = set-aware retrieval recall + assay + embargo |
| `scripts/export_encoder_trt.py` | fp16 → torch.compile → ONNX/TensorRT encoder export (Phase 3) |

Reused as gates: `scripts/bench_seed_extend.py` (assay), `scripts/exp_throughput_breakdown.py`
(speed), `embargo_retrain.py` (swap correctness).

**Boundary discipline:** the neural→classical handoff is **only strings** (read-segment substring +
candidate germline names/sequences). Nothing classical touches tensors; nothing neural touches WFA.
Each side is independently testable.

## Data flow (per batch)

```
1. tokenize(reads) → tokens, mask
2. ENCODER (fp16/compiled) → reps, region_logits, orientation, canon_tokens, match[g]
3. canonicalize reads (orientation head) → canon_seq
4. SEGMENT: decode_boundaries(region_logits) → query start/end; slice segment substring from canon_seq
5. RETRIEVE: match[g].topk(K≥32) ∩ genotype_mask → ranked candidate idxs
6. SEED: kmer(segment) vs ref germlines ∩ genotype → top-m idxs
7. POOL[read,gene] = retrieve_topk ∪ seed_topm (capped, genotype-masked)
8. WFA BATCH: align(segment_str, germline_str[c]) for every c in POOL (ends-free on germline)
            → AlignResult(score, cigar, coords) per (read,gene,c)
9. PICK argmax_c score → call; SET {c: score_top−score_c ≤ ε·T}; COORDS from winner traceback
10. GATE: length-normalized best score < τ → flag is_contaminant (call retained)
11. ASSEMBLE AIRR: calls+sets, query coords (4), germline coords/CIGAR/trims (9),
    junction (V_end..J_start on canon_seq), productive, orientation
```

**Coordinate frames:** all coords produced in the canonical forward frame (query from segmentation
on `canon_seq`; germline from WFA traceback). Orientation mapping back to the presented read reuses
the existing `canonicalize_sequence`/orientation path. This is also where we fix the 0-based-start /
end-position convention the assay flagged, so coord metrics are honest.

**Cost bound:** WFA pairs per batch = Σ_reads |POOL| × genes (~40 candidates × 3 genes at K=32,
m=8), chunked through `align_batch`. The encoder stays dominant once WFA is batched.

## Error handling & edge cases

| case | behavior |
|---|---|
| short/empty segment | segment < k-mer length → skip WFA, keep retrieval order, flag low-confidence |
| empty pool / gene absent (D in light chain) | emit honest empty call, not a forced guess |
| no positive WFA alignment | retain retrieval order, flag low-confidence |
| novel allele neither retrieved nor seeded | residual miss — log it; k-mer admission minimizes it |
| inverted-D | keep `d_inverted` flag; forward-reference WFA can't align it → skip/flag (RC-align deferred) |
| contaminant | length-normalized best score < τ → `is_contaminant=True`, call retained |
| genuine sibling ambiguity | captured by the score-band equivalence set |
| `pywfa` unavailable | graceful fallback to parasail backend |
| germline ends-free overhang | clamp query/germline coords at segment bounds |

## Testing

- *Unit* `align/`: WFA score/CIGAR/coords vs hand-computed alignments (identity, SNP, indel, 5′/3′
  trim); ends-free semantics; parasail-fallback parity.
- *Unit* `seed_prefilter`: admits a known divergent allele a retrieval-miss would drop; index correctness.
- *Unit* `batch`: chunking, order preservation, thread safety.
- *Integration* `predict_reads`: end-to-end, valid AIRR, equivalence sets ⊆ genotype, genotype
  restriction honored (extend `test_dnalignair_infer.py`).
- *Integration* swap test: swap to a reference with a novel allele → callable via the seed path.
- *Regression* coord-frame correctness vs GenAIRR truth.

## Gates (before "done")

- **Accuracy**: assay V/D/J `top1_in_set` ≥ current DP-reader inference (no regression).
- **Swap**: `embargo_retrain.py` held-out recall ≈ control.
- **Speed**: throughput probe hits the phase target.

## Phasing

- **Phase 1** — `align/` package (WFA + seed + batch) + `predict_reads` rewrite + training
  simplification + retrain. Gate: no assay regression, embargo holds, coords correct. (~600–900/s)
- **Phase 2** — fp16 + `torch.compile` + batched WFA + k tuning. (~2–4k/s)
- **Phase 3** — ONNX/TensorRT encoder export. (~10k+/s)

## Strategic note

This abandons the "exact differentiable structured decoder" thesis: the DP becomes a *classical*
aligner, not an in-graph one. Profiling shows that thesis cost ~8× speed for no accuracy edge
(classical SW is in the IgBLAST family, which beat our soft-DP reader on sibling discrimination).
The trade is deliberate and endorsed.

## Open dependencies / decisions deferred to the plan

- `pywfa` (WFA2-lib python bindings) added as a dependency; parasail (already present) is the fallback.
- Exact K (retrieval top-k) and m (seed top-m) tuned empirically in Phase 1/2 against recall vs cost.
- The lightweight per-position contrastive term's exact form chosen during the training-simplification task.
