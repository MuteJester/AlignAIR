# AlignAIR Prediction / Post-Processing Pipeline — Design

**Status:** approved architecture; draft for implementation
**Date:** 2026-07-07
**Branch:** `feature/alignair-pytorch`

## Goal

Port the TF AlignAIR **prediction → alignment** post-processing (model raw outputs → finalized AIRR
records) as a clean, professional PyTorch package. Faithful to TF *behavior*, with an explicit
**functional typed pipeline** (approved) rather than TF's mutable-`PredictObject` steps or its newer
`Stage`/`Runner`/`SlotStore` (both heavier than a fixed 7-stage linear flow needs).

## Architecture (approved)

Pure transform functions over immutable typed dataclasses, composed by one readable `predict()`.
Each stage is independently unit-testable (construct input dataclass → assert output); control flow
is the function body; typed dataclasses provide contract safety at author time. We keep the two good
ideas from TF's newer pipeline — **typed data contracts** (as dataclasses) and an **error taxonomy**
— and drop the slot-store/runner machinery. Reproducibility/provenance, if wanted later, is a thin
wrapper around `predict()`, not a per-stage coupling.

## Module structure (professional, isolated concerns)

```
src/alignair/predict/
  __init__.py        # public API: predict, PredictConfig, key dataclasses
  config.py          # PredictConfig (threshold_pct, cap, genotype, airr_format, max_seq_length)
  state.py           # frozen dataclasses: RawPredictions, Predictions, Coords, Calls, Alignments
  forward.py         # tokenize + batched model forward -> RawPredictions
  clean.py           # clean(): merge batches, decode coords, productive>0.5
  genotype.py        # adjust_for_genotype(): bounded likelihood redistribution (conditional)
  segment.py         # correct_segments(): de-pad, clip, one-directional ordering
  threshold.py       # select_alleles(): MaxLikelihoodPercentageThreshold (+ strategy registry)
  germline.py        # align_germline(): reuse src/alignair/align/ (WFA/parasail); +CIGAR
  pipeline.py        # predict(): the orchestrator
  airr/              # AIRR record assembly (the meaty faithful part) — own subpackage
    __init__.py      # build_airr()
    constants.py     # IMGT_REGIONS, AIRR column order
    alignment.py     # sequence_alignment (IMGT gaps) + germline_alignment
    regions.py       # IMGT regions + junction/CDR3 via J anchor
    quality.py       # stop_codon, vj_in_frame, v_identity
    builder.py       # per-record assembly orchestration
tests/alignair/predict/
  test_threshold.py test_segment.py test_genotype.py test_germline.py test_clean.py
  test_pipeline.py            # end-to-end contract
  airr/ test_alignment.py test_regions.py test_quality.py test_builder.py
```

## Faithful per-stage logic (behavioral ground truth = legacy Steps + AlleleSelector + matcher)

1. **clean** (`clean.py`): vstack per-batch arrays; positions = `argmax(logits)` if logits present
   else the soft-argmax expectation scalar (our model emits expectations → scalar path); `productive
   = prob > 0.5`.
2. **adjust_for_genotype** (`genotype.py`, conditional): drop non-genotype alleles; redistribute
   their mass proportionally onto genotype alleles (`lk + lk·(total_non/total_geno)`), clip 1.0.
3. **correct_segments** (`segment.py`): de-pad (config `max_seq_length`, **not** hardcoded 576),
   `floor`, clip start∈[0,len-1] / end∈[1,len], `end=max(end,start+1)`; then one-directional order
   clamp `d_start=max(d_start,v_end)`, `d_end=max(d_end,d_start+1)`, `j_start=max(j_start,d_end)`, …
   *Deviation:* TF center-pads; our trainer right-pads → the de-pad shift is 0 (handled consciously).
4. **select_alleles** (`threshold.py`) — **the crux**: `MaxLikelihoodPercentageThreshold`, per gene:
   `keep {i : p_i ≥ pct·max(p)}`, sort desc, **cap 3**. `pct` default **0.1**, cap default **3**.
   (Cumulative-confidence variants registered as alternate strategies, not the default.)
5. **align_germline** (`germline.py`): only the **top-1** call per gene is aligned. *Deviation
   (faithful-to-intent):* reuse `src/alignair/align/` (WFA/parasail, banded) instead of TF's
   hand-rolled affine offset search — more principled and yields a real **CIGAR** the TF pipeline
   never produced. Preserve quirks: zero-indel overhang trim, Short-D empty-reference sentinel,
   pure-overhang snap-back semantics (bases beyond the germline window treated as overhang).
6. **build_airr** (`airr/`): faithful port of `Pipeline/AIRR/*`: IMGT-gapped `sequence_alignment`,
   `np1/np2`, `germline_alignment`, junction/CDR3 via the J **anchor**, fixed IMGT region boundaries,
   `stop_codon`/`vj_in_frame`/`v_identity`, **1-based** coord conversion, per-record `try/except`.

## Deviations from TF (all deliberate)
- Functional typed pipeline (vs mutable steps / slot-runner).
- Germline alignment via our `align/` WFA/parasail (+ real CIGAR); TF used a heuristic scorer, no CIGAR.
- Right-padding (our trainer) → de-pad is a no-op; TF center-pads.
- Config-driven `max_seq_length` (TF legacy hardcoded 576 — a latent bug the new TF pipeline fixed).

## Phasing
- **Phase A (this plan): the algorithmic core → benchmarkable calls + coords.** state, forward,
  clean, genotype, segment, threshold, germline, and a `predict()` that emits the benchmark contract
  (v/d/j calls + set, read/germline start-end, CIGAR). Enough to run the IgBLAST head-to-head.
- **Phase B (follow-on): full `airr/` assembly** (sequence_alignment, junction/CDR3, regions, quality)
  for complete AIRR output.

## Testing
Unit per stage (pure function, tiny dataclass in → assert out); `test_pipeline.py` end-to-end on a
few gym reads producing the contract; germline tests assert coords/CIGAR vs hand alignments.
Reuse the existing benchmark (`run_h2h_benchmark`) for the accuracy gate.
