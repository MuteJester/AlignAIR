# Comprehensive Benchmark Assay — Design

**Status:** approved design (agent-reviewed twice); slice 1 ready for planning.
**Date:** 2026-06-29

## Goal

Extend the `alignair.benchmark` suite so it comprehensively evaluates the aligner — especially
**short / adaptive-sequencing reads** (where XAttnAligner is weak: fragment V 0.02–0.29), the
**dynamic-genotype** differentiator (partial genotype, novel alleles — currently only manually
tested), and **multi-locus** (IGH + IGK/IGL + TRB). The existing metric criteria + ~15 scenario axes
are strong; the gaps are in the **conditions generated** and in two **coverage-conditioned metrics**.

## What's already sound (reuse, don't rebuild)
- `core/criteria.py`: rich criteria incl. `set_valued_allele_call`, `topk_candidate_recall`,
  `graceful_degradation`, `fragment_observability`, `contaminant_and_out_of_scope_handling`.
- `evaluation/context.py`: ~15 scenario axes incl. `segment_presence`, `length`
  (≤60/61-90/91-130/131-250/>250 — the 61–130 bins straddle the immunoSEQ band), `genotype_size`,
  `allele_frequency`, `allele_ambiguity`, `orientation`.
- Generation/scoring pipeline, frame handling (canonical vs presented), readiness/coverage planning.

## Two correctness-critical fixes (caught in review — without these the adaptive benchmark measures the WRONG thing)

1. **Germline-anchored one-sided crop, not bp-count crop.** immunoSEQ anchors at a fixed *germline
   position* (FR1/FR2/FR3 multiplex primer), not N bp from the junction. `crop.py:57` already
   recomputes `v_germline_start`. Use it: `v_germline_anchored(g_start)` keeps `[first read position
   whose v_germline ≥ g_start … 3' end]`, with canonical anchors FR1≈0–25, FR2≈70–95, FR3≈195–230.
   The new one-sided path MUST drop the symmetric crop's "D-always-present / V-tail-always-present"
   invariants (`crop.py:35-42`) — the point of adaptive is the 5'-V is gone. Tests assert
   `v_sequence_start==0` and a short residual V on FR3-anchored crops.
2. **Coverage-conditioned observable truth.** On a short window many alleles become observationally
   identical (distinguishing SNPs not in the read). Scoring against full-read truth penalizes the
   model for *irreducible* ambiguity. Recompute the *observable* allele truth set on the cropped
   germline window (the germline coords are in the cropped record) so `fragment_observability` /
   `graceful_degradation` become scoreable. Add a **called-given-segment-present** conditional
   metric (recall conditioned on `segment_presence`) so a correct no-call on absent 5'-V is not a
   miss.

## Sub-specs & build order

**Sub-spec 1 — Adaptive/short-read foundation + fair metrics (build first; IGH).**
- One-sided crop modes in `crop.py`/`generate.py`: `v_germline_anchored(g_start)` (FR1/FR2/FR3),
  `j_anchored(L)` (3'/J-primer), drop the has-D/V-tail invariants on this path.
- Adaptive strata as a **tight ~80–130 bp band** (sample L from the band), labeled distinctly from a
  RACE/full one-sided gradient (200/150/120/100/80/60/40 — those large values are RACE, not adaptive).
- **Reverse-strand × short product cells** (set `orientation_ids` on the one-sided strata) — the
  single highest-value missing cell (orientation head trained on full reads; revcomp short fragment
  is a real collapse risk).
- The two coverage-conditioned metrics above (observable truth + conditional recall).

**Sub-spec 2 — Dynamic genotype (in PARALLEL; the differentiator, and it's invasive).**
The predictor protocol is `Callable[[reads], preds]` with the reference frozen at adapter construction
(`runner.py:27,191`, `model_adapters.py:23`). Supporting `partial_genotype` / `novel_allele` is a
**core protocol change**: a `(reads, genotype)→preds` predictor (or per-genotype instances over case
subsets). `predict_reads_xattn(model, reference_set, ...)` already takes the reference positionally
and `XAttnAligner` re-embeds it, so the model supports it; the *harness* doesn't. Derisk early.

**Sub-spec 3 — Adversarial/realistic strata (cheap, build on 1).**
`adaptive_hard` (short × heavy-SHM × noise), `shm_fragment`, `d_stress`, **chimera/PCR-crossover**
(record-splicer in `generate.py` — directly probes the retrieve+rescore failure mode; truth =
inconsistent V/J support / low-confidence), junction-localized indel, end-adapter contamination,
and a **short-contaminant** stratum (extend `contaminant_and_out_of_scope_handling` contexts to
fragments — the no-call gate is not length-calibrated).

**Sub-spec 4 — Multi-locus scaffolds only (cheap; no tuning).**
Ship `default_igk/igl/trb_spec` (`BenchmarkSpec` with a different `dataconfig_name`; `has_d` plumbing
exists). **Drop SHM strata for TRB** (TCRs don't hypermutate). Do NOT tune per-locus strata or trust
the numbers until trained TCR/light models exist.

## Deferred
Cross-species reference robustness; UMI/duplicate structure (per-read aligner, dedup is upstream).

## Gates / usage
Each new stratum is scored by the existing criteria + the two new coverage-conditioned metrics, sliced
by the existing axes. Slice 1 is immediately runnable against the current IGH XAttnAligner to quantify
the short-read weakness *correctly* (fair to irreducible ambiguity). Reuse `scripts/bench_xattn.py`.

## Key files
`src/alignair/gym/crop.py` (one-sided/germline-anchored crop), `benchmark/generation/strata.py` (new
strata), `benchmark/generation/generate.py:191` (crop wiring, chimera splicer), `benchmark/evaluation/
runner.py:27,191` + `model_adapters.py:23` (genotype protocol), `benchmark/evaluation/context.py:64-76`
(observable-truth / conditional slicing), `benchmark/core/criteria.py:441,646,670`
(fragment_observability, graceful_degradation, contaminant contexts).
