# Amino-Acid AlignAIR (IGH) — Immutable Research Snapshot

**Date:** 2026-07-21 · **Branch:** `experiment/aminoacid-alignair` · **Status:** research milestone,
simulator-validated. **Synthetic architecture iteration is closed.** Next decisive experiment: Phase 3
real-protein validation.

## Defensible conclusion (the claim this snapshot supports — and no more)

> AA AlignAIR accurately recovers simulated IGH V, D, and J assignments across full and partial protein
> coverage and produces finite-sample calibrated allele sets on exchangeable simulator-derived data.

It is **not** yet validated on real protein and the calibration is **simulator-domain only**.

## Provenance

- git commit: `7733312d6ec0adb10d8ec2075a72c2bd3078309c`
- versions: torch 2.11.0+cu128 (CUDA 12.8), GenAIRR 2.2.0, numpy 1.26.4, parasail 1.3.4
- dataconfig: `HUMAN_IGH_OGRDB`; V protein-classes 149, J 7, D alleles 33 (allele posterior, not merged)
- class-map fingerprint (V db_hash): `e1a455b87f614f0d`
- 101 AA tests pass; nucleotide core (613 tests) untouched; `main` clean.

## Frozen models (checkpoints under `experiments/aa/strategic/`, gitignored — SHA-pinned here)

| model | sha256 (20) | role |
|---|---|---|
| `model_vdj_40k.pt` | `1a7962aa45c5c71d558b` | Stage-0 baseline (V/D/J, 40k) |
| `model_stage1_PASS.pt` | `54eef812942e7b96f3d4` | **Stage-1 final** (fine-tune, gate-passed) — the snapshot model |

Reproducible from seed 0: baseline `scripts/aa/train_strategic_igh.py`; Stage-1 fine-tune
`scripts/aa/train_stage1_trunc.py --init model_vdj_40k.pt --lr 1e-4 --p-crop 0.35 --steps 18000`.

## Results (frozen)

**V** (vs alignment *reference*, not a ceiling — model beats it on V-indels):
clean **0.986**, AA-subs-7% 0.953, single-indel 0.980, I/L 0.978, **truncation 0.870**, V-indel 0.71/0.75.
**D** (allele posterior, vs 0.04 freq prior): clean **0.781** allele / **0.841** gene (micro≈macro → balanced).
**J:** clean **0.992**.
Full tables + CIs: `eval_stage1_PASS.json`, `baseline_complete.json`. Failed Stage-1 attempts kept:
`eval_stage1_v1_FAILED.json`, `eval_stage1_v2_FAILED.json`.

**D-evidence ablation:** masking the whole V-end..J-start junction collapses D 0.781→0.033 ≈ prior →
D uses *junction-region sequence evidence* (not V/J-identity/length correlations). NB: masks the entire
junction, so this does not isolate *D-owned* residues specifically.

## Calibration (frozen, `calibration.json`) — SIMULATOR-DOMAIN ONLY

Split-conformal APS, target 95%, per gene, Mondrian by coverage bin + pooled fallback. Distinct
calibration/test cohorts (disjoint gym seeds 1000/9000), coverage-mix.

- Frozen per-gene quantiles serialized: `genes[*].calibration_frozen.{q_pooled, q_mondrian}`.
- Bin defs: full=uncropped; trunc=kept∈[0.6,0.9]; short=kept∈[0.3,0.6). min_group fallback = 500.
- **All three groups were fitted → `pooled_fallback_exercised = False`**; a fallback group would get only
  the *pooled marginal* guarantee.
- Empirical coverage V 0.997 / J 0.991 / D 0.978; set size grows with lost coverage
  (D full 2.2 → short 24.4). V/J class sets expanded to nucleotide-allele sets.

**Guarantee scope:** holds under exchangeability with the calibration distribution, *within* each
Mondrian group. Does **not** cover real-protein distribution shift.

## Research snapshot ≠ production artifact

This freezes the *evaluation* + provenance. A production AA feature additionally requires (not done):
an inference API that loads the frozen quantiles and emits, per read, the class set + nucleotide-allele
set + set size + applied calibration group; and recalibration on representative real inputs (or the
explicit simulator-domain-only label above).

## Next (Phase 3)

Real-protein-repertoire validation with independently defensible labels — the decisive next experiment.
Then recalibrate on real inputs. Frameshift stays shelved (self-inflicted artifact; route DNA-with-indels
to the nucleotide model). Also open: light chains / TCR, full IMGT numbering, RAPS tuning.
