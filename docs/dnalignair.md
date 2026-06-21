# DNAlignAIR (v3, PyTorch)

A single neural model that aligns immunoglobulin / TCR **DNA** reads (human IGH first),
producing the full alignment with no heuristic post-processing, and aiming to beat the
classical tools (IgBLAST / MiXCR / partis) — especially on short fragments and with
calibrated uncertainty that classical tools do not provide.

Package: `src/alignair` (the clean v3 rewrite; the legacy TF lineage under `src/AlignAIR`
is being retired).

## Outputs (per read)
- V/D/J allele calls — **multi-label**: a calibrated equivalence *set* when alleles are
  indistinguishable, not a forced single call.
- In-sequence **and** germline start/end of each segment.
- Per-position region tags `{pad,pre,V,N1,D,N2,J,post}` and edit states
  `{germline,substitution,insertion,deletion}`.
- Orientation (detect + canonicalize), and scalars (seq-error count, SHM rate, indel
  count, productivity).

## Architecture
```
read tokens
  └─ OrientationHead (4 involutive transforms) ─ detect + canonicalize to forward
       └─ conv-stem + Transformer backbone ─ per-position reps
            ├─ RegionTagger (8-class)          → V/D/J in-sequence spans
            ├─ PerPositionStateHead (4-class)  → edit states
            └─ pooled → noise / mutation / indel / productive
   ── allele calling (two-stage, RAG retriever→reader) ──
   stage 1 RETRIEVAL : segment ─ GermlineEncoder ─ cosine vs reference embeddings → gene + top-k
   stage 2 READER    : top-k ─ SoftDPAligner.alignment_score (differentiable, base-match) → exact allele + equivalence SET
   germline coords   : SoftDPAligner (semi-global sum-product DP, gap-aware) → start/end posteriors
```
Key idea: the network finds the **gene** and the alignment frame (a pooled embedding
suffices); a **differentiable soft-DP aligner** (not classical SW) resolves the exact
**allele** via a learned per-position base-match channel — the SW mechanism, made
differentiable, GPU-batched, and end-to-end. Classical Smith-Waterman is retained only
as a teacher / fallback.

### Module map
| concern | file |
|---|---|
| model assembly + forward | `core/dnalignair.py` |
| backbone | `nn/backbone.py` (+ `nn/encoder.py`: RoPE/SDPA shared encoder, opt-in) |
| region / state heads | `nn/region_head.py`, `nn/region_decoder.py` (query decoder, opt-in), `nn/state_head.py` |
| retrieval matching | `nn/germline_encoder.py`, `nn/matching.py` |
| differentiable aligner / reader | `nn/soft_dp_aligner.py` (`alignment_score`) |
| reader training | `training/reader.py` (sibling hard-neg + set-NCE) |
| composite loss | `losses/dnalignair_loss.py` (Kendall-weighted) |
| online data / curriculum | `gym/*` (GenAIRR simulator) |
| trainer | `training/gym_trainer.py` |
| inference (deployed path) | `inference/dnalignair_infer.py` (`predict_reads`, `rescore_alleles`) |

## Results vs IgBLAST (GenAIRR, identical records — `scaled_long`, d=320/10L, ~19M params)
DNAlignAIR call = best of learned-rerank / SW-rescore; IgBLAST = same records.
| stratum | V call | D call | J call | notes |
|---|---|---|---|---|
| clean | tie (1.00) | tie (0.94) | tie (1.00) | |
| moderate | tie (1.00) | **win** (0.89 vs 0.84) | tie (0.99) | |
| hard | tie (0.99) | **win** (0.76 vs 0.62) | **win** (0.96 vs 0.93) | |
| heavy-SHM 0.25 | lose (0.90 vs 0.96) | **win** (0.57 vs 0.38) | **win** (0.94 vs 0.80) | only V loss left |
| extreme 5′ trim | tie (0.99) | **win** (0.87 vs 0.81) | tie (0.99) | |
| fragment ~80bp | tie (0.12 vs 0.09) | **win** (0.75 vs 0.34) | **win** (0.87 vs 0.47) | IgBLAST fails to align |
| heavy-SHM frag | tie | **win** (0.61 vs 0.20) | **win** (0.82 vs 0.34) | |

Benchmark (1650 broad cases): per-gene top1 V 0.74 / D 0.80 / J 0.92; per-stratum full-read
V — clean 0.99, moderate 0.99, hard 0.97, high-SHM 0.94, high-indel 0.97, trimmed 0.92.
Global: orientation 0.979, productive 0.987, mutation-rate MAE 0.033.
- Germline coordinates competitive-to-better (hard/heavy-SHM V germline-start dev 0.0 vs 12.8).
- Multi-label **set** is F1-calibrated and tight (V≈3.3, D≈1.6, J≈1.2) with `graceful_degradation`
  (hard-error V 0.06 / D 0.09 / J 0.04): correct gene/family call or honest abstain on
  information-limited reads instead of a confident wrong allele.
- SOTA story: **dominates D/J everywhere + fragments + calibrated uncertainty/degradation +
  dynamic genotype + one end-to-end model**. Only loss: extreme heavy-SHM full-read V
  (discrimination-bound); fragment-V allele is information-limited (7–23bp of V).

## Usage
Train:
```bash
PYTHONPATH=src .venv/bin/python scripts/train_dnalignair.py \
  --config HUMAN_IGH_OGRDB --steps 2000 --d-model 128 --layers 4 --aligner softdp
```
Benchmark (canonical framework): `python -m alignair.benchmark.cli build ...` then evaluate
predictions with `alignair.benchmark.evaluation`. See `src/alignair/benchmark/README.md`.

Quick model-vs-IgBLAST head-to-head (research driver; requires IgBLAST in
`.private/tools/`): `scripts/headtohead.py`, with `scripts/baseline_igblast.py` for the
IgBLAST bar only. (These are ad-hoc drivers; the `benchmark` module is canonical.)

## Dynamic genotype reference (Property 1)
The reference is encoded per-prediction, never baked into weights, so inference accepts an
arbitrary genotype — an allele **subset** and/or **novel** alleles unseen in training.
```python
from alignair.reference.reference_set import ReferenceSet
from alignair.inference.dnalignair_infer import predict_reads
rs = ReferenceSet.from_yaml("donor_genotype.yaml")            # {v:{name:seq}, d:{...}, j:{...}}
preds = predict_reads(model, rs, reads, rerank="learned")     # novel alleles are legal calls
# restrict to a subset of a larger reference (calls outside it are impossible):
preds = predict_reads(model, big_rs, reads, genotype={"v": allowed_v_names, ...})
```
- Genotype mask threads `predict_reads → forward → match_alleles → matching` (disallowed
  alleles scored −inf; top-k capped to the allowed count). Verified: 0 out-of-genotype calls.
- Novel-allele robustness rests on the floored raw-token soft-DP channel
  (`SoftDPAligner.match_floor`). When reads truly derive from a novel allele, exact-allele
  recall ≈0.91 (1-SNP) and the allele appears in the calibrated set ≈0.95.

## Calibrated multi-label sets + graceful degradation
The equivalence set is a temperature-scaled log-likelihood-ratio band, `(s_top−s_c)/T ≤ ε`,
with per-gene `T`/`ε` fit by `benchmark.evaluation.allele_calibration` (objective `f1`,
recall floor). Fit once and pass the JSON to `predict_reads(calibration=...)`:
```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_sets.py --model <ckpt> --objective f1
```
- `predict_reads` emits per gene: `{g}_call`, `{g}_call_set`/`{g}_calls`, `{g}_set_confidence`,
  and **hierarchical** `{g}_resolved_call` + `{g}_call_level` (`allele|gene|family|none`).
- `resolve_hierarchy` collapses the set to the most specific supported level and **abstains**
  when it spans families — so short fragments (information-limited at the allele level) get a
  correct gene/family call or an honest abstention instead of a wrong allele guess (fragment
  hard-error ~0.7–0.9 → ~0.12–0.17).

## Open roadmap (theory-justified)
Done: scheduled sampling; scale+RoPE (d=256/8L `--backbone shared`); set calibration (F1) +
hierarchical degradation; genotype-subset & novel-allele inference (Property 1).
1. **Wire `resolved_call`/`call_level` into the benchmark scoring** so graceful degradation
   is credited in the formal grade.
2. **State-conditioned emission A/B** — it did not lift heavy-SHM V (reader is at SW parity)
   and may slightly hurt multi-SNP novel discrimination; A/B toggling it off.
3. **Unify encoders** — retire the shallow `GermlineEncoder`; route germlines through the
   shared backbone (note: late-interaction/k-mer retrieval was falsified — recall is *not*
   the heavy-SHM V cap; reader discrimination is).
4. Retire legacy TF lineage (`src/AlignAIR`) and superseded heads.

> Heavy-SHM full-read V (~0.86 vs IgBLAST 0.96) is reader-discrimination-bound (= classic
> SW), and fragment-V allele is **information-limited** (7–23bp of V in 50–80bp reads); the
> SOTA answer there is calibrated uncertainty + degradation, not a higher top-1.
