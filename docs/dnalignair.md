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

## Results vs IgBLAST (GenAIRR, identical records, ~1.45M params)
| stratum | V call | D call | J call | notes |
|---|---|---|---|---|
| clean | tie (1.00) | tie (0.94) | tie (1.00) | |
| moderate | ~tie (0.97–1.00) | tie | tie | |
| hard (heavy SHM) | 0.96–0.99 | **win** (0.69 vs 0.62) | tie | |
| fragment ~80bp | tie (irreducible) | **win** (0.61 vs 0.34) | **win** (0.81 vs 0.47) | IgBLAST often fails to align |

- Germline coordinates competitive-to-better (hard V germline-start dev 0.0 vs 12.8).
- Multi-label **set** output is calibrated: tight (size ≈1) when determinable, widens
  (size ≈6) on fragments — reporting uncertainty instead of guessing.
- The genuine SOTA story is **fragments + calibrated uncertainty + multi-label sets +
  one end-to-end model**, not full-read V (there we *match* IgBLAST).

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

## Open roadmap (theory-justified)
1. **Scheduled sampling** — train match/germline on *predicted* (not teacher-forced)
   regions/allele to remove exposure bias (biggest risk for the fragment story).
2. **Scale + RoPE** — backbone d=256/8L with rotary/relative positions; unify the two
   encoders (capacity + length-generalization for hard reads / fragments).
3. **Multi-label polish** — train the (currently unused) BCE `multilabel_match_loss`
   and calibrate the set threshold for target coverage; improves co-listed-allele recall.
4. **Genotype-subset inference** — wire the candidate mask through `predict_reads`.
5. Retire legacy TF lineage (`src/AlignAIR`) and superseded heads.
