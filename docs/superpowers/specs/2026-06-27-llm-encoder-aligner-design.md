# LLM-Encoder Aligner — Design (Sub-project 2)

**Status:** approved design, ready for implementation planning.
**Date:** 2026-06-27

## Goal

A single neural model that, given an input read **plus a genotype** (the list of V/(D)/J alleles as
`(name, germline_sequence)`), predicts — with SOTA accuracy, better and more reliable than IgBLAST —
the full set of GenAIRR per-record alignment parameters, at **thousands of reads/s on GPU**. The
genotype is an INPUT that can be changed at will (fewer alleles, added alleles, novel alleles) with
**no retraining**.

This is the accuracy+speed end-state that demotes the classical aligner (sub-project 1, the WFA
speed stack) from the hot path. It applies the **architecture essence and training best-practices of
modern LLMs** (Qwen/Llama-style transformer blocks + proper pretraining/optimization), adapted to the
alignment task with the GenAIRR gym as an infinite, perfectly-labelled data source.

## Why this is both fast and accurate (the core idea)

- **One batched forward pass** — no autoregressive decoding, no per-read python loop, no
  per-candidate classical alignment. The thing that capped the WFA caller at ~100/s is gone;
  everything here is batched tensor ops.
- **Token-level cross-attention** (not pooled cosine) does the per-position read↔germline comparison
  that cracks sibling / heavy-SHM discrimination — the exact mechanism the 0.42 heavy-SHM retrieval
  ceiling needed. Speed and accuracy stop being in tension: removing the classical aligner is what
  makes it both fast (no CPU per-read work) and learnable end-to-end.

## Decisions locked during brainstorming

- **Paradigm:** LLM-grade transformer **encoder + per-value heads**, NOT autoregressive generation
  (autoregression + reference-in-prompt is what made the text-LLM too slow/inexact).
- **Scale / compute:** local single RTX 3090Ti (24GB), **from scratch**, ~30–150M params, trained
  for days on the infinite gym.
- **Reference injection:** **retrieve-then-cross-attend** (cheap pooled retrieval narrows to top-k,
  deep token-level cross-attention on those k).
- **Training regime:** single-phase supervised multi-task on GenAIRR's infinite labels. **No MLM**
  (we have infinite perfect labels; the supervised matching task is itself the representation
  learner) and **no WFA teacher** (GenAIRR provides exact germline coordinates directly).
- **CPU 1000/s: DEFERRED** — see "Out of scope / future".

## Architecture

```
Input: read seq  +  genotype = [(allele_name, germline_seq), ...] for V (D) J

ENCODE reference (ONCE per genotype, CACHED):
   germline_seqs ─► transformer encoder ─► per-token germline reps + pooled allele vectors

ENCODE read (per batch):
   read ─► same transformer encoder ─► per-token read reps
        ├─► Orientation head (4-class)
        ├─► Query-span head ─► {v,d,j}_sequence_start/end
        ├─► RETRIEVE: pooled segment · pooled allele vectors ─► top-k     (cheap matmul)
        │   ∪ k-mer SEED admission (reuse align/seed_prefilter.py) ─► top-m (non-learned, novel-safe)
        │        └─► candidate pool (≈8–32, genotype-masked)
        └─► CROSS-ATTEND: read-segment tokens × pool germline tokens
                 ├─► allele match score ─► call + ambiguous set
                 └─► Germline-span head ─► {v,d,j}_germline_start/end
```

Backbone: modern transformer blocks — **RoPE, RMSNorm, SwiGLU, GQA** — sized 30–150M. Reuses/extends
the existing `SharedNucleotideEncoder` design (shared read/germline encoder with type embeddings).

## What the model predicts vs. derives (no colinear tasks)

**Predicted — the only heads (four irreducible structural tasks):**

| head | output |
|---|---|
| Orientation | 4-class (fwd / rc / …) |
| Query span (per gene) | `{v,d,j}_sequence_start/end` |
| Allele (per gene) | call + ambiguous set (retrieve → cross-attend, set-NCE) |
| Germline span (per gene) | `{v,d,j}_germline_start/end` (read off cross-attention) |

**Derived — fast post-processing, no model task** (functions of the four heads + read + called
germline string):

| derived | from |
|---|---|
| `*_trim_5/3`, P-nucleotides | germline span vs. called germline length |
| `np1/np2_length` | gaps between query spans (`d_start − v_end`, …) |
| `mutations`, `mutation_count`, `mutation_rate`, `*_identity` | base-compare read-segment vs. germline over the span |
| `junction`, `junction_aa`, `cdr3` | substring via anchor positions |
| `vj_in_frame`, `stop_codon`, `productive` | translate + frame-check the junction |
| `cigar`, `sequence_alignment`, `germline_alignment` | direct M/X for the no-indel majority; a cheap banded align ONLY for reads where query-span length ≠ germline-span length (indel signal) |

Consequences: `productive`, `mutation_rate`, `np_length`, `junction` are **not** heads (removes
colinear tasks); indels need **no head** (detected from the span-length mismatch; the only surviving
alignment is this minority-case derivation, off the critical path).

## Reference injection & dynamic genotype

- **Encode once, cache.** A genotype's germlines are encoded a single time per run; add/remove/swap →
  re-encode the reference (ms), never the model. Nothing about the allele set lives in the weights.
- **Two admission paths → bounded cross-attention pool:** learned retrieval top-k **∪** non-learned
  k-mer seed top-m (reuses `align/seed_prefilter.py`). The seed path admits *divergent/novel* alleles
  even when learned retrieval misranks them — the weakest-link fix from the WFA review.
- **Remove** → excluded from reference + genotype mask. **Add/swap** → encoded on the fly. **Novel**
  → encoded by the same encoder, admitted by retrieval or seed, matched per-position by cross-attn.
- **Trained, not hoped-for:** the **embargo protocol** holds out a fraction of alleles from the
  model's seen set while still simulating reads from them (GenAIRR truth), forcing the model to call
  alleles it reads from the reference input. `embargo_retrain.py` is the gate (held-out recall ≈
  control).

## Training regime

Single phase, end-to-end multi-task on the infinite gym (`AlignAIRGym` + `StratifiedCurriculum`).

- **Losses (all from GenAIRR truth):** orientation CE; query-span start/end; germline-span start/end;
  allele **set-NCE** over the candidate pool (true set above sibling+random negatives) on the
  cross-attention scores + retrieval **InfoNCE** to keep top-k high-recall. Reuses
  `training/reader.py` (`build_candidates`, `reader_set_nce`); sibling (1–2 SNP) hard-negatives.
- **Loss balancing:** four non-colinear tasks → fixed weights (or light uncertainty weighting; avoid
  full Kendall, which the gym notes show fights the curriculum).
- **Data:** 22-stratum space, hard corner (heavy-SHM, extreme trims, indels, short fragments,
  ambiguous) exposure-concentrated (scaling + hard-corner exposure is the proven lever). Embargo'd
  alleles baked in. `FrozenLattice`/`LatticeEvaluator` exams throughout.
- **LLM best-practices:** AdamW, cosine schedule + warmup, weight decay, **bf16** autocast, grad
  clipping, large effective batch (grad-accum), long run with checkpoint/resume, optional weight EMA.

## Speed engineering (path to thousands/s on GPU)

| lever | effect |
|---|---|
| one batched forward pass | no python per-read loop / no autoregression / no per-candidate align |
| reference encoded once, cached | ~free per read across a run |
| bounded cross-attention (top-k) | cost capped by k≈8–32, independent of reference size |
| bf16 + `torch.compile` + batch 256 | encoder floor at thousands/s (ESM-2 scale) |
| vectorized derivations | trims/np/mutations/identity as batched tensor ops; no per-read python in hot path |

**Risks designed against:** (1) cross-attention is the new cost (read ~300 × top-k germline tokens) —
keep k small, memory-efficient attention, pool germline tokens where possible; watch it on the
throughput probe. (2) derivations must stay vectorized (the WFA caller died on a python per-read
loop); the indel cheap-align is the only CPU step and only for the minority of indel reads.

## Evaluation & gates

- **Per-field correctness** vs GenAIRR truth across all predicted+derived fields — `benchmark/` assay
  (22 strata).
- **Beat IgBLAST** — canonical 4400-case/22-stratum/bootstrap head-to-head; win or tie V/D/J calls +
  all coordinate metrics, decisively beat D/J and fragments.
- **Dynamic-genotype** — `embargo_retrain.py`: held-out & novel-allele recall ≈ control;
  add/remove/swap changes calls correctly with no retrain.
- **Speed** — `exp_throughput_breakdown.py`: ≥ a few thousand reads/s at bf16/compile on the 3090Ti.
- **Per-cell competence** — `FrozenLattice`/`LatticeEvaluator`, especially heavy-SHM-V off 0.42.

## Out of scope / future

- **1000/s on CPU.** The physics caps a CPU-1000/s transformer at ~5–15M params (int8/ONNX), below
  LLM-grade. The intended path is **train big (GPU, SOTA + teacher) → distill to a small ~5–15M
  student** for CPU. Deferred to its own spec once the GPU model is proven.
- ONNX/TensorRT export + quantization for deployment.

## Relationship to sub-project 1 (WFA speed stack)

The WFA speed stack remains as: an **exactness oracle / optional inference refinement** for the
minority indel-cigar derivation, and `align/seed_prefilter.py` is **reused** as the non-learned
novel-allele admission path. The differentiable soft-DP stays removed. This sub-project makes the
all-neural path the primary engine; classical alignment is demoted, not deleted.
