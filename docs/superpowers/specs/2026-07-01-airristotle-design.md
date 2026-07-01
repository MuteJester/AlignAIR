# AIRRistotle — Design

**Status:** design (brainstormed 2026-07-01). A SEPARATE experimental model; does not touch the
current XAttnAligner / DNAlignAIR inference or training.
**Date:** 2026-07-01

## Goal

Train a small, modern LLM ("AIRRistotle") from scratch on GenAIRR's infinite stream so that, given a
**genotype (set of germline alleles) + a read as a prompt**, it produces the **full GenAIRR annotation**
(V/D/J calls, coordinates, junction, productivity). The genotype-as-prompt makes dynamic genotype
(partial genotypes, novel alleles, add/remove anytime) work **in-context, for free** — the cleanest
possible satisfaction of the user-mandated dynamic-genotype requirement (reference as input, novel
alleles handled, add/remove anytime, not memorized). This is an accuracy/flexibility bet (out-*reason*
IgBLAST via joint whole-record inference + a learned mutation model), not primarily a speed bet —
autoregressive decoding trades single-pass speed for flexibility.

## Core thesis

- **F(s, G): read + genotype → record.** The genotype is the prompt, so novel/partial/edited
  genotypes are just different prompts — no retrieval, no per-allele head, no memorization.
- **Beat BLAST-the-system, not Smith-Waterman.** BLAST calls V/D/J with three *independent* searches
  and a generic scoring matrix. AIRRistotle reasons over the **whole rearrangement jointly** (V-end
  constrains D; junction couples segments; genotype gives haplotype priors) and learns the **true SHM
  mutation model** from GenAIRR — the information BLAST discards.
- **Pointer outputs, not generated numbers.** Coordinates and calls are emitted as **pointers into the
  prompt** (decoder cross-attention selects an input position/allele), never free-form digits. This
  gives exact coordinates (selection, not generation) and single-source coords by construction — the
  fix for the boundary jitter and the IgBLAST coordinate gap.

## Architecture

Decoder-only transformer — **architecturally a small Llama/Qwen** — modern converged stack (RMSNorm
pre-norm, RoPE, SwiGLU MLP, grouped-query attention, bias-free, QK-norm), bf16. Same technology as
frontier LLMs (ChatGPT/Claude/Qwen/Llama are all decoder-only); the only task specialization is a copy
head for exact coordinates.

- **Single sequence.** The prompt (genotype allele blocks + query read) and the generated record live
  in ONE causal context window; the model attends over the whole prompt while decoding the record.
- **Two output channels per decode step:**
  - **Vocabulary tokens** for structural/semantic fields (field markers, orientation, productivity) —
    plain next-token generation.
  - **Copy/pointer** for calls + coordinates — a copy head is an attention distribution over the
    PROMPT positions already in context; argmax = the selected position. (a) **Calls** copy the winning
    allele block (dynamic-genotype-native; novel alleles handled). (b) **Coordinates** copy read/germline
    positions (exact, single-source). Copy-over-context is a standard LLM technique, not a departure
    from the decoder-only paradigm.
- **Size (starting point):** ~150M params (≈12 layers, d_model 768, 12 query / 4 KV heads, SwiGLU
  d_ff ≈2048; tune dims to land at ~150M). **Scale toward ~1B only if learning bottlenecks.**
- **Context:** a donor genotype (~50–120 alleles × ~300nt ≈ 15–35k tokens) + one read, in one causal
  context. The long context is the main feasibility risk (see Risks) — its **activations, not the
  params**, are the memory wall when training from scratch on a 24GB GPU. Char-level DNA keeps vocab
  tiny; GQA + FlashAttention manage the KV cost.

## I/O format

**Prompt (encoder input), char-level DNA + structural tokens:**
```
<GENO>
<V> IGHV1-2*02 <S> CAGGTGCAG...ACG
<V> IGHV3-23*01 <S> GAGGTGCAG...
... (donor V alleles, incl. any novel/embargoed ones)
<D> IGHD3-3*01 <S> ...   ... (donor D alleles)
<J> IGHJ4*02 <S> ...     ... (donor J alleles)
<READ> GATCACC...GGA
```

**Output (generated after the prompt, in the same sequence) — compact record; ⟦p⟧ = copy-pointer into the prompt context:**
```
<ORI> fwd
<V> ⟦p:allele-block⟧ <VS> ⟦p:read-pos⟧ <VE> ⟦p:read-pos⟧ <VGS> ⟦p:germ-pos⟧ <VGE> ⟦p:germ-pos⟧
<D> ⟦p⟧ <DS> ⟦p⟧ <DE> ⟦p⟧ <DGS> ⟦p⟧ <DGE> ⟦p⟧
<J> ⟦p⟧ <JS> ⟦p⟧ <JE> ⟦p⟧ <JGS> ⟦p⟧ <JGE> ⟦p⟧
<JUNC> <JNS> ⟦p:read-pos⟧ <JNE> ⟦p:read-pos⟧
<PROD> 1
```
Calls are pointers to allele blocks (copy-by-pointer). In-sequence and germline coordinates are
pointers to read/germline positions. Orientation/productivity are generated scalars. ~30–50 output
tokens → short decode.

## Training pipeline (same stage-structure as frontier labs; verifiable ground truth lets us drop preference alignment)

**Stage 0 — genomic pretraining (OPTIONAL, deferred).** Masked/next-token pretraining on germline +
simulated repertoire DNA to learn sequence representations before the task. Skip for the MVP; revisit
if SFT underfits.

**Stage 1 — supervised (SFT-analog), the bulk.** Teacher-forced on `(prompt → record)` from GenAIRR.
Loss = cross-entropy on vocab tokens + pointer cross-entropy over encoder positions for each pointer
target. **Dynamic-genotype sampling in every batch:** construct each prompt's genotype by sampling the
read's true alleles + distractor alleles + injected **synthetic novel alleles**, and hold a fraction of
real alleles out of training entirely (train/test allele split). This trains AND continuously verifies
the genotype-in-prompt property — the model never sees a fixed allele set.

**Stage 2 — RLVR / GRPO (modern RL post-training; a natural fit).** The output is *verifiable* (the
benchmark scores it). Sample K records per read, reward = benchmark correctness (allele set-match +
coordinate exactness + junction + productivity), critic-free group-relative advantage (GRPO). Refines
exactly what teacher-forcing under-optimizes: hard D, sibling alleles, exact coordinates.

**Infra:** bf16, AdamW + warmup + cosine decay, gradient clipping/checkpointing; FSDP if it grows.

## Data

GenAIRR is the infinite generator. A `prompt-builder` turns each `(read, truth)` into `(prompt,
record-with-pointer-targets)`: build a donor genotype (true alleles + distractors + novel), serialize
genotype + read → prompt, serialize truth → record where every call/coordinate is the *index of its
target token in that prompt*. Streamed, never materialized.

## Evaluation

Score with the existing `alignair.benchmark` suite — including the **dynamic-genotype strata**
(partial_genotype, novel_allele) and the **adaptive/short strata** — via a new `airristotle_predictor`
adapter. Head-to-head against the current XAttnAligner and IgBLAST on the same frozen case sets. The
`novel_allele` / `partial_genotype` cells are the acceptance test for the dynamic-genotype property.

## Isolation

New self-contained package `src/alignair/airristotle/` (model, tokenizer, prompt-builder, training,
inference) + `scripts/train_airristotle.py`. Reuses the GenAIRR gym and the benchmark suite read-only.
Does NOT modify or share weights with the current models. Separate checkpoints under `.private/models/`.

## Risks & open questions

1. **Coordinate exactness** — mitigated by pointer outputs (selection, not generation); still must be
   verified vs IgBLAST's coordinate MAE.
2. **Speed** — autoregressive decode is sequential; this is an accuracy/flexibility experiment, not the
   speed play. Output is short (~40 tokens) and batched; non-autoregressive decoding is a later option.
3. **Context length** — a full donor genotype is 15–35k tokens; the long-context encoder is the main
   feasibility risk. Falls back to a smaller sampled genotype if needed (still dynamic, just fewer
   distractors). Full-reference (400+ alleles) is out of scope for the prompt.
4. **Novel-allele generalization** — a training obligation (Stage-1 sampling + held-out alleles), not
   free as it is for classical alignment; verified on the novel_allele strata.
5. **Does joint whole-record reasoning actually beat independent calling?** The core accuracy premise;
   the MVP tests it.

## Scope / build order (decomposed — this is large)

1. **MVP — prove the hypothesis.** Pointer encoder–decoder + prompt-builder with a FIXED full-reference
   prompt (no dynamic sampling yet) + Stage-1 SFT on clean/moderate reads. Success = it learns V/D/J
   calls + pointer coordinates at all (can an LLM do IG alignment by pointing?). Smallest slice that
   validates the bet.
2. **Dynamic genotype.** Add genotype-subset + novel-allele prompt sampling + held-out alleles; verify
   on the dynamic-genotype strata.
3. **Full difficulty + RLVR.** All strata (adaptive/SHM/indel/…), then GRPO against the benchmark;
   head-to-head vs XAttnAligner + IgBLAST.
4. **Optional.** Genomic pretraining; non-autoregressive decoding for speed.

Each sub-project gets its own plan. This spec covers the whole model; the first plan implements the MVP.

## Decisions encoded here
- **Decoder-only** (like ChatGPT/Claude/Qwen/Llama) — same technology as frontier LLMs. A copy head over
  the in-context prompt handles exact coordinates without leaving the decoder-only paradigm.
- **Copy-mechanism coordinates** (not pure-generative) — for exact coords / single-source consistency.
- **Char-level DNA tokenization** (tiny vocab, long sequences) — vs k-mer BPE; revisit if context bites.
- **~150M param starting size**, scale to ~1B only if learning bottlenecks.
