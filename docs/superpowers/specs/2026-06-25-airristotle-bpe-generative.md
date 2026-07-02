# AIRRistotle v2 — byte-level BPE, generative JSON annotation

**Status:** design agreed 2026-06-25. Supersedes the char-level copy-pointer MVP
(`2026-07-01-airristotle-*`) for the tokenizer + task; keeps the decoder backbone.

## Motivation

The char-level tokenizer makes prompts ~4000 tokens (one token per nucleotide), so a
160M decoder over a genotype-in-prompt is O(L²)-bound: ~0.9 s/step and OOM at batch 8.
The germline sequences and the read are near-duplicates, so a byte-level BPE compresses
the DNA **6–17×** (measured: vocab 512→6.5×, 1024→8.8×, 2048→12.5×, 4096→17×). Prompt
~4000 → ~500 tokens ⇒ attention ~60× cheaper.

## Task

Prompt = (query sequence to align + reference alleles) → the model **generates** the
GenAIRR JSON annotation. No copy-pointer, no offset head — pure causal generation, like
fine-tuning a Llama/Qwen decoder to emit structured JSON. The bet: a well-trained decoder
emits exact integers/names by attending to the reference in-context. Coordinate snapping
(local-align to the predicted germline) is an OPTIONAL postprocess safety net, off by default.

Minimal JSON field set to start (grow later): `v/d/j_call`, per-gene
`sequence_start/end` + `germline_start/end`, `junction_start/end`, `productive`.

## Tokenizer (this deliverable)

Byte-level BPE (`tokenizers`), vocab 1024:
- **Single-digit tokenization** — a `Digits(individual_digits=True)` pre-tokenizer keeps
  every digit its own token so exact coordinate integers are learnable (Llama-3 practice).
- **Byte-level base alphabet** — any character / novel allele name always encodes.
- Structural specials: `<pad> <bos> <sep> <annot> <eos>`.
- Trained deterministically (GenAIRR seed 0) on germline seqs + names + reads + rendered
  JSON, saved to a committed `src/alignair/airristotle/assets/tokenizer.json` so training
  is reproducible. `tokenizers` added as an `[airristotle]` extra.

## Follow-on pieces (not this commit)

1. Generative prompt/target builder: `render_prompt` + `render_json`, completion-only loss mask.
2. Training loop: strip the copy head + dual loss; standard causal CE on the JSON completion.
3. Inference: generate → parse JSON (tolerant), optional coord-snap postprocess.
4. Retrain from scratch (new tokenization ⇒ new model), eval vs the char-level MVP.

Reversible: the char-level tokenizer + copy-pointer path stays in git; this is additive.
