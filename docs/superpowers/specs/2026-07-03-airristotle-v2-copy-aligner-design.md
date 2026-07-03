# AIRRistotle v2 — Copy-Based Open-Reference VDJ Aligner (Design Spec)

**Status:** design, pre-implementation
**Date:** 2026-07-03

## 1. Goal

Given an antibody/TCR read and a reference of germline V/D/J allele **sequences** (the genotype,
supplied at inference), output the specific germline allele **sequences** that recombined to
produce the read — one or more per gene when the read is genuinely ambiguous. Output is the DNA
sequence(s) *copied verbatim from the reference in the prompt* (never a name, never hallucinated),
so downstream code can do the exact alignment / AIRR assembly from known sequences.

The model must:
- work with **any** reference given at inference — novel, renamed, subset, or often-seen alleles,
  with no retraining (in-context / dynamic reference);
- be **single-SNP sensitive** — resolve sibling alleles that differ by 1–2 bases;
- fit an **efficient token budget**;
- be, in architecture and training, a **standard LLM** (no bespoke mechanisms).

## 2. Non-goals

- Emitting the full GenAIRR JSON. The model emits the germline sequences; a deterministic
  post-processor does coordinates, trims, junction, CDR3, and the rest.
- Allele **names**. Names are never produced (they are the memorization trap); post-processing maps
  the copied sequence back to its reference entry.
- Reverse-strand handling in v1 (query is canonicalized to forward first; see Open Questions).

## 3. Approach overview

A pure autoregressive decoder LLM reads a prompt = `[shortlisted reference] + [query]` and generates
the matching germline sequence(s), constrained at decode time to copy exact substrings of the prompt.

```
coarse filter (retrieve top-k V; keep all D,J)  →  build prompt (char-level)  →
pure-LM decode with prompt-trie constraint  →  parse copied seqs  →  post-process alignment
```

Everything is standard LLM practice: RAG-style retrieve-then-read, char tokenization, next-token
training, constrained/structured decoding.

## 4. Model architecture — pure Llama decoder

Reuse the existing, audited AIRRistotle backbone (verified byte-for-byte vs HF Llama:
RMSNorm fp32 / RoPE / SwiGLU / causal GQA / pre-norm / bias-free / init 0.02 + GPT residual
`1/√(2N)` scaling). **Change:** remove the `copy_q`/`copy_k` copy-pointer head and the two-channel
(gen/copy) loss. The model becomes a plain Llama LM: `input_ids → hidden → lm_head → vocab logits`.

No architectural additions. The copy behavior is enforced at *decode* time (§8), not baked into the
model.

## 5. Tokenization — char-level

Vocabulary (~16 tokens): the bases `A C G T N`, plus structure tokens
`<REF> <V> <D> <J> <SEP> <QUERY> <ALIGN> <END> <NONE> <PAD>`.

**Rationale:** single-SNP sensitivity requires the tokenization to preserve exact base identity.
BPE destroys it (measured: 5% mutation → 56% token overlap, 0% positional match); k-mers coarsen a
base difference into a chunk difference and shift frame. Char-level makes a 1-base difference a
1-token difference, so attention can compare query-base ↔ reference-base directly. The token cost of
char-level is paid down by the prompt design (§6), not by coarsening tokens.

## 6. Prompt schema — RAG shortlist

The full reference (~65k bases) does not fit a sane context, so we retrieve then read:

```
<REF>
  <V> v_a <SEP> v_b <SEP> …            (top-k V from the coarse filter; k = v_shortlist)
  <D> d_1 <SEP> d_2 <SEP> …            (all D — small)
  <J> j_1 <SEP> …                      (all J — small)
<QUERY> «read» <ALIGN>
```

- Light chains: omit the `<D>` section entirely.
- Budget at `v_shortlist=16`: ~16·300 + 33·20 + 7·50 + query ≈ **~6.5k tokens** (< `max_seq` 8192).
- Novel/renamed alleles are copyable because they are literally in the prompt; the model never uses
  names or a fixed index.

## 7. Coarse filter (retriever) — swappable v1 stage

A cheap, recall-oriented shortlister that picks the top-`v_shortlist` V candidates (D and J are kept
in full). **v1:** k-mer / minimizer overlap between the query and each V germline (BLAST-like, no
learning). **Contract:** it must include the true allele with high recall — recall is the ceiling on
novel-allele accuracy (if the true allele is not shortlisted, the LM cannot output it).

This stage is deliberately **isolated behind an interface** so it can be improved independently
(better minimizer scheme, learned retriever, hierarchical family→allele filtering) to raise recall
and/or pack more of the reference into the context window. During training the true allele is
**force-included** in the shortlist so there is always a positive target.

## 8. Output & constrained decoding

The model generates, per gene present in the read:
`<V> «seq» [<SEP> «seq2» …] <D> «seq» … <J> «seq» … <END>`, using `<NONE>` for an absent gene.

**Constrained decoding = a trie built over the prompt.** At each step only tokens that continue some
reference substring in the prompt are permitted. Guarantees:
- **No hallucination** — every emitted sequence is an exact copy of a prompt sequence.
- **Work concentrates on SNP branch points** — once a prefix uniquely identifies an allele, the
  remaining tokens are forced (single valid continuation), so a 300-base copy is near-free and the
  model only *decides* where shortlisted siblings diverge.
- **Ambiguity** — multiple sequences per gene are emitted when several alleles are equally
  consistent (set-aware; see §9).

## 9. Training — standard LLM recipe

- **Objective:** pure next-token cross-entropy, **loss masked to the output span** (SFT-style; no
  loss on the prompt). Target = the true allele sequence(s) copied from the prompt.
- **Set-aware ambiguity:** when GenAIRR lists several equally-valid alleles for a gene, the training
  target **enumerates all of them** for that gene, in a canonical order (their order in the prompt),
  each as a `<SEP>`-separated copied sequence — so next-token CE teaches the model to emit the full
  set. Eval is set-aware: an emission is scored correct if it recovers the true set (strict) or
  contains at least one valid allele (loose); both metrics are reported. Canonical ordering keeps the
  autoregressive target deterministic.
- **Generalization:** hold out a fraction of alleles per gene (never in the training shortlist) and
  SNP-augment germlines into novel variants, so the model must copy-align to sequences it never saw.
- **Optimizer / schedule:** the locked standard recipe — AdamW β(0.9, 0.95), wd 0.1 on 2-D params
  only, linear warmup → cosine decay to 10% floor, grad-clip 1.0, bf16.
- **Data:** the online GenAIRR gym (curriculum over SHM / indels / crops) → (prompt, target) pairs.

## 10. Config parameters (`AIRRConfig`)

Keep the audited backbone params (`d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_ff`, `rope_base`,
`init_std`, `max_seq`, `vocab_size`). **Add:**
- `v_shortlist: int = 16` — number of V candidates the coarse filter puts in the prompt.
- (implied) `max_seq` must cover the worst-case prompt at the chosen `v_shortlist`.

## 11. Inference pipeline

1. Canonicalize the read to forward orientation.
2. Coarse filter → top-`v_shortlist` V (+ all D, J) from the caller's reference.
3. Build the char-level prompt (§6).
4. Constrained-decode (§8) → copied V/D/J sequence(s).
5. Map each copied sequence back to its reference entry; run the exact pairwise alignment
   (called allele ↔ read) → germline coords, trims, junction, CDR3 → AIRR record.

## 12. Evaluation

- **Copy validity:** 100% by construction (constrained decode) — assert it in tests.
- **Calling accuracy (set-aware):** the emitted V/D/J is (one of) the true allele(s), split by
  **train vs held-out** alleles and by **SHM stratum** — the held-out column is the novel-allele
  generalization test.
- **Dynamic-reference contract:** canonical / renamed / novel-SNP references over the same reads.
- **Retriever recall@k** per gene per SHM stratum (the accuracy ceiling).
- **Throughput** vs the deployed budget.

## 13. Components / file structure

- `airristotle/config.py` — `AIRRConfig` (+ `v_shortlist`).
- `airristotle/model.py` — pure Llama LM (remove copy head; next-token CE loss).
- `airristotle/tokenizer.py` — char-level tokenizer (bases + structure tokens).
- `airristotle/retriever.py` **(new)** — coarse filter interface + k-mer v1.
- `airristotle/prompt.py` — build prompt + target + loss-mask from a gym record + shortlist.
- `airristotle/data.py` — gym → (prompt, target) example stream + collate.
- `airristotle/decode.py` **(new)** — prompt-trie constrained decoding.
- `airristotle/infer.py` — end-to-end inference + AIRR post-process.
- `scripts/train_airristotle.py` — training (already on the standard recipe).
- `scripts/eval_airristotle.py` — the §12 evals.
- `tests/alignair/airristotle/` — tokenizer, prompt/target, constrained-decode-never-hallucinates,
  set-aware loss, overfit sanity, held-out generalization.

## 14. Open questions & future work

- **Coarse-filter recall** is the novel-allele ceiling; the retriever interface exists to be
  upgraded (learned retriever, hierarchical family filtering, larger shortlist) to pack the context
  better — an explicit follow-on.
- **Full-reference-in-context** (no retriever) needs a 65k+ context; rejected for the 8k budget,
  revisit only if long-context becomes cheap.
- **Reverse-strand / orientation** — v1 canonicalizes to forward; native both-strand handling later.
- **Learnability of char-level alignment attention** — de-risked by the 16-way shortlist, char-level
  exactness, and loss concentrated at SNP branch points; validated empirically on the overfit +
  held-out evals before scaling.

## 15. Success criteria

- Constrained decode: 100% of outputs are exact prompt copies.
- Held-out-allele calling accuracy tracks train accuracy (novel-allele generalization).
- Single-SNP sibling discrimination competitive with the matcher baseline / IgBLAST on V.
- Prompt fits the context budget at the default `v_shortlist`.
