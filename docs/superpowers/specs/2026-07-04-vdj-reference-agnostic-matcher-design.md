# Reference-Agnostic VDJ Matcher — Design (v1: pure late-interaction)

**Date:** 2026-07-04
**Status:** Design approved, pre-implementation
**Package name:** `matcher`.  **Public model name:** `AIRRMatch`.

## One-liner

A fast, fully-learned, reference-agnostic VDJ allele caller. Given any reference set
**R** (V / optionally D / J allele nucleotide sequences, scoped to one locus+species) and a
read **s**, predict which V/(D)/J alleles in R were recombined to produce s, and return
those alleles' sequences — by **pointing into R**, never by generating. v1 is a pure
bi-encoder with salience-weighted late interaction (no reranker, no coordinates).

## Why this shape (context)

This replaces the parked AIRRistotle experiment (heavy autoregressive LLM that learned to
**copy, not select** — see `.private/airristotle/` and memory
`airristotle-copy-not-select-rootcause`). A retrieval/matching model is **selection by
construction**, which structurally removes the copy-not-select failure. The design is
grounded in three SOTA lines: dilated/long-conv DNA encoders (Borzoi/ByteNet, HyenaDNA,
Caduceus), late-interaction retrieval (ColBERTv2 MaxSim + hard negatives), and open-set
few-shot metric learning (prototypical / additive-margin softmax).

## Contract

```
predict(R, s) -> { "V": (names, seq),  "J": (names, seq),  "D": (names, seq) | None }
```

- Each call is a **pointer into R**; the returned sequence is verbatim from R.
- **Set-aware, first-class:** R is preprocessed into a per-gene **`sequence_group` index** —
  `unique_sequence -> [names]`, plus one representative encoded tensor per unique sequence.
  All scoring, softmax, and loss run over **unique sequences**, so N sequence-identical
  alleles never split softmax mass; the winning group's names are expanded only at inference.
  This collapse is a data structure, not a post-hoc step.
- R is scoped to one locus+species (tens to a few hundred alleles). Locus/species is known
  at inference (you hand it the right R) — the model does not infer it.
- **D is optional:** absent for light (IGK/IGL) and α/γ chains → no D output. D-absence is
  structural (no D alleles in R), not a special "none" class.

## Scope & non-goals (v1)

**In scope**
- Multi-locus / multi-species from day one (IGH/IGK/IGL + TCR, ≥2 species). Locus-agnostic:
  variable segment count (3 for heavy/β, 2 for light/α).
- Output: **allele calls + their sequences only.** No coordinates, no segmentation output,
  no junction/productivity.
- Primary success metric: **held-out generalization** (below).

**Explicit non-goals (YAGNI — deferred)**
- No reranker (Approach B/C). v1 is Approach A; the data decides if B/C is ever built.
- No coordinates / segmentation / full AIRR record.
- No ANN / shortlist stage (unnecessary — R is small and fully scored).
- No hand-tuned curriculum (history: naive ramps are unsound — see memory
  `dnalignair-training-protocol`). Natural GenAIRR breadth + RC augmentation only.
- No novel-allele-rejection ("truth not in R") head. v1 assumes truth ∈ R.
- **No independently-inverted D** (disabled/masked in v1 — a single V-anchored read
  orientation cannot resolve it; see Training → Data engine). Re-enabling it is a follow-on.
- **Orientations restricted to forward / reverse-complement.** The gym's complement- and
  reverse-only modes are out of scope for v1 (see Data engine).

## Success criterion

**Set-aware allele-calling accuracy (V/D/J separately) on species/loci never seen in
training.** Train on a subset of configs; reserve whole species/loci as a frozen exam
(`gym/` FrozenLattice). Report held-out **and** in-distribution side by side; IgBLAST as a
yardstick (reuse `scripts/vs_igblast.py`, `scripts/baseline_igblast.py`). Secondary:
reads/sec.

## Architecture

### Data flow

```
chars {A,C,G,T,N} ──┬─ read s   ─► Encoder ─► S  ∈ [L_s, d]    (read salience computed but IGNORED in v1)
                    └─ each a∈R ─► Encoder ─► A_a ∈ [L_a, d]  (+ salience w_a ∈ [L_a])  ◄─ cached per R (unique seqs)

orientation (once per read): o* = argmax_{o ∈ {s, RC(s)}} max_{a ∈ R_V} score(a, o)   (chosen by best V match; V is longest/most reliable)
per gene g ∈ {V, (D), J}, for each candidate a ∈ R_g, in the fixed orientation o*:
    Sim_a = A_a · o*ᵀ                        ∈ [L_a, L_s]      (cosine; embeddings L2-normalized)
    score(a, o*) = Σ_i  w_a,i · max_j Sim_a[i, j]             (salience-weighted MaxSim)
    call_g = argmax_a score(a, o*)  →  return R[a].seq
```

### Encoder (shared weights, read and alleles)

- Char embedding (vocab 5: A/C/G/T/N; +PAD).
- **Dilated-conv residual tower:** depthwise-separable 1-D conv blocks, kernel 3, dilations
  cycling `[1,2,4,8,16]` repeated, fully parallel, no attention. With 16 layers this gives a
  **receptive field ≈ 190 nt — local-to-midrange, not the whole read** — which is exactly
  what *positional* late-interaction needs: each position embeds its local neighborhood for
  matching, and global context is deliberately *not* pooled in (that would blur the SNP).
  Widen the schedule to include `32/64/128` if fuller context proves useful (RF grows
  accordingly). Pre-norm residual, GELU. Default `d_model=256`, `d_match=128`, `n_layers=16`
  (scalable to 512 / 24 — fits 24 GB easily; "many params, still fast").
- Two output heads per position:
  - **Match embedding** `∈ R^d_match` (`d_match≈128`), **L2-normalized** → cosine sim space.
  - **Salience scalar** `w` via linear → softplus, normalized within each sequence.
    **v1 uses salience allele-side only**: it weights allele positions in MaxSim (below); the
    read-side salience output, though produced by the shared encoder, is **ignored in v1**, so
    the head's gradient comes entirely from allele-side use — which is its intended meaning
    ("how informative is this allele position"). Read-side weighting is a later lever.
- Optional later swap: a Hyena/Mamba layer for cheap global context (backbone is isolated
  behind `encoder.py`, so this is a drop-in).

### Scoring (`scoring.py` — pure, unit-testable)

`score(a, s) = Σ_i w_{a,i} · max_j (A_{a,i} · S_j)`  (salience `w` is **allele-side only** in
v1; read positions are unweighted).

- **Salience `w` does double duty** and is the single most important lever:
  1. **Trimming robustness** — down-weights allele positions trimmed off the read (they
     produce only spurious max-sims).
  2. **SNP emphasis** — up-weights discriminating (sibling-distinguishing) positions.
  It is learned end-to-end through the selection loss; nothing hand-labels it.
- **Orientation** is resolved **once per read**: pick the strand ({s, RC(s)}) with the best
  V-gene score, then score all genes in that fixed orientation (V is longest/most reliable,
  so it anchors the strand call). Revcomp can never be a shortcut.

### Tensor / compute notes

- Read ≤ `max_len` (e.g. 576). Alleles cached per R (encode once, reuse across the batch and
  across the stream). Inference per read = one encode + batched matmuls → fast.

### Honest risk (the A→B/C trigger)

- **V** should be strong (long, distinctive).
- **D is the canary** — short (~10–40 nt), heavily trimmed, matches spuriously across the
  read; MaxSim has the least to grip. **J** is intermediate.
- If salience-weighted MaxSim cannot separate J/D siblings on the held-out exam, that is the
  measured signal to add Approach **B** (learned cross-encoder rerank on top-k) or **C**
  (classical WFA/parasail rerank via `align/`). We measure A first.

## Training

### Data engine
- Reuse `gym/` (GenAIRR infinite stream). Each read carries truth alleles (comma-lists →
  set-aware).
- **Batch by config** (same locus+species): encode that config's R **once per batch**, score
  every read in the batch against it. Throughput win.
- **Matcher-specific data bridge (forward / revcomp only).** The existing
  `GymRecordTransformer` can also emit complement- and reverse-only orientations; the v1
  bridge restricts to **forward and reverse-complement**, matching the "one strand per read,
  anchored by V" scoring. The other orientations are out of scope for v1.
- **Inverted-D disabled/masked in v1.** GenAIRR can generate inverted-D (D in the opposite
  orientation to V/J). A single global read orientation anchored by V cannot resolve an
  independently-inverted D, so v1 **disables inverted-D generation** (or masks those reads
  from the D loss). Re-enabling it needs per-segment orientation (likely with reranker B) —
  a follow-on.

### Free hard negatives
- Candidates for gene g = **all of R_g**, which already contains the true allele's siblings.
  A full softmax over R_g therefore has the hard negatives built in — **no ANN, no
  ANCE-style negative refresh** (those exist only for huge corpora; R is small and fully
  scored). Scoped-R bought automatic hard-negative mining.

### Loss
- Per read, per present gene: **additive-margin softmax** (CosFace-style) over the gene's
  **unique sequences** (the `sequence_group` index, not raw alleles — duplicates would split
  softmax mass).
- **Margin is applied in score/cosine space, BEFORE the temperature divide:**
  `logits = (score(a,s) − m · positive_mask) / τ`, then cross-entropy. Subtracting `m` *after*
  dividing by τ makes the effective margin `m·τ` (≈ 0.01 at τ=0.05) — negligible; the margin
  must go in before the scale.
- **Set-aware target:** `positive_mask` covers the set of true unique sequences; numerator =
  logsumexp over that set (any-correct), matching GenAIRR ambiguous comma-lists.
- Light chains: no D unique-sequences in R → no D term.
- Total = mean of V/(D)/J selection losses. Salience learned through this. Defaults
  `τ≈0.05`, `m≈0.2` (cosine-space margin).

### Orientation
- RC augmentation (prob 0.5) + per-read strand pick (best-V) → orientation learned
  implicitly. The data bridge emits forward / revcomp only (see Data engine).

### Generalization protocol
- Episodic references (R varies every batch). **Reserve whole species/loci** (FrozenLattice)
  as the exam; never trained on. E.g. train human+mouse across loci, test a held-out species.

### Optimization defaults
- AdamW, lr 3e-4 + warmup, bf16, grad-clip 1.0, infinite stream, eval every K steps on the
  frozen held-out lattice. GPU: RTX 3090Ti 24 GB (`.venv/bin/python`, `PYTHONPATH=src`).

## Repo structure

```
src/alignair/matcher/
  config.py    # MatcherConfig: d_model, d_match, n_layers, dilations, kernel, tau, margin, max_len, dropout
  encoder.py   # char embed + dilated-conv tower -> per-position match embeddings + salience   (built on nn/primitives)
  scoring.py   # salience-weighted MaxSim, per-gene selection, per-read strand pick (by best-V)   (pure functions)
  loss.py      # additive-margin, set-aware selection cross-entropy
  data.py      # gym bridge: config-homogeneous batches, RC aug, set-aware truth targets, collate
  model.py     # encoder + scoring; predict(R, s); caches reference embeddings per R
  infer.py     # called_names + set-aware sequence->name mapping
scripts/  train_matcher.py,  eval_matcher.py
tests/alignair/matcher/  test_{encoder,scoring,data,loss,model,e2e}.py
```

**Reuses:** `reference/reference_set` (R), `gym/` (stream + FrozenLattice), `nn/primitives`
(conv blocks), `benchmark/` + `scripts/vs_igblast.py` (eval yardstick), `align/` (only if C
is later added). DNA tokenizer: reuse the repo's existing char encoding if present, else a
tiny 5-token tokenizer local to the package. `scoring.py` is a pure function of embeddings so
it is unit-tested in isolation — the boundary that matters most.

## Evaluation & metrics

- **Primary:** set-aware V/D/J allele accuracy on held-out species/loci (FrozenLattice),
  reported next to in-distribution.
- **Yardstick:** IgBLAST via existing scripts.
- **Secondary:** reads/sec.
- **Decision gate (numeric):** build Approach B or C if, after N training steps (N≈50k),
  **held-out J or D top-1-in-set trails IgBLAST by >5 pp**, OR the **in-distribution →
  held-out accuracy drop exceeds 10 pp** on any gene. Otherwise A stands.

## Testing (TDD)

- `scoring`: MaxSim correctness, salience weighting, per-read strand pick (RC read → correct
  strand chosen by best-V) — hand-built tiny tensors with known answers.
- `data`: set-aware targets (comma-list → sequence set), RC augmentation correctness,
  config-homogeneous batching.
- `loss`: additive margin lowers the true logit; gradient flows to salience weights.
- `model` e2e: exact-match read (V+D+J germline concat) → correct pick from a tiny R.
- **Overfit test:** memorize ~30 examples → loss → ~0 (learning sanity check).
- Held-out smoke: train tiny, call on an unseen config without crashing.

## Staged follow-ons (post-v1, data-driven)

1. **B — learned cross-encoder rerank** on top-k per gene (read-region × candidate
   cross-attention) — the primary sibling resolver if A's J/D is weak.
2. **C — classical rerank** (WFA/parasail on top-k via `align/`) — diagnostic oracle /
   lower-risk alternative.
3. Coordinates + segmentation output; full AIRR record.
4. Novel-allele rejection (truth ∉ R) via dummy prototype / threshold.
5. Difficulty concentration on hard strata if held-out lags.
6. Backbone swap to Hyena/Mamba for cheap global context if needed.

## Open risks

- **D resolution** under trimming/SHM (expected weakest — the canary).
- **Salience learnability:** whether one scalar per position can carry both
  trimming-masking and SNP-emphasis; if not, split into two heads.
- **Held-out species drift:** DNA k-mer statistics differ across species; the encoder must
  generalize at the sequence level. Mitigated by multi-species training breadth.
