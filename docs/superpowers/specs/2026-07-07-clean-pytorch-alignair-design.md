# Clean PyTorch AlignAIR Recreation — Design

**Status:** draft for review
**Date:** 2026-07-07
**Branch:** `feature/alignair-pytorch` (off `pytorch-migration`)

## Goal

Recreate the proven TensorFlow **AlignAIR** aligner — the generalized **SingleChain** model first,
then **MultiChain** — as a clean, stable, idiomatic PyTorch implementation that is **architecturally
1:1** with the TF `master` (`main` branch, `src/AlignAIR/Models/`). We retrain from scratch on the
GenAIRR gym stream (no TF weight loading — logic migration, not weight parity), reuse the already-
ported PyTorch infrastructure, and remove the drifted DNAlignAIR/soft-DP experiment once the new
path works end-to-end.

## Why (strategy)

Multiple greenfield architectures (spectral FFT, AIRRMatch, phasor, certified-search) and the
redesigned DNAlignAIR (soft-DP retrieval) either plateaued, were too slow, or drifted so far that
their own checkpoints no longer load. The published AlignAIR architecture is a known-good baseline.
A faithful, clean PyTorch port gives a **stable foundation** to build on, rather than another
speculative redesign. We reuse the solid infra already built during the migration (GenAIRR gym
streaming, `ReferenceSet`, tokenizers, benchmark module) and rebuild only the model.

## Global constraints

- **Faithful architecture (1:1)** with TF SingleChain/MultiChain: same layers, connectivity, heads,
  and loss structure. Deviations only where explicitly flagged in "Fidelity decisions" and approved.
- **Retrain fresh** — no TF→PyTorch weight conversion. Correctness is judged by training convergence
  + benchmark parity vs IgBLAST/legacy AlignAIR, not weight-level reproduction.
- **Reuse, don't re-port**: `reference/`, `gym/`, tokenizers, `benchmark/`, `io/`.
- **Clean idiomatic PyTorch**: `nn.Module`, channels-first convs, standard training loop.
- No Claude/Co-Authored-By in commits. Commit only when the user asks.

---

## Module structure

New package `src/alignair/models/` mirroring TF `Models/`:

| file | responsibility |
|---|---|
| `layers.py` | `TokenAndPositionEmbedding`, `Conv1DBatchNorm`, `ConvResidualFeatureExtractionBlock`, `SoftCutoutLayer`, `RegularizedConstrainedLogVar` |
| `single_chain.py` | `SingleChainAlignAIR(nn.Module)` — the generalized single-chain model (heavy via `has_d=True`, light via `has_d=False`) |
| `multi_chain.py` | `MultiChainAlignAIR` — SingleChain + `chain_type` softmax head |
| `losses.py` | `hierarchical_loss` (segmentation soft-CE + aux, multi-label BCE classification, analysis MAE/BCE, Kendall weighting) |
| `config.py` (in `config/`) | `AlignAIRConfig` dataclass (max_seq_length, allele counts, has_d, latent overrides) |

Training/inference wiring:
- `training/alignair_trainer.py` — clean loop consuming the gym stream; applies weight-constraint
  clamps after each `optimizer.step()`.
- `inference/alignair_infer.py` — `predict(model, reference, seqs)` → AIRR contract (allele set from
  sigmoid heads via cumulative-confidence threshold; coords from soft-argmax expectations).

Tests under `tests/alignair/models/` and `tests/alignair/training/`.

---

## Faithful architecture (SingleChainAlignAIR)

Input: `tokenized_sequence` `(B, L)` int64, vocab_size **6** (PAD/A/C/G/T/N — matches the existing
PyTorch tokenizer), `L = max_seq_length` (default 576).

### Layers (`layers.py`)

**`TokenAndPositionEmbedding(vocab=6, embed=32, maxlen=L)`** — learned token emb (6→32) + learned
absolute position emb (L→32), added. Out `(B, L, 32)`.

**`Conv1DBatchNorm(filters=128, kernel, pool=2, act=tanh)`** — 3× `Conv1d(filters, kernel,
padding='same')` (no per-conv activation between them) → `BatchNorm1d(momentum=0.9, eps=0.8)` → `act`
→ `MaxPool1d(2)`. (Channels-first internally; see TF-isms.)

**`ConvResidualFeatureExtractionBlock(filters=128, N, kernels[len N+1], out=576, conv_act=tanh)`** —
residual conv tower. Forward (exact):
```
res  = Conv1d(128, kernels[-1], padding='same')(x)      # no activation
res  = MaxPool1d(2)(res)                                 # pool #1
f    = Conv1DBatchNorm(kernels[0])(x)                    # (halves L)
res  = LeakyReLU(0.3)(f + res); res = MaxPool1d(2)(res)  # pool #2  (index-0 double pool)
for i in 1..N-1:
    f   = Conv1DBatchNorm(kernels[i])(res)               # halves L
    res = MaxPool1d(2)(res)                              # match length
    res = LeakyReLU(0.3)(f + res)
res  = Flatten(res); res = Linear(-> 576)(res)           # fixed 576-d output
```
Net length halving = `N+1`. Two distinct activations: `conv_act=tanh` inside each `Conv1DBatchNorm`;
`LeakyReLU(0.3)` on the residual sum. Every block Flatten→Linear(576) → global `(B, 576)` vector.

**`SoftCutoutLayer(max_size=L, k=3.0)`** — differentiable soft interval mask:
```
start = clamp(start,0,L); end = max(clamp(end,0,L), start+1)
idx = arange(L)
mask = sigmoid((idx-start)/k) * sigmoid((end-idx)/k)      # (B, L)
```

**`RegularizedConstrainedLogVar()`** — scalar param `log_var` init 0.0, clamped to `[-3, 1]` after
each optimizer step; returns `exp(-log_var)` as the task weight. (See Fidelity decision #1.)

### Feature towers (7)

All `ConvResidualFeatureExtractionBlock(filters=128, conv_act=tanh, out=576)`:

| tower | N | kernels | input |
|---|---|---|---|
| `meta` | 4 | [3,3,3,2,5] | embeddings |
| `v_seg`, `j_seg` (+`d_seg` if D) | 4 | [3,3,3,2,5] | embeddings |
| `v_cls`, `j_cls` | 6 | [3,3,3,2,2,2,5] | masked embeddings |
| `d_cls` (if D) | 4 | [3,3,2,2,5] | masked embeddings |

### Heads & forward

```
emb = TokenAndPositionEmbedding(tokens)                       # (B,L,32)
meta = meta_tower(emb)                                        # (B,576)

# segmentation (per boundary b in v_start,v_end,j_start,j_end [,d_*]):
seg_feat_g = {v: v_seg_tower(emb), j: j_seg_tower(emb) [, d: d_seg_tower(emb)]}
b_logits = Linear(576 -> L)(seg_feat_of_gene(b))             # (B,L)
b_probs  = softmax(b_logits, dim=-1)
b_exp    = sum(b_probs * arange(L), dim=-1, keepdim)         # soft-argmax (B,1)

# analysis (all from meta):
mutation_rate = relu(Linear(576->1, kernel clamp[0,1]))( dropout(gelu(Linear(576->L)(meta))) )   # (B,1)
indel_count   = relu(Linear(576->1, kernel clamp[0,50]))( dropout(gelu(Linear(576->L)(meta))) )   # (B,1)
productive    = sigmoid(Linear(576->1)( dropout(meta) ))                                          # (B,1)

# masked classification (per gene g in V,J [,D]):
mask_g = SoftCutoutLayer(g_start_exp, g_end_exp).unsqueeze(-1)  # (B,L,1)
masked_g = emb * mask_g
cls_feat_g = g_cls_tower(masked_g)                              # (B,576)
g_allele = sigmoid(Linear(count)( swish(Linear(count*2 or latent)(cls_feat_g)) ))   # (B,count) MULTI-LABEL
```

Output dict keys (heavy, has_d): `v_start_logits,v_end_logits,d_start_logits,d_end_logits,
j_start_logits,j_end_logits, v_start,v_end,d_start,d_end,j_start,j_end (soft-argmax expectations),
v_allele,d_allele,j_allele, mutation_rate,indel_count,productive`. Light chain (`has_d=False`) drops
all `d_*` keys.

**Allele heads are SIGMOID multi-label** (a read is compatible with a *set* of alleles). D-allele
vocabulary reserves a trailing `Short-D` class (index −1).

---

## Loss (`hierarchical_loss`)

**Segmentation** (per boundary b): soft-Gaussian target `softmax(-0.5·(idx−gt)²/1.5²)`, then
`cross_entropy(b_logits, soft_target)`; weighted `× exp(-log_var_b)`. Summed over boundaries.
Plus auxiliary (from soft-argmax expectations): `0.1·Huber(len) + 0.1·IoU_loss + 0.05·hinge(span≥1)`
over V, J (+D).

**Classification** (per gene): `BCE(label_smoothing=0.1)(g_allele, multihot_target) × exp(-log_var_g)`.
Plus `short_d_length_penalty = mean( (d_span<5) · P(Short-D) )` (unweighted) — penalizes degenerate
short-D-span while calling Short-D.

**Analysis**: `mutation MAE × exp(-log_var_mut)`, `indel MAE × exp(-log_var_indel)`,
`productive BCE(no smoothing) × exp(-log_var_prod)`.

`total = segmentation + classification + weighted_mutation + weighted_indel + weighted_productive`.

**Task weighting = proper Kendall (decided).** Each weighted term is `loss_i · exp(-log_var_i) +
log_var_i` (the `+log_var_i` counter-term the TF code was missing), `log_var_i` a free scalar param
clamped to `[-3, 1]`. This makes the learned-uncertainty weighting actually functional (vs the TF
collapse-to-uniform). Applies to every weighted head: v/j/d start/end, v/j/d classification,
mutation, indel, productivity (+ chain_type for MultiChain).

---

## Fidelity decisions (need your call)

The extraction found several oddities. Default = faithful; **#1 is the one real decision.**

1. **⭐ Kendall log-variance weighting is degenerate in TF.** The TF custom `train_step` backprops
   only `total_loss` and never adds the counter-term `+log_var` (Keras `self.losses` is never summed),
   so `∂(loss·e^{-log_var})/∂log_var < 0` always → every `log_var` is driven to the clip ceiling `+1`
   → all task weights collapse to `e^{-1} ≈ 0.368` (near-uniform). **Options:**
   - **(a) Replicate faithfully** — keep the `exp(-log_var)`, clamp `[-3,1]`, no counter-term. Weights
     collapse to ~uniform (the log_var machinery is effectively vestigial). Strictly 1:1.
   - **(b) Proper Kendall (recommended)** — add the `+log_var` term so uncertainty weighting actually
     works (`loss·e^{-log_var} + log_var`). A one-line fix aligned with "stable/clean"; deviates from TF.
   - **(c) Fixed uniform weights** — drop the log_var params entirely and weight tasks by tuned
     constants (behaviorally ≈ what (a) converges to, but simpler/cleaner).

   **DECIDED → (b) proper Kendall.** Add the `+log_var` counter-term (`loss·e^{-log_var} + log_var`)
   so uncertainty weighting works. See the Loss section.

2. **Dead L1/L2 kernel regularizers** — inert in TF (same `train_step` reason). **Omit** in the port
   (faithful behavior). No decision needed unless you want real weight decay.

3. **Unusual-but-architectural hyperparameters** — replicate exactly: BatchNorm `eps=0.8`,
   `momentum=0.9` (TF 0.1 inverted), `LeakyReLU(0.3)`, `label_smoothing=0.1` (alleles only),
   productive-BCE without smoothing, mutation/indel head **kernel** clamps `[0,1]`/`[0,50]`.

4. **`padding='same'` with even kernels (2)** — TF pads asymmetrically; must match to avoid a
   ±1 boundary shift in segmentation. Implement explicit asymmetric padding + a parity unit test.

---

## Data contract & reuse

The gym already emits everything (`gym/targets.py:build_targets`): per-gene `sequence_start/end`
(query coords → segmentation targets), `mutation_rate`, `indel_count`, `productive`, plus allele
multi-hot targets via `ReferenceSet` group indices, and (multi-chain) chain type. Mapping:

| gym target | head |
|---|---|
| `{g}_sequence_start/end` | `{g}_start/end` (soft-Gaussian CE + expectation) |
| allele multi-hot (from reference groups) | `{g}_allele` (sigmoid BCE) |
| `mutation_rate` / `indel_count` / `productive` | analysis heads |
| chain type (multi) | `chain_type` (softmax) |

Tokenizer: reuse the existing 6-symbol PyTorch tokenizer (verify PAD/A/C/G/T/N ↔ 0..5 mapping and
that `vocab_size=6`). Allele counts + `has_d` come from the dataconfig/`ReferenceSet`.

---

## MultiChain deltas

Identical to SingleChain plus: `chain_type_mid = gelu(Linear(576→L))→dropout→Linear(→n_types)` off
`meta`, **softmax** head; `chain_type_loss = CategoricalCE × exp(-log_var_chain_type)` added to total.
Allele counts aggregate across chain types (`MultiDataConfigContainer`). **Open:** how D-targets are
masked for D-less chain types in a mixed batch — verify before porting the multi-chain trainer.

---

## Removal plan (after new path works)

Once SingleChain trains + benchmarks, remove the drifted experiment: `core/dnalignair.py`,
`nn/aligner/` (soft-DP: banded_dp, band_head, base_match, diagonal_ops, germline_aligner, soft_dp),
`align/` (WFA), and their training/inference/tests. Keep `reference/gym/benchmark/io/tokenizers`.

## Testing

- Unit (per layer): `TokenAndPositionEmbedding` shape/additivity; `Conv1DBatchNorm` (eps/momentum);
  `ConvResidualFeatureExtractionBlock` output `(B,576)` + halving count; `SoftCutoutLayer` mask shape
  + monotonic ramp + `end≥start+1`; even-kernel same-padding parity.
- Model: forward output-dict keys/shapes for has_d True/False; soft-argmax in `[0,L-1]`.
- Loss: soft-Gaussian target sums to 1; sigmoid-BCE path; log_var clamp; total assembles.
- Training: **overfit smoke** — a handful of reads to ~0 loss (proves the graph learns).
- Benchmark: SingleChain-heavy vs IgBLAST on the IGH assay (accuracy parity target).

## Build order (phases)

1. `layers.py` + unit tests.
2. `single_chain.py` (has_d configurable) + forward tests.
3. `losses.py` + tests.
4. `training/alignair_trainer.py` + overfit smoke; short IGH train.
5. `inference/alignair_infer.py` + benchmark vs IgBLAST (heavy).
6. `multi_chain.py` + chain-type head/loss + multi-chain train.
7. Remove drifted DNAlignAIR/soft-DP/WFA.

## Open dependencies

- Verify the existing PyTorch tokenizer vocab (6 symbols) matches TF token ids.
- `MultiDataConfigContainer` allele-count aggregation + multi-chain D-target masking.
- Confirm `build_targets` emits allele multi-hot (or add the mapping from reference groups).
