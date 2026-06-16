# AlignAIR PyTorch Migration — Design

**Date:** 2026-06-15
**Status:** Approved (structure + phasing); Phase 1 spec ready for implementation planning
**Author:** Thomas (with Claude)

## 1. Context & Goal

AlignAIR is a deep-learning tool for aligning immunoglobulin (IG) and T-cell
receptor (TCR) sequences. The current codebase (`src/AlignAIR/`, package version
2.0.2) is built on **TensorFlow / Keras 3**. The recent git history (TF 2.17.1,
Keras V3 saving) confirms it is still a TF product despite scattered partial
PyTorch experiments.

**Goal of this effort:** Migrate *all* model/training/inference logic to modern
stable **PyTorch**, removing TensorFlow/Keras entirely, and restructure the code
into a clean, professional, concern-separated package. This is a **logic +
structure migration**, not a feature change. New training schemes and a newer
GenAIRR integration come in *later* efforts, once the codebase is fully on
PyTorch.

### Decisions (confirmed with user)

| Decision | Choice |
| --- | --- |
| Weight parity with existing TF models? | **No.** Faithful logic/architecture port; retrain from scratch later. |
| New package vs in-place refactor? | **New** `src/alignair/` (lowercase), built fresh alongside the old package; delete old package only at the end. |
| Which models to keep? | **Unified Single/Multi-chain only.** Drop legacy `HeavyChain`/`LightChain` (the unified models subsume them via the `has_d_gene` flag). |
| Existing `src/AlignAIR/Pytorch/` skeleton? | **Delete it outright** — do not reference or build on it. |
| Phase-1 scope? | **Full standalone package** is the end goal; decomposed into phases. Phase 1 = DL core only. |
| Metrics implementation? | **Lightweight custom** accumulators, no new dependency (no torchmetrics). |
| Loss/metric validation? | **Numeric-equivalence tests vs the legacy TF loss** (TF still installed during migration). |
| Naming/structure constraint? | All new folders and file names **lowercase**, separated by concern. |

## 2. Overall Strategy

Build `src/alignair/` bottom-up in dependency order. `src/AlignAIR/` remains
untouched and working as a reference until the new package reaches parity, then
is deleted in the final phase. Each phase is a working, tested increment — no
big-bang cutover.

Key structural wins captured by the rewrite:

- **Collapse Single/Multi duplication.** The legacy `SingleChainAlignAIR` and
  `MultiChainAlignAIR` are ~90% identical; `MultiChain` only adds a `chain_type`
  head/loss. The rewrite uses one shared base with a thin multi-chain extension.
- **Loss as its own `nn.Module`** that owns the uncertainty-weighting
  parameters — testable in isolation, decoupled from the model.
- **Metrics live in the training loop, not the model.** Keras forced metric
  state into the model (`self.loss_tracker`, `train_step`, etc.); PyTorch should
  keep `forward()` pure (tensors in, tensors out) and compute metrics externally.
- **Config-driven construction.** A typed `ModelConfig` (derived from GenAIRR
  `DataConfig`) drives model construction, so models rebuild deterministically
  from a saved config.

## 3. Target Package Layout

```
src/alignair/
  config/         model + training config dataclasses (typed, serializable)
  nn/             reusable building blocks (framework primitives)
                    embedding.py   token + position embedding
                    conv.py        Conv1d+BN, residual feature extractor
                    heads.py       segmentation / classification / analysis heads
                    masking.py     soft-cutout differentiable mask
                    weighting.py   Kendall uncertainty log-var
  core/           model definitions (compose nn/ blocks)
                    base.py        shared AlignAIR nn.Module
                    single_chain.py / multi_chain.py
  losses/         hierarchical.py  AlignAIRLoss as its own nn.Module
  metrics/        boundary.py, entropy.py, allele_auc.py, average_last_label.py
  data/           dataset.py, tokenizer.py, encoders.py, collate.py, simulation.py
  training/       trainer.py (AMP, grad-clip, resume), callbacks.py, loop.py
  serialization/  bundle.py (state_dict + config.json + dataconfig + meta), hub.py
  inference/      predictor.py, pipeline.py, postprocessing/
  preprocessing/  orientation.py (PyTorch), sequence prep, dataconfig loading
  io/             sequence_reader.py (CSV/TSV/FASTA)
  reporting/      reports + plots
  cli/            app.py (typer)
  utils/
```

Framework-agnostic legacy modules (post-processing, reporting, io, much of
preprocessing/pipeline) are **moved and reorganized**, not rewritten. Only the
~33 TF-coupled files get true rewrites.

## 4. Phase Decomposition

Each phase gets its own spec → implementation plan → build cycle.

1. **DL core** (this spec) — `config/`, `nn/`, `core/`, `losses/`, `metrics/`.
   Pure PyTorch model + blocks + loss + metrics.
   *Done when:* forward pass runs on dummy input with correct shapes; loss is
   finite and backprops; numeric-equivalence + unit tests green.
2. **Data + Training** — `data/` (torch `Dataset`/`DataLoader`, tokenizer,
   encoders) + `training/` (trainer loop, callbacks, AMP, checkpoint/resume).
   *Done when:* model trains for N steps on `tests/data` sample CSV and loss
   decreases.
3. **Serialization + Inference** — `serialization/` (state_dict bundle replacing
   TF SavedModel), `inference/` (predictor + pipeline + postprocessing port),
   PyTorch orientation model.
   *Done when:* save a trained model, reload, predict end-to-end on CSV/FASTA.
4. **CLI + Reporting + Hub + teardown** (low priority — user will revise CLI
   after) — port `app.py`/CLI, reports, HF hub; flip entry points; **delete
   `src/AlignAIR`**; update `pyproject.toml` (drop `tensorflow`, add `torch`).
   *Done when:* full CLI workflow runs on the new package and the old one is gone.

## 5. Phase 1 Detailed Spec — DL Core

### 5.1 Modules

**`config/model_config.py`**
- `ModelConfig` dataclass: `max_seq_length`, `v_allele_count`, `j_allele_count`,
  `d_allele_count` (optional), `has_d_gene`, latent sizes (`v/d/j_allele_latent_size`),
  `latent_size_factor=2`, activations (`classification_middle="swish"`,
  `fblock="tanh"`), and multi-chain fields (`number_of_chains`, `chain_types`).
- `ModelConfig.from_dataconfig(dataconfig)` and
  `ModelConfig.from_dataconfigs(container)` classmethods to derive from GenAIRR
  configs. JSON round-trippable (`to_dict` / `from_dict`).

**`nn/` — reusable blocks (each a plain `nn.Module`):**
- `embedding.py` → `TokenPositionEmbedding(vocab_size=6, embed_dim=32, max_len)`:
  token embedding + learned position embedding, summed.
- `conv.py` →
  - `Conv1dBatchNorm(in_ch, out_ch, kernel, activation)`: Conv1d → BatchNorm1d →
    activation. Operates channel-first internally.
  - `ConvResidualFeatureExtractor(filter_size=128, num_conv_layers, kernel_sizes,
    max_pool_size, activation)`: the legacy `ConvResidualFeatureExtractionBlock`
    — stacked conv-bn layers with residual/skip connections, max-pool, flatten →
    `LazyLinear`. **Note:** `LazyLinear` requires a dummy forward before
    `save`/`.to(device)`; the model init contract performs this.
- `heads.py` →
  - `SegmentationHead(in_features, max_seq_length)`: `Linear(L)` producing
    per-position boundary logits (softmax applied in the model/loss, not here).
  - `AlleleClassificationHead(in_features, latent_dim, num_alleles, activation,
    regularizer=None)`: mid `Linear` (swish) → `Linear` (num_alleles) → sigmoid.
  - `MutationRateHead`, `IndelCountHead` (mid GELU + dropout + constrained
    output), `ProductivityHead` (flatten + dropout + sigmoid),
    `ChainTypeHead(num_types)` (mid GELU + dropout + softmax).
- `masking.py` → `SoftCutout(gene, max_size, k=3.0)`: differentiable soft mask
  computed from start/end position expectations (preserves gradients near
  boundaries, respects `[start:end)`).
- `weighting.py` → `UncertaintyWeight` (legacy `RegularizedConstrainedLogVar`):
  Kendall uncertainty weighting with a constrained log-variance parameter and
  the `+0.5*log_var` regularization penalty.

**`core/`**
- `base.py` → `BaseAlignAIR(nn.Module)` composing: embedding → meta + V/J (+ D)
  segmentation feature extractors → segmentation heads → expectations → soft
  masks → masked classification feature extractors → allele heads → analysis
  heads. `forward(tokenized_sequence)` returns an `AlignAIROutput` of **pure
  tensors**: `{v/j(/d)_start_logits, *_end_logits, *_start, *_end (expectations),
  v/j(/d)_allele, mutation_rate, indel_count, productive}`. `has_d_gene` is a
  config conditional.
- `single_chain.py` → `SingleChainAlignAIR(BaseAlignAIR)` — config only.
- `multi_chain.py` → `MultiChainAlignAIR(BaseAlignAIR)` — adds `chain_type` head
  + output field.
- `AlignAIROutput` typed container (dataclass or `TypedDict`) for clarity over
  the legacy raw dict.

**`losses/hierarchical.py`** → `AlignAIRLoss(nn.Module)` owning all
`UncertaintyWeight` params. Faithfully ports legacy `hierarchical_loss`:
- Segmentation: soft-target cross-entropy per boundary (Gaussian soft targets,
  `sigma=1.5`), each weighted by its `UncertaintyWeight`.
- Auxiliary segmentation: `0.1*Huber(length)` + `0.1*(1-IoU)` + `0.05*hinge`.
- Classification: BCE with `label_smoothing=0.1`, per-gene uncertainty-weighted;
  short-D length penalty when `has_d_gene`.
- Analysis: mutation-rate MAE, indel-count MAE, productivity BCE — each
  uncertainty-weighted.
- Chain type: categorical CE (multi-chain only), uncertainty-weighted.
- Per-layer Keras `kernel_regularizer` (l1/l2) terms that PyTorch lacks built-in
  are **folded into the loss explicitly** (not silently dropped).
- Returns `(total_loss, components_dict)`.
- Helpers: `soft_targets`, `expectation_from_logits`, `interval_iou_loss`.

**`metrics/`** — lightweight stateful accumulators (`update`/`compute`/`reset`,
mirroring `keras.metrics.Mean` semantics), no new dependency:
- `boundary.py` → boundary MAE, exact-match accuracy, ±1nt accuracy from logits +
  ground-truth scalar.
- `entropy.py` → allele prediction entropy.
- `allele_auc.py` → multi-label AUC (custom implementation).
- `average_last_label.py` → legacy `AverageLastLabel`.

### 5.2 Porting conventions (applied throughout Phase 1)

- **Tensor layout:** External interface accepts `(B, L)` integer tokens. TF conv
  is channel-last `(B,L,C)`; PyTorch `Conv1d` is channel-first `(B,C,L)` →
  internal transposes at conv boundaries.
- **Initializers:** Glorot/GlorotUniform → `nn.init.xavier_uniform_`.
- **Activations:** swish → `SiLU`, gelu → `GELU`, plus `tanh`/`relu`/`sigmoid`.
- **Constraints:** `MinMaxValueConstraint` → weight clamping (parametrization);
  `unit_norm` → normalized parametrization where used.
- **Regularizers:** l1/l2 per-layer → explicit terms in `AlignAIRLoss`.
- **LazyLinear:** model init runs a dummy forward to materialize lazy params.

### 5.3 Testing strategy

- **Shape/forward tests:** `SingleChain` and `MultiChain`, with and without D
  gene, on synthetic dummy token tensors; assert every output key and shape.
- **Loss tests:** finite total loss; `backward()` produces finite grads for all
  parameters (grad-flow check); component breakdown keys present.
- **Numeric-equivalence tests vs legacy TF** (TF still installed during
  migration): run the legacy `hierarchical_loss` and the new `AlignAIRLoss` on
  identical toy inputs (same weights where applicable, or compare unweighted
  sub-terms) and assert agreement within tolerance. Same approach for metrics.
- **Per-block tests:** each `nn/` block independently shape-checked.
- Tests live under `tests/` following the existing unit/integration layout.

### 5.4 Out of scope for Phase 1

- No `data/`, `training/`, `serialization/`, `inference/`, `cli/` (later phases).
- No real data loading or training run (synthetic tensors only).
- No weight conversion from TF models.

## 6. Risks & Notes

- **Numeric-equivalence tolerance:** softmax/CE and reductions differ subtly
  between TF and PyTorch; tests assert closeness within tolerance, not bitwise
  equality.
- **LazyLinear ordering:** forgetting the dummy forward before `.to(device)` or
  save is a known footgun — encoded into the model contract and covered by a test.
- **Regularizer fidelity:** l1/l2 terms must be reproduced in the loss to keep
  training dynamics comparable; explicitly enumerated from the legacy heads.
