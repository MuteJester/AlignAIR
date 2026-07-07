# Clean PyTorch AlignAIR — Implementation Plan (Core: trainable SingleChain)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Build a faithful, clean PyTorch `SingleChainAlignAIR` that trains on the GenAIRR gym stream
(overfit smoke passes). Design: [docs/superpowers/specs/2026-07-07-clean-pytorch-alignair-design.md](../specs/2026-07-07-clean-pytorch-alignair-design.md).
Architecture detail: the code-explorer extraction (this session) — refer to the spec for exact numbers.

**Architecture:** 1:1 with TF `src/AlignAIR/Models/` (on `main`): token+position embedding → 7
`ConvResidualFeatureExtractionBlock` towers → soft-argmax segmentation → differentiable soft-cutout
masking → sigmoid multi-label allele heads + mutation/indel/productivity heads; proper-Kendall loss.

**Tech stack:** PyTorch (`nn.Module`, channels-first conv). Reuse `alignair.reference`, `alignair.gym`,
tokenizers, `alignair.benchmark`.

## Global constraints
- Faithful architecture 1:1. Retrain fresh (no TF weights). **Proper Kendall** weighting
  (`loss·exp(-log_var) + log_var`, `log_var` clamped `[-3,1]`).
- Replicate exactly: BatchNorm `eps=0.8`, `momentum=0.9`; `LeakyReLU(0.3)`; `tanh` conv activation;
  allele heads **sigmoid** + `BCE(label_smoothing=0.1)`; productive BCE **no** smoothing; mutation/indel
  head **kernel** clamps `[0,1]`/`[0,50]`; even-kernel `same` padding matched to TF (asymmetric).
- Omit L1/L2 kernel regularizers (inert in TF). New code in `src/alignair/models/`.
- No Claude trailer in commits. Commit only when the user asks (steps say "commit" — stage, and the
  human decides when to actually commit).
- Run tests with `PYTHONPATH=src .venv/bin/python -m pytest`.

---

### Task 1: Config + TokenAndPositionEmbedding

**Files:** Create `src/alignair/config/alignair_config.py`, `src/alignair/models/__init__.py`,
`src/alignair/models/layers.py`; Test `tests/alignair/models/test_layers.py`.

**Interfaces — Produces:** `AlignAIRConfig(max_seq_length=576, v_allele_count, d_allele_count,
j_allele_count, has_d, vocab_size=6, embed_dim=32, v/d/j_allele_latent_size=None)`;
`TokenAndPositionEmbedding(vocab_size, embed_dim, maxlen)` → `(B, L, embed_dim)`.

- [ ] **Step 1: failing test**
```python
import torch
from alignair.models.layers import TokenAndPositionEmbedding
def test_token_pos_embedding_shape_and_additivity():
    emb = TokenAndPositionEmbedding(vocab_size=6, embed_dim=32, maxlen=16)
    x = torch.randint(0, 6, (4, 16))
    out = emb(x)
    assert out.shape == (4, 16, 32)
    # position component is input-independent: difference of two token seqs == token-emb difference only
    x2 = torch.randint(0, 6, (4, 16))
    assert torch.allclose((emb(x) - emb(x2)), emb.token(x) - emb.token(x2), atol=1e-6)
```
- [ ] **Step 2:** `pytest tests/alignair/models/test_layers.py::test_token_pos_embedding_shape_and_additivity -v` → FAIL (import).
- [ ] **Step 3: implement** — `AlignAIRConfig` dataclass; `TokenAndPositionEmbedding(nn.Module)` with
  `self.token = nn.Embedding(vocab_size, embed_dim)`, `self.pos = nn.Embedding(maxlen, embed_dim)`;
  `forward(x)`: `self.token(x.long()) + self.pos(torch.arange(maxlen, device=x.device))`.
- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** commit `feat(models): AlignAIRConfig + TokenAndPositionEmbedding`.

---

### Task 2: Conv1DBatchNorm (TF-faithful: eps/momentum/same-padding)

**Files:** Modify `src/alignair/models/layers.py`; Test `tests/alignair/models/test_layers.py`.

**Interfaces — Produces:** `Conv1DBatchNorm(filters=128, kernel, pool=2, act='tanh')` operating on
channels-first `(B, C, L)` → `(B, filters, L//pool)`. 3 stacked convs (no interleaved activation) →
BN(`eps=0.8, momentum=0.9`) → act → MaxPool1d(pool).

- [ ] **Step 1: failing test** — verify output shape + that even-kernel `same` padding preserves L
  before pooling (parity vs a hand-padded conv):
```python
from alignair.models.layers import Conv1DBatchNorm, same_pad1d
def test_conv_bn_shapes_and_even_kernel_same_padding():
    blk = Conv1DBatchNorm(filters=8, kernel=2, pool=2).eval()
    x = torch.randn(2, 8, 17)   # in_channels must match first conv; use filters=in for test simplicity
    y = blk(x)
    assert y.shape == (2, 8, 8)          # 17 -> same(17) -> pool2 -> 8
    # TF 'same' for even kernel pads (0 left, 1 right) for stride 1
    assert same_pad1d(kernel=2) == (0, 1)
    assert same_pad1d(kernel=3) == (1, 1)
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** — `same_pad1d(kernel, dilation=1)` returning `(left, right)` matching TF
  (`total = dilation*(kernel-1)`, `left = total//2`, `right = total-left`). `Conv1DBatchNorm`: three
  `nn.Conv1d(in?, filters, kernel)` with manual `F.pad` using `same_pad1d` (lazy first-conv in-channels
  via `nn.LazyConv1d` or pass `in_channels`), `nn.BatchNorm1d(filters, eps=0.8, momentum=0.9)`,
  `act` (`torch.tanh`), `nn.MaxPool1d(pool)`. First conv `in_channels` from a ctor arg.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(models): Conv1DBatchNorm with TF-faithful eps/momentum/same-padding`.

---

### Task 3: ConvResidualFeatureExtractionBlock

**Files:** Modify `layers.py`; Test `test_layers.py`.

**Interfaces — Produces:** `ConvResidualFeatureExtractionBlock(in_channels, filters=128, N,
kernels(len N+1), out=576, conv_act='tanh')` → `(B, 576)`; internal length halving = `N+1`.

- [ ] **Step 1: failing test**
```python
from alignair.models.layers import ConvResidualFeatureExtractionBlock
def test_resblock_output_and_halving():
    blk = ConvResidualFeatureExtractionBlock(in_channels=32, filters=128, N=4,
                                             kernels=[3,3,3,2,5], out=576).eval()
    x = torch.randn(2, 512, 32)          # (B, L, C) input convention at model level
    y = blk(x)
    assert y.shape == (2, 576)
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** per spec "ConvResidualFeatureExtractionBlock" forward (index-0 double pool
  via reused `MaxPool1d(2)`, `LeakyReLU(0.3)` after each residual `Add`, residual channel =
  `Conv1d(filters, kernels[-1], same)` no activation). Accept `(B, L, C)`, transpose to channels-first
  internally, `Flatten` + `nn.Linear(filters * L_final, out)`. Compute `L_final = L >> (N+1)`.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(models): ConvResidualFeatureExtractionBlock (residual conv tower)`.

---

### Task 4: SoftCutoutLayer + RegularizedConstrainedLogVar

**Files:** Modify `layers.py`; Test `test_layers.py`.

**Interfaces — Produces:** `SoftCutoutLayer(max_size, k=3.0)(start,end)` → `(B, max_size)` soft mask;
`RegularizedConstrainedLogVar()` with `.weight()` → `exp(-log_var)` and `.kendall(loss)` →
`loss*exp(-log_var) + log_var`, `log_var` clamped `[-3,1]` via a `clamp_params()` hook.

- [ ] **Step 1: failing test**
```python
from alignair.models.layers import SoftCutoutLayer, RegularizedConstrainedLogVar
def test_soft_cutout_and_logvar():
    m = SoftCutoutLayer(max_size=10, k=3.0)
    start = torch.tensor([[2.0]]); end = torch.tensor([[6.0]])
    mask = m(start, end)
    assert mask.shape == (1, 10)
    assert mask[0, 4] > mask[0, 0] and mask[0, 4] > mask[0, 9]   # peak inside interval
    lv = RegularizedConstrainedLogVar()
    loss = torch.tensor(2.0)
    k = lv.kendall(loss)
    assert torch.isclose(k, loss * torch.exp(-lv.log_var) + lv.log_var)
    lv.log_var.data.fill_(5.0); lv.clamp_params()
    assert lv.log_var.item() == 1.0                              # clamp ceiling
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** `SoftCutoutLayer` per spec (`end = max(clamp(end), clamp(start)+1)`,
  `sigmoid((idx-start)/k)*sigmoid((end-idx)/k)`). `RegularizedConstrainedLogVar(nn.Module)` with
  `self.log_var = nn.Parameter(torch.zeros(()))`, `weight()`, `kendall(loss)`, `clamp_params()` doing
  `self.log_var.data.clamp_(-3, 1)`.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(models): SoftCutoutLayer + proper-Kendall RegularizedConstrainedLogVar`.

---

### Task 5: SingleChainAlignAIR (forward + heads)

**Files:** Create `src/alignair/models/single_chain.py`; Test `tests/alignair/models/test_single_chain.py`.

**Interfaces — Consumes:** all of `layers.py`, `AlignAIRConfig`. **Produces:**
`SingleChainAlignAIR(cfg)(batch)` where `batch["tokenized_sequence"]:(B,L) long` → output dict per spec.

- [ ] **Step 1: failing test** — output keys/shapes for has_d True and False, and soft-argmax range:
```python
from alignair.config.alignair_config import AlignAIRConfig
from alignair.models.single_chain import SingleChainAlignAIR
def test_singlechain_forward_heavy_and_light():
    for has_d in (True, False):
        cfg = AlignAIRConfig(max_seq_length=64, v_allele_count=30, d_allele_count=8,
                             j_allele_count=6, has_d=has_d)
        m = SingleChainAlignAIR(cfg).eval()
        out = m({"tokenized_sequence": torch.randint(0, 6, (3, 64))})
        for k in ["v_start","v_end","j_start","j_end","v_allele","j_allele",
                  "mutation_rate","indel_count","productive"]:
            assert k in out
        assert out["v_allele"].shape == (3, 30)
        assert (out["v_start"] >= 0).all() and (out["v_start"] <= 63).all()
        assert ("d_allele" in out) == has_d
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** `SingleChainAlignAIR` per spec "Heads & forward": build 7 towers (skip d_*
  when `not has_d`), segmentation `nn.Linear(576, L)` per boundary → softmax → soft-argmax expectation
  (`(probs * arange(L)).sum(-1, keepdim=True)`), analysis heads (gelu mid → dropout → head; mutation
  relu + kernel-clamp registered for `clamp_params`, indel same), soft-cutout mask → masked embeddings
  → cls towers → allele heads (swish mid → sigmoid). Provide `clamp_params()` clamping mutation/indel
  head weights + all log_vars.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(models): SingleChainAlignAIR faithful forward + heads`.

---

### Task 6: Loss (hierarchical, proper Kendall)

**Files:** Create `src/alignair/models/losses.py`; Test `tests/alignair/models/test_losses.py`.

**Interfaces — Produces:** `hierarchical_loss(out, targets, cfg, logvars) -> (total, parts_dict)`.
`logvars`: an `nn.ModuleDict` of `RegularizedConstrainedLogVar` per weighted head (owned by trainer).

- [ ] **Step 1: failing test** — soft-Gaussian target sums to 1; total is finite + decreases when
  predictions match targets:
```python
from alignair.models.losses import soft_gaussian_target, hierarchical_loss
def test_soft_gaussian_and_total():
    t = soft_gaussian_target(torch.tensor([5.0, 10.0]), L=32, sigma=1.5)
    assert t.shape == (2, 32)
    assert torch.allclose(t.sum(-1), torch.ones(2), atol=1e-5)
    # (build a minimal out/targets/logvars dict; assert total is a finite scalar > 0)
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** per spec "Loss": `soft_gaussian_target`; per-boundary CE(logits, soft
  target) → `logvars[b].kendall(...)`; aux len-Huber/IoU/hinge (0.1/0.1/0.05, from expectations);
  classification `BCEWithLogits`? — heads already sigmoid, so use `F.binary_cross_entropy` with
  `label_smoothing` applied manually (`y*(1-0.1)+0.05`) → `logvars[g].kendall`; `short_d_length_penalty`;
  analysis MAE (mutation/indel) + productive BCE(no smoothing), each `.kendall`. Sum → total.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(models): hierarchical multi-task loss (proper Kendall)`.

---

### Task 7: Trainer + overfit smoke

**Files:** Create `src/alignair/training/alignair_trainer.py`, `scripts/train_alignair.py`;
Test `tests/alignair/training/test_alignair_overfit.py`.

**Interfaces — Consumes:** gym stream (`alignair.gym`), `SingleChainAlignAIR`, `hierarchical_loss`,
`ReferenceSet`. **Produces:** a training step that calls `optimizer.step()` then `model.clamp_params()`
+ `logvars` clamps; `train(...)` loop.

- [ ] **Step 1: failing test** — overfit a tiny fixed batch to near-zero segmentation+classification
  loss in <300 steps (proves the graph learns end-to-end):
```python
def test_overfit_tiny_batch():
    # build cfg from HUMAN_IGH_OGRDB reference; draw ONE small gym batch; repeat it.
    # train 300 steps; assert final total loss < 0.5 * initial loss (learning signal).
    ...
```
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement** `alignair_trainer.train_step`/`train`: map gym batch → model input +
  targets (allele multi-hot from reference groups, coords, mutation/indel/productive); AdamW; after
  `opt.step()` call `model.clamp_params()`; log per-part losses. `scripts/train_alignair.py` CLI
  (dataconfig, steps, batch-size, out).
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(training): AlignAIR trainer + overfit smoke`.

---

## Definition of done (this plan)
- All 7 tasks' tests green; overfit smoke passes.
- A short real IGH training run reduces held-out segmentation+allele loss (sanity, not gated here).

## Follow-on plans (separate)
1. **Inference + benchmark**: `inference/alignair_infer.py` (allele set via cumulative-confidence;
   coords from expectations) → AIRR contract → `run_h2h_benchmark` vs IgBLAST (heavy). Accuracy parity.
2. **MultiChain**: `multi_chain.py` + `chain_type` softmax head/loss + multi-chain trainer.
3. **Remove drift**: delete `core/dnalignair.py`, `nn/aligner/`, `align/` + their training/inference/tests.

## Self-review notes
- Spec coverage: Layers (T1-4), SingleChain (T5), loss incl. proper Kendall (T6), trainer+smoke (T7). ✓
- Open items deferred to follow-on plans (inference/benchmark, MultiChain, removal) — intentional.
- Verify during T7: `build_targets` emits allele multi-hot (else add reference-group mapping); tokenizer
  vocab is 6 (PAD/A/C/G/T/N ↔ 0..5).
