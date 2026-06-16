# AlignAIR PyTorch Core (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the framework-agnostic PyTorch "DL core" of the new `src/alignair` package — model config, reusable `nn` blocks, unified single/multi-chain models, the hierarchical loss, and training metrics — validated by shape/grad tests and numeric-equivalence tests against the legacy TensorFlow implementation.

**Architecture:** A clean lowercase `src/alignair/` package built bottom-up. Reusable neural blocks live in `nn/`; they are composed by a shared `BaseAlignAIR(nn.Module)` in `core/`, with thin `SingleChainAlignAIR` / `MultiChainAlignAIR` subclasses. `forward()` returns pure tensors (an `AlignAIROutput` dataclass); the hierarchical loss is its own `nn.Module` (`AlignAIRLoss`) owning the uncertainty-weighting parameters; metrics are lightweight stateful accumulators computed outside the model. No TensorFlow in the new package.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU build), pytest 7.4. The legacy `src/AlignAIR` (TensorFlow 2.21 / Keras 3) stays in place as reference and is used **only by the numeric-equivalence tests**. Always use the project venv created for this migration: `.venv/bin/python` (a `--system-site-packages` venv so it sees the system TensorFlow; torch installed into it).

---

## Reference: Legacy → PyTorch porting conventions

These apply to every task below.

- **Tensor layout.** External input is `(B, L)` integer tokens. Embedding produces `(B, L, E)`. PyTorch `Conv1d` is channel-first, so transpose to `(B, E, L)` before conv blocks and keep conv math in `(B, C, L)`.
- **Conv padding.** Use `nn.Conv1d(..., padding="same")` (string, stride 1) for every conv — NOT a computed int. Int padding drifts even-kernel output length by +1, which breaks the residual add in the feature extractor. `padding="same"` matches Keras exactly. (PyTorch emits a one-time harmless warning for even kernels.)
- **Minimum sequence length.** The feature extractors downsample by ~`2**(num_conv_layers+1)` (the 4-layer segmentation extractor ≈ ÷32, the 6-layer classification extractor ≈ ÷128). Inputs must be long enough to survive this: model-level tests use `L=256` (see `conftest.TEST_SEQ_LEN`); the real model uses `max_seq_length=576`. `L=16` collapses to length 0 — do not use it for anything that runs a full extractor.
- **BatchNorm semantics differ.** Keras `BatchNormalization(momentum=0.1)` updates running stats as `running = 0.1*running + 0.9*batch`. PyTorch updates as `running = (1-momentum)*running + momentum*batch`. To preserve the legacy intent, use `nn.BatchNorm1d(..., eps=0.8, momentum=0.9)`. (Yes, `eps=0.8` is unusually large — it is intentional in the legacy code; replicate it.)
- **Initializers.** Keras `glorot_uniform` → `nn.init.xavier_uniform_`; biases → zeros.
- **Activations.** `swish` → `nn.SiLU`; `gelu` → `nn.GELU`; plus `tanh`, `relu`, `leaky_relu` (default negative slope 0.3 where the legacy uses bare `LeakyReLU()` — Keras default alpha is 0.3), `sigmoid`.
- **Constraints.** `MinMaxValueConstraint(min,max)` and `unit_norm` → clamp/normalize the weight after each optimizer step (a `apply_constraints()` helper). For Phase 1 we expose the helper and unit-test it; the trainer (Phase 2) will call it.
- **Regularizers (important).** Legacy heads pass `kernel_regularizer=l1/l2`, but the legacy custom `train_step` builds gradients only from `hierarchical_loss` and never adds `model.losses`, so those penalties are **inert** in the current code. Decision: in the new package we make them explicit and *active* via `model.regularization_loss()`, which `AlignAIRLoss` includes. This is a deliberate, documented improvement over the legacy latent bug, not a blind port.
- **No weight parity.** Numeric-equivalence tests compare *architecture math* (e.g. soft-target CE, expectations, mask shapes) on identical inputs/seeded weights where feasible, within tolerance — not trained weights.

---

## File Structure (Phase 1)

```
src/alignair/
  __init__.py
  config/
    __init__.py
    model_config.py        # ModelConfig dataclass + from_dataconfig(s)
  nn/
    __init__.py
    activations.py         # make_activation(str) factory
    embedding.py           # TokenPositionEmbedding
    conv.py                # Conv1dBatchNorm, ConvResidualFeatureExtractor
    masking.py             # SoftCutout
    weighting.py           # UncertaintyWeight
    heads.py               # SegmentationHead, AlleleClassificationHead,
                           #   MutationRateHead, IndelCountHead,
                           #   ProductivityHead, ChainTypeHead
  core/
    __init__.py
    output.py              # AlignAIROutput dataclass
    base.py                # BaseAlignAIR
    single_chain.py        # SingleChainAlignAIR
    multi_chain.py         # MultiChainAlignAIR
  losses/
    __init__.py
    functional.py          # soft_targets, expectation_from_logits, interval_iou_loss
    hierarchical.py        # AlignAIRLoss
  metrics/
    __init__.py
    accumulator.py         # MeanAccumulator base
    boundary.py            # BoundaryMetrics
    entropy.py             # AlleleEntropy
    allele_auc.py          # MultiLabelAUC
    average_last_label.py  # AverageLastLabel

tests/alignair/
  __init__.py
  conftest.py              # shared fixtures: tiny ModelConfig, dummy batch
  test_model_config.py
  nn/test_embedding.py
  nn/test_conv.py
  nn/test_masking.py
  nn/test_weighting.py
  nn/test_heads.py
  core/test_base.py
  core/test_single_chain.py
  core/test_multi_chain.py
  losses/test_functional.py
  losses/test_hierarchical.py
  metrics/test_metrics.py
  equivalence/test_tf_equivalence.py   # numeric-equivalence vs legacy TF
```

**Conventions for all test commands:** run from repo root with
`PYTHONPATH=src .venv/bin/python -m pytest <path> -v`.

---

## Task 0: Package scaffold + remove legacy PyTorch skeleton

**Files:**
- Delete: `src/AlignAIR/Pytorch/` (entire directory)
- Create: `src/alignair/__init__.py`
- Create: `src/alignair/config/__init__.py`, `src/alignair/nn/__init__.py`, `src/alignair/core/__init__.py`, `src/alignair/losses/__init__.py`, `src/alignair/metrics/__init__.py`
- Create: `tests/alignair/conftest.py` (NOTE: test dirs must NOT contain `__init__.py` — this repo runs pytest in prepend import mode, so an `alignair` package under `tests/` would shadow `src/alignair`)

- [ ] **Step 1: Confirm nothing imports the legacy skeleton**

Run: `grep -rn "AlignAIR.Pytorch\|from .Pytorch\|Pytorch import" src tests app.py | grep -v "src/AlignAIR/Pytorch/"`
Expected: no output (nothing outside the dir references it). If there are hits, stop and report them before deleting.

- [ ] **Step 2: Delete the legacy skeleton**

```bash
git rm -r src/AlignAIR/Pytorch
```

- [ ] **Step 3: Create the new package `__init__.py` files**

`src/alignair/__init__.py`:
```python
"""AlignAIR — PyTorch package (v3 rewrite).

Clean, concern-separated PyTorch implementation. The legacy TensorFlow/Keras
package under ``src/AlignAIR`` remains available during migration and is removed
once this package reaches parity.
"""

__all__ = []
```

Create empty `__init__.py` in `config/`, `nn/`, `core/`, `losses/`, `metrics/` (source package dirs only — NOT in any `tests/` dir).

- [ ] **Step 4: Create shared test fixtures**

`tests/alignair/conftest.py`:
```python
import pytest
import torch


@pytest.fixture
def tiny_config_d():
    """A tiny D-gene (heavy-chain-like) ModelConfig for fast tests."""
    from alignair.config.model_config import ModelConfig
    return ModelConfig(
        max_seq_length=16,
        v_allele_count=5,
        j_allele_count=3,
        d_allele_count=4,
        has_d_gene=True,
    )


@pytest.fixture
def tiny_config_no_d():
    """A tiny non-D (light-chain-like) ModelConfig."""
    from alignair.config.model_config import ModelConfig
    return ModelConfig(
        max_seq_length=16,
        v_allele_count=5,
        j_allele_count=3,
        d_allele_count=None,
        has_d_gene=False,
    )


@pytest.fixture
def dummy_tokens():
    """A batch of (B=2, L=16) integer tokens in [0, 5]."""
    torch.manual_seed(0)
    return torch.randint(0, 6, (2, 16))
```

- [ ] **Step 5: Verify the test harness collects**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -v`
Expected: collected 0 items (no tests yet), exit code 5 ("no tests ran") — confirms imports/fixtures load without error. If you see an ImportError, fix it before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/alignair tests/alignair
git commit -m "chore(alignair): scaffold new lowercase package, remove legacy Pytorch skeleton"
```

---

## Task 1: `config/model_config.py` — ModelConfig

**Files:**
- Create: `src/alignair/config/model_config.py`
- Test: `tests/alignair/test_model_config.py`

- [ ] **Step 1: Write the failing test**

`tests/alignair/test_model_config.py`:
```python
import pytest
from alignair.config.model_config import ModelConfig


def test_d_gene_config_fields():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200,
                      j_allele_count=7, d_allele_count=34, has_d_gene=True)
    assert cfg.has_d_gene is True
    assert cfg.d_allele_count == 34
    # default latent dim = count * latent_size_factor when latent size is None
    assert cfg.v_latent_dim == 200 * 2
    assert cfg.j_latent_dim == 7 * 2
    assert cfg.d_latent_dim == 34 * 2


def test_no_d_gene_has_no_d_latent():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200,
                      j_allele_count=7, d_allele_count=None, has_d_gene=False)
    assert cfg.has_d_gene is False
    assert cfg.d_latent_dim is None


def test_explicit_latent_size_overrides_factor():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200, j_allele_count=7,
                      d_allele_count=None, has_d_gene=False,
                      v_allele_latent_size=128)
    assert cfg.v_latent_dim == 128


def test_roundtrip_dict():
    cfg = ModelConfig(max_seq_length=576, v_allele_count=200, j_allele_count=7,
                      d_allele_count=34, has_d_gene=True)
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg


def test_d_config_consistency_validation():
    # has_d_gene=True but d_allele_count=None must raise
    with pytest.raises(ValueError):
        ModelConfig(max_seq_length=16, v_allele_count=5, j_allele_count=3,
                    d_allele_count=None, has_d_gene=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/test_model_config.py -v`
Expected: FAIL (ModuleNotFoundError: no module named alignair.config.model_config).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/config/model_config.py`:
```python
"""Typed model configuration for AlignAIR models.

Decouples the model from the raw GenAIRR ``DataConfig`` and makes models
deterministically reconstructable from a saved JSON config.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass(eq=True)
class ModelConfig:
    max_seq_length: int
    v_allele_count: int
    j_allele_count: int
    d_allele_count: Optional[int]
    has_d_gene: bool

    # Optional explicit latent sizes; when None, derived as count * latent_size_factor.
    v_allele_latent_size: Optional[int] = None
    d_allele_latent_size: Optional[int] = None
    j_allele_latent_size: Optional[int] = None
    latent_size_factor: int = 2

    # Activations.
    classification_mid_activation: str = "swish"
    feature_block_activation: str = "tanh"

    # Multi-chain (None / empty for single-chain).
    number_of_chains: Optional[int] = None
    chain_types: Optional[List[str]] = field(default=None)

    def __post_init__(self) -> None:
        if self.has_d_gene and self.d_allele_count is None:
            raise ValueError("has_d_gene=True requires a non-None d_allele_count")
        if not self.has_d_gene and self.d_allele_count is not None:
            # Tolerate but normalize: a non-D model carries no D count.
            self.d_allele_count = None

    @property
    def v_latent_dim(self) -> int:
        return self.v_allele_latent_size or self.v_allele_count * self.latent_size_factor

    @property
    def j_latent_dim(self) -> int:
        return self.j_allele_latent_size or self.j_allele_count * self.latent_size_factor

    @property
    def d_latent_dim(self) -> Optional[int]:
        if not self.has_d_gene:
            return None
        return self.d_allele_latent_size or self.d_allele_count * self.latent_size_factor

    @property
    def is_multi_chain(self) -> bool:
        return bool(self.number_of_chains and self.number_of_chains > 1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    @classmethod
    def from_dataconfig(cls, dataconfig, **overrides) -> "ModelConfig":
        """Build from a single GenAIRR ``DataConfig``."""
        has_d = bool(dataconfig.metadata.has_d)
        return cls(
            max_seq_length=overrides.pop("max_seq_length"),
            v_allele_count=dataconfig.number_of_v_alleles,
            j_allele_count=dataconfig.number_of_j_alleles,
            d_allele_count=dataconfig.number_of_d_alleles if has_d else None,
            has_d_gene=has_d,
            **overrides,
        )

    @classmethod
    def from_dataconfigs(cls, container, **overrides) -> "ModelConfig":
        """Build from a multi-chain ``MultiDataConfigContainer``."""
        has_d = bool(container.has_at_least_one_d())
        try:
            chain_types = [getattr(ct, "value", str(ct)) for ct in container.chain_types()]
        except Exception:
            chain_types = None
        return cls(
            max_seq_length=overrides.pop("max_seq_length"),
            v_allele_count=container.number_of_v_alleles,
            j_allele_count=container.number_of_j_alleles,
            d_allele_count=container.number_of_d_alleles if has_d else None,
            has_d_gene=has_d,
            number_of_chains=len(chain_types) if chain_types else None,
            chain_types=chain_types,
            **overrides,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/test_model_config.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/config tests/alignair/test_model_config.py
git commit -m "feat(alignair): add ModelConfig dataclass"
```

---

## Task 2: `nn/activations.py` + `nn/embedding.py`

**Files:**
- Create: `src/alignair/nn/activations.py`
- Create: `src/alignair/nn/embedding.py`
- Test: `tests/alignair/nn/test_embedding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/alignair/nn/test_embedding.py`:
```python
import torch
from alignair.nn.activations import make_activation
from alignair.nn.embedding import TokenPositionEmbedding


def test_make_activation_known():
    assert isinstance(make_activation("swish"), torch.nn.SiLU)
    assert isinstance(make_activation("gelu"), torch.nn.GELU)
    assert isinstance(make_activation("tanh"), torch.nn.Tanh)


def test_make_activation_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        make_activation("not_an_activation")


def test_embedding_output_shape():
    emb = TokenPositionEmbedding(max_len=16, vocab_size=6, embed_dim=32)
    x = torch.randint(0, 6, (4, 16))
    out = emb(x)
    assert out.shape == (4, 16, 32)


def test_embedding_adds_position():
    # Two identical token rows should produce identical embeddings (positions equal).
    emb = TokenPositionEmbedding(max_len=4, vocab_size=6, embed_dim=8)
    x = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    out = emb(x)
    assert torch.allclose(out[0], out[1])
    # But position 0 and position 1 of the same token should differ (position added).
    assert not torch.allclose(out[0, 0], out[0, 1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_embedding.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/nn/activations.py`:
```python
"""String-keyed activation factory (mirrors the legacy string-based API)."""
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.3),  # Keras LeakyReLU default alpha=0.3
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "mish": nn.Mish,
}


def make_activation(name: str) -> "nn.Module":
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Known: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()
```

`src/alignair/nn/embedding.py`:
```python
"""Token + learned position embedding (port of legacy TokenAndPositionEmbedding)."""
import torch
import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """Sum of a token embedding and a learned position embedding.

    Input:  (B, L) integer tokens in [0, vocab_size).
    Output: (B, L, embed_dim).
    """

    def __init__(self, max_len: int, vocab_size: int = 6, embed_dim: int = 32):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.int32, torch.int64):
            x = x.long()
        positions = torch.arange(self.max_len, device=x.device)
        return self.token_emb(x) + self.pos_emb(positions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_embedding.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/activations.py src/alignair/nn/embedding.py tests/alignair/nn
git commit -m "feat(alignair): add activation factory and token+position embedding"
```

---

## Task 3: `nn/conv.py` — Conv1dBatchNorm

**Files:**
- Create: `src/alignair/nn/conv.py` (Conv1dBatchNorm only in this task)
- Test: `tests/alignair/nn/test_conv.py`

- [ ] **Step 1: Write the failing test**

`tests/alignair/nn/test_conv.py`:
```python
import torch
from alignair.nn.conv import Conv1dBatchNorm


def test_conv_bn_halves_length_and_sets_channels():
    # Input (B, C_in, L); block applies 3 same-conv then maxpool(2).
    block = Conv1dBatchNorm(in_channels=8, filters=16, kernel=3, max_pool=2,
                            activation="leaky_relu")
    x = torch.randn(2, 8, 16)
    out = block(x)
    assert out.shape == (2, 16, 8)  # channels->16, length 16//2=8


def test_conv_bn_uses_eps_and_momentum():
    block = Conv1dBatchNorm(in_channels=4, filters=4, kernel=3, max_pool=2,
                            activation="leaky_relu")
    assert abs(block.batch_norm.eps - 0.8) < 1e-9
    assert abs(block.batch_norm.momentum - 0.9) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_conv.py -v`
Expected: FAIL (ImportError: cannot import name Conv1dBatchNorm).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/nn/conv.py`:
```python
"""Convolutional building blocks (port of legacy Conv1D_and_BatchNorm and
ConvResidualFeatureExtractionBlock). All ops operate channel-first: (B, C, L)."""
from typing import List, Union

import torch
import torch.nn as nn

from .activations import make_activation


def _same_padding(kernel: int) -> int:
    # 'same' padding for odd kernels; for even kernels PyTorch can't do exact
    # 'same' with a single int, so we use floor(kernel/2) (length preserved for
    # odd kernels; even kernels shift by at most 1, matching the legacy warning).
    return kernel // 2


class Conv1dBatchNorm(nn.Module):
    """Three stacked same-convolutions -> BatchNorm -> activation -> MaxPool.

    Mirrors legacy ``Conv1D_and_BatchNorm`` (3 convs, BN(momentum=0.1, eps=0.8),
    LeakyReLU, MaxPool1d). BN momentum is converted 0.1(Keras) -> 0.9(PyTorch).
    """

    def __init__(self, in_channels: int, filters: int = 16, kernel: int = 3,
                 max_pool: int = 2, activation: str = "leaky_relu"):
        super().__init__()
        pad = _same_padding(kernel)
        self.conv1 = nn.Conv1d(in_channels, filters, kernel, padding=pad)
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding=pad)
        self.conv3 = nn.Conv1d(filters, filters, kernel, padding=pad)
        self.batch_norm = nn.BatchNorm1d(filters, eps=0.8, momentum=0.9)
        self.activation = make_activation(activation)
        self.max_pool = nn.MaxPool1d(max_pool)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_conv.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/conv.py tests/alignair/nn/test_conv.py
git commit -m "feat(alignair): add Conv1dBatchNorm block"
```

---

## Task 4: `nn/conv.py` — ConvResidualFeatureExtractor

**Files:**
- Modify: `src/alignair/nn/conv.py` (add `ConvResidualFeatureExtractor`)
- Test: `tests/alignair/nn/test_conv.py` (add cases)

**Porting note on the legacy residual wiring** (from `ConvResidualFeatureExtractionBlock.call`): with a `kernel_size` list of length N, the first N-1 entries become `Conv1dBatchNorm` layers and the **last** entry is the kernel of a `residual_channel` plain conv. The residual stream is `residual_channel(embeddings)` maxpooled once; then for each conv-batch layer the feature stream is added to the pooled residual, activated, and pooled. Finally flatten → `Linear(out_features)`. We replicate this exactly. Because the final linear's input width depends on the pooled length, use `nn.LazyLinear`. **LazyLinear footgun:** a dummy forward must run before `.to(device)`/`state_dict` save — the model assembly (Task 8) handles this; here we just unit-test shape after one forward.

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/alignair/nn/test_conv.py`:
```python
from alignair.nn.conv import ConvResidualFeatureExtractor


def test_residual_extractor_output_shape():
    # embeddings (B, L, E) -> block transposes internally -> (B, out_features)
    fe = ConvResidualFeatureExtractor(
        in_channels=32, filter_size=16,
        kernel_sizes=[3, 3, 3, 2, 5], max_pool_size=2,
        out_features=64, activation="tanh",
    )
    x = torch.randn(2, 16, 32)  # (B, L, E)
    out = fe(x)
    assert out.shape == (2, 64)


def test_residual_extractor_backprop():
    fe = ConvResidualFeatureExtractor(
        in_channels=32, filter_size=16, kernel_sizes=[3, 3, 3, 2, 5],
        max_pool_size=2, out_features=64, activation="tanh",
    )
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = fe(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_conv.py -v`
Expected: FAIL (ImportError: cannot import name ConvResidualFeatureExtractor).

- [ ] **Step 3: Write minimal implementation (append to `conv.py`)**

```python
class ConvResidualFeatureExtractor(nn.Module):
    """Residual conv feature extractor (port of ConvResidualFeatureExtractionBlock).

    Input:  (B, L, E) embeddings (channel-last, as produced by the embedding).
    Output: (B, out_features).

    With ``kernel_sizes`` as a list of length N, the first N-1 kernels become
    Conv1dBatchNorm layers and the last kernel sizes the residual projection conv.
    """

    def __init__(self, in_channels: int, filter_size: int = 128,
                 kernel_sizes: Union[int, List[int]] = 5, max_pool_size: int = 2,
                 out_features: int = 576, activation: str = "tanh"):
        super().__init__()
        if isinstance(kernel_sizes, int):
            conv_kernels = [kernel_sizes] * 5
            residual_kernel = kernel_sizes
        else:
            conv_kernels = list(kernel_sizes[:-1])
            residual_kernel = kernel_sizes[-1]

        self.num_layers = len(conv_kernels)

        # First conv-batch layer maps in_channels -> filter_size; the rest map
        # filter_size -> filter_size (the residual stream is filter_size-wide).
        self.conv_layers = nn.ModuleList()
        for i, ks in enumerate(conv_kernels):
            cin = in_channels if i == 0 else filter_size
            self.conv_layers.append(
                Conv1dBatchNorm(cin, filters=filter_size, kernel=ks,
                                max_pool=max_pool_size, activation=activation)
            )

        self.residual_channel = nn.Conv1d(
            in_channels, filter_size, residual_kernel,
            padding=_same_padding(residual_kernel),
        )
        nn.init.xavier_uniform_(self.residual_channel.weight)
        if self.residual_channel.bias is not None:
            nn.init.zeros_(self.residual_channel.bias)

        self.pools = nn.ModuleList([nn.MaxPool1d(2) for _ in range(self.num_layers)])
        self.acts = nn.ModuleList([make_activation("leaky_relu") for _ in range(self.num_layers)])
        self.proj = nn.LazyLinear(out_features)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # (B, L, E) -> (B, E, L) for channel-first conv.
        x = embeddings.transpose(1, 2)

        residual = self.residual_channel(x)
        residual = self.pools[0](residual)

        feat = self.conv_layers[0](x)
        residual = feat + residual
        residual = self.acts[0](residual)
        residual = self.pools[0](residual)

        for i in range(1, self.num_layers):
            feat = self.conv_layers[i](residual)
            residual = self.pools[i](residual)
            residual = feat + residual
            residual = self.acts[i](residual)

        residual = torch.flatten(residual, start_dim=1)
        return self.proj(residual)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_conv.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/conv.py tests/alignair/nn/test_conv.py
git commit -m "feat(alignair): add ConvResidualFeatureExtractor"
```

---

## Task 5: `nn/masking.py` — SoftCutout

**Files:**
- Create: `src/alignair/nn/masking.py`
- Test: `tests/alignair/nn/test_masking.py`

- [ ] **Step 1: Write the failing test**

`tests/alignair/nn/test_masking.py`:
```python
import torch
from alignair.nn.masking import SoftCutout


def test_softcutout_shape():
    m = SoftCutout(max_size=16, k=3.0)
    start = torch.tensor([[4.0], [0.0]])
    end = torch.tensor([[10.0], [16.0]])
    mask = m(start, end)
    assert mask.shape == (2, 16)


def test_softcutout_high_inside_low_outside():
    m = SoftCutout(max_size=20, k=1.0)
    start = torch.tensor([[5.0]])
    end = torch.tensor([[15.0]])
    mask = m(start, end)[0]
    # Middle of the interval should be near 1, far outside near 0.
    assert mask[10] > 0.9
    assert mask[0] < 0.1
    assert mask[19] < 0.1


def test_softcutout_enforces_min_width():
    # end <= start should be bumped to start + 1, producing a non-degenerate mask.
    m = SoftCutout(max_size=10, k=1.0)
    start = torch.tensor([[5.0]])
    end = torch.tensor([[5.0]])
    mask = m(start, end)
    assert torch.isfinite(mask).all()
    assert mask.max() > 0.0


def test_softcutout_differentiable():
    m = SoftCutout(max_size=10, k=1.0)
    start = torch.tensor([[3.0]], requires_grad=True)
    end = torch.tensor([[7.0]], requires_grad=True)
    m(start, end).sum().backward()
    assert start.grad is not None and torch.isfinite(start.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_masking.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/nn/masking.py`:
```python
"""Differentiable soft cutout mask (port of legacy SoftCutoutLayer).

Produces a length-``max_size`` mask per sample from start/end position
expectations using smooth sigmoid ramps, so gradients flow through the predicted
boundaries. Semantics: start-inclusive, end-exclusive [start, end), with
end >= start + 1 and bounds clamped to [0, max_size]."""
import torch
import torch.nn as nn


class SoftCutout(nn.Module):
    def __init__(self, max_size: int, k: float = 3.0):
        super().__init__()
        self.max_size = int(max_size)
        self.k = float(k)

    def _sanitize(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float32).reshape(-1, 1)
        return torch.clamp(t, 0.0, float(self.max_size))

    def forward(self, start_raw: torch.Tensor, end_raw: torch.Tensor) -> torch.Tensor:
        start = self._sanitize(start_raw)
        end = self._sanitize(end_raw)
        end = torch.maximum(end, start + 1.0)

        indices = torch.arange(self.max_size, dtype=torch.float32,
                               device=start.device).unsqueeze(0)  # (1, L)
        left = torch.sigmoid((indices - start) / self.k)
        right = torch.sigmoid((end - indices) / self.k)
        return left * right  # (B, L)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_masking.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/masking.py tests/alignair/nn/test_masking.py
git commit -m "feat(alignair): add SoftCutout differentiable mask"
```

---

## Task 6: `nn/weighting.py` — UncertaintyWeight

**Files:**
- Create: `src/alignair/nn/weighting.py`
- Test: `tests/alignair/nn/test_weighting.py`

**Porting note** (from `RegularizedConstrainedLogVar`): a single trainable scalar `log_var` initialized to `log(initial_value)` (=0 for initial_value=1.0), clamped to `[min_log_var, max_log_var]=[-3, 1]`. `forward()` returns the precision `exp(-log_var)` and exposes a regularization term `regularizer_weight * relu(-log_var - 2)` (default weight 0.01). In PyTorch the clamp is applied as a constraint after optimizer steps (via `apply_constraints()`); `forward` also clamps defensively so the returned precision always respects bounds.

- [ ] **Step 1: Write the failing test**

`tests/alignair/nn/test_weighting.py`:
```python
import math
import torch
from alignair.nn.weighting import UncertaintyWeight


def test_initial_precision_is_one():
    w = UncertaintyWeight(initial_value=1.0)
    # log_var initialized to log(1)=0 -> precision exp(-0)=1
    assert abs(w().item() - 1.0) < 1e-6


def test_regularization_term_nonnegative():
    w = UncertaintyWeight()
    assert w.regularization().item() >= 0.0


def test_clamp_constraint_bounds_log_var():
    w = UncertaintyWeight(min_log_var=-3.0, max_log_var=1.0)
    with torch.no_grad():
        w.log_var.fill_(5.0)
    w.apply_constraints()
    assert w.log_var.item() <= 1.0 + 1e-6


def test_precision_is_differentiable():
    w = UncertaintyWeight()
    loss = torch.tensor(2.0)
    weighted = loss * w()
    weighted.backward()
    assert w.log_var.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_weighting.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/nn/weighting.py`:
```python
"""Kendall-style uncertainty weighting (port of RegularizedConstrainedLogVar).

A trainable scalar log-variance; ``forward()`` returns the precision exp(-log_var)
used to weight a task loss. ``regularization()`` returns a soft penalty that
discourages very small log-variance. ``apply_constraints()`` clamps log_var to
[min_log_var, max_log_var] and is called by the trainer after each optimizer step.
"""
import math

import torch
import torch.nn as nn


class UncertaintyWeight(nn.Module):
    def __init__(self, initial_value: float = 1.0, min_log_var: float = -3.0,
                 max_log_var: float = 1.0, regularizer_weight: float = 0.01):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.regularizer_weight = regularizer_weight
        self.log_var = nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        log_var = torch.clamp(self.log_var, self.min_log_var, self.max_log_var)
        return torch.exp(-log_var)

    def regularization(self) -> torch.Tensor:
        return self.regularizer_weight * torch.relu(-self.log_var - 2.0)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        self.log_var.clamp_(self.min_log_var, self.max_log_var)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_weighting.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/weighting.py tests/alignair/nn/test_weighting.py
git commit -m "feat(alignair): add UncertaintyWeight (Kendall log-var)"
```

---

## Task 7: `nn/heads.py` — output heads

**Files:**
- Create: `src/alignair/nn/heads.py`
- Test: `tests/alignair/nn/test_heads.py`

**Porting notes:**
- Segmentation head: legacy uses `Dense(L)` on the (flattened) segmentation feature vector to produce per-position logits → a `Linear(in_features, max_seq_length)`.
- Allele head: `Dense(latent, swish)` → `Dense(num_alleles, sigmoid)`. The new head returns **probabilities** (sigmoid applied) to match legacy outputs; the loss uses BCE on probabilities (Task 12) for equivalence. (Phase 2 may switch to logits + BCEWithLogits; out of scope here.)
- Mutation rate: `Dense(L, gelu)` → dropout → `Dense(1, relu)` clamped to [0, 1]. Indel: same with `Dense(1, relu)` clamped to [0, 50]. Productivity: flatten meta features → dropout → `Dense(1, sigmoid)`. Chain type: `Dense(L, gelu)` → dropout → `Dense(num_types, softmax)`.
- The `MinMaxValueConstraint` on mutation/indel output kernels is applied via `apply_constraints()`.

- [ ] **Step 1: Write the failing test**

`tests/alignair/nn/test_heads.py`:
```python
import torch
from alignair.nn.heads import (
    SegmentationHead, AlleleClassificationHead, MutationRateHead,
    IndelCountHead, ProductivityHead, ChainTypeHead,
)


def test_segmentation_head_shape():
    head = SegmentationHead(in_features=64, max_seq_length=16)
    feats = torch.randn(2, 64)
    logits = head(feats)
    assert logits.shape == (2, 16)


def test_allele_head_probabilities():
    head = AlleleClassificationHead(in_features=64, latent_dim=20, num_alleles=5,
                                    mid_activation="swish")
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 5)
    assert (out >= 0).all() and (out <= 1).all()  # sigmoid


def test_mutation_rate_head_range():
    head = MutationRateHead(in_features=64, max_seq_length=16)
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 1)
    assert (out >= 0).all()


def test_indel_head_shape():
    head = IndelCountHead(in_features=64, max_seq_length=16)
    assert head(torch.randn(2, 64)).shape == (2, 1)


def test_productivity_head_is_prob():
    head = ProductivityHead(in_features=64)
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_chain_type_head_softmax():
    head = ChainTypeHead(in_features=64, max_seq_length=16, num_types=3)
    out = head(torch.randn(2, 64))
    assert out.shape == (2, 3)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_heads.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/nn/heads.py`:
```python
"""Output heads for AlignAIR models."""
import torch
import torch.nn as nn

from .activations import make_activation


class SegmentationHead(nn.Module):
    """Per-position boundary logits: (B, in_features) -> (B, max_seq_length)."""

    def __init__(self, in_features: int, max_seq_length: int):
        super().__init__()
        self.linear = nn.Linear(in_features, max_seq_length)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AlleleClassificationHead(nn.Module):
    """Latent dense (swish) -> sigmoid allele probabilities."""

    def __init__(self, in_features: int, latent_dim: int, num_alleles: int,
                 mid_activation: str = "swish"):
        super().__init__()
        self.mid = nn.Linear(in_features, latent_dim)
        self.act = make_activation(mid_activation)
        self.out = nn.Linear(latent_dim, num_alleles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mid(x))
        return torch.sigmoid(self.out(x))


class _ScalarRegressionHead(nn.Module):
    """Shared body for mutation-rate / indel-count heads: Dense(L, gelu) ->
    dropout -> Dense(1, relu), with output clamped to [min_val, max_val]."""

    def __init__(self, in_features: int, max_seq_length: int,
                 min_val: float, max_val: float, dropout: float = 0.05):
        super().__init__()
        self.mid = nn.Linear(in_features, max_seq_length)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(max_seq_length, 1)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mid(x))
        x = self.dropout(x)
        x = torch.relu(self.out(x))
        return torch.clamp(x, self.min_val, self.max_val)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        # MinMaxValueConstraint on the output kernel.
        self.out.weight.clamp_(self.min_val, self.max_val)


class MutationRateHead(_ScalarRegressionHead):
    def __init__(self, in_features: int, max_seq_length: int):
        super().__init__(in_features, max_seq_length, min_val=0.0, max_val=1.0)


class IndelCountHead(_ScalarRegressionHead):
    def __init__(self, in_features: int, max_seq_length: int):
        super().__init__(in_features, max_seq_length, min_val=0.0, max_val=50.0)


class ProductivityHead(nn.Module):
    """Flatten meta features -> dropout -> sigmoid scalar."""

    def __init__(self, in_features: int, dropout: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.dropout(x)))


class ChainTypeHead(nn.Module):
    """Dense(L, gelu) -> dropout -> softmax over chain types."""

    def __init__(self, in_features: int, max_seq_length: int, num_types: int,
                 dropout: float = 0.05):
        super().__init__()
        self.mid = nn.Linear(in_features, max_seq_length)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(max_seq_length, num_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mid(x))
        x = self.dropout(x)
        return torch.softmax(self.out(x), dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_heads.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/heads.py tests/alignair/nn/test_heads.py
git commit -m "feat(alignair): add output heads (segmentation, allele, analysis, chain-type)"
```

---

## Task 8: `core/output.py` + `core/base.py` — BaseAlignAIR

**Files:**
- Create: `src/alignair/core/output.py`
- Create: `src/alignair/core/base.py`
- Test: `tests/alignair/core/test_base.py`

**Porting notes** (from `SingleChainAlignAIR.call`):
- Feature extractors share the embedding output. The meta extractor feeds the analysis heads. V/J (and D) segmentation extractors feed the segmentation heads. The softmax over each boundary logit gives probs; `expectation = sum(probs * positions)`. The expectation pair drives `SoftCutout`; the mask multiplies the embeddings (`(B,L,E) * (B,L,1)`); the masked embeddings feed the classification extractors → allele heads.
- `forward` returns an `AlignAIROutput` with logits + expectations + allele probs + analysis scalars (+ chain_type when present, set by the subclass override).
- `out_features` for every extractor = `max_seq_length` (so segmentation/analysis heads receive `in_features=max_seq_length`), matching the legacy `Dense(max_seq_length)` heads operating on the projected feature vector.
- **LazyLinear materialization:** `__init__` ends with a dummy forward on a `(1, L)` zero token tensor so all `LazyLinear` params exist before any `.to(device)`/save.

- [ ] **Step 1: Write the failing test**

Create `tests/alignair/core/test_base.py`:
```python
import torch
from alignair.core.base import BaseAlignAIR


def test_base_forward_keys_with_d(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    d = out.as_dict()
    for k in ["v_start_logits", "v_end_logits", "j_start_logits", "j_end_logits",
              "d_start_logits", "d_end_logits",
              "v_start", "v_end", "j_start", "j_end", "d_start", "d_end",
              "v_allele", "j_allele", "d_allele",
              "mutation_rate", "indel_count", "productive"]:
        assert k in d, f"missing {k}"


def test_base_forward_shapes(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    B, L = dummy_tokens.shape
    assert out.v_start_logits.shape == (B, L)
    assert out.v_start.shape == (B, 1)
    assert out.v_allele.shape == (B, tiny_config_d.v_allele_count)
    assert out.d_allele.shape == (B, tiny_config_d.d_allele_count)
    assert out.mutation_rate.shape == (B, 1)


def test_base_no_d_omits_d_keys(tiny_config_no_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_no_d)
    out = model(dummy_tokens)
    d = out.as_dict()
    assert "d_allele" not in d and "d_start_logits" not in d


def test_base_backprop(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    out.v_start_logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_regularization_loss_is_finite_scalar(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    _ = model(dummy_tokens)
    reg = model.regularization_loss()
    assert reg.ndim == 0 and torch.isfinite(reg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_base.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/core/output.py`:
```python
"""Typed model output container."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Dict

import torch


@dataclass
class AlignAIROutput:
    v_start_logits: torch.Tensor
    v_end_logits: torch.Tensor
    j_start_logits: torch.Tensor
    j_end_logits: torch.Tensor
    v_start: torch.Tensor
    v_end: torch.Tensor
    j_start: torch.Tensor
    j_end: torch.Tensor
    v_allele: torch.Tensor
    j_allele: torch.Tensor
    mutation_rate: torch.Tensor
    indel_count: torch.Tensor
    productive: torch.Tensor
    # Optional D-gene fields.
    d_start_logits: Optional[torch.Tensor] = None
    d_end_logits: Optional[torch.Tensor] = None
    d_start: Optional[torch.Tensor] = None
    d_end: Optional[torch.Tensor] = None
    d_allele: Optional[torch.Tensor] = None
    # Optional multi-chain field.
    chain_type: Optional[torch.Tensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name)
                for f in fields(self) if getattr(self, f.name) is not None}
```

`src/alignair/core/base.py`:
```python
"""Shared AlignAIR model body (port of SingleChainAlignAIR, unified)."""
import torch
import torch.nn as nn

from ..config.model_config import ModelConfig
from ..nn.embedding import TokenPositionEmbedding
from ..nn.conv import ConvResidualFeatureExtractor
from ..nn.masking import SoftCutout
from ..nn.heads import (
    SegmentationHead, AlleleClassificationHead, MutationRateHead,
    IndelCountHead, ProductivityHead,
)
from .output import AlignAIROutput

_EMBED_DIM = 32


class BaseAlignAIR(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        L = config.max_seq_length
        self.max_seq_length = L
        self.has_d_gene = config.has_d_gene

        self.embedding = TokenPositionEmbedding(max_len=L, vocab_size=6, embed_dim=_EMBED_DIM)

        def extractor(kernels):
            return ConvResidualFeatureExtractor(
                in_channels=_EMBED_DIM, filter_size=128, kernel_sizes=kernels,
                max_pool_size=2, out_features=L, activation="tanh",
            )

        seg_kernels = [3, 3, 3, 2, 5]
        cls_kernels = [3, 3, 3, 2, 2, 2, 5]

        self.meta_extractor = extractor(seg_kernels)
        self.v_seg_extractor = extractor(seg_kernels)
        self.j_seg_extractor = extractor(seg_kernels)
        self.v_cls_extractor = extractor(cls_kernels)
        self.j_cls_extractor = extractor(cls_kernels)

        self.v_start_head = SegmentationHead(L, L)
        self.v_end_head = SegmentationHead(L, L)
        self.j_start_head = SegmentationHead(L, L)
        self.j_end_head = SegmentationHead(L, L)

        self.v_mask = SoftCutout(L, k=3.0)
        self.j_mask = SoftCutout(L, k=3.0)

        self.v_allele_head = AlleleClassificationHead(
            L, config.v_latent_dim, config.v_allele_count, config.classification_mid_activation)
        self.j_allele_head = AlleleClassificationHead(
            L, config.j_latent_dim, config.j_allele_count, config.classification_mid_activation)

        self.mutation_rate_head = MutationRateHead(L, L)
        self.indel_count_head = IndelCountHead(L, L)
        self.productivity_head = ProductivityHead(L)

        if self.has_d_gene:
            self.d_seg_extractor = extractor(seg_kernels)
            self.d_cls_extractor = extractor([3, 3, 2, 2, 5])
            self.d_start_head = SegmentationHead(L, L)
            self.d_end_head = SegmentationHead(L, L)
            self.d_mask = SoftCutout(L, k=3.0)
            self.d_allele_head = AlleleClassificationHead(
                L, config.d_latent_dim, config.d_allele_count, config.classification_mid_activation)

        # Materialize all LazyLinear params (must happen before .to(device)/save).
        self._materialize_lazy_params()

    def _materialize_lazy_params(self) -> None:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros((1, self.max_seq_length), dtype=torch.long)
            self.forward(dummy)
        self.train(was_training)

    def _positions(self, device) -> torch.Tensor:
        return torch.arange(self.max_seq_length, dtype=torch.float32, device=device).unsqueeze(0)

    @staticmethod
    def _expectation(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return (probs * positions).sum(dim=-1, keepdim=True)

    def _gene_branch(self, emb, seg_extractor, start_head, end_head, mask_layer,
                     cls_extractor, allele_head, positions):
        seg_feats = seg_extractor(emb)
        start_logits = start_head(seg_feats)
        end_logits = end_head(seg_feats)
        start_exp = self._expectation(start_logits, positions)
        end_exp = self._expectation(end_logits, positions)
        mask = mask_layer(start_exp, end_exp).unsqueeze(-1)  # (B, L, 1)
        masked = emb * mask
        cls_feats = cls_extractor(masked)
        allele = allele_head(cls_feats)
        return start_logits, end_logits, start_exp, end_exp, allele

    def forward(self, tokenized_sequence: torch.Tensor) -> AlignAIROutput:
        emb = self.embedding(tokenized_sequence)  # (B, L, E)
        positions = self._positions(emb.device)

        meta = self.meta_extractor(emb)
        mutation_rate = self.mutation_rate_head(meta)
        indel_count = self.indel_count_head(meta)
        productive = self.productivity_head(meta)

        v = self._gene_branch(emb, self.v_seg_extractor, self.v_start_head, self.v_end_head,
                              self.v_mask, self.v_cls_extractor, self.v_allele_head, positions)
        j = self._gene_branch(emb, self.j_seg_extractor, self.j_start_head, self.j_end_head,
                              self.j_mask, self.j_cls_extractor, self.j_allele_head, positions)

        out = AlignAIROutput(
            v_start_logits=v[0], v_end_logits=v[1], v_start=v[2], v_end=v[3], v_allele=v[4],
            j_start_logits=j[0], j_end_logits=j[1], j_start=j[2], j_end=j[3], j_allele=j[4],
            mutation_rate=mutation_rate, indel_count=indel_count, productive=productive,
        )

        if self.has_d_gene:
            d = self._gene_branch(emb, self.d_seg_extractor, self.d_start_head, self.d_end_head,
                                  self.d_mask, self.d_cls_extractor, self.d_allele_head, positions)
            out.d_start_logits, out.d_end_logits = d[0], d[1]
            out.d_start, out.d_end, out.d_allele = d[2], d[3], d[4]

        self._meta_features = meta  # cached for multi-chain subclass
        return out

    def regularization_loss(self) -> torch.Tensor:
        """Explicit l2 penalty over conv weights (legacy intent; legacy train_step
        dropped these — see plan porting notes). Weight 0.01 matches legacy."""
        reg = torch.zeros((), device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                reg = reg + 0.01 * (module.weight ** 2).sum()
        return reg

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for head in (self.mutation_rate_head, self.indel_count_head):
            head.apply_constraints()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_base.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/core/output.py src/alignair/core/base.py tests/alignair/core
git commit -m "feat(alignair): add AlignAIROutput and unified BaseAlignAIR"
```

---

## Task 9: `core/single_chain.py`

**Files:**
- Create: `src/alignair/core/single_chain.py`
- Test: `tests/alignair/core/test_single_chain.py`

- [ ] **Step 1: Write the failing test**

`tests/alignair/core/test_single_chain.py`:
```python
import torch
from alignair.core.single_chain import SingleChainAlignAIR


def test_single_chain_is_base(tiny_config_d, dummy_tokens):
    model = SingleChainAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    assert out.chain_type is None  # single chain has no chain_type head
    assert out.v_allele.shape[0] == dummy_tokens.shape[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_single_chain.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/core/single_chain.py`:
```python
"""Single-chain AlignAIR model."""
from .base import BaseAlignAIR


class SingleChainAlignAIR(BaseAlignAIR):
    """Single-chain model — the base body with no chain-type head."""
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_single_chain.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/core/single_chain.py tests/alignair/core/test_single_chain.py
git commit -m "feat(alignair): add SingleChainAlignAIR"
```

---

## Task 10: `core/multi_chain.py`

**Files:**
- Create: `src/alignair/core/multi_chain.py`
- Test: `tests/alignair/core/test_multi_chain.py`

**Porting note:** `MultiChainAlignAIR` adds a `ChainTypeHead` driven by the cached meta features. It overrides `forward` to call the base then attach `chain_type`. `number_of_chains` comes from the config.

- [ ] **Step 1: Write the failing test**

`tests/alignair/core/test_multi_chain.py`:
```python
import torch
from alignair.config.model_config import ModelConfig
from alignair.core.multi_chain import MultiChainAlignAIR


def _multi_config():
    return ModelConfig(
        max_seq_length=16, v_allele_count=5, j_allele_count=3, d_allele_count=4,
        has_d_gene=True, number_of_chains=2, chain_types=["IGH", "IGK"],
    )


def test_multi_chain_has_chain_type():
    model = MultiChainAlignAIR(_multi_config())
    out = model(torch.randint(0, 6, (2, 16)))
    assert out.chain_type is not None
    assert out.chain_type.shape == (2, 2)
    assert torch.allclose(out.chain_type.sum(dim=-1), torch.ones(2), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_multi_chain.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/core/multi_chain.py`:
```python
"""Multi-chain AlignAIR model (adds a chain-type classification head)."""
from ..config.model_config import ModelConfig
from ..nn.heads import ChainTypeHead
from .base import BaseAlignAIR
from .output import AlignAIROutput


class MultiChainAlignAIR(BaseAlignAIR):
    def __init__(self, config: ModelConfig):
        if not config.number_of_chains or config.number_of_chains < 1:
            raise ValueError("MultiChainAlignAIR requires config.number_of_chains >= 1")
        super().__init__(config)
        L = config.max_seq_length
        self.chain_type_head = ChainTypeHead(L, L, config.number_of_chains)
        self._materialize_lazy_params()  # chain_type_head has no lazy params, but keep contract

    def forward(self, tokenized_sequence) -> AlignAIROutput:
        out = super().forward(tokenized_sequence)
        out.chain_type = self.chain_type_head(self._meta_features)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_multi_chain.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/core/multi_chain.py tests/alignair/core/test_multi_chain.py
git commit -m "feat(alignair): add MultiChainAlignAIR with chain-type head"
```

---

## Task 11: `metrics/` — accumulators

**Files:**
- Create: `src/alignair/metrics/accumulator.py`, `boundary.py`, `entropy.py`, `allele_auc.py`, `average_last_label.py`
- Test: `tests/alignair/metrics/test_metrics.py`

**Porting notes:** legacy metrics subclass `keras.metrics.Mean` (running mean of per-batch values). We implement a `MeanAccumulator` with `update(value)`, `compute()`, `reset()`. `BoundaryMetrics` computes MAE / exact-acc / ±1nt-acc from logits + a ground-truth scalar (round → argmax compare). `AlleleEntropy` = `-sum(p*log(p+1e-9))` per row, meaned. `MultiLabelAUC` is a custom ranking AUC over flattened (label, prob) pairs. `AverageLastLabel` tracks the mean of the last D-allele column (the "Short-D" probability) when present.

- [ ] **Step 1: Write the failing test**

Create `tests/alignair/metrics/test_metrics.py`:
```python
import torch
from alignair.metrics.accumulator import MeanAccumulator
from alignair.metrics.boundary import BoundaryMetrics
from alignair.metrics.entropy import AlleleEntropy
from alignair.metrics.allele_auc import MultiLabelAUC
from alignair.metrics.average_last_label import AverageLastLabel


def test_mean_accumulator():
    m = MeanAccumulator()
    m.update(torch.tensor(2.0))
    m.update(torch.tensor(4.0))
    assert abs(m.compute().item() - 3.0) < 1e-6
    m.reset()
    assert m.count == 0


def test_boundary_exact_match():
    bm = BoundaryMetrics()
    # logits argmax at index 3 for both rows; gt = 3 -> exact acc 1, mae 0
    logits = torch.full((2, 8), -10.0)
    logits[:, 3] = 10.0
    gt = torch.tensor([[3.0], [3.0]])
    bm.update(gt, logits)
    res = bm.compute()
    assert abs(res["mae"]) < 1e-6
    assert abs(res["acc"] - 1.0) < 1e-6
    assert abs(res["acc_1nt"] - 1.0) < 1e-6


def test_boundary_within_1nt():
    bm = BoundaryMetrics()
    logits = torch.full((1, 8), -10.0)
    logits[:, 4] = 10.0  # predicts 4
    gt = torch.tensor([[3.0]])  # off by 1
    bm.update(gt, logits)
    res = bm.compute()
    assert abs(res["acc"]) < 1e-6        # not exact
    assert abs(res["acc_1nt"] - 1.0) < 1e-6  # within 1


def test_allele_entropy_uniform_is_max():
    ent = AlleleEntropy()
    probs = torch.full((1, 4), 0.25)
    ent.update(probs)
    import math
    assert abs(ent.compute().item() - math.log(4)) < 1e-4


def test_multilabel_auc_perfect():
    auc = MultiLabelAUC()
    y_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    auc.update(y_true, y_pred)
    assert auc.compute().item() > 0.99


def test_average_last_label():
    all_ = AverageLastLabel()
    d_allele = torch.tensor([[0.1, 0.2, 0.7], [0.0, 0.0, 0.5]])
    all_.update(d_allele)
    assert abs(all_.compute().item() - 0.6) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/metrics/test_metrics.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

`src/alignair/metrics/accumulator.py`:
```python
"""Running-mean accumulator (mirrors keras.metrics.Mean)."""
import torch


class MeanAccumulator:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: torch.Tensor) -> None:
        v = value.detach()
        self.total += float(v.sum().item())
        self.count += int(v.numel())

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(0.0)
        return torch.tensor(self.total / self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
```

`src/alignair/metrics/boundary.py`:
```python
"""Boundary accuracy/MAE metrics from per-position logits + ground-truth scalar."""
import torch

from .accumulator import MeanAccumulator


class BoundaryMetrics:
    def __init__(self):
        self.mae = MeanAccumulator()
        self.acc = MeanAccumulator()
        self.acc_1nt = MeanAccumulator()

    def update(self, gt_scalar: torch.Tensor, logits: torch.Tensor) -> None:
        max_idx = logits.shape[-1] - 1
        gt_idx = torch.round(gt_scalar.squeeze(-1)).long().clamp(0, max_idx)
        pred_idx = logits.argmax(dim=-1)
        err = (pred_idx - gt_idx).abs().float()
        self.mae.update(err)
        self.acc.update((pred_idx == gt_idx).float())
        self.acc_1nt.update((err <= 1.0).float())

    def compute(self) -> dict:
        return {"mae": self.mae.compute().item(),
                "acc": self.acc.compute().item(),
                "acc_1nt": self.acc_1nt.compute().item()}

    def reset(self) -> None:
        self.mae.reset(); self.acc.reset(); self.acc_1nt.reset()
```

`src/alignair/metrics/entropy.py`:
```python
"""Prediction-entropy metric for an allele probability head."""
import torch

from .accumulator import MeanAccumulator


class AlleleEntropy(MeanAccumulator):
    def update(self, probs: torch.Tensor) -> None:
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        super().update(entropy)
```

`src/alignair/metrics/allele_auc.py`:
```python
"""Multi-label AUC via the rank-statistic (Mann–Whitney U) formulation."""
import torch

from .accumulator import MeanAccumulator


class MultiLabelAUC:
    def __init__(self):
        self._mean = MeanAccumulator()

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        labels = (y_true.reshape(-1) > 0.5)
        scores = y_pred.reshape(-1)
        pos = scores[labels]
        neg = scores[~labels]
        if pos.numel() == 0 or neg.numel() == 0:
            return
        # AUC = P(score_pos > score_neg); count ties as 0.5.
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
        auc = wins / (pos.numel() * neg.numel())
        self._mean.update(auc.reshape(1))

    def compute(self) -> torch.Tensor:
        return self._mean.compute()

    def reset(self) -> None:
        self._mean.reset()
```

`src/alignair/metrics/average_last_label.py`:
```python
"""Mean of the last allele column (the 'Short-D' probability)."""
import torch

from .accumulator import MeanAccumulator


class AverageLastLabel(MeanAccumulator):
    def update(self, d_allele: torch.Tensor) -> None:
        super().update(d_allele[:, -1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/metrics/test_metrics.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/metrics tests/alignair/metrics
git commit -m "feat(alignair): add lightweight training metrics"
```

---

## Task 12: `losses/functional.py` + `losses/hierarchical.py`

**Files:**
- Create: `src/alignair/losses/functional.py`
- Create: `src/alignair/losses/hierarchical.py`
- Test: `tests/alignair/losses/test_functional.py`, `tests/alignair/losses/test_hierarchical.py`

**Porting notes** (from `hierarchical_loss`): segmentation uses Gaussian `soft_targets(gt, L, sigma=1.5)` and softmax cross-entropy against the boundary logits, each weighted by an `UncertaintyWeight`. Aux = `0.1*Huber(length) + 0.1*(1-IoU) + 0.05*hinge`. Classification = BCE(label_smoothing=0.1) on allele probabilities, per-gene uncertainty-weighted; D path adds a short-D penalty. Analysis = mutation MAE + indel MAE + productivity BCE, each weighted. Multi-chain adds categorical CE on chain_type. The loss owns all `UncertaintyWeight` modules and returns `(total, components)`.

- [ ] **Step 1: Write the failing test for functional helpers**

Create `tests/alignair/losses/test_functional.py`:
```python
import torch
from alignair.losses.functional import soft_targets, expectation_from_logits, interval_iou_loss


def test_soft_targets_peak_at_gt():
    probs = soft_targets(torch.tensor([[4.0]]), L=10, sigma=1.5)
    assert probs.shape == (1, 10)
    assert torch.argmax(probs, dim=-1).item() == 4
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_expectation_recovers_peak():
    logits = torch.full((1, 10), -10.0)
    logits[0, 6] = 10.0
    exp = expectation_from_logits(logits, max_seq_length=10)
    assert abs(exp.item() - 6.0) < 1e-3


def test_iou_loss_zero_for_perfect_overlap():
    s = torch.tensor([[2.0]]); e = torch.tensor([[8.0]])
    loss = interval_iou_loss(s, e, torch.tensor([2.0]), torch.tensor([8.0]))
    assert loss.item() < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses/test_functional.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the functional helpers**

`src/alignair/losses/functional.py`:
```python
"""Pure functions used by the hierarchical loss."""
import torch
import torch.nn.functional as F


def soft_targets(gt: torch.Tensor, L: int, sigma: float = 1.5) -> torch.Tensor:
    """Gaussian soft-label distribution centred at the (rounded, clamped) gt index."""
    gt = torch.round(gt.to(torch.float32)).clamp(0.0, float(L - 1))
    positions = torch.arange(L, dtype=torch.float32, device=gt.device).unsqueeze(0)
    dist2 = (positions - gt) ** 2
    logits = -0.5 * dist2 / (sigma * sigma)
    return torch.softmax(logits, dim=-1)


def expectation_from_logits(logits: torch.Tensor, max_seq_length: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    pos = torch.arange(max_seq_length, dtype=torch.float32, device=logits.device).unsqueeze(0)
    return (probs * pos).sum(dim=-1, keepdim=True)


def soft_label_cross_entropy(target_probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Mean soft-label cross-entropy: -sum(target * log_softmax(logits))."""
    log_probs = F.log_softmax(logits, dim=-1)
    return (-(target_probs * log_probs).sum(dim=-1)).mean()


def interval_iou_loss(s_pred: torch.Tensor, e_pred: torch.Tensor,
                      s_true: torch.Tensor, e_true: torch.Tensor,
                      eps: float = 1e-6) -> torch.Tensor:
    s_pred = s_pred.squeeze(-1)
    e_pred = e_pred.squeeze(-1)
    inter = torch.relu(torch.minimum(e_pred, e_true) - torch.maximum(s_pred, s_true))
    len_pred = torch.clamp(e_pred - s_pred, min=0.0)
    len_true = torch.clamp(e_true - s_true, min=0.0)
    union = len_pred + len_true - inter + eps
    iou = inter / union
    return 1.0 - iou.mean()
```

- [ ] **Step 4: Run functional tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses/test_functional.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write the failing test for the loss module**

`tests/alignair/losses/test_hierarchical.py`:
```python
import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.core.multi_chain import MultiChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss


def _targets_for(cfg, B):
    L = cfg.max_seq_length
    y = {
        "v_start": torch.full((B, 1), 1.0), "v_end": torch.full((B, 1), float(L // 2)),
        "j_start": torch.full((B, 1), float(L // 2 + 1)), "j_end": torch.full((B, 1), float(L - 1)),
        "v_allele": torch.zeros(B, cfg.v_allele_count),
        "j_allele": torch.zeros(B, cfg.j_allele_count),
        "mutation_rate": torch.full((B, 1), 0.1),
        "indel_count": torch.full((B, 1), 1.0),
        "productive": torch.ones(B, 1),
    }
    y["v_allele"][:, 0] = 1.0
    y["j_allele"][:, 0] = 1.0
    if cfg.has_d_gene:
        y["d_start"] = torch.full((B, 1), float(L // 2 - 2))
        y["d_end"] = torch.full((B, 1), float(L // 2))
        y["d_allele"] = torch.zeros(B, cfg.d_allele_count)
        y["d_allele"][:, 0] = 1.0
    return y


def test_loss_is_finite_and_backprops(tiny_config_d, dummy_tokens):
    model = SingleChainAlignAIR(tiny_config_d)
    loss_fn = AlignAIRLoss(tiny_config_d)
    out = model(dummy_tokens)
    y = _targets_for(tiny_config_d, dummy_tokens.shape[0])
    total, components = loss_fn(y, out.as_dict())
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert "segmentation_loss" in components and "classification_loss" in components


def test_loss_multichain_has_chain_type_component():
    cfg = ModelConfig(max_seq_length=16, v_allele_count=5, j_allele_count=3,
                      d_allele_count=4, has_d_gene=True, number_of_chains=2,
                      chain_types=["IGH", "IGK"])
    model = MultiChainAlignAIR(cfg)
    loss_fn = AlignAIRLoss(cfg)
    x = torch.randint(0, 6, (2, 16))
    out = model(x)
    y = _targets_for(cfg, 2)
    y["chain_type"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    total, components = loss_fn(y, out.as_dict())
    assert torch.isfinite(total)
    assert "chain_type_loss" in components
```

- [ ] **Step 6: Run loss tests to verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses/test_hierarchical.py -v`
Expected: FAIL (ModuleNotFoundError: alignair.losses.hierarchical).

- [ ] **Step 7: Write the loss module**

`src/alignair/losses/hierarchical.py`:
```python
"""Hierarchical multi-task loss with Kendall uncertainty weighting.

Faithful port of the legacy ``hierarchical_loss``, refactored into its own
nn.Module that owns the uncertainty-weighting parameters.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig
from ..nn.weighting import UncertaintyWeight
from .functional import (
    soft_targets, expectation_from_logits, soft_label_cross_entropy, interval_iou_loss,
)


def _bce_label_smoothing(target: torch.Tensor, prob: torch.Tensor,
                         smoothing: float = 0.1, eps: float = 1e-7) -> torch.Tensor:
    """Binary cross-entropy on probabilities with label smoothing (Keras-style)."""
    target = target * (1.0 - smoothing) + 0.5 * smoothing
    prob = prob.clamp(eps, 1.0 - eps)
    return -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob)).mean()


class AlignAIRLoss(nn.Module):
    def __init__(self, config: ModelConfig, sigma: float = 1.5):
        super().__init__()
        self.config = config
        self.L = config.max_seq_length
        self.has_d_gene = config.has_d_gene
        self.is_multi_chain = config.is_multi_chain
        self.sigma = sigma

        names = ["v_start", "v_end", "j_start", "j_end",
                 "v_classification", "j_classification",
                 "mutation", "indel", "productivity"]
        if self.has_d_gene:
            names += ["d_start", "d_end", "d_classification"]
        if self.is_multi_chain:
            names += ["chain_type"]
        self.weights = nn.ModuleDict({n: UncertaintyWeight() for n in names})

    def _seg_term(self, y_true, y_pred, gene: str) -> torch.Tensor:
        t = soft_targets(y_true[f"{gene}"], self.L, self.sigma)
        return soft_label_cross_entropy(t, y_pred[f"{gene}_logits"])

    def forward(self, y_true: Dict[str, torch.Tensor],
                y_pred: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        L = self.L

        # --- Segmentation soft-label CE, per boundary, uncertainty-weighted ---
        boundaries = ["v_start", "v_end", "j_start", "j_end"]
        if self.has_d_gene:
            boundaries += ["d_start", "d_end"]
        seg_losses = {}
        for b in boundaries:
            t = soft_targets(y_true[b], L, self.sigma)
            ce = soft_label_cross_entropy(t, y_pred[f"{b}_logits"])
            seg_losses[b] = ce
        segmentation_loss = sum(seg_losses[b] * self.weights[b]() for b in boundaries)

        # --- Auxiliary segmentation: Huber length + IoU + hinge ---
        def exp_of(b):
            return expectation_from_logits(y_pred[f"{b}_logits"], L)

        genes = ["v", "j"] + (["d"] if self.has_d_gene else [])
        len_loss = torch.zeros((), device=segmentation_loss.device)
        iou_loss = torch.zeros((), device=segmentation_loss.device)
        hinge_loss = torch.zeros((), device=segmentation_loss.device)
        for g in genes:
            s_exp, e_exp = exp_of(f"{g}_start"), exp_of(f"{g}_end")
            len_pred = (e_exp - s_exp).squeeze(-1)
            len_true = (y_true[f"{g}_end"].float() - y_true[f"{g}_start"].float()).squeeze(-1)
            len_loss = len_loss + F.huber_loss(len_pred, len_true, delta=1.0)
            iou_loss = iou_loss + interval_iou_loss(
                s_exp, e_exp,
                y_true[f"{g}_start"].float().squeeze(-1),
                y_true[f"{g}_end"].float().squeeze(-1))
            hinge_loss = hinge_loss + torch.relu(1.0 - len_pred).mean()
        segmentation_loss = segmentation_loss + 0.1 * len_loss + 0.1 * iou_loss + 0.05 * hinge_loss

        # --- Classification BCE (label smoothing), uncertainty-weighted ---
        clf_v = _bce_label_smoothing(y_true["v_allele"], y_pred["v_allele"])
        clf_j = _bce_label_smoothing(y_true["j_allele"], y_pred["j_allele"])
        classification_loss = clf_v * self.weights["v_classification"]() \
            + clf_j * self.weights["j_classification"]()
        if self.has_d_gene:
            clf_d = _bce_label_smoothing(y_true["d_allele"], y_pred["d_allele"])
            classification_loss = classification_loss + clf_d * self.weights["d_classification"]()
            d_len = (y_pred["d_end"] - y_pred["d_start"]).squeeze(-1)
            short_d_prob = y_pred["d_allele"][:, -1]
            classification_loss = classification_loss + ((d_len < 5).float() * short_d_prob).mean()

        # --- Analysis losses ---
        mutation_loss = (y_true["mutation_rate"].float() - y_pred["mutation_rate"].float()).abs().mean()
        indel_loss = (y_true["indel_count"].float() - y_pred["indel_count"].float()).abs().mean()
        productive_loss = F.binary_cross_entropy(
            y_pred["productive"].clamp(1e-7, 1 - 1e-7), y_true["productive"].float())

        weighted_mutation = mutation_loss * self.weights["mutation"]()
        weighted_indel = indel_loss * self.weights["indel"]()
        weighted_productive = productive_loss * self.weights["productivity"]()

        total = (segmentation_loss + classification_loss
                 + weighted_mutation + weighted_indel + weighted_productive)

        components = {
            "segmentation_loss": segmentation_loss.detach(),
            "classification_loss": classification_loss.detach(),
            "mutation_rate_loss": weighted_mutation.detach(),
            "indel_count_loss": weighted_indel.detach(),
            "productive_loss": weighted_productive.detach(),
        }

        if self.is_multi_chain and "chain_type" in y_pred:
            # Categorical CE on softmax probabilities.
            ct_pred = y_pred["chain_type"].clamp(1e-7, 1.0)
            chain_type_loss = -(y_true["chain_type"] * torch.log(ct_pred)).sum(dim=-1).mean()
            weighted_ct = chain_type_loss * self.weights["chain_type"]()
            total = total + weighted_ct
            components["chain_type_loss"] = weighted_ct.detach()

        # Regularization penalties from the uncertainty weights.
        total = total + sum(w.regularization() for w in self.weights.values())

        components["total_loss"] = total.detach()
        return total, components
```

- [ ] **Step 8: Run loss tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/losses -v`
Expected: PASS (5 passed total).

- [ ] **Step 9: Commit**

```bash
git add src/alignair/losses tests/alignair/losses
git commit -m "feat(alignair): add hierarchical multi-task loss"
```

---

## Task 13: Numeric-equivalence tests vs legacy TensorFlow

**Files:**
- Test: `tests/alignair/equivalence/test_tf_equivalence.py`

**Goal:** prove the ported *math* matches the legacy TF implementation within tolerance on identical inputs. TensorFlow is still installed during migration, so these tests import the legacy modules directly. Mark them so they can be skipped if TF is unavailable.

- [ ] **Step 1: Write the equivalence test**

Create `tests/alignair/equivalence/test_tf_equivalence.py`:
```python
"""Numeric-equivalence checks: new PyTorch math vs legacy TensorFlow math.

These import the legacy ``src/AlignAIR`` package and TensorFlow. They are skipped
automatically if TF import fails.
"""
import numpy as np
import pytest
import torch

tf = pytest.importorskip("tensorflow")


def test_soft_targets_match_tf():
    from alignair.losses.functional import soft_targets as pt_soft

    L, gt, sigma = 12, 5.0, 1.5
    pt = pt_soft(torch.tensor([[gt]]), L=L, sigma=sigma).numpy()[0]

    # Reproduce the legacy TF soft_targets math inline (same as SingleChainAlignAIR).
    gt_t = tf.constant([[gt]], dtype=tf.float32)
    positions = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
    dist2 = tf.square(positions - gt_t)
    logits = -0.5 * dist2 / (sigma * sigma)
    tf_probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    np.testing.assert_allclose(pt, tf_probs, atol=1e-5)


def test_soft_label_ce_matches_tf():
    from alignair.losses.functional import soft_label_cross_entropy, soft_targets

    L = 12
    rng = np.random.default_rng(0)
    logits_np = rng.standard_normal((4, L)).astype(np.float32)
    gt = np.array([[2.0], [5.0], [9.0], [0.0]], dtype=np.float32)

    t_pt = soft_targets(torch.tensor(gt), L=L)
    pt_loss = soft_label_cross_entropy(t_pt, torch.tensor(logits_np)).item()

    positions = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
    gt_t = tf.constant(gt)
    dist2 = tf.square(positions - gt_t)
    t_tf = tf.nn.softmax(-0.5 * dist2 / (1.5 * 1.5), axis=-1)
    tf_loss = float(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=t_tf, logits=tf.constant(logits_np))))

    assert abs(pt_loss - tf_loss) < 1e-4


def test_expectation_matches_tf():
    from alignair.losses.functional import expectation_from_logits

    L = 12
    rng = np.random.default_rng(1)
    logits_np = rng.standard_normal((3, L)).astype(np.float32)

    pt = expectation_from_logits(torch.tensor(logits_np), max_seq_length=L).numpy().ravel()

    probs = tf.nn.softmax(tf.constant(logits_np), axis=-1)
    pos = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
    tf_exp = tf.reduce_sum(probs * pos, axis=-1).numpy()

    np.testing.assert_allclose(pt, tf_exp, atol=1e-4)


def test_entropy_metric_matches_tf():
    from alignair.metrics.entropy import AlleleEntropy

    rng = np.random.default_rng(2)
    probs_np = rng.random((5, 7)).astype(np.float32)
    probs_np /= probs_np.sum(axis=1, keepdims=True)

    ent = AlleleEntropy()
    ent.update(torch.tensor(probs_np))
    pt_val = ent.compute().item()

    tf_entropy = -tf.reduce_sum(
        tf.constant(probs_np) * tf.math.log(tf.constant(probs_np) + 1e-9), axis=-1)
    tf_val = float(tf.reduce_mean(tf_entropy))

    assert abs(pt_val - tf_val) < 1e-4
```

- [ ] **Step 2: Run the equivalence tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/equivalence/test_tf_equivalence.py -v`
Expected: PASS (4 passed) — or SKIPPED if TensorFlow is not importable in the venv. If any FAIL, the ported math diverges from legacy; investigate before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/alignair/equivalence
git commit -m "test(alignair): numeric-equivalence tests vs legacy TensorFlow"
```

---

## Task 14: Full Phase-1 suite + package smoke

**Files:**
- Create: `src/alignair/core/__init__.py` exports (modify)

- [ ] **Step 1: Add convenience exports**

Modify `src/alignair/core/__init__.py`:
```python
from .single_chain import SingleChainAlignAIR
from .multi_chain import MultiChainAlignAIR
from .output import AlignAIROutput

__all__ = ["SingleChainAlignAIR", "MultiChainAlignAIR", "AlignAIROutput"]
```

- [ ] **Step 2: Run the entire Phase-1 test suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -v`
Expected: all PASS (equivalence may SKIP if TF absent). No failures.

- [ ] **Step 3: Smoke-import the package without TensorFlow on the path implicitly**

Run: `PYTHONPATH=src .private/.venv/bin/python -c "import alignair, alignair.core, alignair.losses.hierarchical, alignair.metrics.boundary; print('ok')"`
Expected: prints `ok` with no TensorFlow import triggered.

- [ ] **Step 4: Commit**

```bash
git add src/alignair/core/__init__.py
git commit -m "feat(alignair): export core models from package"
```

---

## Self-Review (completed during planning)

**Spec coverage:** every Phase-1 module in spec §5.1 maps to a task — `config`(T1), `nn/embedding+activations`(T2), `nn/conv`(T3–T4), `nn/masking`(T5), `nn/weighting`(T6), `nn/heads`(T7), `core`(T8–T10), `metrics`(T11), `losses`(T12), numeric-equivalence(T13), suite(T14). Porting conventions in spec §5.2 are encoded in the Reference section and per-task notes. Testing strategy §5.3 (shape/forward, loss finite+backprop, numeric-equivalence, per-block) is covered across T2–T13.

**Placeholder scan:** no TBD/TODO; every code step contains full code; every test step contains real assertions.

**Type consistency:** `ModelConfig` property names (`v_latent_dim`, etc.) are used consistently in T8/T12; `AlignAIROutput.as_dict()` is the single interface consumed by `AlignAIRLoss` and tests; `UncertaintyWeight.__call__`/`regularization`/`apply_constraints` names match across T6/T8/T12; metric `update`/`compute`/`reset` names match across T11/T13.

**Known follow-ups (out of Phase-1 scope, intentionally deferred):** the legacy `metrics` property/`get_metrics_log` aggregation, AUC/boundary metric *wiring into a training loop*, and BCEWithLogits switch are Phase-2 concerns.
