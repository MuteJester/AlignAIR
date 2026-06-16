# AlignAIR Phase 3a — Serialization (state_dict bundle) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the TF SavedModel serialization with a PyTorch `state_dict` bundle and a `PretrainedMixin` so any AlignAIR model can `save_pretrained`/`from_pretrained` with exact forward-output round-trip parity.

**Architecture:** `alignair/serialization/bundle.py` owns all on-disk IO (`model.pt` state_dict, `model_config.json`, optional `dataconfig.pkl`, `training_meta.json`, `VERSION`, `fingerprint.txt`) plus a SHA-256 fingerprint and tamper check. `alignair/serialization/pretrained.py` adds `PretrainedMixin` (mixed into `BaseAlignAIR`) that reconstructs the right model class from the saved `ModelConfig` and loads weights. No `ModelBundleConfig` — `ModelConfig` already holds every structural field; the model class is chosen via `ModelConfig.is_multi_chain`.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), pytest. Venv: `.venv/bin/python`. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs contain NO `__init__.py`.

---

## Bundle on-disk layout

```
bundle_dir/
  model.pt            # torch.save(model.state_dict())
  model_config.json   # ModelConfig.to_dict()
  dataconfig.pkl      # optional (GenAIRR DataConfig; needed later for AIRR/germline)
  training_meta.json  # TrainingMeta
  VERSION             # bundle format version (int)
  fingerprint.txt     # SHA-256 over the other files
```

## File structure (Plan 3a)

```
src/alignair/serialization/
  __init__.py
  bundle.py      TrainingMeta, BUNDLE_FORMAT_VERSION, compute_fingerprint, save_bundle, load_bundle
  pretrained.py  PretrainedMixin (save_pretrained / from_pretrained / load_dataconfig)
src/alignair/core/base.py   [MODIFY: BaseAlignAIR(PretrainedMixin, nn.Module)]
tests/alignair/serialization/
  test_bundle.py test_pretrained.py
```

---

## Task 1: `serialization/bundle.py` — bundle IO + fingerprint

**Files:** Create `src/alignair/serialization/__init__.py` (empty), `src/alignair/serialization/bundle.py`;
Test `tests/alignair/serialization/test_bundle.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import torch
import pytest
from alignair.config.model_config import ModelConfig
from alignair.serialization.bundle import (
    save_bundle, load_bundle, compute_fingerprint, TrainingMeta, BUNDLE_FORMAT_VERSION,
)


def _cfg():
    return ModelConfig(max_seq_length=16, v_allele_count=5, j_allele_count=3,
                       d_allele_count=4, has_d_gene=True)


def test_save_creates_expected_files(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(3)},
                dataconfig={"ref": "x"}, training_meta=TrainingMeta(epochs_trained=2))
    for name in ("model.pt", "model_config.json", "dataconfig.pkl",
                 "training_meta.json", "VERSION", "fingerprint.txt"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "VERSION").read_text().strip() == str(BUNDLE_FORMAT_VERSION)


def test_roundtrip_config_dataconfig_meta(tmp_path):
    cfg = _cfg()
    save_bundle(tmp_path, model_config=cfg, state_dict={"w": torch.ones(2)},
                dataconfig={"ref": "abc"}, training_meta=TrainingMeta(epochs_trained=5, final_loss=1.5))
    loaded_cfg, dataconfig, meta = load_bundle(tmp_path)
    assert loaded_cfg == cfg
    assert dataconfig == {"ref": "abc"}
    assert meta.epochs_trained == 5 and meta.final_loss == 1.5


def test_no_dataconfig_loads_none(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(1)})
    _, dataconfig, _ = load_bundle(tmp_path)
    assert dataconfig is None


def test_fingerprint_detects_tampering(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(1)})
    # Corrupt the config after fingerprinting.
    (tmp_path / "model_config.json").write_text(json.dumps({"max_seq_length": 999}))
    with pytest.raises(ValueError):
        load_bundle(tmp_path)


def test_fingerprint_stable_for_unchanged_bundle(tmp_path):
    save_bundle(tmp_path, model_config=_cfg(), state_dict={"w": torch.zeros(1)})
    stored = (tmp_path / "fingerprint.txt").read_text().strip()
    assert compute_fingerprint(tmp_path) == stored
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`src/alignair/serialization/__init__.py`: empty.

`src/alignair/serialization/bundle.py`:
```python
"""PyTorch model-bundle IO: state_dict + config + dataconfig + meta + fingerprint."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch

from ..config.model_config import ModelConfig

BUNDLE_FORMAT_VERSION = 1
_REQUIRED = ("model.pt", "model_config.json", "VERSION", "fingerprint.txt")


@dataclass
class TrainingMeta:
    epochs_trained: int = 0
    best_loss: Optional[float] = None
    final_loss: Optional[float] = None
    metrics_summary: dict = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMeta":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def compute_fingerprint(bundle_dir) -> str:
    """SHA-256 over every bundle file except fingerprint.txt, in name order."""
    h = hashlib.sha256()
    for p in sorted(Path(bundle_dir).iterdir()):
        if not p.is_file() or p.name == "fingerprint.txt":
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def save_bundle(bundle_dir, *, model_config: ModelConfig, state_dict,
                dataconfig=None, training_meta: Optional[TrainingMeta] = None) -> None:
    d = Path(bundle_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, d / "model.pt")
    (d / "model_config.json").write_text(
        json.dumps(model_config.to_dict(), indent=2, sort_keys=True))
    if dataconfig is not None:
        with (d / "dataconfig.pkl").open("wb") as f:
            pickle.dump(dataconfig, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = training_meta or TrainingMeta()
    (d / "training_meta.json").write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True))
    (d / "VERSION").write_text(str(BUNDLE_FORMAT_VERSION))
    (d / "fingerprint.txt").write_text(compute_fingerprint(d))


def load_bundle(bundle_dir):
    """Return (ModelConfig, dataconfig | None, TrainingMeta); verify fingerprint."""
    d = Path(bundle_dir)
    missing = [n for n in _REQUIRED if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(f"bundle missing required files: {missing}")
    expected = (d / "fingerprint.txt").read_text().strip()
    if compute_fingerprint(d) != expected:
        raise ValueError(f"bundle fingerprint mismatch — {d} was modified or is corrupt")

    model_config = ModelConfig.from_dict(json.loads((d / "model_config.json").read_text()))
    dataconfig = None
    if (d / "dataconfig.pkl").exists():
        with (d / "dataconfig.pkl").open("rb") as f:
            dataconfig = pickle.load(f)
    meta = TrainingMeta()
    if (d / "training_meta.json").exists():
        meta = TrainingMeta.from_dict(json.loads((d / "training_meta.json").read_text()))
    return model_config, dataconfig, meta
```

- [ ] **Step 4: Run — expect PASS** (5 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/serialization/__init__.py src/alignair/serialization/bundle.py tests/alignair/serialization/test_bundle.py
git commit -m "feat(alignair): add state_dict bundle IO with fingerprint"
```

---

## Task 2: `serialization/pretrained.py` + wire into BaseAlignAIR

**Files:** Create `src/alignair/serialization/pretrained.py`; Modify `src/alignair/core/base.py`;
Test `tests/alignair/serialization/test_pretrained.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.core.multi_chain import MultiChainAlignAIR


def _single_cfg():
    return ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                       d_allele_count=4, has_d_gene=True)


def _multi_cfg():
    return ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                       d_allele_count=4, has_d_gene=True, number_of_chains=2,
                       chain_types=["IGH", "IGK"])


def _forward_parity(model_cls, cfg, tmp_path):
    torch.manual_seed(0)
    model = model_cls(cfg).eval()
    x = torch.randint(0, 6, (2, cfg.max_seq_length))
    with torch.no_grad():
        before = model(x).as_dict()
    model.save_pretrained(tmp_path, dataconfig={"ref": "abc"})

    reloaded = model_cls.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(x).as_dict()
    for k in before:
        assert torch.allclose(before[k], after[k], atol=1e-6), f"mismatch in {k}"


def test_single_chain_roundtrip_parity(tmp_path):
    _forward_parity(SingleChainAlignAIR, _single_cfg(), tmp_path)


def test_multi_chain_roundtrip_parity(tmp_path):
    _forward_parity(MultiChainAlignAIR, _multi_cfg(), tmp_path)
    assert "chain_type" in MultiChainAlignAIR.from_pretrained(tmp_path)(
        torch.randint(0, 6, (1, 256))).as_dict()


def test_from_pretrained_picks_class_by_config(tmp_path):
    MultiChainAlignAIR(_multi_cfg()).eval().save_pretrained(tmp_path)
    # Even calling via the base/single entry, config.is_multi_chain selects MultiChain.
    reloaded = SingleChainAlignAIR.from_pretrained(tmp_path)
    assert isinstance(reloaded, MultiChainAlignAIR)


def test_load_dataconfig(tmp_path):
    SingleChainAlignAIR(_single_cfg()).save_pretrained(tmp_path, dataconfig={"k": 1})
    assert SingleChainAlignAIR.load_dataconfig(tmp_path) == {"k": 1}
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `pretrained.py`**

```python
"""PretrainedMixin: save/load AlignAIR models as state_dict bundles."""
from pathlib import Path

import torch

from .bundle import save_bundle, load_bundle, TrainingMeta


class PretrainedMixin:
    """Mixed into BaseAlignAIR. Requires ``self.config`` to be a ModelConfig."""

    def save_pretrained(self, bundle_dir, *, dataconfig=None, training_meta=None) -> None:
        save_bundle(bundle_dir, model_config=self.config, state_dict=self.state_dict(),
                    dataconfig=dataconfig, training_meta=training_meta)

    @classmethod
    def from_pretrained(cls, bundle_dir):
        model_config, _dataconfig, _meta = load_bundle(bundle_dir)
        # Choose the concrete class from the saved config (ignore the calling cls).
        from ..core.single_chain import SingleChainAlignAIR
        from ..core.multi_chain import MultiChainAlignAIR
        model_cls = MultiChainAlignAIR if model_config.is_multi_chain else SingleChainAlignAIR
        model = model_cls(model_config)
        state = torch.load(Path(bundle_dir) / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    @staticmethod
    def load_dataconfig(bundle_dir):
        _cfg, dataconfig, _meta = load_bundle(bundle_dir)
        return dataconfig
```

- [ ] **Step 4: Wire into BaseAlignAIR** — in `src/alignair/core/base.py`:

Change the import block to add:
```python
from ..serialization.pretrained import PretrainedMixin
```
and change the class declaration:
```python
class BaseAlignAIR(PretrainedMixin, nn.Module):
```
(`PretrainedMixin` only does lazy imports of core inside `from_pretrained`, so there is
no import cycle at module-load time.)

- [ ] **Step 5: Run — expect PASS** (5 passed).
- [ ] **Step 6: Commit**
```bash
git add src/alignair/serialization/pretrained.py src/alignair/core/base.py tests/alignair/serialization/test_pretrained.py
git commit -m "feat(alignair): add PretrainedMixin (save_pretrained/from_pretrained)"
```

---

## Task 3: Package exports + full suite

**Files:** Create/Modify `src/alignair/serialization/__init__.py`

- [ ] **Step 1: Add exports** — `src/alignair/serialization/__init__.py`:
```python
from .bundle import save_bundle, load_bundle, compute_fingerprint, TrainingMeta, BUNDLE_FORMAT_VERSION
from .pretrained import PretrainedMixin

__all__ = ["save_bundle", "load_bundle", "compute_fingerprint", "TrainingMeta",
           "BUNDLE_FORMAT_VERSION", "PretrainedMixin"]
```

- [ ] **Step 2: Run the full alignair suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`
Expected: all green (existing 94 + new serialization tests).

- [ ] **Step 3: Smoke — round-trip via save/load and confirm no import cycle.**
```
PYTHONPATH=src .venv/bin/python -c "
import alignair.core, alignair.serialization
from alignair.core import SingleChainAlignAIR
print('ok', hasattr(SingleChainAlignAIR, 'from_pretrained'))
"
```
Expected: `ok True`.

- [ ] **Step 4: Commit**
```bash
git add src/alignair/serialization/__init__.py
git commit -m "feat(alignair): export serialization API"
```

---

## Self-Review

**Spec coverage (design §3):** bundle layout + `save_bundle`/`load_bundle` + fingerprint → Task 1;
`PretrainedMixin.save_pretrained`/`from_pretrained`/`load_dataconfig` + class selection via
`ModelConfig.is_multi_chain` → Task 2; exports → Task 3. "Done when" round-trip forward parity +
fingerprint stability + multi-chain chain_type round-trip are all covered by Task 2 tests + Task 1 tests.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `save_bundle(model_config, state_dict, dataconfig, training_meta)` /
`load_bundle -> (ModelConfig, dataconfig, TrainingMeta)` identical across Tasks 1/2; `TrainingMeta`
`to_dict`/`from_dict` consistent; `PretrainedMixin` method names (`save_pretrained`/`from_pretrained`/
`load_dataconfig`) match the tests; `weights_only=True` load is valid for a pure tensor state_dict.

**Known notes:** models must be in `.eval()` for forward parity (dropout off, BN uses running stats —
both captured in the round-trip). `dataconfig` is optional; absent → `load_dataconfig` returns None.
No `ModelBundleConfig` — `ModelConfig` is the single structural source of truth.
```
