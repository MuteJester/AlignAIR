# Amino-Acid AlignAIR — Phase 1 (Minimal Learning Proof) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that an AlignAIR-style model learns V/J protein-equivalence-class prediction and residue-level segmentation from amino-acid IGH reads — by overfitting one batch (capacity/plumbing proof) and showing a generalization signal on a held-out set — as a **thin AA module that reuses the core building blocks and leaves the shipped nucleotide model, loss, and trainer untouched.**

**Architecture:** A new `AAAlignAIR` composes reused core blocks — `TokenAndPositionEmbedding`, `build_tower`/`GeneBranch.segment`, `soft_gaussian_target`/`_soft_ce` — with a **softmax single-label** class head over protein-equivalence classes. No orientation head, no meta heads, no DNA gym/curriculum. The D branch is instantiated (`has_d=True`) but **unsupervised** — excluded from the loss until Phase 1b. A minimal AA data adapter turns GenAIRR `sequence_aa` + projected boundaries + class targets into batches; a minimal overfit/eval loop delivers the learning proof.

**Tech Stack:** Python 3, PyTorch, GenAIRR 2.2.0 (`HUMAN_IGH_OGRDB`), pytest. GPU (RTX 3090Ti) available for the real run; tests run on CPU.

## Global Constraints

- **Branch:** all work on `experiment/aminoacid-alignair`. Never commit to `main`.
- **Do NOT modify shipped code:** `core/model.py`, `core/losses.py`, `core/gene_branch.py`, `train/**`, `data/**` are **read-only** here. Reuse by import only. All new code lives under `src/alignair/aa/`.
- **Reuse, don't duplicate:** import `TokenAndPositionEmbedding` (`core/layers.py`), `build_tower`/`GeneBranch`/`MetaHead` (`core/gene_branch.py`), `_SEG_KERNELS` (`core/config.py`), `AlignAIRConfig` (`core/config.py`), `soft_gaussian_target`/`_soft_ce` (`core/losses.py`), and the Phase-0 `aa/` modules (`tokenizer`, `translation`, `reference`, `equivalence`).
- **Softmax single-label class heads:** class head emits **logits** over protein-equivalence classes; loss is `cross_entropy(logits, true_class_id)`. Never sigmoid, never multi-hot.
- **D is unsupervised in Phase 1:** the D branch exists and produces outputs; it is **excluded** from the loss (`supervised_genes=("v","j")`). Its class-head width is a placeholder (raw D allele count).
- **Half-open `[start, end)`** boundary semantics; AA residue coordinates (never nt).
- **Right-padding** to exactly `max_seq_length` (192); AA reads are ≤133 residues so this never truncates.
- **Determinism:** explicit `seed` on every GenAIRR stream.
- **Commits:** conventional messages; **never** a `Co-Authored-By: Claude` line or any Claude/Anthropic mention.
- **Run env:** `PYTHONPATH=src .venv/bin/python …`. Tests under `tests/alignair/aa/` (no `__init__.py`). Tests that load a real dataconfig / stream / train are `@pytest.mark.slow`.
- **Verified API facts:** `AAGeneBranch` can subclass `GeneBranch` — its `__init__` already builds `seg_tower`, `start_head`, `end_head`, `cutout`, `cls_tower`, `cls_mid`, `cls_head`; only `classify` needs overriding to drop the sigmoid. `GeneBranch.segment(emb, mask)` returns `(s_log, e_log, s_exp, e_exp)`. The nt model feeds branches the raw `emb` (the meta tower is only for meta heads), so AA needs no meta tower. GenAIRR record fields: `sequence_aa`, `junction_start`, `v_call`/`j_call`, `v/d/j_sequence_start/end`, `productive`, `sequence` (nt). Class-head output dim must equal the class count (V=149, J=7 for `HUMAN_IGH_OGRDB`); class ids are contiguous `0..n-1`.

---

### Task 1: AA model — softmax class head + orientation/meta-free assembly (`aa/model.py`)

**Files:**
- Create: `src/alignair/aa/model.py`
- Test: `tests/alignair/aa/test_aa_model.py`

**Interfaces:**
- Consumes: `TokenAndPositionEmbedding` (`core/layers.py`), `GeneBranch` (`core/gene_branch.py`), `AlignAIRConfig` (`core/config.py`).
- Produces:
  - `class AAGeneBranch(GeneBranch)` overriding `classify(emb, s_exp, e_exp) -> Tensor` to return **class logits** (no sigmoid).
  - `class AAAlignAIR(nn.Module)(cfg: AlignAIRConfig)` whose `forward(batch: dict) -> dict` emits, per gene `g` in `v/d/j`: `{g}_start_logits`, `{g}_end_logits`, `{g}_start`, `{g}_end`, `{g}_class_logits`, plus `position_mask`. No `orientation*`, no `mutation_rate`/`indel_count`/`productive`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_model.py
import torch
from alignair.aa.model import AAAlignAIR, AAGeneBranch
from alignair.core.config import AlignAIRConfig


def _cfg(v=5, j=3, d=4):
    # max_seq_length must be >= 128 for the V/J classification tower (N=6 -> needs 1<<7).
    return AlignAIRConfig(v_allele_count=v, j_allele_count=j, d_allele_count=d,
                          has_d=True, num_chain_types=1, max_seq_length=192, vocab_size=23)


def test_forward_emits_expected_keys_and_shapes():
    cfg = _cfg()
    model = AAAlignAIR(cfg).eval()
    tokens = torch.randint(1, 23, (2, 192))          # valid AA ids (1..22), no pad
    out = model({"tokenized_sequence": tokens})
    for g, n_cls in (("v", 5), ("j", 3), ("d", 4)):
        assert out[f"{g}_start_logits"].shape == (2, 192)
        assert out[f"{g}_end_logits"].shape == (2, 192)
        assert out[f"{g}_start"].shape == (2, 1)
        assert out[f"{g}_end"].shape == (2, 1)
        assert out[f"{g}_class_logits"].shape == (2, n_cls)
    assert out["position_mask"].shape == (2, 192)


def test_no_orientation_or_meta_heads_present():
    out = AAAlignAIR(_cfg()).eval()({"tokenized_sequence": torch.randint(1, 23, (2, 192))})
    for absent in ("orientation_logits", "orientation", "mutation_rate", "indel_count", "productive"):
        assert absent not in out


def test_class_head_returns_logits_not_probabilities():
    # softmax head: raw logits can be negative / exceed 1 (a sigmoid head could not).
    out = AAAlignAIR(_cfg()).eval()({"tokenized_sequence": torch.randint(1, 23, (2, 192))})
    logits = out["v_class_logits"]
    assert logits.min() < 0.0 or logits.max() > 1.0


def test_pad_positions_masked_out():
    cfg = _cfg()
    tokens = torch.zeros(1, 192, dtype=torch.long)   # all pad
    tokens[0, :10] = torch.randint(1, 23, (10,))
    out = AAAlignAIR(cfg).eval()({"tokenized_sequence": tokens})
    assert out["position_mask"][0, :10].all() and not out["position_mask"][0, 10:].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.model'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/model.py
"""Thin amino-acid AlignAIR model. Reuses the core embedding + conv towers + per-gene
segmentation branch; swaps the sigmoid multi-label allele head for a SOFTMAX single-label
protein-equivalence-class head. No orientation head, no meta heads, no meta tower (the nt
model's meta tower only feeds meta heads, which AA does not have)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import AlignAIRConfig
from ..core.gene_branch import GeneBranch
from ..core.layers import TokenAndPositionEmbedding


class AAGeneBranch(GeneBranch):
    """GeneBranch with a softmax single-label class head: ``classify`` returns class LOGITS
    (the loss applies cross-entropy; inference applies softmax)."""

    def classify(self, emb: torch.Tensor, s_exp: torch.Tensor, e_exp: torch.Tensor) -> torch.Tensor:
        mask = self.cutout(s_exp, e_exp).unsqueeze(-1)               # (B, L, 1)
        f = self.cls_tower(emb * mask)
        return self.cls_head(F.silu(self.cls_mid(f)))               # class logits (no sigmoid)


class AAAlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = TokenAndPositionEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)
        self.branches = nn.ModuleDict({s.name: AAGeneBranch(s, cfg) for s in cfg.gene_specs})

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        mask = tokens != 0                                          # non-pad
        emb = self.embedding(tokens)                               # no orientation correction
        out: dict = {"position_mask": mask}
        exp: dict = {}
        for name, branch in self.branches.items():
            s_log, e_log, s_exp, e_exp = branch.segment(emb, mask)
            out[f"{name}_start_logits"], out[f"{name}_end_logits"] = s_log, e_log
            out[f"{name}_start"], out[f"{name}_end"] = s_exp, e_exp
            exp[name] = (s_exp, e_exp)
        for name, branch in self.branches.items():
            out[f"{name}_class_logits"] = branch.classify(emb, *exp[name])
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/model.py tests/alignair/aa/test_aa_model.py
git commit -m "feat(aa): thin AAAlignAIR model with softmax class head (no orientation/meta)"
```

---

### Task 2: AA loss — soft-Gaussian segmentation CE + softmax class CE (`aa/loss.py`)

**Files:**
- Create: `src/alignair/aa/loss.py`
- Test: `tests/alignair/aa/test_aa_loss.py`

**Interfaces:**
- Consumes: `soft_gaussian_target`, `_soft_ce` (`core/losses.py`).
- Produces:
  - `aa_loss(out: dict, targets: dict, supervised_genes=("v","j")) -> tuple[Tensor, dict]` — total loss + parts `{"segmentation","classification"}`. Boundary CE is masked to `out["position_mask"]`; D is excluded (not in `supervised_genes`).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_loss.py
import torch
from alignair.aa.loss import aa_loss

L = 192


def _out(bstart, bend, vlogits, jlogits, mask=None):
    B = vlogits.shape[0]
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool)
    o = {"position_mask": mask}
    for g in ("v", "j"):
        o[f"{g}_start_logits"] = bstart.clone()
        o[f"{g}_end_logits"] = bend.clone()
        o[f"{g}_start"] = torch.zeros(B, 1)
        o[f"{g}_end"] = torch.zeros(B, 1)
    o["v_class_logits"], o["j_class_logits"] = vlogits, jlogits
    return o


def test_loss_is_finite_scalar_and_has_parts():
    B = 4
    out = _out(torch.randn(B, L), torch.randn(B, L), torch.randn(B, 5), torch.randn(B, 3))
    targets = {"v_start": torch.zeros(B, 1), "v_end": torch.full((B, 1), 40.0),
               "j_start": torch.full((B, 1), 60.0), "j_end": torch.full((B, 1), 70.0),
               "v_class": torch.randint(0, 5, (B,)), "j_class": torch.randint(0, 3, (B,))}
    total, parts = aa_loss(out, targets)
    assert total.ndim == 0 and torch.isfinite(total)
    assert set(parts) == {"segmentation", "classification"}


def test_confident_correct_class_gives_near_zero_classification():
    B = 3
    vlogits = torch.full((B, 5), -10.0); jlogits = torch.full((B, 3), -10.0)
    vlogits[range(B), 1] = 10.0; jlogits[range(B), 2] = 10.0    # confidently class 1 / class 2
    out = _out(torch.zeros(B, L), torch.zeros(B, L), vlogits, jlogits)
    targets = {"v_start": torch.zeros(B, 1), "v_end": torch.zeros(B, 1),
               "j_start": torch.zeros(B, 1), "j_end": torch.zeros(B, 1),
               "v_class": torch.full((B,), 1), "j_class": torch.full((B,), 2)}
    _, parts = aa_loss(out, targets)
    assert parts["classification"] < 1e-3


def test_d_is_excluded_from_loss():
    # a D key error must NOT occur — D is not in supervised_genes.
    B = 2
    out = _out(torch.randn(B, L), torch.randn(B, L), torch.randn(B, 5), torch.randn(B, 3))
    targets = {"v_start": torch.zeros(B, 1), "v_end": torch.zeros(B, 1),
               "j_start": torch.zeros(B, 1), "j_end": torch.zeros(B, 1),
               "v_class": torch.zeros(B, dtype=torch.long), "j_class": torch.zeros(B, dtype=torch.long)}
    total, _ = aa_loss(out, targets)                # no d_* keys anywhere -> must not raise
    assert torch.isfinite(total)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_loss.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.loss'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/loss.py
"""Minimal AA loss: soft-Gaussian segmentation cross-entropy (reused primitives) + softmax
single-label class cross-entropy. Equal-weighted sum (no Kendall) for the learning proof.
D is excluded (not in ``supervised_genes``)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.losses import soft_gaussian_target, _soft_ce


def aa_loss(out: dict, targets: dict, supervised_genes=("v", "j")):
    L = out["v_start_logits"].shape[-1]
    pos_mask = out.get("position_mask")

    seg = out["v_start_logits"].new_zeros(())
    for g in supervised_genes:
        for b in ("start", "end"):
            tgt = soft_gaussian_target(targets[f"{g}_{b}"], L)
            seg = seg + _soft_ce(out[f"{g}_{b}_logits"], tgt, pos_mask)

    clf = out["v_start_logits"].new_zeros(())
    for g in supervised_genes:
        clf = clf + F.cross_entropy(out[f"{g}_class_logits"], targets[f"{g}_class"])

    total = seg + clf
    return total, {"segmentation": seg, "classification": clf}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_loss.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/loss.py tests/alignair/aa/test_aa_loss.py
git commit -m "feat(aa): minimal AA loss (soft-Gaussian seg CE + softmax class CE, D excluded)"
```

---

### Task 3: AA config factory (`aa/config.py`)

**Files:**
- Create: `src/alignair/aa/config.py`
- Test: `tests/alignair/aa/test_aa_config.py`

**Interfaces:**
- Consumes: `AlignAIRConfig` (`core/config.py`), `AA_VOCAB_SIZE` (`aa/tokenizer.py`).
- Produces: `aa_igh_config(n_v_classes: int, n_j_classes: int, n_d_placeholder: int, max_seq_length: int = 192) -> AlignAIRConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_config.py
from alignair.aa.config import aa_igh_config
from alignair.aa.tokenizer import AA_VOCAB_SIZE


def test_config_wires_class_counts_and_aa_vocab():
    cfg = aa_igh_config(n_v_classes=149, n_j_classes=7, n_d_placeholder=33)
    assert cfg.vocab_size == AA_VOCAB_SIZE == 23
    assert cfg.v_allele_count == 149 and cfg.j_allele_count == 7 and cfg.d_allele_count == 33
    assert cfg.has_d is True
    assert cfg.max_seq_length == 192
    # gene_specs carries V, D, J in that order (class-head widths follow class counts)
    assert tuple(s.name for s in cfg.gene_specs) == ("v", "d", "j")
    assert tuple(s.allele_count for s in cfg.gene_specs) == (149, 33, 7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/config.py
"""AA model config factory. Reuses AlignAIRConfig purely for sizing the reused building blocks;
the classification heads are sized by PROTEIN-CLASS counts, not raw allele counts. D count is a
placeholder (raw D alleles) since the D branch is unsupervised in Phase 1."""
from __future__ import annotations

from ..core.config import AlignAIRConfig
from .tokenizer import AA_VOCAB_SIZE


def aa_igh_config(n_v_classes: int, n_j_classes: int, n_d_placeholder: int,
                  max_seq_length: int = 192) -> AlignAIRConfig:
    return AlignAIRConfig(
        v_allele_count=n_v_classes,
        j_allele_count=n_j_classes,
        d_allele_count=n_d_placeholder,
        has_d=True,
        num_chain_types=1,
        max_seq_length=max_seq_length,
        vocab_size=AA_VOCAB_SIZE,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/config.py tests/alignair/aa/test_aa_config.py
git commit -m "feat(aa): AA IGH config factory (class-count-sized heads, AA vocab)"
```

---

### Task 4: AA data adapter — records → class + boundary targets (`aa/data.py`)

**Files:**
- Create: `src/alignair/aa/data.py`
- Test: `tests/alignair/aa/test_aa_data.py`

**Interfaces:**
- Consumes: `AAProteinTokenizer` (`aa/tokenizer.py`), `project_codons`, `frame_offset_of` (`aa/translation.py`), `EquivalenceClassMap` (`aa/equivalence.py`).
- Produces:
  - `record_to_sample(rec: dict, class_maps: dict, tokenizer, max_len: int) -> dict | None` — one usable sample `{tokens, v_class, j_class, v_start, v_end, j_start, j_end}`, or `None` if not usable.
  - `collate_aa(samples: list, max_len: int, device="cpu") -> tuple[dict, dict]` — `(batch_in, targets)`.
  - `stream_aa_samples(n, seed, class_maps, tokenizer, max_len, rate=0.05, dc_name="HUMAN_IGH_OGRDB")` — generator of usable samples.
  - `build_class_maps(dc_name="HUMAN_IGH_OGRDB") -> tuple[dict, dict]` — `(class_maps, meta)` where `class_maps={"V":EquivalenceClassMap,"J":...}` and `meta={"n_v_classes","n_j_classes","n_d","db_hash"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_data.py
import numpy as np
import pytest
import torch

from alignair.aa.data import record_to_sample, collate_aa
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.aa.equivalence import build_equivalence_classes, db_hash


def _class_maps():
    vmap = build_equivalence_classes("V", {"IGHV1-1*01": "QVQLC", "IGHV2-2*01": "EVQLC"}, db_hash([("x", "a")]))
    jmap = build_equivalence_classes("J", {"IGHJ1*01": "WGQGT", "IGHJ2*01": "WGKGT"}, db_hash([("y", "b")]))
    return {"V": vmap, "J": jmap}


def test_record_to_sample_projects_boundaries_and_maps_class():
    cmaps = _class_maps()
    tok = AAProteinTokenizer()
    # frame 0; V=[0,12) J=[12,24). sequence length 24 -> 8 residues. V owns r0..r3, J r4..r7.
    rec = {"sequence": "A" * 24, "sequence_aa": "QVQLCXXX", "junction_start": 0, "productive": True,
           "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01",
           "v_sequence_start": 0, "v_sequence_end": 12,
           "d_sequence_start": 12, "d_sequence_end": 12,
           "j_sequence_start": 12, "j_sequence_end": 24}
    s = record_to_sample(rec, cmaps, tok, max_len=192)
    assert s is not None
    assert s["v_class"] == cmaps["V"].class_of("IGHV1-1*01")
    assert s["j_class"] == cmaps["J"].class_of("IGHJ1*01")
    assert (s["v_start"], s["v_end"]) == (0, 4)     # 4 fully-owned V residues
    assert (s["j_start"], s["j_end"]) == (4, 8)
    assert isinstance(s["tokens"], np.ndarray)


def test_record_to_sample_rejects_nonproductive_and_missing_aa():
    cmaps, tok = _class_maps(), AAProteinTokenizer()
    base = {"sequence": "A" * 24, "sequence_aa": "QVQLCXXX", "junction_start": 0, "productive": True,
            "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01",
            "v_sequence_start": 0, "v_sequence_end": 12, "d_sequence_start": 12,
            "d_sequence_end": 12, "j_sequence_start": 12, "j_sequence_end": 24}
    assert record_to_sample({**base, "productive": False}, cmaps, tok, 192) is None
    assert record_to_sample({**base, "sequence_aa": ""}, cmaps, tok, 192) is None
    assert record_to_sample({**base, "junction_start": None}, cmaps, tok, 192) is None
    assert record_to_sample({**base, "v_call": "IGHV9-9*99"}, cmaps, tok, 192) is None  # unknown allele


def test_collate_aa_shapes_and_dtypes():
    samples = [
        {"tokens": np.array([1, 2, 3], dtype=np.int64), "v_class": 0, "j_class": 1,
         "v_start": 0, "v_end": 2, "j_start": 2, "j_end": 3},
        {"tokens": np.array([4, 5], dtype=np.int64), "v_class": 1, "j_class": 0,
         "v_start": 0, "v_end": 1, "j_start": 1, "j_end": 2},
    ]
    bi, tg = collate_aa(samples, max_len=192)
    assert bi["tokenized_sequence"].shape == (2, 192)
    assert bi["tokenized_sequence"][0, :3].tolist() == [1, 2, 3]
    assert bi["tokenized_sequence"][1, 2:].sum().item() == 0     # right-padded
    assert tg["v_class"].tolist() == [0, 1] and tg["v_class"].dtype == torch.long
    assert tg["v_start"].shape == (2, 1) and tg["v_start"].dtype == torch.float32


@pytest.mark.slow
def test_stream_aa_samples_real_igh_yields_valid_samples():
    from alignair.aa.data import build_class_maps, stream_aa_samples
    from alignair.aa.tokenizer import AAProteinTokenizer
    cmaps, meta = build_class_maps("HUMAN_IGH_OGRDB")
    assert meta["n_v_classes"] == 149 and meta["n_j_classes"] == 7
    tok = AAProteinTokenizer()
    got = []
    for s in stream_aa_samples(n=2000, seed=0, class_maps=cmaps, tokenizer=tok, max_len=192):
        got.append(s)
        if len(got) >= 32:
            break
    assert len(got) == 32
    for s in got:
        assert 0 <= s["v_class"] < 149 and 0 <= s["j_class"] < 7
        assert 0 <= s["v_start"] < s["v_end"] <= 192
        assert len(s["tokens"]) <= 192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_data.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.data'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/data.py
"""AA data adapter: GenAIRR productive IGH records -> (tokens, class targets, projected boundary
targets). Reads GenAIRR's `sequence_aa` directly (reads are NOT translated here); only the
nucleotide segment coordinates are projected into AA residue boundaries (with straddling-codon
masking) via `project_codons`. True V/J alleles are mapped to protein-equivalence class ids."""
from __future__ import annotations

import numpy as np
import torch

from .translation import frame_offset_of, project_codons
from .equivalence import db_hash, build_equivalence_classes

_SEG_COLS = {
    "V": ("v_sequence_start", "v_sequence_end"),
    "D": ("d_sequence_start", "d_sequence_end"),
    "J": ("j_sequence_start", "j_sequence_end"),
}


def _first(call: str) -> str:
    return str(call).split(",")[0].strip()


def _segments(rec) -> dict:
    segs = {}
    for seg, (a, b) in _SEG_COLS.items():
        s, e = rec.get(a), rec.get(b)
        if s is not None and e is not None:
            segs[seg] = (int(s), int(e))
    return segs


def record_to_sample(rec: dict, class_maps: dict, tokenizer, max_len: int):
    """One GenAIRR record -> a usable sample dict, or None if unusable (non-productive, no AA,
    no frame, unknown allele, missing V/J boundary, or over length)."""
    js = rec.get("junction_start")
    aa = rec.get("sequence_aa")
    if js is None or not aa or not rec.get("productive"):
        return None
    try:
        v_class = class_maps["V"].class_of(_first(rec["v_call"]))
        j_class = class_maps["J"].class_of(_first(rec["j_call"]))
    except KeyError:
        return None                                     # allele not in the (quarantine-filtered) class map
    proj = project_codons(len(rec["sequence"]), frame_offset_of(js), _segments(rec))
    vb, jb = proj.aa_bounds.get("V"), proj.aa_bounds.get("J")
    if vb is None or jb is None:
        return None
    tokens = tokenizer.encode(aa)
    if len(tokens) > max_len:
        return None
    return {"tokens": tokens, "v_class": v_class, "j_class": j_class,
            "v_start": vb[0], "v_end": vb[1], "j_start": jb[0], "j_end": jb[1]}


def collate_aa(samples: list, max_len: int, device: str = "cpu"):
    """Right-pad tokens to ``max_len``; build model input + AA targets (class ids long, boundaries float)."""
    B = len(samples)
    tokens = torch.zeros(B, max_len, dtype=torch.long)
    for i, s in enumerate(samples):
        t = s["tokens"]
        tokens[i, :len(t)] = torch.from_numpy(np.asarray(t, dtype=np.int64))
    batch_in = {"tokenized_sequence": tokens.to(device)}
    targets = {
        "v_class": torch.tensor([s["v_class"] for s in samples], dtype=torch.long, device=device),
        "j_class": torch.tensor([s["j_class"] for s in samples], dtype=torch.long, device=device),
    }
    for key in ("v_start", "v_end", "j_start", "j_end"):
        targets[key] = torch.tensor([[float(s[key])] for s in samples],
                                    dtype=torch.float32, device=device)
    return batch_in, targets


def build_class_maps(dc_name: str = "HUMAN_IGH_OGRDB"):
    """Build the frozen V/J protein-equivalence class maps from a GenAIRR dataconfig."""
    import GenAIRR as ga

    from .reference import build_aa_germlines
    dc = getattr(ga, dc_name)
    gset = build_aa_germlines(dc, genes=("V", "J"))
    dbh = db_hash([(a.name, a.ungapped_seq) for a in dc.allele_list("v")]
                  + [(a.name, a.ungapped_seq) for a in dc.allele_list("j")])
    class_maps = {}
    for gene in ("V", "J"):
        aa_by_name = {n_: g.aa_seq for n_, g in gset.genes[gene].items()}
        class_maps[gene] = build_equivalence_classes(gene, aa_by_name, dbh)
    meta = {"n_v_classes": class_maps["V"].n_classes(),
            "n_j_classes": class_maps["J"].n_classes(),
            "n_d": len(dc.allele_list("d")), "db_hash": dbh}
    return class_maps, meta


def stream_aa_samples(n, seed, class_maps, tokenizer, max_len, rate=0.05,
                      dc_name: str = "HUMAN_IGH_OGRDB"):
    """Yield usable AA samples from a GenAIRR IGH stream (productive-only, s5f SHM)."""
    import GenAIRR as ga
    exp = ga.Experiment.on(dc_name).recombine().mutate(model="s5f", rate=rate)
    for rec in exp.stream_records(n=n, seed=seed):
        s = record_to_sample(rec, class_maps, tokenizer, max_len)
        if s is not None:
            yield s
```

- [ ] **Step 4: Run tests to verify they pass**

Run (fast): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_data.py -q -m "not slow"`
Expected: PASS.
Run (slow): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_data.py -q -m slow`
Expected: PASS (149 V / 7 J classes; samples valid).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/data.py tests/alignair/aa/test_aa_data.py
git commit -m "feat(aa): AA data adapter (records -> class + projected boundary targets)"
```

---

### Task 5: Overfit-one-batch learning proof + eval metrics (`aa/train.py`)

**Files:**
- Create: `src/alignair/aa/train.py`
- Test: `tests/alignair/aa/test_aa_train.py`

**Interfaces:**
- Consumes: `aa_loss` (`aa/loss.py`), `AAAlignAIR` (`aa/model.py`).
- Produces:
  - `aa_eval_metrics(out: dict, targets: dict, genes=("v","j")) -> dict` — `{g}_class_acc`, `{g}_start_mae`, `{g}_end_mae`.
  - `overfit_one_batch(model, batch_in, targets, *, steps=300, lr=1e-3, genes=("v","j")) -> dict` — `{initial_loss, final_loss, ...metrics}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_train.py
import pytest
import torch

from alignair.aa.train import aa_eval_metrics, overfit_one_batch
from alignair.aa.model import AAAlignAIR
from alignair.aa.config import aa_igh_config


def test_aa_eval_metrics_perfect_predictions():
    B, L = 3, 192
    out = {"position_mask": torch.ones(B, L, dtype=torch.bool)}
    for g in ("v", "j"):
        logits = torch.full((B, 4), -10.0); logits[range(B), 2] = 10.0
        out[f"{g}_class_logits"] = logits
        out[f"{g}_start"] = torch.full((B, 1), 5.0)
        out[f"{g}_end"] = torch.full((B, 1), 40.0)
    targets = {"v_class": torch.full((B,), 2), "j_class": torch.full((B,), 2),
               "v_start": torch.full((B, 1), 5.0), "v_end": torch.full((B, 1), 40.0),
               "j_start": torch.full((B, 1), 5.0), "j_end": torch.full((B, 1), 40.0)}
    m = aa_eval_metrics(out, targets)
    assert m["v_class_acc"] == 1.0 and m["j_class_acc"] == 1.0
    assert m["v_start_mae"] < 1e-4


@pytest.mark.slow
def test_overfit_one_batch_learns_classes_and_boundaries():
    torch.manual_seed(0)
    from alignair.aa.data import build_class_maps, stream_aa_samples, collate_aa
    from alignair.aa.tokenizer import AAProteinTokenizer
    cmaps, meta = build_class_maps("HUMAN_IGH_OGRDB")
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"], meta["n_d"])
    tok = AAProteinTokenizer()
    samples = []
    for s in stream_aa_samples(n=5000, seed=1, class_maps=cmaps, tokenizer=tok, max_len=192):
        samples.append(s)
        if len(samples) >= 16:
            break
    batch_in, targets = collate_aa(samples, max_len=192)
    model = AAAlignAIR(cfg)
    res = overfit_one_batch(model, batch_in, targets, steps=300, lr=1e-3)
    # capacity/plumbing proof: the model can memorize one batch's classes + boundaries.
    assert res["final_loss"] < 0.5 * res["initial_loss"]
    assert res["v_class_acc"] >= 0.9 and res["j_class_acc"] >= 0.9
    assert res["v_start_mae"] < 5.0 and res["j_end_mae"] < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_train.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/train.py
"""Minimal AA training helpers for the Phase-1 learning proof: overfit one batch (capacity/
plumbing) and per-task eval metrics. Full curriculum/checkpointing is deferred (Phase 2)."""
from __future__ import annotations

import torch

from .loss import aa_loss


@torch.no_grad()
def aa_eval_metrics(out: dict, targets: dict, genes=("v", "j")) -> dict:
    m = {}
    for g in genes:
        pred = out[f"{g}_class_logits"].argmax(-1)
        m[f"{g}_class_acc"] = float((pred == targets[f"{g}_class"]).float().mean())
        for b in ("start", "end"):
            m[f"{g}_{b}_mae"] = float(
                (out[f"{g}_{b}"].squeeze(-1) - targets[f"{g}_{b}"].squeeze(-1)).abs().mean())
    return m


def overfit_one_batch(model, batch_in, targets, *, steps=300, lr=1e-3, genes=("v", "j")) -> dict:
    """Drive one batch to memorization; returns initial/final loss + final metrics."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    initial = None
    for _ in range(steps):
        out = model(batch_in)
        total, _ = aa_loss(out, targets, supervised_genes=genes)
        if initial is None:
            initial = float(total.detach())
        opt.zero_grad()
        total.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(batch_in)
        final = float(aa_loss(out, targets, supervised_genes=genes)[0])
        metrics = aa_eval_metrics(out, targets, genes)
    return {"initial_loss": initial, "final_loss": final, **metrics}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (fast): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_train.py -q -m "not slow"`
Expected: PASS.
Run (slow, trains on CPU): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_train.py -q -m slow`
Expected: PASS — the model overfits one batch (V/J class acc ≥ 0.9).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/train.py tests/alignair/aa/test_aa_train.py
git commit -m "feat(aa): overfit-one-batch learning proof + AA eval metrics"
```

---

### Task 6: Phase-1 runner + real learning proof, record results (`aa/phase1.py`, `scripts/aa/phase1_proof.py`)

Wire it together into the Phase-1 deliverable: (a) overfit one batch (memorization/capacity proof), (b) a short train on the GenAIRR stream with a **held-out** eval (generalization signal). Run it, record results, and append the Phase-1 verdict to the design spec.

**Files:**
- Create: `src/alignair/aa/phase1.py`
- Create: `scripts/aa/phase1_proof.py`
- Test: `tests/alignair/aa/test_aa_phase1.py`
- Modify (Step 6 below): `docs/superpowers/specs/2026-07-18-aminoacid-alignair-design.md`

**Interfaces:**
- Consumes: `build_class_maps`, `stream_aa_samples`, `collate_aa` (`aa/data.py`), `aa_igh_config` (`aa/config.py`), `AAAlignAIR` (`aa/model.py`), `aa_loss` (`aa/loss.py`), `aa_eval_metrics`, `overfit_one_batch` (`aa/train.py`).
- Produces:
  - `run_learning_proof(*, overfit_batch=16, overfit_steps=300, train_steps=400, batch_size=32, lr=1e-3, seed=0, device="cpu", dc_name="HUMAN_IGH_OGRDB") -> dict` — `{overfit: {...}, generalization: {train_final: {...}, heldout: {...}}, meta: {...}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_phase1.py
import pytest


@pytest.mark.slow
def test_run_learning_proof_smoke():
    from alignair.aa.phase1 import run_learning_proof
    r = run_learning_proof(overfit_batch=8, overfit_steps=60, train_steps=40,
                           batch_size=16, lr=1e-3, seed=0, device="cpu")
    # overfit result present and improves
    assert r["overfit"]["final_loss"] < r["overfit"]["initial_loss"]
    # generalization block computed on a disjoint held-out batch
    for g in ("v", "j"):
        assert 0.0 <= r["generalization"]["heldout"][f"{g}_class_acc"] <= 1.0
    assert r["meta"]["n_v_classes"] == 149 and r["meta"]["n_j_classes"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_phase1.py -q -m slow`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.phase1'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase1.py
"""Phase-1 learning proof: (a) overfit one batch (capacity), (b) short train + held-out eval
(generalization signal). Deliberately minimal — full training is Phase 2."""
from __future__ import annotations

import itertools

import torch

from .config import aa_igh_config
from .data import build_class_maps, collate_aa, stream_aa_samples
from .loss import aa_loss
from .model import AAAlignAIR
from .tokenizer import AAProteinTokenizer
from .train import aa_eval_metrics, overfit_one_batch


def _take(stream, k):
    return list(itertools.islice(stream, k))


def run_learning_proof(*, overfit_batch=16, overfit_steps=300, train_steps=400,
                       batch_size=32, lr=1e-3, seed=0, device="cpu",
                       dc_name="HUMAN_IGH_OGRDB") -> dict:
    torch.manual_seed(seed)
    class_maps, meta = build_class_maps(dc_name)
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"], meta["n_d"])
    tok = AAProteinTokenizer()

    def stream(sd):
        return stream_aa_samples(n=None, seed=sd, class_maps=class_maps, tokenizer=tok,
                                 max_len=cfg.max_seq_length, dc_name=dc_name)

    # (a) overfit one batch
    ob = _take(stream(seed + 1), overfit_batch)
    obi, obt = collate_aa(ob, cfg.max_seq_length, device)
    overfit = overfit_one_batch(AAAlignAIR(cfg).to(device), obi, obt,
                                steps=overfit_steps, lr=lr)

    # (b) short train + held-out eval (disjoint seeds)
    model = AAAlignAIR(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tr = stream(seed + 100)
    model.train()
    for _ in range(train_steps):
        batch = _take(tr, batch_size)
        if len(batch) < batch_size:
            break
        bi, bt = collate_aa(batch, cfg.max_seq_length, device)
        out = model(bi)
        total, _ = aa_loss(out, bt)
        opt.zero_grad(); total.backward(); opt.step()
    with torch.no_grad():
        train_final = aa_eval_metrics(model(bi), bt)
    hb = _take(stream(seed + 999), batch_size)
    hbi, hbt = collate_aa(hb, cfg.max_seq_length, device)
    model.eval()
    with torch.no_grad():
        heldout = aa_eval_metrics(model(hbi), hbt)

    return {"overfit": overfit,
            "generalization": {"train_final": train_final, "heldout": heldout},
            "meta": {**meta, "train_steps": train_steps, "batch_size": batch_size, "lr": lr,
                     "seed": seed, "max_seq_length": cfg.max_seq_length}}
```

```python
# scripts/aa/phase1_proof.py
"""CLI: run the amino-acid AlignAIR Phase-1 learning proof and print a JSON summary.

Run (GPU): PYTHONPATH=src .venv/bin/python scripts/aa/phase1_proof.py --device cuda --train-steps 2000
"""
import argparse
import json

from alignair.aa.phase1 import run_learning_proof


def main() -> None:
    ap = argparse.ArgumentParser(description="Amino-acid AlignAIR Phase-1 learning proof")
    ap.add_argument("--overfit-batch", type=int, default=32)
    ap.add_argument("--overfit-steps", type=int, default=400)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/aa/phase1/proof.json")
    args = ap.parse_args()

    r = run_learning_proof(overfit_batch=args.overfit_batch, overfit_steps=args.overfit_steps,
                           train_steps=args.train_steps, batch_size=args.batch_size, lr=args.lr,
                           seed=args.seed, device=args.device)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2, sort_keys=True)
    print(json.dumps(r, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_phase1.py -q -m slow`
Expected: PASS (smoke: overfit improves, held-out metrics computed, 149/7 classes).

- [ ] **Step 5: Run the real learning proof (GPU) and the full AA suite**

Run the proof at a meaningful scale:
`PYTHONPATH=src .venv/bin/python scripts/aa/phase1_proof.py --device cuda --overfit-batch 32 --overfit-steps 400 --train-steps 3000 --batch-size 64 --out experiments/aa/phase1/proof.json`
Expected: prints a JSON summary. Record: overfit V/J class accuracy (should approach 1.0 — capacity proof), and held-out V/J class accuracy + boundary MAE (the generalization signal).

Then confirm no regressions:
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/ -q` (all AA tests)
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q -m "not slow" --ignore=tests/alignair/aa` (nucleotide suite unaffected)

- [ ] **Step 6: Record the Phase-1 verdict and commit**

Copy the frozen proof to a tracked path (`experiments/` is gitignored) and append a "Phase 1 — Learning-Proof Results" section to the design spec with the real overfit + held-out numbers and a verdict (does it learn V/J classes + boundaries; is the generalization signal strong enough to proceed to Phase 1b / Phase 2).

```bash
mkdir -p docs/superpowers/artifacts/2026-07-18-aa-phase1
cp experiments/aa/phase1/proof.json docs/superpowers/artifacts/2026-07-18-aa-phase1/proof.json
git add src/alignair/aa/phase1.py scripts/aa/phase1_proof.py tests/alignair/aa/test_aa_phase1.py \
        docs/superpowers/artifacts/2026-07-18-aa-phase1/proof.json \
        docs/superpowers/specs/2026-07-18-aminoacid-alignair-design.md
git commit -m "feat(aa): Phase-1 learning proof (overfit + held-out) + results verdict"
```

---

## Self-Review

**Spec coverage (design-spec Phase 1 bullets):**
- "Right-padded AA tokenizer" → reused from Phase 0 (`aa/tokenizer.py`); `collate_aa` right-pads. ✅
- "V/J equivalence-class heads" → `AAGeneBranch` softmax class head over class counts (Task 1); class targets from the frozen class map (Task 4). ✅
- "No orientation or meta-heads" → `AAAlignAIR` omits both (Task 1); `aa_loss` has no meta/orientation terms (Task 2). ✅
- "Overfit one batch, then a small held-out simulation set" → `overfit_one_batch` (Task 5) + `run_learning_proof` held-out eval (Task 6). ✅
- "Demonstrate boundaries and class probabilities actually learn" → overfit asserts V/J class acc ≥ 0.9 + boundary MAE < 5 (Task 5); held-out generalization recorded (Task 6). ✅
- Decision "softmax single-label + CE" → `AAGeneBranch.classify` returns logits; `aa_loss` uses `cross_entropy`. ✅
- Decision "keep D branch, unsupervised" → `has_d=True` config (Task 3); D excluded via `supervised_genes=("v","j")` (Task 2). ✅
- Decision "thin AA module, reuse blocks, don't modify shipped code" → all new files under `aa/`; shipped model/loss/trainer imported only. ✅
- Leakage note (spec §6): the overfit/held-out batches use **disjoint seeds** (`seed+1`, `seed+100`, `seed+999`). Full clonal-lineage splitting is a Phase-2 concern (the minimal proof uses independent-rearrangement streams); flagged here, not silently skipped.

**Placeholder scan:** none — every code step is complete and runnable; every run step has an exact command + expected outcome. Task 6 Step 6's "record the real numbers" is the experimental deliverable, not a code placeholder.

**Type consistency:** `AAAlignAIR.forward` emits `{g}_class_logits`, `{g}_start_logits`, `{g}_end_logits`, `{g}_start`, `{g}_end`, `position_mask` — consumed identically by `aa_loss` (Task 2) and `aa_eval_metrics` (Task 5). `record_to_sample` keys (`tokens,v_class,j_class,v_start,v_end,j_start,j_end`) match `collate_aa`'s reads (Task 4), whose target keys (`v_class,j_class,v_start,v_end,j_start,j_end`) match `aa_loss`/`aa_eval_metrics`. `aa_igh_config` counts feed `gene_specs.allele_count` → class-head widths (Task 3 ↔ Task 1). `build_class_maps` returns `(class_maps, meta)` consumed by Task 5/6. ✅

**Scope check:** one subsystem (the thin AA learning-proof module), one plan, delivers a runnable proof + recorded verdict. Phase 1b (supervised D source classification), full training/curriculum, checkpointing, and the §4 shared-model unification are explicitly out of scope and get their own plans.
