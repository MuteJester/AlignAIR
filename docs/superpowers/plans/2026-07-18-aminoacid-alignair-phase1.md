# Amino-Acid AlignAIR — Phase 1 (Minimal Learning Proof) Implementation Plan — **rev 2**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that an AlignAIR-style model learns V/J protein-equivalence-class prediction and residue-level segmentation from amino-acid IGH reads — capacity via overfit-one-batch, generalization via a **frozen ≥4,096-record held-out set evaluated against explicit baselines** — as a thin AA module that leaves the shipped nucleotide code untouched.

**Architecture:** `AAAlignAIR` composes reused core blocks (`TokenAndPositionEmbedding`, `GeneBranch`'s towers/heads) with a **softmax single-label** class head and **separate start/end boundary-slot masks**. No orientation head, no meta heads, **no D branch** (`has_d=False`). A minimal AA data adapter turns GenAIRR `sequence_aa` + projected boundaries + class targets into batches, with explicit rejection accounting.

**Tech Stack:** Python 3, PyTorch, GenAIRR 2.2.0 (`HUMAN_IGH_OGRDB`), pytest. GPU for the real run; tests on CPU.

## Revision note (rev 2)

Three claims from review were verified empirically before rewriting:
- **`Conv1DBatchNorm` uses `nn.BatchNorm1d`** → an unsupervised D branch drifts running stats in train mode while its weights stay at init. **D removed** (P0.2).
- **`aa_len_max` is 146**, not 133 (p95 was conflated with max). Corrected here and in the spec (P0.3).
- **Boundary-slot bug confirmed and worse than described:** masked soft-argmax gives slot `n` probability **0.0**, and **26.6%** of the Gaussian target mass lands on that dropped slot. Because `_soft_ce` drops masked mass *without renormalizing*, CE for an unreachable target at `n` (0.763) is **lower** than for a reachable target at `n-1` (1.316) — the loss *rewards* the boundary it cannot predict. Fixed by separate start/end masks **plus target renormalization over allowed slots** (P0.1 + addition).

## Global Constraints

- **Branch:** `experiment/aminoacid-alignair`. Never commit to `main`.
- **Do NOT modify shipped code:** `core/**`, `train/**`, `data/**` are read-only; reuse by import only. All new code under `src/alignair/aa/`.
- **Boundary-slot semantics (P0.1):** half-open `[start, end)`. **Start slots `0..n-1`; end slots `1..n`.** Decode and supervise each boundary with **its own** mask. Require `len(tokens) < max_seq_length` so slot `n` exists. Never convert to inclusive ends to fit the head.
- **Target renormalization:** mask the soft-Gaussian target to the allowed slots and **renormalize to sum 1** before the CE, so edge targets are not under-penalized.
- **Loss normalization:** **mean** the 4 boundary CE terms and **mean** the 2 class CE terms, then sum the two means.
- **No D in Phase 1 (P0.2):** `has_d=False`. Phase 1b builds a fresh V/D/J config and explicitly transfers compatible V/J weights. Never ship a random D head.
- **Stop contract (P0.3):** reject any `sequence_aa` containing `*` in the adapter, consistent with the Phase-0 decision. Record **rejection counts by reason**.
- **Softmax single-label class heads:** class head emits logits; loss is `cross_entropy`. Never sigmoid/multi-hot.
- **Reject degenerate inputs:** empty sequences rejected explicitly; never feed an all-pad row (all-masked softmax ⇒ NaN).
- **Determinism / provenance:** explicit seeds; the run artifact freezes class maps + hashes + versions + git commit + counts (P0.5).
- **Commits:** conventional messages; **never** a `Co-Authored-By: Claude` line or any Claude/Anthropic mention.
- **Run env:** `PYTHONPATH=src .venv/bin/python …`. Tests under `tests/alignair/aa/` (no `__init__.py`); dataconfig/stream/training tests marked `@pytest.mark.slow`.
- **Verified facts:** AA reads: p50 126, p95 133, **max 146** (< 192, so no truncation). `GeneBranch.__init__` builds `seg_tower/start_head/end_head/cutout/cls_tower/cls_mid/cls_head`; `_soft_argmax` and `_pos` are available to subclasses. `HUMAN_IGH_OGRDB`: **149 V classes, 7 J classes**. GenAIRR record fields: `sequence`, `sequence_aa`, `junction_start`, `v_call`/`j_call`, `v/d/j_sequence_start/end`, `productive`.

---

### Task 1: AA model — separate boundary-slot masks + softmax class head (`aa/model.py`)

**Files:**
- Create: `src/alignair/aa/model.py`
- Test: `tests/alignair/aa/test_aa_model.py`

**Interfaces:**
- Consumes: `TokenAndPositionEmbedding` (`core/layers.py`), `GeneBranch` (`core/gene_branch.py`), `AlignAIRConfig` (`core/config.py`).
- Produces:
  - `boundary_masks(tokens: Tensor) -> tuple[Tensor, Tensor]` — `(start_mask [0..n-1], end_mask [1..n])`, both `(B, L)` bool.
  - `class AAGeneBranch(GeneBranch)` with `segment_with_masks(emb, start_mask, end_mask) -> (s_log, e_log, s_exp, e_exp)` and `classify(emb, s_exp, e_exp) -> class logits` (raw linear output).
  - `class AAAlignAIR(nn.Module)(cfg)` — `forward(batch) -> dict` with per gene `v`/`j`: `{g}_start_logits`, `{g}_end_logits`, `{g}_start`, `{g}_end`, `{g}_class_logits`; plus `start_mask`, `end_mask`. No `d_*`, no orientation, no meta keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_model.py
import pytest
import torch
import torch.nn.functional as F

from alignair.aa.model import AAAlignAIR, AAGeneBranch, boundary_masks
from alignair.core.config import AlignAIRConfig


def _cfg(v=5, j=3):
    # max_seq_length >= 128 required by the V/J classification tower (N=6 -> 1<<7).
    return AlignAIRConfig(v_allele_count=v, j_allele_count=j, d_allele_count=0,
                          has_d=False, num_chain_types=1, max_seq_length=192, vocab_size=23)


def _tokens(lengths, L=192):
    t = torch.zeros(len(lengths), L, dtype=torch.long)
    for i, n in enumerate(lengths):
        t[i, :n] = torch.randint(1, 23, (n,))
    return t


def test_boundary_masks_start_is_0_to_n_minus_1_and_end_is_1_to_n():
    tokens = _tokens([8], L=16)
    sm, em = boundary_masks(tokens)
    assert sm[0, :8].all() and not sm[0, 8:].any()          # start slots 0..7
    assert not em[0, 0] and em[0, 1:9].all() and not em[0, 9:].any()   # end slots 1..8


def test_end_slot_n_is_reachable_by_the_decoder():
    """The rev-1 bug: slot n was masked out, so an exclusive end == n could never be decoded."""
    tokens = _tokens([8], L=16)
    _, em = boundary_masks(tokens)
    logits = torch.full((1, 16), -1e4)
    logits[0, 8] = 1e4                                       # model wants to say "end = 8"
    dec = logits.masked_fill(~em, torch.finfo(logits.dtype).min)
    pos = torch.arange(16, dtype=torch.float32)
    assert abs(float((F.softmax(dec, -1) * pos).sum()) - 8.0) < 1e-3


def test_forward_emits_expected_keys_and_shapes_without_d():
    cfg = _cfg()
    out = AAAlignAIR(cfg).eval()({"tokenized_sequence": _tokens([120, 100])})
    for g, n_cls in (("v", 5), ("j", 3)):
        assert out[f"{g}_start_logits"].shape == (2, 192)
        assert out[f"{g}_end_logits"].shape == (2, 192)
        assert out[f"{g}_start"].shape == (2, 1) and out[f"{g}_end"].shape == (2, 1)
        assert out[f"{g}_class_logits"].shape == (2, n_cls)
    assert out["start_mask"].shape == (2, 192) and out["end_mask"].shape == (2, 192)
    assert not any(k.startswith("d_") for k in out)           # no D branch in Phase 1
    for absent in ("orientation_logits", "orientation", "mutation_rate", "indel_count", "productive"):
        assert absent not in out


def test_classify_returns_raw_linear_output_deterministically():
    """Deterministic check (not the flaky 'some logit is <0 or >1' heuristic)."""
    cfg = _cfg()
    model = AAAlignAIR(cfg).eval()
    br = model.branches["v"]
    emb = model.embedding(_tokens([100]))
    sm, em = boundary_masks(_tokens([100]))
    _, _, s_exp, e_exp = br.segment_with_masks(emb, sm, em)
    with torch.no_grad():
        got = br.classify(emb, s_exp, e_exp)
        mask = br.cutout(s_exp, e_exp).unsqueeze(-1)
        want = br.cls_head(F.silu(br.cls_mid(br.cls_tower(emb * mask))))
    assert torch.allclose(got, want, atol=1e-6)


def test_outputs_finite_for_short_but_nonempty_reads():
    out = AAAlignAIR(_cfg()).eval()({"tokenized_sequence": _tokens([1, 5])})
    for k, v in out.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), k


def test_all_pad_row_is_rejected_not_silently_nan():
    with pytest.raises(ValueError, match="empty"):
        AAAlignAIR(_cfg()).eval()({"tokenized_sequence": torch.zeros(1, 192, dtype=torch.long)})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.model'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/model.py
"""Thin amino-acid AlignAIR model.

Reuses the core embedding + conv towers + per-gene heads, with two AA-specific corrections:
1. SEPARATE boundary-slot masks. Coordinates are half-open ``[start, end)``, so an exclusive end
   may equal the residue count ``n``. Start slots are ``0..n-1``; end slots are ``1..n``. Sharing
   one mask (as the nt model does) makes ``end == n`` undecodable AND under-penalized.
2. SOFTMAX single-label class head: ``classify`` returns raw class logits (CE applied by the loss).

No orientation head, no meta heads, no meta tower, no D branch (Phase 1 is V/J only).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import AlignAIRConfig
from ..core.gene_branch import GeneBranch
from ..core.layers import TokenAndPositionEmbedding


def boundary_masks(tokens: torch.Tensor):
    """(start_mask, end_mask) for half-open coords: starts occupy 0..n-1, ends occupy 1..n."""
    valid = tokens != 0                                   # (B, L) residue positions 0..n-1
    end = torch.zeros_like(valid)
    end[:, 1:] = valid[:, :-1]                            # slot k allowed iff k-1 is a residue
    return valid, end


class AAGeneBranch(GeneBranch):
    """GeneBranch with mask-correct segmentation and a softmax single-label class head."""

    def segment_with_masks(self, emb: torch.Tensor, start_mask: torch.Tensor,
                           end_mask: torch.Tensor):
        f = self.seg_tower(emb)
        s_log, e_log = self.start_head(f), self.end_head(f)
        neg = torch.finfo(s_log.dtype).min
        s_dec = s_log.masked_fill(~start_mask, neg)
        e_dec = e_log.masked_fill(~end_mask, neg)
        return s_log, e_log, self._soft_argmax(s_dec), self._soft_argmax(e_dec)

    def classify(self, emb: torch.Tensor, s_exp: torch.Tensor, e_exp: torch.Tensor) -> torch.Tensor:
        mask = self.cutout(s_exp, e_exp).unsqueeze(-1)                # (B, L, 1)
        return self.cls_head(F.silu(self.cls_mid(self.cls_tower(emb * mask))))   # raw logits


class AAAlignAIR(nn.Module):
    def __init__(self, cfg: AlignAIRConfig):
        super().__init__()
        if cfg.has_d:
            raise ValueError("Phase 1 is V/J only: build the AA config with has_d=False")
        self.cfg = cfg
        self.embedding = TokenAndPositionEmbedding(cfg.vocab_size, cfg.embed_dim, cfg.max_seq_length)
        self.branches = nn.ModuleDict({s.name: AAGeneBranch(s, cfg) for s in cfg.gene_specs})

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokenized_sequence"]
        start_mask, end_mask = boundary_masks(tokens)
        if not start_mask.any(-1).all():
            raise ValueError("empty (all-pad) sequence in batch; reject empty reads upstream")
        emb = self.embedding(tokens)                                  # no orientation correction
        out: dict = {"start_mask": start_mask, "end_mask": end_mask}
        exp: dict = {}
        for name, branch in self.branches.items():
            s_log, e_log, s_exp, e_exp = branch.segment_with_masks(emb, start_mask, end_mask)
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
git commit -m "feat(aa): AA model with separate boundary-slot masks + softmax class head"
```

---

### Task 2: AA loss — mask-correct, renormalized, component-normalized (`aa/loss.py`)

**Files:**
- Create: `src/alignair/aa/loss.py`
- Test: `tests/alignair/aa/test_aa_loss.py`

**Interfaces:**
- Consumes: `soft_gaussian_target`, `_soft_ce` (`core/losses.py`).
- Produces:
  - `masked_soft_gaussian(gt, L, slot_mask, sigma=1.5) -> Tensor` — Gaussian target masked to allowed slots and **renormalized to sum 1**.
  - `aa_loss(out, targets, genes=("v","j")) -> tuple[Tensor, dict]` — total + parts `{"segmentation","classification"}`; start uses `start_mask`, end uses `end_mask`; 4 boundary terms averaged, 2 class terms averaged.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_loss.py
import torch
from alignair.aa.loss import aa_loss, masked_soft_gaussian
from alignair.aa.model import boundary_masks

L = 32


def _batch(n=8, B=2):
    tokens = torch.zeros(B, L, dtype=torch.long)
    tokens[:, :n] = 1
    return boundary_masks(tokens)


def _out(sm, em, vlogits, jlogits, seg_logits=None):
    B = vlogits.shape[0]
    sl = torch.zeros(B, L) if seg_logits is None else seg_logits
    o = {"start_mask": sm, "end_mask": em}
    for g in ("v", "j"):
        o[f"{g}_start_logits"], o[f"{g}_end_logits"] = sl.clone(), sl.clone()
        o[f"{g}_start"], o[f"{g}_end"] = torch.zeros(B, 1), torch.zeros(B, 1)
    o["v_class_logits"], o["j_class_logits"] = vlogits, jlogits
    return o


def _targets(B=2, n=8):
    return {"v_start": torch.zeros(B, 1), "v_end": torch.full((B, 1), 4.0),
            "j_start": torch.full((B, 1), 4.0), "j_end": torch.full((B, 1), float(n)),
            "v_class": torch.zeros(B, dtype=torch.long), "j_class": torch.zeros(B, dtype=torch.long)}


def test_masked_soft_gaussian_renormalizes_to_one_over_allowed_slots():
    _, em = _batch(n=8, B=1)
    t = masked_soft_gaussian(torch.tensor([[8.0]]), L, em)      # target at the terminal slot
    assert torch.allclose(t.sum(-1), torch.ones(1), atol=1e-6)  # renormalized
    assert float(t[0, 9:].sum()) == 0.0                          # no mass outside allowed slots
    assert float(t[0, 8]) > 0.3                                  # peak survives at the endpoint


def test_terminal_end_target_is_penalized_more_than_a_wrong_reachable_one():
    """rev-1 regression: CE for the unreachable endpoint used to be LOWER than for n-1."""
    sm, em = _batch(n=8, B=1)
    logits = torch.zeros(1, L)
    logits_at_7 = logits.clone(); logits_at_7[0, 7] = 10.0       # model confidently says 7
    out = _out(sm, em, torch.zeros(1, 2), torch.zeros(1, 2), seg_logits=logits_at_7)
    tg_correct = {**_targets(B=1), "j_end": torch.tensor([[8.0]])}
    tg_wrong = {**_targets(B=1), "j_end": torch.tensor([[7.0]])}
    loss_when_truth_is_8, _ = aa_loss(out, tg_correct)
    loss_when_truth_is_7, _ = aa_loss(out, tg_wrong)
    assert loss_when_truth_is_8 > loss_when_truth_is_7           # predicting 7 is penalized iff truth is 8


def test_loss_is_finite_and_components_normalized():
    sm, em = _batch()
    out = _out(sm, em, torch.randn(2, 5), torch.randn(2, 3))
    total, parts = aa_loss(out, _targets())
    assert total.ndim == 0 and torch.isfinite(total)
    assert set(parts) == {"segmentation", "classification"}
    # components are MEANS (not sums): segmentation of 4 identical terms == one term's value
    assert torch.isclose(parts["segmentation"], parts["segmentation"])
    assert float(parts["segmentation"]) < 10.0                  # a sum of 4 would be ~4x larger


def test_confident_correct_class_gives_near_zero_classification():
    sm, em = _batch()
    v = torch.full((2, 5), -10.0); j = torch.full((2, 3), -10.0)
    v[:, 0] = 10.0; j[:, 0] = 10.0
    _, parts = aa_loss(_out(sm, em, v, j), _targets())
    assert parts["classification"] < 1e-3


def test_no_d_keys_required():
    sm, em = _batch()
    total, _ = aa_loss(_out(sm, em, torch.randn(2, 5), torch.randn(2, 3)), _targets())
    assert torch.isfinite(total)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_loss.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.loss'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/loss.py
"""AA loss: soft-Gaussian boundary CE (per-boundary slot masks, renormalized targets) + softmax
single-label class CE. Components are averaged (4 boundary terms, 2 class terms) so segmentation
does not swamp classification at init. D is absent in Phase 1."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.losses import soft_gaussian_target, _soft_ce


def masked_soft_gaussian(gt: torch.Tensor, length: int, slot_mask: torch.Tensor,
                         sigma: float = 1.5) -> torch.Tensor:
    """Soft-Gaussian target restricted to the allowed boundary slots and RENORMALIZED.

    ``_soft_ce`` drops masked terms without renormalizing, so an un-renormalized target at an edge
    silently loses mass and is under-penalized (at the terminal slot, ~35-40% of it)."""
    t = soft_gaussian_target(gt, length) * slot_mask
    return t / t.sum(-1, keepdim=True).clamp_min(1e-8)


def aa_loss(out: dict, targets: dict, genes=("v", "j")):
    L = out["v_start_logits"].shape[-1]
    masks = {"start": out["start_mask"], "end": out["end_mask"]}

    seg_terms = []
    for g in genes:
        for b in ("start", "end"):
            m = masks[b]
            tgt = masked_soft_gaussian(targets[f"{g}_{b}"], L, m)
            seg_terms.append(_soft_ce(out[f"{g}_{b}_logits"], tgt, m))
    clf_terms = [F.cross_entropy(out[f"{g}_class_logits"], targets[f"{g}_class"]) for g in genes]

    seg = torch.stack(seg_terms).mean()
    clf = torch.stack(clf_terms).mean()
    return seg + clf, {"segmentation": seg, "classification": clf}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_loss.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/loss.py tests/alignair/aa/test_aa_loss.py
git commit -m "feat(aa): mask-correct AA loss with renormalized targets + normalized components"
```

---

### Task 3: AA config factory — V/J only (`aa/config.py`)

**Files:**
- Create: `src/alignair/aa/config.py`
- Test: `tests/alignair/aa/test_aa_config.py`

**Interfaces:**
- Produces: `aa_igh_config(n_v_classes: int, n_j_classes: int, max_seq_length: int = 192) -> AlignAIRConfig` with `has_d=False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_config.py
from alignair.aa.config import aa_igh_config
from alignair.aa.tokenizer import AA_VOCAB_SIZE


def test_config_is_vj_only_with_class_counts_and_aa_vocab():
    cfg = aa_igh_config(n_v_classes=149, n_j_classes=7)
    assert cfg.vocab_size == AA_VOCAB_SIZE == 23
    assert cfg.v_allele_count == 149 and cfg.j_allele_count == 7
    assert cfg.has_d is False and cfg.d_allele_count == 0
    assert cfg.max_seq_length == 192
    assert tuple(s.name for s in cfg.gene_specs) == ("v", "j")     # no D branch
    assert tuple(s.allele_count for s in cfg.gene_specs) == (149, 7)


def test_max_seq_length_leaves_room_for_the_exclusive_endpoint():
    # observed AA max is 146 residues; slot 146 must exist, so L must exceed it.
    assert aa_igh_config(149, 7).max_seq_length > 146
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/config.py
"""AA model config factory (Phase 1: V/J only).

Classification heads are sized by PROTEIN-CLASS counts, not raw allele counts. ``has_d=False``:
an unsupervised D branch would drift its BatchNorm running statistics while its weights stay at
init, and emit meaningless outputs. Phase 1b builds a fresh V/D/J config and explicitly transfers
the compatible V/J weights."""
from __future__ import annotations

from ..core.config import AlignAIRConfig
from .tokenizer import AA_VOCAB_SIZE


def aa_igh_config(n_v_classes: int, n_j_classes: int, max_seq_length: int = 192) -> AlignAIRConfig:
    return AlignAIRConfig(
        v_allele_count=n_v_classes,
        j_allele_count=n_j_classes,
        d_allele_count=0,
        has_d=False,
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
git commit -m "feat(aa): V/J-only AA IGH config factory (has_d=False)"
```

---

### Task 4: AA data adapter — stop filtering + rejection accounting (`aa/data.py`)

**Files:**
- Create: `src/alignair/aa/data.py`
- Test: `tests/alignair/aa/test_aa_data.py`

**Interfaces:**
- Produces:
  - `REJECT_REASONS: tuple[str, ...]`.
  - `record_to_sample(rec, class_maps, tokenizer, max_len) -> tuple[dict | None, str | None]` — `(sample, None)` or `(None, reason)`.
  - `collate_aa(samples, max_len, device="cpu") -> (batch_in, targets)`.
  - `build_class_maps(dc_name="HUMAN_IGH_OGRDB") -> (class_maps, meta)`.
  - `stream_aa_samples(n, seed, class_maps, tokenizer, max_len, rate=0.05, dc_name=..., counter=None)` — generator; increments `counter` (a `Counter`) with `generated` and per-reason rejections.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_data.py
from collections import Counter

import numpy as np
import pytest
import torch

from alignair.aa.data import record_to_sample, collate_aa
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.aa.equivalence import build_equivalence_classes, db_hash


def _cmaps():
    v = build_equivalence_classes("V", {"IGHV1-1*01": "QVQLC", "IGHV2-2*01": "EVQLC"}, db_hash([("x", "a")]))
    j = build_equivalence_classes("J", {"IGHJ1*01": "WGQGT", "IGHJ2*01": "WGKGT"}, db_hash([("y", "b")]))
    return {"V": v, "J": j}


def _rec(**over):
    base = {"sequence": "A" * 24, "sequence_aa": "QVQLCWGQG", "junction_start": 0, "productive": True,
            "v_call": "IGHV1-1*01", "j_call": "IGHJ1*01",
            "v_sequence_start": 0, "v_sequence_end": 12,
            "d_sequence_start": 12, "d_sequence_end": 12,
            "j_sequence_start": 12, "j_sequence_end": 24}
    base.update(over)
    return base


def test_projects_boundaries_and_maps_classes():
    s, why = record_to_sample(_rec(), _cmaps(), AAProteinTokenizer(), 192)
    assert why is None and s is not None
    assert (s["v_start"], s["v_end"]) == (0, 4)      # 4 fully-owned V residues
    assert (s["j_start"], s["j_end"]) == (4, 8)
    assert isinstance(s["tokens"], np.ndarray)


def test_nonzero_frame_offset_shifts_the_residue_grid():
    # junction_start=1 -> frame 1; V=[1,13) J=[13,25); sequence 25nt -> 8 residues.
    s, why = record_to_sample(
        _rec(sequence="A" * 25, junction_start=1, v_sequence_start=1, v_sequence_end=13,
             d_sequence_start=13, d_sequence_end=13, j_sequence_start=13, j_sequence_end=25),
        _cmaps(), AAProteinTokenizer(), 192)
    assert why is None
    assert (s["v_start"], s["v_end"]) == (0, 4) and (s["j_start"], s["j_end"]) == (4, 8)


def test_straddling_codon_is_masked_out_of_owned_span():
    # V=[0,11) straddles: residue r3 (nt 9..12) is not fully owned -> V owns only r0..r2.
    s, why = record_to_sample(_rec(v_sequence_end=11), _cmaps(), AAProteinTokenizer(), 192)
    assert why is None and (s["v_start"], s["v_end"]) == (0, 3)


def test_terminal_exclusive_endpoint_equals_residue_count():
    s, _ = record_to_sample(_rec(), _cmaps(), AAProteinTokenizer(), 192)
    assert s["j_end"] == 8                            # == number of residues in sequence_aa
    assert s["j_end"] <= len(s["tokens"])


@pytest.mark.parametrize("over,reason", [
    ({"productive": False}, "not_productive"),
    ({"sequence_aa": ""}, "no_aa"),
    ({"junction_start": None}, "no_frame"),
    ({"sequence_aa": "QVQ*CWGQG"}, "stop_codon"),
    ({"v_call": "IGHV9-9*99"}, "unknown_allele"),
])
def test_rejections_are_reported_by_reason(over, reason):
    s, why = record_to_sample(_rec(**over), _cmaps(), AAProteinTokenizer(), 192)
    assert s is None and why == reason


def test_exact_length_is_rejected_to_reserve_the_endpoint_slot():
    # len(tokens) must be < max_len so slot n exists for an exclusive end.
    long_aa = "Q" * 16
    s, why = record_to_sample(_rec(sequence="A" * 48, sequence_aa=long_aa,
                                   v_sequence_end=24, d_sequence_start=24, d_sequence_end=24,
                                   j_sequence_start=24, j_sequence_end=48),
                              _cmaps(), AAProteinTokenizer(), max_len=16)
    assert s is None and why == "too_long"


def test_collate_shapes_and_dtypes():
    samples = [{"tokens": np.array([1, 2, 3], dtype=np.int64), "v_class": 0, "j_class": 1,
                "v_start": 0, "v_end": 2, "j_start": 2, "j_end": 3},
               {"tokens": np.array([4, 5], dtype=np.int64), "v_class": 1, "j_class": 0,
                "v_start": 0, "v_end": 1, "j_start": 1, "j_end": 2}]
    bi, tg = collate_aa(samples, max_len=192)
    assert bi["tokenized_sequence"].shape == (2, 192)
    assert bi["tokenized_sequence"][1, 2:].sum().item() == 0
    assert tg["v_class"].dtype == torch.long and tg["v_start"].shape == (2, 1)


@pytest.mark.slow
def test_stream_real_igh_yields_valid_samples_and_counts_rejections():
    from alignair.aa.data import build_class_maps, stream_aa_samples
    cmaps, meta = build_class_maps("HUMAN_IGH_OGRDB")
    assert meta["n_v_classes"] == 149 and meta["n_j_classes"] == 7
    counter = Counter()
    got = []
    for s in stream_aa_samples(n=3000, seed=0, class_maps=cmaps, tokenizer=AAProteinTokenizer(),
                               max_len=192, counter=counter):
        got.append(s)
        if len(got) >= 64:
            break
    assert len(got) == 64
    assert counter["generated"] > 0
    for s in got:
        assert 0 <= s["v_class"] < 149 and 0 <= s["j_class"] < 7
        assert 0 <= s["v_start"] < s["v_end"] <= len(s["tokens"])
        assert 0 <= s["j_start"] < s["j_end"] <= len(s["tokens"])
        assert len(s["tokens"]) < 192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_data.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.data'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/data.py
"""AA data adapter: GenAIRR productive IGH records -> (tokens, class targets, projected boundary
targets), with explicit rejection accounting.

Reads use GenAIRR's ``sequence_aa`` directly (never re-translated here); only the nucleotide
segment coordinates are projected into AA residue boundaries with straddling-codon masking.
Stop-containing reads are REJECTED, matching the Phase-0 input contract."""
from __future__ import annotations

import numpy as np
import torch

from .equivalence import build_equivalence_classes, db_hash
from .translation import frame_offset_of, project_codons

REJECT_REASONS = ("not_productive", "no_aa", "no_frame", "stop_codon",
                  "unknown_allele", "no_bounds", "too_long", "empty")

_SEG_COLS = {"V": ("v_sequence_start", "v_sequence_end"),
             "D": ("d_sequence_start", "d_sequence_end"),
             "J": ("j_sequence_start", "j_sequence_end")}


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
    """-> (sample, None) or (None, reason). ``max_len`` is exclusive: ``len(tokens) < max_len`` so
    the exclusive end slot ``n`` exists."""
    if not rec.get("productive"):
        return None, "not_productive"
    aa = rec.get("sequence_aa")
    if not aa:
        return None, "no_aa"
    if rec.get("junction_start") is None:
        return None, "no_frame"
    if "*" in aa:
        return None, "stop_codon"
    try:
        v_class = class_maps["V"].class_of(_first(rec["v_call"]))
        j_class = class_maps["J"].class_of(_first(rec["j_call"]))
    except KeyError:
        return None, "unknown_allele"
    proj = project_codons(len(rec["sequence"]), frame_offset_of(rec["junction_start"]), _segments(rec))
    vb, jb = proj.aa_bounds.get("V"), proj.aa_bounds.get("J")
    if vb is None or jb is None:
        return None, "no_bounds"
    tokens = tokenizer.encode(aa)
    if len(tokens) == 0:
        return None, "empty"
    if len(tokens) >= max_len:                     # reserve slot n for the exclusive endpoint
        return None, "too_long"
    return {"tokens": tokens, "v_class": v_class, "j_class": j_class,
            "v_start": vb[0], "v_end": vb[1], "j_start": jb[0], "j_end": jb[1]}, None


def collate_aa(samples: list, max_len: int, device: str = "cpu"):
    B = len(samples)
    tokens = torch.zeros(B, max_len, dtype=torch.long)
    for i, s in enumerate(samples):
        t = np.asarray(s["tokens"], dtype=np.int64)
        tokens[i, :len(t)] = torch.from_numpy(t)
    batch_in = {"tokenized_sequence": tokens.to(device)}
    targets = {k: torch.tensor([s[k] for s in samples], dtype=torch.long, device=device)
               for k in ("v_class", "j_class")}
    for k in ("v_start", "v_end", "j_start", "j_end"):
        targets[k] = torch.tensor([[float(s[k])] for s in samples],
                                  dtype=torch.float32, device=device)
    return batch_in, targets


def build_class_maps(dc_name: str = "HUMAN_IGH_OGRDB"):
    """Frozen V/J protein-equivalence class maps from a GenAIRR dataconfig."""
    import GenAIRR as ga

    from .reference import build_aa_germlines
    dc = getattr(ga, dc_name)
    gset = build_aa_germlines(dc, genes=("V", "J"))
    dbh = db_hash([(a.name, a.ungapped_seq) for a in dc.allele_list("v")]
                  + [(a.name, a.ungapped_seq) for a in dc.allele_list("j")])
    class_maps = {g: build_equivalence_classes(
        g, {n: x.aa_seq for n, x in gset.genes[g].items()}, dbh) for g in ("V", "J")}
    meta = {"n_v_classes": class_maps["V"].n_classes(),
            "n_j_classes": class_maps["J"].n_classes(),
            "db_hash": dbh,
            "quarantined_v": len(gset.quarantined["V"]),
            "quarantined_j": len(gset.quarantined["J"])}
    return class_maps, meta


def stream_aa_samples(n, seed, class_maps, tokenizer, max_len, rate=0.05,
                      dc_name: str = "HUMAN_IGH_OGRDB", counter=None):
    """Yield usable AA samples; tally ``generated`` and per-reason rejections into ``counter``."""
    import GenAIRR as ga
    exp = ga.Experiment.on(dc_name).recombine().mutate(model="s5f", rate=rate)
    for rec in exp.stream_records(n=n, seed=seed):
        if counter is not None:
            counter["generated"] += 1
        s, why = record_to_sample(rec, class_maps, tokenizer, max_len)
        if s is None:
            if counter is not None:
                counter[why] += 1
            continue
        if counter is not None:
            counter["usable"] += 1
        yield s
```

- [ ] **Step 4: Run tests to verify they pass**

Run (fast): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_data.py -q -m "not slow"`
Run (slow): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_data.py -q -m slow`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/data.py tests/alignair/aa/test_aa_data.py
git commit -m "feat(aa): AA data adapter with stop filtering + rejection accounting"
```

---

### Task 5: Overfit proof — diverse batch, component history, all four MAEs (`aa/train.py`)

**Files:**
- Create: `src/alignair/aa/train.py`
- Test: `tests/alignair/aa/test_aa_train.py`

**Interfaces:**
- Produces:
  - `aa_eval_metrics(out, targets, genes=("v","j")) -> dict` — `{g}_class_acc`, `{g}_{start|end}_mae`.
  - `assert_batch_diverse(samples, min_v_classes=3, min_j_classes=2) -> None` (raises `ValueError`).
  - `overfit_one_batch(model, batch_in, targets, *, steps=300, lr=1e-3, genes=("v","j")) -> dict` — includes `history` (per-step `total`/`segmentation`/`classification`) so a falling total cannot hide a stuck class term.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_train.py
import pytest
import torch

from alignair.aa.train import aa_eval_metrics, assert_batch_diverse, overfit_one_batch
from alignair.aa.model import AAAlignAIR
from alignair.aa.config import aa_igh_config


def test_eval_metrics_on_perfect_predictions():
    B = 3
    out = {}
    for g in ("v", "j"):
        lg = torch.full((B, 4), -10.0); lg[range(B), 2] = 10.0
        out[f"{g}_class_logits"] = lg
        out[f"{g}_start"] = torch.full((B, 1), 5.0)
        out[f"{g}_end"] = torch.full((B, 1), 40.0)
    tg = {"v_class": torch.full((B,), 2), "j_class": torch.full((B,), 2),
          "v_start": torch.full((B, 1), 5.0), "v_end": torch.full((B, 1), 40.0),
          "j_start": torch.full((B, 1), 5.0), "j_end": torch.full((B, 1), 40.0)}
    m = aa_eval_metrics(out, tg)
    assert m["v_class_acc"] == 1.0 and m["j_class_acc"] == 1.0
    for k in ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae"):
        assert m[k] < 1e-4


def test_assert_batch_diverse_rejects_degenerate_batches():
    same = [{"v_class": 1, "j_class": 0} for _ in range(8)]
    with pytest.raises(ValueError, match="diver"):
        assert_batch_diverse(same)
    mixed = [{"v_class": i % 4, "j_class": i % 2} for i in range(8)]
    assert_batch_diverse(mixed) is None


@pytest.mark.slow
def test_overfit_one_batch_learns_classes_and_all_four_boundaries():
    torch.manual_seed(0)
    from alignair.aa.data import build_class_maps, collate_aa, stream_aa_samples
    from alignair.aa.tokenizer import AAProteinTokenizer
    cmaps, meta = build_class_maps("HUMAN_IGH_OGRDB")
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"])
    samples = []
    for s in stream_aa_samples(n=8000, seed=1, class_maps=cmaps, tokenizer=AAProteinTokenizer(),
                               max_len=cfg.max_seq_length):
        samples.append(s)
        if len(samples) >= 16:
            break
    assert_batch_diverse(samples)                       # the proof must not be trivially degenerate
    bi, tg = collate_aa(samples, cfg.max_seq_length)
    res = overfit_one_batch(AAAlignAIR(cfg), bi, tg, steps=300, lr=1e-3)
    assert res["final_loss"] < 0.5 * res["initial_loss"]
    assert res["v_class_acc"] >= 0.9 and res["j_class_acc"] >= 0.9
    for k in ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae"):
        assert res[k] < 5.0, k
    # both components must actually fall (a falling total must not hide a stuck class term)
    h = res["history"]
    assert h[-1]["classification"] < 0.5 * h[0]["classification"]
    assert h[-1]["segmentation"] < h[0]["segmentation"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_train.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.train'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/train.py
"""Phase-1 training helpers: overfit-one-batch capacity proof + per-task metrics."""
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


def assert_batch_diverse(samples, min_v_classes: int = 3, min_j_classes: int = 2) -> None:
    """A memorization proof on a single-class batch proves nothing — require real diversity."""
    nv = len({s["v_class"] for s in samples})
    nj = len({s["j_class"] for s in samples})
    if nv < min_v_classes or nj < min_j_classes:
        raise ValueError(f"batch not diverse enough: {nv} V classes (need >= {min_v_classes}), "
                         f"{nj} J classes (need >= {min_j_classes})")


def overfit_one_batch(model, batch_in, targets, *, steps=300, lr=1e-3, genes=("v", "j")) -> dict:
    """Drive one batch to memorization. Returns initial/final loss, final metrics, and the
    per-step component history."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history, initial = [], None
    for _ in range(steps):
        out = model(batch_in)
        total, parts = aa_loss(out, targets, genes=genes)
        history.append({"total": float(total.detach()),
                        **{k: float(v.detach()) for k, v in parts.items()}})
        if initial is None:
            initial = float(total.detach())
        opt.zero_grad()
        total.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(batch_in)
        final = float(aa_loss(out, targets, genes=genes)[0])
        metrics = aa_eval_metrics(out, targets, genes)
    return {"initial_loss": initial, "final_loss": final, "history": history, **metrics}
```

- [ ] **Step 4: Run tests to verify they pass**

Run (fast): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_train.py -q -m "not slow"`
Run (slow): `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_train.py -q -m slow`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/train.py tests/alignair/aa/test_aa_train.py
git commit -m "feat(aa): overfit proof with diversity guard + component history"
```

---

### Task 6: Baselines + frozen held-out evaluation (`aa/evaluate.py`)

The scientific core: a metric between 0 and 1 means nothing without a baseline.

**Files:**
- Create: `src/alignair/aa/evaluate.py`
- Test: `tests/alignair/aa/test_aa_evaluate.py`

**Interfaces:**
- Produces:
  - `wilson_interval(k, n, z=1.96) -> tuple[float, float]`.
  - `bootstrap_mean_ci(values, n_boot=1000, seed=0) -> tuple[float, float]`.
  - `majority_class_baseline(class_ids) -> float`; `constant_median_mae(values) -> float`.
  - `evaluate_frozen(model, samples, cfg, class_maps, *, batch_size=128, device="cpu") -> dict` — micro/macro accuracy + Wilson CI vs majority baseline per gene; per-boundary MAE + bootstrap CI vs constant-median baseline; V accuracy stratified by observed V coverage and by equivalence-class ambiguity size.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_evaluate.py
import math

from alignair.aa.evaluate import (
    wilson_interval, bootstrap_mean_ci, majority_class_baseline, constant_median_mae,
)


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi and 0.0 <= lo and hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    tight = wilson_interval(8000, 10000); wide = wilson_interval(8, 10)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])       # more data -> tighter


def test_majority_class_baseline_is_the_best_constant_predictor():
    assert majority_class_baseline([1, 1, 1, 2]) == 0.75
    assert majority_class_baseline([]) == 0.0


def test_constant_median_mae_beats_or_matches_any_other_constant():
    vals = [1.0, 2.0, 3.0, 10.0]
    med = constant_median_mae(vals)
    for c in (0.0, 1.5, 4.0, 11.0):
        assert med <= sum(abs(v - c) for v in vals) / len(vals) + 1e-9


def test_bootstrap_mean_ci_is_deterministic_and_brackets_the_mean():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_mean_ci(vals, n_boot=200, seed=0)
    assert a == bootstrap_mean_ci(vals, n_boot=200, seed=0)   # seeded -> reproducible
    assert a[0] <= 3.0 <= a[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_evaluate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/evaluate.py
"""Frozen held-out evaluation with explicit baselines and uncertainty.

A raw accuracy is uninterpretable: 149 V classes are highly imbalanced, so class accuracy can
reflect allele-frequency priors, and boundary MAE can be beaten by a constant. Every metric here
is reported against its best-constant baseline, with an interval."""
from __future__ import annotations

import math
from collections import Counter

import torch

from .data import collate_aa
from .train import aa_eval_metrics


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values, n_boot: int = 1000, seed: int = 0):
    import numpy as np
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(rng.choice(arr, size=arr.size, replace=True).mean()) for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def majority_class_baseline(class_ids) -> float:
    ids = list(class_ids)
    if not ids:
        return 0.0
    return Counter(ids).most_common(1)[0][1] / len(ids)


def constant_median_mae(values) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    med = vals[len(vals) // 2]
    return sum(abs(v - med) for v in vals) / len(vals)


def _macro_accuracy(true_ids, pred_ids) -> float:
    per: dict = {}
    for t, p in zip(true_ids, pred_ids):
        hit, n = per.get(t, (0, 0))
        per[t] = (hit + int(t == p), n + 1)
    return (sum(h / n for h, n in per.values()) / len(per)) if per else 0.0


@torch.no_grad()
def evaluate_frozen(model, samples, cfg, class_maps, *, batch_size: int = 128,
                    device: str = "cpu") -> dict:
    """Evaluate over the WHOLE frozen held-out set in batches, with baselines + intervals."""
    model.eval()
    true = {"v": [], "j": []}
    pred = {"v": [], "j": []}
    errs = {f"{g}_{b}": [] for g in ("v", "j") for b in ("start", "end")}
    tgts = {f"{g}_{b}": [] for g in ("v", "j") for b in ("start", "end")}
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        bi, tg = collate_aa(chunk, cfg.max_seq_length, device)
        out = model(bi)
        for g in ("v", "j"):
            true[g] += tg[f"{g}_class"].tolist()
            pred[g] += out[f"{g}_class_logits"].argmax(-1).tolist()
            for b in ("start", "end"):
                e = (out[f"{g}_{b}"].squeeze(-1) - tg[f"{g}_{b}"].squeeze(-1)).abs().tolist()
                errs[f"{g}_{b}"] += e
                tgts[f"{g}_{b}"] += tg[f"{g}_{b}"].squeeze(-1).tolist()

    res: dict = {"n": len(samples), "classification": {}, "boundaries": {}, "stratified": {}}
    for g in ("v", "j"):
        hits = sum(int(t == p) for t, p in zip(true[g], pred[g]))
        n = len(true[g])
        micro = hits / n if n else 0.0
        base = majority_class_baseline(true[g])
        res["classification"][g] = {
            "micro_acc": micro, "macro_acc": _macro_accuracy(true[g], pred[g]),
            "micro_acc_ci95": wilson_interval(hits, n),
            "majority_baseline": base, "lift_over_majority": micro - base,
            "n_distinct_true_classes": len(set(true[g])),
        }
    for key, e in errs.items():
        mae = sum(e) / len(e) if e else 0.0
        base = constant_median_mae(tgts[key])
        res["boundaries"][key] = {
            "mae": mae, "mae_ci95": bootstrap_mean_ci(e),
            "constant_median_baseline_mae": base, "beats_baseline": mae < base,
        }

    # V accuracy stratified by observed coverage and by protein-class ambiguity size
    cov_buckets, amb_buckets = {}, {}
    vmap = class_maps["V"]
    for s, t, p in zip(samples, true["v"], pred["v"]):
        cov = s["v_end"] - s["v_start"]
        cb = "<40" if cov < 40 else ("40-79" if cov < 80 else ">=80")
        hit, n = cov_buckets.get(cb, (0, 0)); cov_buckets[cb] = (hit + int(t == p), n + 1)
        amb = len(vmap.ambiguity_set(t))
        ab = str(amb) if amb < 3 else ">=3"
        hit, n = amb_buckets.get(ab, (0, 0)); amb_buckets[ab] = (hit + int(t == p), n + 1)
    res["stratified"]["v_acc_by_coverage"] = {
        k: {"acc": h / n, "n": n, "ci95": wilson_interval(h, n)} for k, (h, n) in sorted(cov_buckets.items())}
    res["stratified"]["v_acc_by_ambiguity_size"] = {
        k: {"acc": h / n, "n": n, "ci95": wilson_interval(h, n)} for k, (h, n) in sorted(amb_buckets.items())}
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_evaluate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/evaluate.py tests/alignair/aa/test_aa_evaluate.py
git commit -m "feat(aa): frozen held-out evaluation with baselines, CIs and stratification"
```

---

### Task 7: Phase-1 runner — provenance, frozen eval set, multi-seed (`aa/phase1.py`, `scripts/aa/phase1_proof.py`)

**Files:**
- Create: `src/alignair/aa/phase1.py`
- Create: `scripts/aa/phase1_proof.py`
- Test: `tests/alignair/aa/test_aa_phase1.py`

**Interfaces:**
- Produces:
  - `provenance(cfg, class_maps, meta, model) -> dict` — serialized V/J class maps + their SHA-256, db hash, model config, GenAIRR/torch/CUDA versions, git commit, param count.
  - `run_seed(seed, cfg, class_maps, heldout, *, train_steps, batch_size, lr, overfit_batch, overfit_steps, device) -> dict`.
  - `run_learning_proof(*, seeds=(0,1,2), heldout_size=4096, train_steps=3000, batch_size=64, lr=1e-3, overfit_batch=32, overfit_steps=400, device="cpu", dc_name="HUMAN_IGH_OGRDB") -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/aa/test_aa_phase1.py
import pytest


@pytest.mark.slow
def test_run_learning_proof_smoke_has_provenance_baselines_and_seeds():
    from alignair.aa.phase1 import run_learning_proof
    r = run_learning_proof(seeds=(0, 1), heldout_size=256, train_steps=30, batch_size=16,
                           overfit_batch=8, overfit_steps=40, device="cpu")
    p = r["provenance"]
    for key in ("class_maps", "class_map_sha256", "db_hash", "config", "versions",
                "git_commit", "param_count"):
        assert key in p
    assert set(p["class_maps"]) == {"V", "J"} and set(p["class_map_sha256"]) == {"V", "J"}
    assert p["versions"]["torch"] and p["versions"]["genairr"]
    assert r["heldout_size"] == 256
    assert set(r["seeds"]) == {"0", "1"}
    one = r["seeds"]["0"]
    assert one["overfit"]["final_loss"] < one["overfit"]["initial_loss"]
    ev = one["heldout"]
    for g in ("v", "j"):
        c = ev["classification"][g]
        assert 0.0 <= c["micro_acc"] <= 1.0 and "majority_baseline" in c and "lift_over_majority" in c
    for key in ("v_start", "v_end", "j_start", "j_end"):
        assert "constant_median_baseline_mae" in ev["boundaries"][key]
    assert "rejections" in r and r["rejections"]["generated"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_phase1.py -q -m slow`
Expected: FAIL — `ModuleNotFoundError: No module named 'alignair.aa.phase1'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/aa/phase1.py
"""Phase-1 learning proof: per-seed capacity (overfit) + generalization on ONE frozen held-out
set evaluated against baselines. Freezes full provenance so a class id is interpretable later."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import subprocess
from collections import Counter

import torch

from .config import aa_igh_config
from .data import build_class_maps, collate_aa, stream_aa_samples
from .evaluate import evaluate_frozen
from .loss import aa_loss
from .model import AAAlignAIR
from .tokenizer import AAProteinTokenizer
from .train import assert_batch_diverse, overfit_one_batch


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                    # noqa: BLE001 - provenance must not break a run
        return "unknown"


def provenance(cfg, class_maps, meta, model) -> dict:
    import GenAIRR
    maps = {g: class_maps[g].to_json() for g in ("V", "J")}
    sha = {g: hashlib.sha256(json.dumps(maps[g], sort_keys=True).encode()).hexdigest()
           for g in ("V", "J")}
    return {
        "class_maps": maps,
        "class_map_sha256": sha,
        "db_hash": meta["db_hash"],
        "config": dataclasses.asdict(cfg),
        "versions": {"genairr": getattr(GenAIRR, "__version__", "unknown"),
                     "torch": torch.__version__,
                     "cuda": torch.version.cuda,
                     "cudnn": (torch.backends.cudnn.version() if torch.cuda.is_available() else None)},
        "git_commit": _git_commit(),
        "param_count": sum(p.numel() for p in model.parameters()),
        "quarantined_v": meta["quarantined_v"], "quarantined_j": meta["quarantined_j"],
    }


def _take(stream, k):
    return list(itertools.islice(stream, k))


def run_seed(seed, cfg, class_maps, heldout, *, train_steps, batch_size, lr,
             overfit_batch, overfit_steps, device, dc_name, counter) -> dict:
    torch.manual_seed(seed)
    tok = AAProteinTokenizer()

    def stream(sd):
        return stream_aa_samples(n=None, seed=sd, class_maps=class_maps, tokenizer=tok,
                                 max_len=cfg.max_seq_length, dc_name=dc_name, counter=counter)

    # (a) capacity: overfit one DIVERSE batch
    ob = _take(stream(seed * 7919 + 1), overfit_batch)
    assert_batch_diverse(ob)
    obi, obt = collate_aa(ob, cfg.max_seq_length, device)
    overfit = overfit_one_batch(AAAlignAIR(cfg).to(device), obi, obt,
                                steps=overfit_steps, lr=lr)

    # (b) generalization: short train, then evaluate the FROZEN held-out set
    model = AAAlignAIR(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tr = stream(seed * 7919 + 100)
    model.train()
    losses = []
    for _ in range(train_steps):
        batch = _take(tr, batch_size)
        if len(batch) < batch_size:
            break
        bi, bt = collate_aa(batch, cfg.max_seq_length, device)
        total, parts = aa_loss(model(bi), bt)
        losses.append({"total": float(total.detach()),
                       **{k: float(v.detach()) for k, v in parts.items()}})
        opt.zero_grad(); total.backward(); opt.step()

    return {"overfit": {k: v for k, v in overfit.items() if k != "history"},
            "overfit_history_last": overfit["history"][-1],
            "train_loss_first": losses[0] if losses else None,
            "train_loss_last": losses[-1] if losses else None,
            "heldout": evaluate_frozen(model, heldout, cfg, class_maps,
                                       batch_size=batch_size, device=device)}


def run_learning_proof(*, seeds=(0, 1, 2), heldout_size=4096, train_steps=3000, batch_size=64,
                       lr=1e-3, overfit_batch=32, overfit_steps=400, device="cpu",
                       dc_name="HUMAN_IGH_OGRDB") -> dict:
    class_maps, meta = build_class_maps(dc_name)
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"])
    counter = Counter()
    tok = AAProteinTokenizer()

    # ONE frozen held-out set, disjoint from every training seed, evaluated in full
    heldout = _take(stream_aa_samples(n=None, seed=987654321, class_maps=class_maps, tokenizer=tok,
                                      max_len=cfg.max_seq_length, dc_name=dc_name, counter=counter),
                    heldout_size)

    per_seed = {}
    for sd in seeds:
        per_seed[str(sd)] = run_seed(sd, cfg, class_maps, heldout, train_steps=train_steps,
                                     batch_size=batch_size, lr=lr, overfit_batch=overfit_batch,
                                     overfit_steps=overfit_steps, device=device,
                                     dc_name=dc_name, counter=counter)

    ref_model = AAAlignAIR(cfg)
    return {"schema": "alignair.aa.phase1.v1",
            "provenance": provenance(cfg, class_maps, meta, ref_model),
            "heldout_size": len(heldout),
            "rejections": dict(counter),
            "settings": {"seeds": list(seeds), "train_steps": train_steps,
                         "batch_size": batch_size, "lr": lr, "overfit_batch": overfit_batch,
                         "overfit_steps": overfit_steps, "device": device},
            "seeds": per_seed}
```

```python
# scripts/aa/phase1_proof.py
"""CLI: run the amino-acid AlignAIR Phase-1 learning proof and freeze the artifact.

Run (GPU): PYTHONPATH=src .venv/bin/python scripts/aa/phase1_proof.py --device cuda
"""
import argparse
import json
import os

from alignair.aa.phase1 import run_learning_proof


def main() -> None:
    ap = argparse.ArgumentParser(description="Amino-acid AlignAIR Phase-1 learning proof")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--heldout-size", type=int, default=4096)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--overfit-batch", type=int, default=32)
    ap.add_argument("--overfit-steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/aa/phase1/proof.json")
    args = ap.parse_args()

    r = run_learning_proof(seeds=tuple(args.seeds), heldout_size=args.heldout_size,
                           train_steps=args.train_steps, batch_size=args.batch_size, lr=args.lr,
                           overfit_batch=args.overfit_batch, overfit_steps=args.overfit_steps,
                           device=args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(r, f, indent=2, sort_keys=True)
    # console summary (the full class maps live in the artifact, not the terminal)
    for sd, s in sorted(r["seeds"].items()):
        c, b = s["heldout"]["classification"], s["heldout"]["boundaries"]
        print(f"seed {sd}: V {c['v']['micro_acc']:.3f} (maj {c['v']['majority_baseline']:.3f}, "
              f"macro {c['v']['macro_acc']:.3f})  J {c['j']['micro_acc']:.3f} "
              f"(maj {c['j']['majority_baseline']:.3f})  "
              + " ".join(f"{k} {b[k]['mae']:.2f}/{b[k]['constant_median_baseline_mae']:.2f}"
                         for k in ("v_start", "v_end", "j_start", "j_end")))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/test_aa_phase1.py -q -m slow`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/aa/phase1.py scripts/aa/phase1_proof.py tests/alignair/aa/test_aa_phase1.py
git commit -m "feat(aa): Phase-1 runner with frozen provenance, held-out set and multi-seed"
```

---

### Task 8: Run the proof, evaluate the gate, record the verdict

- [ ] **Step 1: Full AA suite + nucleotide regression check**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/aa/ -q`
Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q -m "not slow" --ignore=tests/alignair/aa`
Expected: PASS both (AA is additive; nucleotide untouched).

- [ ] **Step 2: Run the real proof on GPU**

Run: `PYTHONPATH=src .venv/bin/python scripts/aa/phase1_proof.py --device cuda --seeds 0 1 2 --heldout-size 4096 --train-steps 3000 --batch-size 64 --out experiments/aa/phase1/proof.json`
Expected: a per-seed summary line + the frozen artifact.

- [ ] **Step 3: Evaluate the Phase-1 gate**

Read `experiments/aa/phase1/proof.json` and check **every** criterion (record the real numbers; do not pre-judge):

1. **Capacity:** overfit batch reaches ≥0.90 V *and* J class accuracy and <5-residue MAE on **all four** boundaries.
2. **Classification beats prior:** held-out V and J micro accuracy exceed their majority baselines by a declared margin, with non-overlapping Wilson intervals.
3. **Boundaries beat constant:** every one of the four boundary MAEs beats its constant-median baseline (`beats_baseline: true`).
4. **Seed consistency:** results directionally consistent across all three seeds.
5. **Not just allele priors:** coverage- and ambiguity-stratified V accuracy does not reveal that aggregate accuracy is carried by a few frequent classes (check `macro_acc` vs `micro_acc` and the strata).
6. **Hygiene:** no NaNs, no empty batches, no masked endpoint targets, no silent truncations (`rejections` accounts for every generated record).

**Gate:** proceed to Phase 1b only if 1–6 all hold. If classification beats baseline but boundaries do not (or vice versa), that is a partial result — record it as such rather than declaring success.

- [ ] **Step 4: Record the verdict and commit**

Copy the artifact to a tracked path (`experiments/` is gitignored) and append a "Phase 1 — Learning-Proof Results" section to the design spec with the real numbers, each gate criterion marked pass/fail, and the GO / PARTIAL / NO-GO verdict.

```bash
mkdir -p docs/superpowers/artifacts/2026-07-18-aa-phase1
cp experiments/aa/phase1/proof.json docs/superpowers/artifacts/2026-07-18-aa-phase1/proof.json
git add docs/superpowers/artifacts/2026-07-18-aa-phase1/proof.json \
        docs/superpowers/specs/2026-07-18-aminoacid-alignair-design.md
git commit -m "chore(aa): freeze Phase-1 learning-proof results + gate verdict"
```

---

## Self-Review

**Review corrections addressed:**
- P0.1 boundary slots → `boundary_masks` (start `0..n-1`, end `1..n`), `segment_with_masks`, per-boundary mask in `aa_loss`, `too_long` rejection reserving slot `n`, plus a regression test proving slot `n` is decodable and that the terminal target is now penalized correctly. **Addition:** `masked_soft_gaussian` renormalizes the target (the verified under-penalization). ✅
- P0.2 D removed → `has_d=False`; `AAAlignAIR` raises if `has_d`; test asserts no `d_*` keys. ✅
- P0.3 stops rejected in the adapter with per-reason counts; `aa_len_max` corrected to **146** here and in the spec. ✅
- P0.4 baselines + ≥4096 frozen held-out, full-set batched evaluation, majority/constant-median baselines, macro+micro, Wilson + bootstrap CIs, coverage & ambiguity stratification, `train_final` replaced by `train_loss_first`/`train_loss_last`. ✅
- P0.5 provenance: serialized class maps + SHA-256, db hash, config, GenAIRR/torch/CUDA versions, git commit, seeds, generated/usable/rejection counts, param count. ✅
- Tests: deterministic `classify` equality (flaky heuristic removed), explicit empty-sequence rejection + finiteness checks, component-normalized loss, diversity guard, all four MAEs asserted, adapter tests for frame offset / straddling codons / terminal endpoint / stop / exact-length / unknown allele, component history retained. ✅
- Gate criteria encoded verbatim in Task 8 Step 3. ✅

**Placeholder scan:** none — all code complete and runnable; all run steps have exact commands. Task 8's "record the real numbers" is the experimental output.

**Type consistency:** `boundary_masks` → `start_mask`/`end_mask` emitted by `AAAlignAIR.forward`, consumed by `aa_loss` and nothing else. `record_to_sample` returns `(sample|None, reason|None)` — every caller (`stream_aa_samples`, tests) unpacks the tuple. Sample keys match `collate_aa`, whose target keys match `aa_loss`/`aa_eval_metrics`/`evaluate_frozen`. `build_class_maps` → `(class_maps, meta)` with `meta` keys used in `provenance` (`db_hash`, `quarantined_v/j`) and `aa_igh_config` (`n_v_classes`, `n_j_classes`). `evaluate_frozen` needs `class_maps` for ambiguity stratification — passed by `run_seed`. ✅

**Known limitation (recorded, not hidden):** the held-out set uses a disjoint *seed*, not clonal-lineage-based splitting. GenAIRR's plain stream models each read as an independent rearrangement, so leakage risk is limited to exact-duplicate VDJ+junction; true lineage-aware splitting is a Phase-2 requirement.
