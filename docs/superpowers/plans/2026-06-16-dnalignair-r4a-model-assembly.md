# DNAlignAIR R4a — Unified Model Assembly — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the validated R1–R3 components into a single `DNAlignAIR(nn.Module)` whose forward produces every output (orientation, per-position region + state, scalar aggregates, per-gene allele-match scores) and whose `germline_coords` aligns a gene segment to a chosen allele's germline — all from a tokenized batch + an encoded `ReferenceSet`.

**Architecture:** `encode_reference(reference_set)` encodes every allele's germline sequence once into pooled embeddings + per-position reps (cached, recomputable for training). `forward(tokens, mask, ref_emb)` runs orientation → backbone → region/state/scalar heads, then per gene pools the backbone reps over the predicted region positions, projects to the germline-embedding space, and scores against the gene's reference embeddings. `germline_coords` gathers a gene's contiguous segment reps and runs the diagonal-correlation aligner against a chosen allele's per-position germline reps.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), GenAIRR 2.2.0, pytest. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs have NO `__init__.py`. d_model is shared across the backbone, germline encoder, and matching space (no cross-space projection needed beyond a learned segment projection).

**Note on the old Phase-1 model:** `core/base.py` / `single_chain.py` / `multi_chain.py` (the superseded architecture) and their dependents stay for now; the new model lives in `core/dnalignair.py`. Removing the old architecture is a post-R4 cleanup.

**Per-position state scheme (R4 decision):** 4 classes `{germline, substitution, insertion, deletion}`.
Update `nn/state_head.py` accordingly in Task 1.

---

## File structure (R4a)

```
src/alignair/nn/state_head.py        [MODIFY: STATES -> germline/substitution/insertion/deletion]
src/alignair/config/dnalignair_config.py   DNAlignAIRConfig
src/alignair/core/dnalignair.py      DNAlignAIR, DNAlignAIROutput, ReferenceEmbeddings, extract_segment
tests/alignair/nn/test_state_head.py [MODIFY: new state names]
tests/alignair/config/test_dnalignair_config.py
tests/alignair/core/test_dnalignair.py
```

---

## Task 1: per-position state scheme + `DNAlignAIRConfig`

**Files:** Modify `src/alignair/nn/state_head.py`, `tests/alignair/nn/test_state_head.py`;
Create `src/alignair/config/dnalignair_config.py`, `tests/alignair/config/test_dnalignair_config.py`

- [ ] **Step 1: Update the state-head scheme** — in `src/alignair/nn/state_head.py` change the STATES
tuple and the counts (substitution replaces mutation/noise per position; aggregate noise/mutation come
from scalar heads):
```python
STATES = ("germline", "substitution", "insertion", "deletion")
STATE_INDEX = {name: i for i, name in enumerate(STATES)}
```
and replace `state_counts` with:
```python
def state_counts(state_logits: torch.Tensor, mask: torch.Tensor) -> dict:
    """Per-sample counts of substitution / indel from argmax states (padding ignored)."""
    labels = state_logits.argmax(dim=-1)
    valid = mask

    def count(name: str) -> torch.Tensor:
        return ((labels == STATE_INDEX[name]) & valid).sum(dim=-1)

    return {
        "substitution_count": count("substitution"),
        "indel_count": count("insertion") + count("deletion"),
    }
```

- [ ] **Step 2: Update the state-head test** — replace `tests/alignair/nn/test_state_head.py` body:
```python
import torch
from alignair.nn.state_head import PerPositionStateHead, state_counts, STATES, STATE_INDEX


def test_state_head_shape_and_backprop():
    head = PerPositionStateHead(d_model=16)
    h = torch.randn(2, 7, 16)
    logits = head(h)
    assert logits.shape == (2, 7, len(STATES))
    assert len(STATES) == 4
    logits.sum().backward()
    assert head.fc.weight.grad is not None


def test_state_counts_from_labels():
    L = 6
    names = ["germline", "substitution", "substitution", "insertion", "deletion", "germline"]
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, STATE_INDEX[nm]] = 10.0
    counts = state_counts(logits, torch.ones(1, L, dtype=torch.bool))
    assert counts["substitution_count"].tolist() == [2]
    assert counts["indel_count"].tolist() == [2]


def test_state_counts_respects_mask():
    L = 4
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i in range(L):
        logits[0, i, STATE_INDEX["substitution"]] = 10.0
    counts = state_counts(logits, torch.tensor([[True, True, False, False]]))
    assert counts["substitution_count"].tolist() == [2]
```

- [ ] **Step 3: Run state-head tests — expect PASS** (3 passed).
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_state_head.py -q`

- [ ] **Step 4: Write the failing config test**

`tests/alignair/config/test_dnalignair_config.py`:
```python
from alignair.config.dnalignair_config import DNAlignAIRConfig


def test_defaults_and_roundtrip():
    cfg = DNAlignAIRConfig(d_model=128)
    assert cfg.d_model == 128 and cfg.n_layers >= 1 and cfg.nhead >= 1
    assert cfg.n_regions == 8 and cfg.n_states == 4
    assert DNAlignAIRConfig.from_dict(cfg.to_dict()) == cfg
```

- [ ] **Step 5: Implement the config**

`src/alignair/config/dnalignair_config.py`:
```python
"""Configuration for the unified DNAlignAIR model."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(eq=True)
class DNAlignAIRConfig:
    d_model: int = 128
    n_layers: int = 4
    nhead: int = 8
    dim_feedforward: int = 512
    max_len: int = 1024
    orientation_dim: int = 64
    n_regions: int = 8     # len(REGIONS)
    n_states: int = 4      # germline/substitution/insertion/deletion

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DNAlignAIRConfig":
        return cls(**d)
```

- [ ] **Step 6: Run config test — expect PASS** (1 passed). Create `tests/alignair/config/` dir via the
test file (no `__init__.py`).

- [ ] **Step 7: Run the full suite to confirm the state-scheme change broke nothing else.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`
Expected: all PASS (only the state-head tests reference the old names, already updated).

- [ ] **Step 8: Commit**
```bash
git add src/alignair/nn/state_head.py tests/alignair/nn/test_state_head.py src/alignair/config/dnalignair_config.py tests/alignair/config/test_dnalignair_config.py
git commit -m "feat(alignair): 4-class per-position state scheme + DNAlignAIRConfig"
```

---

## Task 2: `DNAlignAIR` core — dense + scalar outputs

**Files:** Create `src/alignair/core/dnalignair.py`; Test `tests/alignair/core/test_dnalignair.py`

The model + its forward producing orientation logits, region logits, state logits, and the four scalar
heads (matching + germline come in Tasks 3/4).

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.region_head import REGIONS
from alignair.nn.state_head import STATES


def test_dense_and_scalar_outputs():
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    model = DNAlignAIR(cfg)
    tokens, mask = pad_tokenize(["ACGTACGTACGT", "ACGTAC"])
    out = model.forward_dense(tokens, mask)
    B, L = tokens.shape
    assert out["orientation_logits"].shape == (B, 4)
    assert out["region_logits"].shape == (B, L, len(REGIONS))
    assert out["state_logits"].shape == (B, L, len(STATES))
    for k in ("noise_count", "mutation_rate", "indel_count", "productive"):
        assert out[k].shape == (B, 1)
    # bounded scalars
    assert (out["mutation_rate"] >= 0).all() and (out["mutation_rate"] <= 1).all()
    assert (out["productive"] >= 0).all() and (out["productive"] <= 1).all()
    assert (out["noise_count"] >= 0).all()
    # backbone reps exposed for downstream heads
    assert out["reps"].shape == (B, L, cfg.d_model)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement (dense part)**

`src/alignair/core/dnalignair.py`:
```python
"""Unified DNAlignAIR model: assembles orientation + backbone + region/state/scalar
heads + allele matching + germline alignment over a ReferenceSet."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.dnalignair_config import DNAlignAIRConfig
from ..nn.orientation import OrientationHead
from ..nn.backbone import SequenceBackbone
from ..nn.region_head import RegionTagger, REGION_INDEX
from ..nn.state_head import PerPositionStateHead
from ..nn.germline_encoder import GermlineEncoder
from ..nn.matching import AlleleMatchingHead
from ..nn.germline_aligner import GermlineAligner


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class DNAlignAIR(nn.Module):
    def __init__(self, config: DNAlignAIRConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.orientation_head = OrientationHead(d=config.orientation_dim)
        self.backbone = SequenceBackbone(
            d_model=d, n_layers=config.n_layers, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, max_len=config.max_len)
        self.region_tagger = RegionTagger(d_model=d)
        self.state_head = PerPositionStateHead(d_model=d)
        self.germline_encoder = GermlineEncoder(embed_dim=d)
        self.matching = AlleleMatchingHead()
        self.aligner = GermlineAligner(d_model=d)
        self.seg_proj = nn.Linear(d, d)
        self.noise_head = nn.Linear(d, 1)
        self.mutation_head = nn.Linear(d, 1)
        self.indel_head = nn.Linear(d, 1)
        self.productive_head = nn.Linear(d, 1)

    def forward_dense(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        orientation_logits = self.orientation_head(tokens, mask)
        reps = self.backbone(tokens, mask)
        region_logits = self.region_tagger(reps)
        state_logits = self.state_head(reps)
        pooled = _masked_mean(reps, mask)
        return {
            "orientation_logits": orientation_logits,
            "region_logits": region_logits,
            "state_logits": state_logits,
            "noise_count": F.relu(self.noise_head(pooled)),
            "mutation_rate": torch.sigmoid(self.mutation_head(pooled)),
            "indel_count": F.relu(self.indel_head(pooled)),
            "productive": torch.sigmoid(self.productive_head(pooled)),
            "reps": reps,
        }
```

- [ ] **Step 4: Run — expect PASS** (1 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/core/dnalignair.py tests/alignair/core/test_dnalignair.py
git commit -m "feat(alignair): DNAlignAIR core dense + scalar outputs"
```

---

## Task 3: reference encoding + allele matching

**Files:** Modify `src/alignair/core/dnalignair.py`; Test `tests/alignair/core/test_dnalignair.py` (add)

`encode_reference` encodes the ReferenceSet's germline sequences per gene; `forward` adds per-gene match
scores by pooling reps over predicted region positions.

- [ ] **Step 1: Write the failing test (append)**

```python
import pytest


def _tiny_refset():
    genairr = pytest.importorskip("GenAIRR")
    import GenAIRR.data as gdata
    from alignair.reference.reference_set import ReferenceSet
    return ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)  # V/J only, smaller


def test_encode_reference_and_match():
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    model = DNAlignAIR(cfg)
    refset = _tiny_refset()
    ref_emb = model.encode_reference(refset)
    assert set(ref_emb) == {"V", "J"}
    nV = len(refset.gene("V").names)
    assert ref_emb["V"]["embeddings"].shape == (nV, cfg.d_model)

    tokens, mask = pad_tokenize(["ACGTACGTACGT", "ACGTAC"])
    out = model(tokens, mask, ref_emb)
    assert out["match"]["V"].shape == (2, nV)
    assert out["match"]["J"].shape == (2, len(refset.gene("J").names))
    # dense outputs still present
    assert out["region_logits"].shape[0] == 2
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — append to `DNAlignAIR` in `core/dnalignair.py`:

```python
    def encode_reference(self, reference_set) -> dict:
        """Encode each gene's germline sequences -> pooled embeddings + per-position reps (cached)."""
        from ..data.tokenizer import pad_tokenize
        out = {}
        for gene, ref in reference_set.genes.items():
            tok, msk = pad_tokenize(ref.sequences)
            device = next(self.parameters()).device
            tok, msk = tok.to(device), msk.to(device)
            out[gene] = {
                "embeddings": self.germline_encoder(tok, msk),            # (K, d) normalized
                "pos_reps": self.germline_encoder.forward_positions(tok, msk),  # (K, Lg, d)
                "pos_mask": msk,                                          # (K, Lg)
            }
        return out

    def _segment_rep(self, reps: torch.Tensor, mask: torch.Tensor,
                     region_labels: torch.Tensor, gene: str) -> torch.Tensor:
        """Masked mean of reps over the positions tagged as `gene` -> (B, d), normalized query."""
        gid = REGION_INDEX[gene]
        gene_mask = (region_labels == gid) & mask
        seg = _masked_mean(reps, gene_mask)
        return F.normalize(self.seg_proj(seg), dim=-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, ref_emb: dict) -> dict:
        out = self.forward_dense(tokens, mask)
        region_labels = out["region_logits"].argmax(dim=-1)
        match = {}
        for gene, emb in ref_emb.items():
            query = self._segment_rep(out["reps"], mask, region_labels, gene)
            match[gene] = self.matching(query, emb["embeddings"])
        out["match"] = match
        return out
```

- [ ] **Step 4: Run — expect PASS** (2 passed in that file).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/core/dnalignair.py tests/alignair/core/test_dnalignair.py
git commit -m "feat(alignair): DNAlignAIR reference encoding + allele matching"
```

---

## Task 4: segment extraction + germline coordinates; exports

**Files:** Modify `src/alignair/core/dnalignair.py`; Create `tests/alignair/core/test_dnalignair_germline.py`;
Modify `src/alignair/core/__init__.py`

`extract_segment` gathers a gene's contiguous region positions from `reps` into a left-aligned padded
tensor; `germline_coords` runs the aligner against a chosen allele's per-position germline reps.

- [ ] **Step 1: Write the failing test**

`tests/alignair/core/test_dnalignair_germline.py`:
```python
import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment
from alignair.nn.region_head import REGION_INDEX


def test_extract_segment_left_aligns_gene_positions():
    B, L, d = 1, 8, 4
    reps = torch.arange(B * L * d, dtype=torch.float32).reshape(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    # region labels: positions 2,3,4 are V
    labels = torch.zeros(B, L, dtype=torch.long)
    labels[0, 2:5] = REGION_INDEX["V"]
    seg, seg_mask = extract_segment(reps, mask, labels, "V")
    assert seg.shape[0] == 1 and seg_mask[0].sum().item() == 3
    # first extracted row equals reps at original position 2
    assert torch.allclose(seg[0, 0], reps[0, 2])
    assert torch.allclose(seg[0, 2], reps[0, 4])


def test_germline_coords_shapes():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=2, nhead=4, dim_feedforward=64)
    model = DNAlignAIR(cfg)
    B, Ls, Lg = 2, 5, 12
    seg = torch.randn(B, Ls, cfg.d_model)
    seg_mask = torch.ones(B, Ls, dtype=torch.bool)
    germ = torch.randn(B, Lg, cfg.d_model)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    start_logits, end_logits = model.germline_coords(seg, seg_mask, germ, germ_mask)
    assert start_logits.shape == (B, Lg) and end_logits.shape == (B, Lg)
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** — append `extract_segment` (module function) and the `germline_coords`
method in `core/dnalignair.py`:

```python
def extract_segment(reps: torch.Tensor, mask: torch.Tensor,
                    region_labels: torch.Tensor, gene: str):
    """Gather each sample's positions tagged `gene` into a left-aligned padded
    (B, Smax, d) tensor + (B, Smax) bool mask. Smax is the batch's longest gene run."""
    from ..nn.region_head import REGION_INDEX
    gid = REGION_INDEX[gene]
    sel = (region_labels == gid) & mask                 # (B, L)
    counts = sel.sum(dim=1)                              # (B,)
    smax = int(counts.max().item()) if counts.numel() else 0
    B, L, d = reps.shape
    smax = max(smax, 1)
    seg = reps.new_zeros(B, smax, d)
    seg_mask = torch.zeros(B, smax, dtype=torch.bool, device=reps.device)
    for b in range(B):
        idx = sel[b].nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n:
            seg[b, :n] = reps[b, idx]
            seg_mask[b, :n] = True
    return seg, seg_mask
```

and inside `DNAlignAIR`:
```python
    def germline_coords(self, seg_reps, seg_mask, germ_reps, germ_mask):
        """Align a gene's segment reps to a chosen allele's per-position germline reps."""
        return self.aligner(seg_reps, seg_mask, germ_reps, germ_mask)
```

- [ ] **Step 4: Run — expect PASS** (2 passed).

- [ ] **Step 5: Add exports** — `src/alignair/core/__init__.py` (append; keep existing lines):
```python
from .dnalignair import DNAlignAIR, extract_segment
```
and add `"DNAlignAIR"`, `"extract_segment"` to `__all__`.

- [ ] **Step 6: Run the full suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 7: Commit**
```bash
git add src/alignair/core/dnalignair.py src/alignair/core/__init__.py tests/alignair/core/test_dnalignair_germline.py
git commit -m "feat(alignair): segment extraction + germline_coords; export DNAlignAIR"
```

---

## Self-Review

**Spec coverage (R4 design §3 assembly):** orientation+backbone+region+state+scalar heads → Task 2;
reference encoding + matching → Task 3; segment extraction + germline coords → Task 4; the 4-class state
scheme + config → Task 1. R4b (gym + provenance GT) and R4c (composite loss + training) are separate plans.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `forward_dense(tokens, mask) -> dict` (keys used in Tasks 2/3); `encode_reference(
reference_set) -> {gene: {embeddings, pos_reps, pos_mask}}` consumed by `forward` (Task 3);
`extract_segment(reps, mask, region_labels, gene) -> (seg, seg_mask)` and `germline_coords(seg, seg_mask,
germ, germ_mask) -> (start_logits, end_logits)` consistent (Task 4); `DNAlignAIRConfig` field names match
Task 2's constructor usage; `REGION_INDEX`/`STATES` from the nn modules.

**Known notes:** the old Phase-1 `core/base.py` model stays (removed in a post-R4 cleanup). Germline
coordinates use a chosen allele's reps supplied by the caller (the trainer teacher-forces the true allele
in R4c). Matching pools reps over *predicted* region positions; with an untrained model these are noisy,
but shapes are exercised here — learning happens in R4c on the gym. `noise_count`/`indel_count` are
non-negative (relu); `mutation_rate`/`productive` are in [0,1] (sigmoid).
```
