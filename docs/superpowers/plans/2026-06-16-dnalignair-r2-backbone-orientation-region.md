# DNAlignAIR R2 — Backbone + Orientation + Region Tagging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the input encoder (conv stem + Transformer, full per-position resolution), the in-model orientation mechanism (token transforms + 4-class head + canonicalize), and the per-position region tagger that yields exact in-sequence V/D/J boundaries.

**Architecture:** Tokens → (light) orientation head predicts one of 4 strand transforms → canonicalize to forward → conv-stem + Transformer encoder produces per-position reps `H (B,L,d)` (right-pad + attention mask, no center-pad) → a per-position region classifier over `{pad, pre, V, N1, D, N2, J, post}`; exact boundaries are the contiguous region runs. The 4 transforms (identity / reverse-complement / complement / reverse) are involutions, so canonicalizing = re-applying the predicted transform.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), pytest. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs have NO `__init__.py`. Token ids: A=1, T=2, G=3, C=4, N=5, pad=0.

**Scope note:** R2 validates the mechanisms with unit tests + a toy region-tagging *learnability* test (exact boundaries on composition-biased region blocks). Full orientation/region training on real GenAIRR-oriented data is R4 (the gym provides directional ground truth). R2 builds the modules; R3 wires them with germline coords + state; the full-model assembly + matching-from-backbone follows.

---

## File structure (R2)

```
src/alignair/nn/orientation.py   COMPLEMENT, apply_orientation, OrientationHead
src/alignair/nn/backbone.py      SequenceBackbone (conv stem + Transformer)
src/alignair/nn/region_head.py   REGIONS, REGION_INDEX, RegionTagger, decode_boundaries
tests/alignair/nn/test_orientation.py
tests/alignair/nn/test_backbone.py
tests/alignair/nn/test_region_head.py
tests/alignair/integration/test_region_tagging_learns.py
```

---

## Task 1: `nn/orientation.py` — transforms + orientation head

**Files:** Create `src/alignair/nn/orientation.py`; Test `tests/alignair/nn/test_orientation.py`

Transform ids: `0=identity, 1=reverse-complement, 2=complement, 3=reverse`. All are involutions.
Complement map (by token id): A(1)↔T(2), G(3)↔C(4), N(5)→N, pad(0)→pad. Reverse flips only the valid
(unpadded) prefix per sample, keeping pad at the end.

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.orientation import apply_orientation, complement, reverse_valid, OrientationHead

# helper to decode tokens back to letters for readability
INV = {0: "-", 1: "A", 2: "T", 3: "G", 4: "C", 5: "N"}
def to_str(tok, msk):
    return "".join(INV[int(t)] for t, m in zip(tok, msk) if m)


def test_complement():
    tok, msk = pad_tokenize(["ACGT"])
    c = complement(tok)
    assert to_str(c[0], msk[0]) == "TGCA"  # A->T C->G G->C T->A


def test_reverse_valid_respects_padding():
    tok, msk = pad_tokenize(["ACGT", "AC"])  # second is right-padded to len 4
    r = reverse_valid(tok, msk)
    assert to_str(r[0], msk[0]) == "TGCA"
    assert to_str(r[1], msk[1]) == "CA"     # only the 2 valid bases reversed
    # padding stays at the end
    assert r[1].tolist()[2:] == [0, 0]


def test_reverse_complement():
    tok, msk = pad_tokenize(["AAAC"])
    out = apply_orientation(tok, msk, torch.tensor([1]))  # revcomp
    assert to_str(out[0], msk[0]) == "GTTT"  # comp(AAAC)=TTTG, reverse=GTTT


def test_transforms_are_involutions():
    tok, msk = pad_tokenize(["ACGTACG", "ACG"])
    for tid in (0, 1, 2, 3):
        ids = torch.full((2,), tid)
        once = apply_orientation(tok, msk, ids)
        twice = apply_orientation(once, msk, ids)
        assert torch.equal(twice, tok), f"transform {tid} is not an involution"


def test_orientation_head_shape_and_backprop():
    head = OrientationHead(d=32)
    tok, msk = pad_tokenize(["ACGTACGT", "ACG"])
    logits = head(tok, msk)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert all(p.grad is not None for p in head.parameters())
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""In-model orientation: token transforms (identity / reverse-complement /
complement / reverse) and a 4-class orientation head. All transforms are
involutions, so canonicalizing = re-applying the predicted transform."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Complement lookup by token id: A(1)<->T(2), G(3)<->C(4), N(5)->N, pad(0)->pad.
_COMPLEMENT = torch.tensor([0, 2, 1, 4, 3, 5], dtype=torch.long)

# Transform ids
IDENTITY, REVERSE_COMPLEMENT, COMPLEMENT, REVERSE = 0, 1, 2, 3
NUM_ORIENTATIONS = 4


def complement(tokens: torch.Tensor) -> torch.Tensor:
    return _COMPLEMENT.to(tokens.device)[tokens]


def reverse_valid(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse only the valid (unpadded) prefix of each row; pad stays at the end."""
    B, L = tokens.shape
    lengths = mask.sum(dim=1, keepdim=True)              # (B,1)
    ar = torch.arange(L, device=tokens.device).unsqueeze(0)  # (1,L)
    valid = ar < lengths
    rev_idx = (lengths - 1 - ar).clamp(min=0)            # (B,L)
    reversed_tokens = torch.gather(tokens, 1, rev_idx)
    return torch.where(valid, reversed_tokens, torch.zeros_like(tokens))


def apply_orientation(tokens: torch.Tensor, mask: torch.Tensor,
                      transform_ids: torch.Tensor) -> torch.Tensor:
    """Apply a per-row transform id in {0,1,2,3} to the token batch."""
    comp = complement(tokens)
    rev = reverse_valid(tokens, mask)
    revcomp = reverse_valid(comp, mask)
    variants = torch.stack([tokens, revcomp, comp, rev], dim=0)  # (4,B,L)
    B = tokens.shape[0]
    return variants[transform_ids, torch.arange(B, device=tokens.device)]


class OrientationHead(nn.Module):
    """Light encoder -> 4-class orientation logits."""

    def __init__(self, d: int = 64, vocab_size: int = 6):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.conv = nn.Conv1d(d, d, 7, padding="same")
        self.fc = nn.Linear(d, NUM_ORIENTATIONS)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        h = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2) * m
        pooled = h.sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)
```

- [ ] **Step 4: Run — expect PASS** (5 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/nn/orientation.py tests/alignair/nn/test_orientation.py
git commit -m "feat(alignair): add orientation transforms + 4-class OrientationHead"
```

---

## Task 2: `nn/backbone.py` — conv stem + Transformer

**Files:** Create `src/alignair/nn/backbone.py`; Test `tests/alignair/nn/test_backbone.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.backbone import SequenceBackbone


def test_backbone_output_shape_and_padding_zeroed():
    bb = SequenceBackbone(d_model=64, n_layers=2, nhead=4)
    tok, msk = pad_tokenize(["ACGTACGT", "ACG"])
    h = bb(tok, msk)
    assert h.shape == (2, 8, 64)
    # padded positions are zeroed in the output
    assert torch.allclose(h[1, 3:], torch.zeros(5, 64), atol=1e-6)


def test_backbone_padding_invariance():
    bb = SequenceBackbone(d_model=64, n_layers=2, nhead=4).eval()
    t1, m1 = pad_tokenize(["ACGTACGT"])
    t2, m2 = pad_tokenize(["ACGTACGT", "AAAAAAAAAAAAAAAA"])  # forces longer padding on row 0
    with torch.no_grad():
        h1 = bb(t1, m1)[0]
        h2 = bb(t2, m2)[0, :8]
    assert torch.allclose(h1, h2, atol=1e-5)  # valid positions independent of padding
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Sequence backbone: conv stem (local motifs) + Transformer (long-range), full
per-position resolution, right-pad + attention mask aware."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceBackbone(nn.Module):
    def __init__(self, d_model: int = 128, vocab_size: int = 6, n_layers: int = 4,
                 nhead: int = 8, dim_feedforward: int = 512,
                 stem_kernels=(7, 5), max_len: int = 1024):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.stem = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding="same") for k in stem_kernels])
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, n_layers)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L = tokens.shape[1]
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds backbone max_len {self.max_len}")
        positions = torch.arange(L, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m                                    # zero padded positions (clean stem input)
        h = x.transpose(1, 2)
        for conv in self.stem:
            h = F.gelu(conv(h)) * m.transpose(1, 2)
        h = h.transpose(1, 2)
        h = self.transformer(h, src_key_padding_mask=~mask)  # True = ignore (pad)
        return h * m
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/nn/backbone.py tests/alignair/nn/test_backbone.py
git commit -m "feat(alignair): add SequenceBackbone (conv stem + Transformer)"
```

---

## Task 3: `nn/region_head.py` — region tagger + boundary decode

**Files:** Create `src/alignair/nn/region_head.py`; Test `tests/alignair/nn/test_region_head.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.nn.region_head import RegionTagger, decode_boundaries, REGIONS, REGION_INDEX


def test_region_tagger_shape():
    tagger = RegionTagger(d_model=32)
    h = torch.randn(2, 10, 32)
    logits = tagger(h)
    assert logits.shape == (2, 10, len(REGIONS))


def test_decode_boundaries_from_labels():
    # build one sample whose argmax labels are: pre pre V V V N1 D D J J
    L = 10
    names = ["pre", "pre", "V", "V", "V", "N1", "D", "D", "J", "J"]
    logits = torch.full((1, L, len(REGIONS)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, REGION_INDEX[nm]] = 10.0
    mask = torch.ones(1, L, dtype=torch.bool)
    rec = decode_boundaries(logits, mask, has_d=True)[0]
    assert rec["v_start"] == 2 and rec["v_end"] == 5   # V at [2,5)
    assert rec["d_start"] == 6 and rec["d_end"] == 8   # D at [6,8)
    assert rec["j_start"] == 8 and rec["j_end"] == 10  # J at [8,10)


def test_decode_absent_gene_is_minus_one():
    L = 6
    names = ["V", "V", "V", "N1", "J", "J"]  # no D
    logits = torch.full((1, L, len(REGIONS)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, REGION_INDEX[nm]] = 10.0
    rec = decode_boundaries(logits, torch.ones(1, L, dtype=torch.bool), has_d=True)[0]
    assert rec["d_start"] == -1 and rec["d_end"] == -1
    assert rec["v_start"] == 0 and rec["j_end"] == 6
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Per-position region tagging and exact-boundary decoding."""
import torch
import torch.nn as nn

REGIONS = ("pad", "pre", "V", "N1", "D", "N2", "J", "post")
REGION_INDEX = {name: i for i, name in enumerate(REGIONS)}


class RegionTagger(nn.Module):
    """Per-position region classifier: (B, L, d) -> (B, L, len(REGIONS)) logits."""

    def __init__(self, d_model: int, n_regions: int = len(REGIONS)):
        super().__init__()
        self.fc = nn.Linear(d_model, n_regions)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def decode_boundaries(region_logits: torch.Tensor, mask: torch.Tensor,
                      has_d: bool = True) -> list:
    """Argmax region labels -> per-gene [start, end) from the contiguous run.

    Returns a list (one dict per sample) with v/j(/d)_start/_end; -1 if absent."""
    labels = region_logits.argmax(dim=-1)  # (B, L)
    B = labels.shape[0]
    genes = ["V", "J"] + (["D"] if has_d else [])
    out = []
    for b in range(B):
        valid = mask[b]
        rec = {}
        for g in genes:
            gid = REGION_INDEX[g]
            pos = ((labels[b] == gid) & valid).nonzero(as_tuple=True)[0]
            key = g.lower()
            if pos.numel() == 0:
                rec[f"{key}_start"], rec[f"{key}_end"] = -1, -1
            else:
                rec[f"{key}_start"] = int(pos.min().item())
                rec[f"{key}_end"] = int(pos.max().item()) + 1
        out.append(rec)
    return out
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/nn/region_head.py tests/alignair/nn/test_region_head.py
git commit -m "feat(alignair): add RegionTagger + exact boundary decode"
```

---

## Task 4: Region-tagging learns exact boundaries (toy) + exports

**Files:** Create `tests/alignair/integration/test_region_tagging_learns.py`; Modify `src/alignair/nn/__init__.py`

R2 "done when": backbone + region tagger learn to tag composition-biased region blocks and recover
**exact** V/D/J boundaries (mean deviation small). Each region uses a biased base distribution so the
task is learnable from local composition.

- [ ] **Step 1: Write the learnability test**

```python
import numpy as np
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.backbone import SequenceBackbone
from alignair.nn.region_head import RegionTagger, decode_boundaries, REGION_INDEX

# dominant base per region (others sampled uniformly with low prob) -> learnable composition
DOM = {"pre": "T", "V": "A", "N1": "N", "D": "G", "N2": "N", "J": "C", "post": "T"}


def _gen(rng):
    layout = [("pre", rng.integers(2, 6)), ("V", rng.integers(40, 56)),
              ("N1", rng.integers(2, 6)), ("D", rng.integers(8, 14)),
              ("N2", rng.integers(2, 6)), ("J", rng.integers(20, 28)),
              ("post", rng.integers(2, 6))]
    seq, labels = [], []
    for name, n in layout:
        dom = DOM[name]
        for _ in range(int(n)):
            base = dom if rng.random() < 0.7 else "ACGT"[rng.integers(0, 4)]
            seq.append(base)
            labels.append(REGION_INDEX[name])
    return "".join(seq), labels


def _batch(rng, B):
    seqs, lab = zip(*[_gen(rng) for _ in range(B)])
    tokens, mask = pad_tokenize(list(seqs))
    L = tokens.shape[1]
    label_t = torch.zeros(B, L, dtype=torch.long)  # 0 = pad
    for i, ls in enumerate(lab):
        label_t[i, :len(ls)] = torch.tensor(ls)
    return tokens, mask, label_t


def test_region_tagging_recovers_exact_boundaries():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    bb = SequenceBackbone(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    tagger = RegionTagger(d_model=64)
    opt = torch.optim.Adam(list(bb.parameters()) + list(tagger.parameters()), lr=2e-3)
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(120):
        tokens, mask, labels = _batch(rng, 16)
        logits = tagger(bb(tokens, mask))
        loss = ce(logits[mask], labels[mask])
        opt.zero_grad(); loss.backward(); opt.step()

    # evaluate exact-boundary deviation on a fresh batch
    bb.eval(); tagger.eval()
    with torch.no_grad():
        tokens, mask, labels = _batch(rng, 32)
        logits = tagger(bb(tokens, mask))
    pred = decode_boundaries(logits, mask, has_d=True)
    # ground-truth boundaries from labels
    devs = []
    for b in range(32):
        for g in ("V", "D", "J"):
            gid = REGION_INDEX[g]
            gt = (labels[b] == gid).nonzero(as_tuple=True)[0]
            gs, ge = int(gt.min()), int(gt.max()) + 1
            devs.append(abs(pred[b][f"{g.lower()}_start"] - gs))
            devs.append(abs(pred[b][f"{g.lower()}_end"] - ge))
    mean_dev = float(np.mean(devs))
    assert mean_dev <= 2.0, f"mean boundary deviation too high: {mean_dev}"
```

- [ ] **Step 2: Run — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/integration/test_region_tagging_learns.py -v`
(If mean_dev marginally > 2.0 under seeding, raise steps to 200 — the mechanism must clearly learn.)

- [ ] **Step 3: Add exports** — append to `src/alignair/nn/__init__.py` (keep existing lines):
```python
from .orientation import apply_orientation, OrientationHead, NUM_ORIENTATIONS
from .backbone import SequenceBackbone
from .region_head import RegionTagger, decode_boundaries, REGIONS, REGION_INDEX
```
and extend `__all__` with: `"apply_orientation", "OrientationHead", "NUM_ORIENTATIONS",
"SequenceBackbone", "RegionTagger", "decode_boundaries", "REGIONS", "REGION_INDEX"`.

- [ ] **Step 4: Run the full alignair suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 5: Commit**
```bash
git add tests/alignair/integration/test_region_tagging_learns.py src/alignair/nn/__init__.py
git commit -m "feat(alignair): region tagging learns exact boundaries (toy) + exports"
```

---

## Self-Review

**Spec coverage (design §3.3 backbone + orientation, §3.4.1 region tagging):** orientation transforms +
4-class head → Task 1; conv-stem+Transformer full-resolution encoder → Task 2; per-position region head +
exact boundary decode → Task 3; "exact in-sequence boundaries" learnability → Task 4. Wiring orientation
canonicalize into a full forward, real GenAIRR-oriented training, and the monotonic CRF refinement are
R3/R4.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `apply_orientation(tokens, mask, transform_ids)` and `OrientationHead(d).forward(
tokens, mask) -> (B,4)` consistent (Task 1); `SequenceBackbone(d_model,...).forward(tokens, mask) ->
(B,L,d)` consistent (Tasks 2/4); `RegionTagger(d_model).forward(h) -> (B,L,len(REGIONS))` and
`decode_boundaries(region_logits, mask, has_d) -> list[dict]` consistent (Tasks 3/4); `REGION_INDEX`
shared.

**Known notes:** transforms are involutions (canonicalize = re-apply predicted transform). Backbone is
padding-invariant because padded token embeddings (padding_idx=0) + pos emb are zeroed pre-stem and the
Transformer uses `src_key_padding_mask`. Toy region learnability uses composition bias; real directional
orientation/region training is validated in R4 on GenAIRR data. `max_len=1024` default covers IGH/TCR.
```
