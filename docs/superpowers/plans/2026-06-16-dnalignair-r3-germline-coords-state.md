# DNAlignAIR R3 — Germline Coordinates + Per-Position State — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the input↔germline cross-attention aligner that emits **exact germline start/end** (the trims) for a matched allele, plus a per-position state head separating germline / SHM-mutation / sequencing-noise / insertion / deletion.

**Architecture:** The germline encoder gains per-position reps `G (B, Lg, d)`. The aligner derives a *start query* from the observed segment's first position rep and an *end query* from its last position rep, each pointing (dot-product attention) at germline positions → distributions over germline length → argmax gives exact `germline_start/end`. The state head is a per-position 5-class classifier on the backbone reps `H`; counts of noise/mutation/indel fall out as sums.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), pytest. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs have NO `__init__.py`. Token ids: A=1 T=2 G=3 C=4 N=5 pad=0; `pad_tokenize` right-pads (index 0 = first base).

**Scope note:** R3 validates the germline-coordinate mechanism with a learnability test (predict trims of a mutated substring, deviation ≈ 0). The state head is built and unit-tested here; its *learning to separate mutation-vs-noise* is mechanically identical to region tagging (already validated) and is trained on real provenance ground truth in R4.

---

## File structure (R3)

```
src/alignair/nn/germline_encoder.py   [MODIFY: add forward_positions]
src/alignair/nn/germline_aligner.py   GermlineAligner, decode_germline_coords
src/alignair/nn/state_head.py         STATES, STATE_INDEX, PerPositionStateHead, state_counts
tests/alignair/nn/test_germline_positions.py
tests/alignair/nn/test_germline_aligner.py
tests/alignair/nn/test_state_head.py
tests/alignair/integration/test_germline_coords_learn.py
```

---

## Task 1: `GermlineEncoder.forward_positions` — per-position reps

**Files:** Modify `src/alignair/nn/germline_encoder.py`; Test `tests/alignair/nn/test_germline_positions.py`

Refactor so `forward` (pooled, normalized embedding) reuses a new `forward_positions` that returns the
masked per-position conv reps `(B, L, d)` (the keys/values for cross-attention).

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.germline_encoder import GermlineEncoder


def test_forward_positions_shape_and_padding_zeroed():
    enc = GermlineEncoder(embed_dim=48)
    tokens, mask = pad_tokenize(["ACGTACGT", "ACG"])
    pos = enc.forward_positions(tokens, mask)
    assert pos.shape == (2, 8, 48)
    assert torch.allclose(pos[1, 3:], torch.zeros(5, 48), atol=1e-6)


def test_pooled_forward_still_normalized():
    enc = GermlineEncoder(embed_dim=48)
    tokens, mask = pad_tokenize(["ACGTACGT"])
    emb = enc(tokens, mask)
    assert emb.shape == (1, 48)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(1), atol=1e-5)
```

- [ ] **Step 2: Run — expect FAIL** (no attribute `forward_positions`).

- [ ] **Step 3: Refactor `germline_encoder.py`** — replace the body of the class's `forward` and add
`forward_positions`:

```python
"""GermlineEncoder: nucleotide sequence -> per-position reps and a pooled
L2-normalized embedding (for matching)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GermlineEncoder(nn.Module):
    """Conv stack producing per-position reps; masked mean-pool + proj -> unit-norm embedding.

    Input:  tokens (B, L) long, mask (B, L) bool (True = valid position).
    forward_positions -> (B, L, embed_dim) masked per-position reps.
    forward -> (B, embed_dim) L2-normalized pooled embedding.
    """

    def __init__(self, embed_dim: int = 128, vocab_size: int = 6,
                 n_conv: int = 3, kernel: int = 5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, kernel, padding="same") for _ in range(n_conv)])
        self.act = nn.GELU()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward_positions(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)                 # (B, L, E)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = self.act(conv(h)) * m.transpose(1, 2)
        return h.transpose(1, 2)                    # (B, L, E), padded positions zero

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.forward_positions(tokens, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)
```

- [ ] **Step 4: Run — expect PASS** (2 passed). Also re-run the existing encoder test:
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_germline_encoder.py tests/alignair/nn/test_germline_positions.py -q`
Expected: all PASS (the refactor preserves pooled behavior).

- [ ] **Step 5: Commit**
```bash
git add src/alignair/nn/germline_encoder.py tests/alignair/nn/test_germline_positions.py
git commit -m "feat(alignair): GermlineEncoder.forward_positions (per-position reps)"
```

---

## Task 2: `nn/germline_aligner.py` — cross-attention germline coordinates

**Files:** Create `src/alignair/nn/germline_aligner.py`; Test `tests/alignair/nn/test_germline_aligner.py`

The start/end queries come from the observed segment's first/last valid position reps; each points at
germline positions by dot product → `(B, Lg)` logits. `germline_start = argmax(start_logits)`,
`germline_end = argmax(end_logits) + 1` (end-exclusive).

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.nn.germline_aligner import GermlineAligner, decode_germline_coords


def test_aligner_output_shapes_and_germline_masking():
    d, B, Ls, Lg = 16, 2, 6, 10
    aligner = GermlineAligner(d_model=d)
    seg = torch.randn(B, Ls, d)
    seg_mask = torch.ones(B, Ls, dtype=torch.bool)
    germ = torch.randn(B, Lg, d)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    germ_mask[:, 7:] = False  # last 3 germline positions are padding
    sl, el = aligner(seg, seg_mask, germ, germ_mask)
    assert sl.shape == (B, Lg) and el.shape == (B, Lg)
    # masked germline positions never win the argmax
    gs, ge = decode_germline_coords(sl, el)
    assert (gs < 7).all() and (ge <= 7).all()


def test_decode_germline_coords_argmax():
    sl = torch.full((1, 8), -10.0); sl[0, 2] = 10.0
    el = torch.full((1, 8), -10.0); el[0, 6] = 10.0
    gs, ge = decode_germline_coords(sl, el)
    assert gs.tolist() == [2] and ge.tolist() == [7]  # end-exclusive = argmax+1
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Germline coordinate aligner: cross-attention from the observed segment's
endpoints to the matched allele's per-position germline reps -> exact trims."""
import torch
import torch.nn as nn


class GermlineAligner(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_start = nn.Linear(d_model, d_model)
        self.q_end = nn.Linear(d_model, d_model)

    def forward(self, seg_reps: torch.Tensor, seg_mask: torch.Tensor,
                germ_reps: torch.Tensor, germ_mask: torch.Tensor):
        """seg_reps (B,Ls,d) right-padded; germ_reps (B,Lg,d). Returns
        (start_logits (B,Lg), end_logits (B,Lg))."""
        B = seg_reps.shape[0]
        lengths = seg_mask.sum(dim=1)
        first = seg_reps[:, 0]                                  # first valid position
        last = seg_reps[torch.arange(B, device=seg_reps.device),
                        (lengths - 1).clamp(min=0)]             # last valid position
        qs = self.q_start(first)                                # (B,d)
        qe = self.q_end(last)
        start_logits = torch.einsum("bd,bld->bl", qs, germ_reps)  # (B,Lg)
        end_logits = torch.einsum("bd,bld->bl", qe, germ_reps)
        neg = torch.finfo(start_logits.dtype).min
        start_logits = start_logits.masked_fill(~germ_mask, neg)
        end_logits = end_logits.masked_fill(~germ_mask, neg)
        return start_logits, end_logits


def decode_germline_coords(start_logits: torch.Tensor, end_logits: torch.Tensor):
    """Exact germline_start (argmax) and germline_end (argmax+1, end-exclusive)."""
    gs = start_logits.argmax(dim=-1)
    ge = end_logits.argmax(dim=-1) + 1
    return gs, ge
```

- [ ] **Step 4: Run — expect PASS** (2 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/nn/germline_aligner.py tests/alignair/nn/test_germline_aligner.py
git commit -m "feat(alignair): add GermlineAligner (cross-attention germline coords)"
```

---

## Task 3: Germline coordinates *learn* exact trims (the R3 keystone)

**Files:** Create `tests/alignair/integration/test_germline_coords_learn.py`

R3 "done when": given a mutated substring `germline[gs:ge]` of an allele, the encoder + aligner predict
`gs`/`ge` with deviation ≈ 0 after training.

- [ ] **Step 1: Write the learnability test**

```python
import numpy as np
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.germline_encoder import GermlineEncoder
from alignair.nn.germline_aligner import GermlineAligner, decode_germline_coords

BASES = "ACGT"


def _rand(rng, n):
    return "".join(BASES[i] for i in rng.integers(0, 4, size=n))


def _mutate(rng, s, p=0.05):
    return "".join(BASES[rng.integers(0, 4)] if rng.random() < p else c for c in s)


def _make_batch(rng, B, glen=80):
    germs, starts, ends, obs = [], [], [], []
    for _ in range(B):
        g = _rand(rng, glen)
        gs = int(rng.integers(0, 15))
        ge = glen - int(rng.integers(0, 15))   # ge in [65, 80], always > gs
        germs.append(g); starts.append(gs); ends.append(ge)
        obs.append(_mutate(rng, g[gs:ge], 0.05))
    gt, gm = pad_tokenize(germs)
    ot, om = pad_tokenize(obs)
    return gt, gm, ot, om, torch.tensor(starts), torch.tensor(ends)


def test_germline_aligner_learns_exact_trims():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    enc = GermlineEncoder(embed_dim=64)
    aligner = GermlineAligner(d_model=64)
    opt = torch.optim.Adam(list(enc.parameters()) + list(aligner.parameters()), lr=2e-3)
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(250):
        gt, gm, ot, om, gs, ge = _make_batch(rng, 16)
        G = enc.forward_positions(gt, gm)
        S = enc.forward_positions(ot, om)
        sl, el = aligner(S, om, G, gm)
        loss = ce(sl, gs) + ce(el, ge - 1)   # end target is last aligned position
        opt.zero_grad(); loss.backward(); opt.step()

    enc.eval(); aligner.eval()
    with torch.no_grad():
        gt, gm, ot, om, gs, ge = _make_batch(rng, 64)
        G = enc.forward_positions(gt, gm)
        S = enc.forward_positions(ot, om)
        sl, el = aligner(S, om, G, gm)
        pgs, pge = decode_germline_coords(sl, el)
    start_dev = (pgs - gs).abs().float().mean().item()
    end_dev = (pge - ge).abs().float().mean().item()
    assert start_dev <= 1.0, f"germline start deviation too high: {start_dev}"
    assert end_dev <= 1.0, f"germline end deviation too high: {end_dev}"
```

- [ ] **Step 2: Run — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/integration/test_germline_coords_learn.py -v`
(If deviation marginally > 1.0 under seeding, raise steps to 350 — the mechanism must clearly learn.)

- [ ] **Step 3: Commit**
```bash
git add tests/alignair/integration/test_germline_coords_learn.py
git commit -m "test(alignair): germline aligner learns exact trims (R3 keystone)"
```

---

## Task 4: `nn/state_head.py` — per-position state + counts; exports

**Files:** Create `src/alignair/nn/state_head.py`; Modify `src/alignair/nn/__init__.py`;
Test `tests/alignair/nn/test_state_head.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from alignair.nn.state_head import PerPositionStateHead, state_counts, STATES, STATE_INDEX


def test_state_head_shape_and_backprop():
    head = PerPositionStateHead(d_model=16)
    h = torch.randn(2, 7, 16)
    logits = head(h)
    assert logits.shape == (2, 7, len(STATES))
    logits.sum().backward()
    assert head.fc.weight.grad is not None


def test_state_counts_from_labels():
    L = 8
    names = ["germline", "germline", "mutation", "noise", "noise", "insertion", "deletion", "germline"]
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, STATE_INDEX[nm]] = 10.0
    mask = torch.ones(1, L, dtype=torch.bool)
    counts = state_counts(logits, mask)
    assert counts["noise_count"].tolist() == [2]
    assert counts["mutation_count"].tolist() == [1]
    assert counts["indel_count"].tolist() == [2]   # 1 insertion + 1 deletion


def test_state_counts_respects_mask():
    L = 4
    names = ["noise", "noise", "noise", "noise"]
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, STATE_INDEX[nm]] = 10.0
    mask = torch.tensor([[True, True, False, False]])
    counts = state_counts(logits, mask)
    assert counts["noise_count"].tolist() == [2]   # padded positions ignored
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Per-position state head: germline / SHM-mutation / sequencing-noise / insertion / deletion."""
import torch
import torch.nn as nn

STATES = ("germline", "mutation", "noise", "insertion", "deletion")
STATE_INDEX = {name: i for i, name in enumerate(STATES)}


class PerPositionStateHead(nn.Module):
    """(B, L, d) -> (B, L, len(STATES)) per-position state logits."""

    def __init__(self, d_model: int, n_states: int = len(STATES)):
        super().__init__()
        self.fc = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def state_counts(state_logits: torch.Tensor, mask: torch.Tensor) -> dict:
    """Per-sample counts of noise / mutation / indel from argmax states (padding ignored)."""
    labels = state_logits.argmax(dim=-1)  # (B, L)
    valid = mask

    def count(name: str) -> torch.Tensor:
        return ((labels == STATE_INDEX[name]) & valid).sum(dim=-1)

    return {
        "noise_count": count("noise"),
        "mutation_count": count("mutation"),
        "indel_count": count("insertion") + count("deletion"),
    }
```

- [ ] **Step 4: Run — expect PASS** (3 passed).

- [ ] **Step 5: Add exports** — append to `src/alignair/nn/__init__.py` (keep existing lines):
```python
from .germline_aligner import GermlineAligner, decode_germline_coords
from .state_head import PerPositionStateHead, state_counts, STATES, STATE_INDEX
```
and extend `__all__` with: `"GermlineAligner", "decode_germline_coords", "PerPositionStateHead",
"state_counts", "STATES", "STATE_INDEX"`.

- [ ] **Step 6: Run the full alignair suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 7: Commit**
```bash
git add src/alignair/nn/state_head.py src/alignair/nn/__init__.py tests/alignair/nn/test_state_head.py
git commit -m "feat(alignair): add per-position state head + counts; nn exports"
```

---

## Self-Review

**Spec coverage (design §3.4.4 germline coords, §3.4.2 per-position state):** per-position germline reps →
Task 1; cross-attention aligner → exact germline coords → Task 2; the germline-coordinate learnability
(deviation ≈ 0) → Task 3; per-position state head + noise/mutation/indel counts → Task 4. The full
per-position alignment matrix (optional richer target) and training the state head on real provenance GT
are R4.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `GermlineEncoder.forward_positions(tokens, mask) -> (B,L,d)` used in Tasks 1/2/3;
`GermlineAligner(d_model).forward(seg, seg_mask, germ, germ_mask) -> (start_logits, end_logits)` and
`decode_germline_coords(sl, el) -> (gs, ge)` consistent across Tasks 2/3; `PerPositionStateHead(d_model)`,
`state_counts(logits, mask)`, `STATE_INDEX` consistent across Task 4.

**Known notes:** start/end queries use the segment's first/last *valid* position reps (right-pad → index 0
is first). `germline_end` is argmax+1 (end-exclusive). The aligner aligns by representation similarity in
the germline encoder's space, so it learns from the (shared-encoder) endpoint reps. State-head learning on
mutation-vs-noise (which needs germline comparison) is exercised end-to-end in R4 with provenance GT.
```
