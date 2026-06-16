# DNAlignAIR R1 — Reference & Allele-Matching Keystone — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the keystone of the redesign — a `ReferenceSet` that unions 1..N GenAIRR dataconfigs into per-gene allele lists + germline sequences, a `GermlineEncoder` that maps a nucleotide sequence to an L2-normalized embedding, and an `AlleleMatchingHead` that scores a query against candidate allele embeddings (multi-label, genotype-restrictable) — and prove the mechanism *learns to identify alleles* on noised inputs.

**Architecture:** Allele classification is metric-learning/retrieval, not a fixed head. The germline encoder embeds each candidate allele's germline sequence; a query (later: a backbone segment representation; in R1: a noised allele sequence through the same encoder) is scored by cosine similarity (scaled by a learned temperature) against the candidate embeddings; per-candidate sigmoid gives a multi-label call. A genotype is a candidate-column mask. No "Short-D": absence of a confident D = all D scores below threshold.

**Tech Stack:** Python 3.12, PyTorch 2.12 (CPU), GenAIRR 2.2.0 (`.venv` editable), pytest. Tests run `PYTHONPATH=src .venv/bin/python -m pytest <path> -v`. Test dirs contain NO `__init__.py`. GenAIRR-dependent tests start with `pytest.importorskip("GenAIRR")`.

**GenAIRR facts (probed):** `cfg.allele_list('v'|'d'|'j')` → allele objects with `.name` (str) and `.ungapped_seq` (germline nucleotides, lowercase); light chains (e.g. `HUMAN_IGK_OGRDB`) return `[]` for `'d'` and `cfg.metadata.has_d is False`. `HUMAN_IGH_OGRDB`: V=198, D=33, J=7.

---

## File structure (R1)

```
src/alignair/reference/
  __init__.py
  reference_set.py     GeneReference, ReferenceSet (build from dataconfigs, genotype mask)
src/alignair/data/tokenizer.py   [MODIFY: add pad_tokenize (right-pad + mask)]
src/alignair/nn/germline_encoder.py   GermlineEncoder
src/alignair/nn/matching.py           AlleleMatchingHead, multilabel_match_loss
tests/alignair/reference/test_reference_set.py
tests/alignair/data/test_pad_tokenize.py
tests/alignair/nn/test_germline_encoder.py
tests/alignair/nn/test_matching.py
tests/alignair/integration/test_matching_learns.py
```

---

## Task 1: `reference/reference_set.py` — ReferenceSet

**Files:** Create `src/alignair/reference/__init__.py` (empty), `src/alignair/reference/reference_set.py`;
Test `tests/alignair/reference/test_reference_set.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet


def test_build_from_igh():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    assert rs.has_d is True
    assert set(rs.genes) == {"V", "D", "J"}
    assert len(rs.gene("V").names) == 198 and len(rs.gene("J").names) == 7
    assert len(rs.gene("D").names) == 33  # real D alleles only, no 'Short-D'
    # germline sequences are uppercased nucleotides aligned with names
    v0 = rs.gene("V")
    assert v0.sequences[0] == v0.sequences[0].upper()
    assert set(v0.sequences[0]) <= set("ACGTN")
    assert v0.index[v0.names[0]] == 0


def test_light_chain_has_no_d():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    assert rs.has_d is False
    assert set(rs.genes) == {"V", "J"}


def test_union_of_two_dataconfigs():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB, gdata.HUMAN_IGK_OGRDB)
    # union V = IGH V (198) + IGK V (168), names unique
    assert len(rs.gene("V").names) == 198 + 168
    assert rs.has_d is True  # at least one chain has D
    assert len(set(rs.gene("V").names)) == len(rs.gene("V").names)


def test_genotype_mask():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    names = rs.gene("V").names
    allowed = {names[0], names[5], names[10]}
    mask = rs.genotype_mask("V", allowed)
    assert mask.dtype == torch.bool and mask.shape == (len(names),)
    assert mask.sum().item() == 3
    assert mask[0] and mask[5] and mask[10]
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

`src/alignair/reference/__init__.py`: empty.

`src/alignair/reference/reference_set.py`:
```python
"""ReferenceSet: union 1..N GenAIRR dataconfigs into per-gene allele references."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


@dataclass
class GeneReference:
    names: List[str]          # ordered allele names
    sequences: List[str]      # germline nucleotide seqs (uppercased), aligned with names
    index: Dict[str, int]     # name -> row index

    def __len__(self) -> int:
        return len(self.names)


class ReferenceSet:
    """Per-gene union allele references built from one or more GenAIRR DataConfigs."""

    def __init__(self, genes: Dict[str, GeneReference], has_d: bool):
        self.genes = genes
        self.has_d = has_d

    def gene(self, g: str) -> GeneReference:
        return self.genes[g.upper()]

    @classmethod
    def from_dataconfigs(cls, *dataconfigs) -> "ReferenceSet":
        has_d = any(dc.metadata.has_d for dc in dataconfigs)
        wanted = ["v", "j"] + (["d"] if has_d else [])
        genes: Dict[str, GeneReference] = {}
        for g in wanted:
            names: List[str] = []
            sequences: List[str] = []
            index: Dict[str, int] = {}
            for dc in dataconfigs:
                if g == "d" and not dc.metadata.has_d:
                    continue
                for allele in dc.allele_list(g):
                    if allele.name in index:
                        continue
                    index[allele.name] = len(names)
                    names.append(allele.name)
                    sequences.append(allele.ungapped_seq.upper())
            genes[g.upper()] = GeneReference(names, sequences, index)
        return cls(genes, has_d)

    def genotype_mask(self, gene: str, allowed_names: Iterable[str]) -> torch.Tensor:
        ref = self.gene(gene)
        allowed = set(allowed_names)
        return torch.tensor([n in allowed for n in ref.names], dtype=torch.bool)
```

- [ ] **Step 4: Run — expect PASS** (4 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/reference tests/alignair/reference/test_reference_set.py
git commit -m "feat(alignair): add ReferenceSet (union dataconfigs -> per-gene allele refs)"
```

---

## Task 2: `data/tokenizer.pad_tokenize` + `nn/germline_encoder.py`

**Files:** Modify `src/alignair/data/tokenizer.py`; Create `src/alignair/nn/germline_encoder.py`;
Test `tests/alignair/data/test_pad_tokenize.py`, `tests/alignair/nn/test_germline_encoder.py`

- [ ] **Step 1: Write the failing test (pad_tokenize)**

`tests/alignair/data/test_pad_tokenize.py`:
```python
import torch
from alignair.data.tokenizer import pad_tokenize


def test_pad_tokenize_shapes_and_mask():
    tokens, mask = pad_tokenize(["ACGT", "ACG"])
    assert tokens.shape == (2, 4) and mask.shape == (2, 4)
    assert tokens.dtype == torch.long and mask.dtype == torch.bool
    assert tokens[0].tolist() == [1, 4, 3, 2]   # A C G T per TOKEN_DICT
    assert tokens[1].tolist() == [1, 4, 3, 0]   # right-padded with 0
    assert mask[0].tolist() == [True, True, True, True]
    assert mask[1].tolist() == [True, True, True, False]


def test_pad_tokenize_unknown_to_n():
    tokens, _ = pad_tokenize(["AXN"])
    assert tokens[0].tolist() == [1, 5, 5]  # X -> N(5), N -> 5
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement pad_tokenize** — append to `src/alignair/data/tokenizer.py`:

First add `import torch` at the top of the file (next to `import numpy as np`). Then append:
```python
def pad_tokenize(sequences, token_dict: dict | None = None):
    """Right-pad a batch of nucleotide strings to the batch max length.

    Returns (tokens LongTensor (B, Lmax), mask BoolTensor (B, Lmax) with True=valid).
    Unknown characters map to N; pad token is 0.
    """
    td = token_dict or TOKEN_DICT
    n = td["N"]
    encoded = [[td.get(c, n) for c in s.upper()] for s in sequences]
    lmax = max((len(e) for e in encoded), default=0)
    tokens = torch.zeros(len(encoded), lmax, dtype=torch.long)
    mask = torch.zeros(len(encoded), lmax, dtype=torch.bool)
    for i, e in enumerate(encoded):
        if e:
            tokens[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            mask[i, :len(e)] = True
    return tokens, mask
```

- [ ] **Step 4: Run — expect PASS** (2 passed).

- [ ] **Step 5: Write the failing test (germline encoder)**

`tests/alignair/nn/test_germline_encoder.py`:
```python
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.germline_encoder import GermlineEncoder


def test_encoder_output_is_normalized_embedding():
    enc = GermlineEncoder(embed_dim=64)
    tokens, mask = pad_tokenize(["ACGTACGTAC", "ACGT", "TTGGCCAATT"])
    emb = enc(tokens, mask)
    assert emb.shape == (3, 64)
    # L2-normalized rows (unit norm)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_encoder_masking_ignores_padding():
    enc = GermlineEncoder(embed_dim=32).eval()
    # same sequence, one padded longer in a batch with a longer neighbor
    t1, m1 = pad_tokenize(["ACGTACGT"])
    t2, m2 = pad_tokenize(["ACGTACGT", "AAAAAAAAAAAAAAAA"])
    with torch.no_grad():
        e1 = enc(t1, m1)[0]
        e2 = enc(t2, m2)[0]  # first row identical seq, but batch padded to len 16
    assert torch.allclose(e1, e2, atol=1e-5)  # padding must not change the embedding
```

- [ ] **Step 6: Run — expect FAIL.**

- [ ] **Step 7: Implement `germline_encoder.py`**

```python
"""GermlineEncoder: nucleotide sequence -> L2-normalized embedding (for matching)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GermlineEncoder(nn.Module):
    """Conv stack + masked mean-pool + projection -> unit-norm embedding.

    Input:  tokens (B, L) long, mask (B, L) bool (True = valid position).
    Output: (B, embed_dim) L2-normalized.
    """

    def __init__(self, embed_dim: int = 128, vocab_size: int = 6,
                 n_conv: int = 3, kernel: int = 5):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, embed_dim, kernel, padding="same") for _ in range(n_conv)])
        self.act = nn.GELU()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)                 # (B, L, E)
        m = mask.unsqueeze(-1).to(x.dtype)         # (B, L, 1)
        x = x * m                                  # zero padded positions pre-conv
        h = x.transpose(1, 2)                      # (B, E, L)
        for conv in self.convs:
            h = self.act(conv(h)) * m.transpose(1, 2)  # re-mask after each conv
        h = h.transpose(1, 2)                      # (B, L, E)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # masked mean
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)
```

- [ ] **Step 8: Run — expect PASS** (2 passed).
- [ ] **Step 9: Commit**
```bash
git add src/alignair/data/tokenizer.py src/alignair/nn/germline_encoder.py tests/alignair/data/test_pad_tokenize.py tests/alignair/nn/test_germline_encoder.py
git commit -m "feat(alignair): add pad_tokenize and GermlineEncoder (sequence -> embedding)"
```

---

## Task 3: `nn/matching.py` — AlleleMatchingHead + loss

**Files:** Create `src/alignair/nn/matching.py`; Test `tests/alignair/nn/test_matching.py`

- [ ] **Step 1: Write the failing test**

```python
import math
import torch
from alignair.nn.matching import AlleleMatchingHead, multilabel_match_loss


def test_scores_shape_and_self_match_is_high():
    head = AlleleMatchingHead(init_temp=0.1)
    # 3 orthonormal candidate embeddings; queries equal to candidates
    E = torch.eye(3)                    # (K=3, d=3), unit-norm rows
    Q = torch.eye(3)                    # (B=3, d=3)
    scores = head(Q, E)
    assert scores.shape == (3, 3)
    # each query's own candidate scores highest
    assert (scores.argmax(dim=1) == torch.arange(3)).all()


def test_genotype_mask_excludes_candidates():
    head = AlleleMatchingHead(init_temp=0.1)
    E = torch.eye(4)
    Q = torch.eye(4)
    allowed = torch.tensor([True, False, True, False])
    scores = head(Q, E, candidate_mask=allowed)
    assert torch.isneginf(scores[:, 1]).all() and torch.isneginf(scores[:, 3]).all()
    assert torch.isfinite(scores[:, 0]).all() and torch.isfinite(scores[:, 2]).all()


def test_multilabel_loss_finite_and_backprops():
    head = AlleleMatchingHead()
    E = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    Q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1).requires_grad_(True)
    scores = head(Q, E)
    target = torch.zeros(2, 5)
    target[0, 1] = 1.0
    target[1, 3] = 1.0
    target[1, 4] = 1.0  # multi-label row (two true alleles)
    loss = multilabel_match_loss(scores, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert Q.grad is not None and torch.isfinite(Q.grad).all()
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement**

```python
"""Allele-embedding matching head: cosine similarity (temperature-scaled) ->
multi-label allele scores, with optional genotype candidate masking."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlleleMatchingHead(nn.Module):
    """Score normalized queries (B, d) against normalized candidate embeddings (K, d).

    Returns (B, K) logits = cosine_sim / temperature. A genotype is a (K,) bool mask
    of allowed candidates (disallowed -> -inf, i.e. sigmoid 0).
    """

    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

    def forward(self, query: torch.Tensor, candidates: torch.Tensor,
                candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = (query @ candidates.t()) / self.log_temp.exp().clamp(min=1e-4)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.unsqueeze(0), float("-inf"))
        return scores


def multilabel_match_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-label BCE over candidates. ``target`` is a (B, K) multi-hot of true alleles.

    Columns with -inf scores (genotype-masked) are ignored so they never contribute."""
    finite = torch.isfinite(scores)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
    per_elem = F.binary_cross_entropy_with_logits(safe_scores, target, reduction="none")
    per_elem = per_elem * finite.to(per_elem.dtype)
    return per_elem.sum() / finite.to(per_elem.dtype).sum().clamp(min=1.0)
```

- [ ] **Step 4: Run — expect PASS** (3 passed).
- [ ] **Step 5: Commit**
```bash
git add src/alignair/nn/matching.py tests/alignair/nn/test_matching.py
git commit -m "feat(alignair): add AlleleMatchingHead + multi-label match loss"
```

---

## Task 4: Keystone — the matching mechanism *learns* to identify alleles

**Files:** Create `tests/alignair/integration/test_matching_learns.py`; Create/Modify
`src/alignair/reference/__init__.py`, `src/alignair/nn/__init__.py` (exports)

This is the R1 "done when": on a small reference of distinct germline-like sequences, training the
encoder + matching head makes a *noised* query of allele i retrieve allele i.

- [ ] **Step 1: Write the learnability test**

```python
import torch
from alignair.data.tokenizer import pad_tokenize, TOKEN_DICT
from alignair.nn.germline_encoder import GermlineEncoder
from alignair.nn.matching import AlleleMatchingHead, multilabel_match_loss

BASES = "ACGT"


def _rand_seq(rng, n):
    return "".join(BASES[i] for i in rng.integers(0, 4, size=n))


def _noise(rng, seq, p=0.1):
    out = list(seq)
    for i in range(len(out)):
        if rng.random() < p:
            out[i] = BASES[rng.integers(0, 4)]
    return "".join(out)


def test_matching_head_learns_to_identify_alleles():
    import numpy as np
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    K = 12
    refs = [_rand_seq(rng, 80) for _ in range(K)]  # distinct "germline alleles"

    enc = GermlineEncoder(embed_dim=64)
    head = AlleleMatchingHead(init_temp=0.1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)

    ref_tokens, ref_mask = pad_tokenize(refs)
    target = torch.eye(K)

    for _ in range(200):
        E = enc(ref_tokens, ref_mask)                 # (K, d)
        queries = [_noise(rng, r, p=0.1) for r in refs]
        q_tokens, q_mask = pad_tokenize(queries)
        Q = enc(q_tokens, q_mask)                      # (K, d)
        scores = head(Q, E)                            # (K, K)
        loss = multilabel_match_loss(scores, target)
        opt.zero_grad(); loss.backward(); opt.step()

    # after training, a noised query of allele i should retrieve i
    enc.eval()
    with torch.no_grad():
        E = enc(ref_tokens, ref_mask)
        queries = [_noise(rng, r, p=0.1) for r in refs]
        qt, qm = pad_tokenize(queries)
        scores = head(enc(qt, qm), E)
    top1 = (scores.argmax(dim=1) == torch.arange(K)).float().mean().item()
    assert top1 >= 0.8, f"retrieval top-1 too low: {top1}"
```

- [ ] **Step 2: Run — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/integration/test_matching_learns.py -v`
(If top-1 is marginally <0.8 due to seeding, increase steps to 300 — the mechanism must clearly learn.)

- [ ] **Step 3: Add exports**

`src/alignair/reference/__init__.py`:
```python
from .reference_set import ReferenceSet, GeneReference

__all__ = ["ReferenceSet", "GeneReference"]
```

Append to `src/alignair/nn/__init__.py` (create if absent) the matching/encoder exports:
```python
from .germline_encoder import GermlineEncoder
from .matching import AlleleMatchingHead, multilabel_match_loss
```
(If `nn/__init__.py` already has content, just add these lines; do not remove existing exports.)

- [ ] **Step 4: Run the full alignair suite — expect PASS.**
`PYTHONPATH=src .venv/bin/python -m pytest tests/alignair -q`

- [ ] **Step 5: Commit**
```bash
git add tests/alignair/integration/test_matching_learns.py src/alignair/reference/__init__.py src/alignair/nn/__init__.py
git commit -m "feat(alignair): keystone matching learnability test + exports"
```

---

## Self-Review

**Spec coverage (design §3.2 ReferenceSet, allele-matching keystone):** ReferenceSet union +
genotype mask → Task 1; germline encoder (allele → embedding) → Task 2; matching head (multi-label,
genotype-restrictable) → Task 3; the keystone "matching learns to identify alleles" → Task 4. Per-position
germline reps `G_a` (for R3 cross-attention) and the backbone-produced query are intentionally out of R1.

**Placeholder scan:** none — every step has complete code and real assertions.

**Type consistency:** `ReferenceSet.from_dataconfigs(*cfgs)` / `.gene(g)` / `.genotype_mask(gene, names)`
consistent across Tasks 1/4; `pad_tokenize(seqs) -> (tokens, mask)` consistent (Tasks 2/4);
`GermlineEncoder(embed_dim).forward(tokens, mask) -> (B,d)` normalized consistent (Tasks 2/4);
`AlleleMatchingHead(init_temp).forward(query, candidates, candidate_mask)` and
`multilabel_match_loss(scores, target)` consistent across Tasks 3/4.

**Known notes:** R1 uses the germline encoder for *both* references and queries (self-retrieval) to
validate the mechanism without the backbone; R2 replaces the query with the backbone's per-gene segment
representation projected into the same space. No "Short-D" class — absent D = all D scores below
threshold. References are a few hundred alleles, so full-candidate scoring is cheap.
```
