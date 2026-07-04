# AIRRMatch VDJ Matcher v1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `AIRRMatch` — a fast, reference-agnostic VDJ allele caller that, given a reference set R (scoped to one locus+species) and a read s, predicts which V/(D)/J alleles in R produced s and returns their sequences, via salience-weighted late interaction (no reranker, no coordinates).

**Architecture:** One shared length-preserving dilated-conv encoder embeds the read and every allele to per-position L2-normalized vectors + a per-position salience scalar. Per gene, each candidate is scored by salience-weighted MaxSim over the read; argmax over the gene's *unique sequences* is the call. Trained on GenAIRR's infinite multi-locus/species stream with an additive-margin set-aware softmax over the (small, sibling-containing) reference — hard negatives come free. Judged on held-out species/loci.

**Tech Stack:** PyTorch, GenAIRR (via `alignair.gym`), `alignair.reference.ReferenceSet`. Python 3.12.

Spec: [docs/superpowers/specs/2026-07-04-vdj-reference-agnostic-matcher-design.md](../specs/2026-07-04-vdj-reference-agnostic-matcher-design.md)

## Global Constraints

- **Package:** `src/alignair/matcher/`; public model class name `AIRRMatch`.
- **Run everything** with `PYTHONPATH=src .venv/bin/python` (GPU RTX 3090Ti 24 GB; do not touch the user's Terraria GPU process).
- **v1 output:** allele **calls + sequences only** — NO coordinates, NO segmentation, NO junction/productivity.
- **Multi-locus/species from day one**, R **scoped to one locus+species** at inference (no ANN/shortlist).
- **Scoring runs over UNIQUE sequences** (per-gene `sequence_group` index); N sequence-identical alleles never split softmax mass.
- **Additive margin in score space, BEFORE the τ divide:** `logits = (score − m·positive_mask) / τ`. Defaults `τ=0.05`, `m=0.2`.
- **Salience is allele-side only**; the read-side salience output is computed but ignored in v1.
- **Orientations restricted to forward / reverse-complement**; **inverted-D disabled** via `invert_d_prob=0.0`.
- **Encoder:** length-preserving dilated depthwise-separable conv, **LayerNorm** (not BatchNorm), **no positional encoding**.
- **Primary metric:** set-aware V/D/J allele accuracy on **held-out** species/loci.
- **Commits:** conventional messages, **no `Co-Authored-By` / Claude trailer** (user convention). Work on a feature branch off `pytorch-migration` (do not commit to `main`).

---

## Task 1: Package scaffold — config + DNA tokenizer

**Files:**
- Create: `src/alignair/matcher/__init__.py`
- Create: `src/alignair/matcher/config.py`
- Create: `src/alignair/matcher/tokenizer.py`
- Test: `tests/alignair/matcher/__init__.py`, `tests/alignair/matcher/test_tokenizer.py`

**Interfaces:**
- Produces: `MatcherConfig` dataclass (fields below). `tokenizer.encode(seq: str) -> LongTensor[L]`, `tokenizer.revcomp(seq: str) -> str`, module constants `PAD=0, VOCAB_SIZE=6`.

- [ ] **Step 1: Write the failing test**

`tests/alignair/matcher/test_tokenizer.py`:
```python
import torch
from alignair.matcher import tokenizer as tk
from alignair.matcher.config import MatcherConfig


def test_encode_maps_bases_and_unknown_to_N():
    ids = tk.encode("ACGTNx")
    assert ids.tolist() == [1, 2, 3, 4, 5, 5]  # x -> N
    assert ids.dtype == torch.long


def test_revcomp_is_reverse_complement():
    assert tk.revcomp("AAATTTGC") == "GCAAATTT"
    assert tk.revcomp(tk.revcomp("ACGTACGT")) == "ACGTACGT"


def test_config_defaults_match_spec():
    c = MatcherConfig()
    assert (c.vocab_size, c.d_model, c.d_match, c.n_layers) == (6, 256, 128, 16)
    assert c.dilations == (1, 2, 4, 8, 16)
    assert (c.tau, c.margin, c.max_len) == (0.05, 0.2, 576)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_tokenizer.py -q`
Expected: FAIL — `ModuleNotFoundError: alignair.matcher`.

- [ ] **Step 3: Write minimal implementation**

`src/alignair/matcher/__init__.py`:
```python
"""AIRRMatch — reference-agnostic VDJ allele matcher (late-interaction)."""
```

`src/alignair/matcher/config.py`:
```python
from dataclasses import dataclass


@dataclass
class MatcherConfig:
    vocab_size: int = 6          # PAD, A, C, G, T, N
    d_model: int = 256
    d_match: int = 128
    n_layers: int = 16
    kernel: int = 3              # odd; keeps length exact under dilation
    dilations: tuple = (1, 2, 4, 8, 16)
    dropout: float = 0.1
    tau: float = 0.05
    margin: float = 0.2
    max_len: int = 576
```

`src/alignair/matcher/tokenizer.py`:
```python
import torch

PAD, A, C, G, T, N = 0, 1, 2, 3, 4, 5
VOCAB_SIZE = 6
_CHAR2ID = {"A": A, "C": C, "G": G, "T": T, "N": N}
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def encode(seq: str) -> torch.Tensor:
    return torch.tensor([_CHAR2ID.get(ch, N) for ch in seq.upper()], dtype=torch.long)


def revcomp(seq: str) -> str:
    return "".join(_COMP.get(ch, "N") for ch in reversed(seq.upper()))
```

Also create empty `tests/alignair/matcher/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_tokenizer.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/matcher/__init__.py src/alignair/matcher/config.py src/alignair/matcher/tokenizer.py tests/alignair/matcher/
git commit -m "feat(matcher): package scaffold, config, DNA tokenizer"
```

---

## Task 2: Encoder — length-preserving dilated-conv tower

**Files:**
- Create: `src/alignair/matcher/encoder.py`
- Test: `tests/alignair/matcher/test_encoder.py`

**Interfaces:**
- Consumes: `MatcherConfig`.
- Produces: `MatcherEncoder(cfg)` with `forward(ids: LongTensor[B,L], mask: BoolTensor[B,L]) -> (emb: FloatTensor[B,L,d_match] (L2-normalized), sal: FloatTensor[B,L] (>0))`. Length L is preserved exactly.

- [ ] **Step 1: Write the failing test**

`tests/alignair/matcher/test_encoder.py`:
```python
import torch
from alignair.matcher.config import MatcherConfig
from alignair.matcher.encoder import MatcherEncoder


def _tiny_cfg():
    return MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2))


def test_output_shapes_and_length_preserved():
    cfg = _tiny_cfg()
    enc = MatcherEncoder(cfg).eval()
    B, L = 3, 40
    ids = torch.randint(1, 6, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    emb, sal = enc(ids, mask)
    assert emb.shape == (B, L, cfg.d_match)      # length preserved (no pooling)
    assert sal.shape == (B, L)


def test_match_embeddings_are_l2_normalized_and_salience_positive():
    cfg = _tiny_cfg()
    enc = MatcherEncoder(cfg).eval()
    ids = torch.randint(1, 6, (2, 25))
    mask = torch.ones(2, 25, dtype=torch.bool)
    emb, sal = enc(ids, mask)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert (sal > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_encoder.py -q`
Expected: FAIL — `ModuleNotFoundError: ...encoder`.

- [ ] **Step 3: Write minimal implementation**

`src/alignair/matcher/encoder.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MatcherConfig


class DilatedConvBlock(nn.Module):
    """Length-preserving depthwise-separable dilated residual block. (B, C, L) -> (B, C, L).

    Uses LayerNorm over channels (per position, per sample) so padding and batch
    composition never pollute the normalization statistics — important because D
    alleles are short and heavily padded within a gene batch.
    """

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        assert kernel % 2 == 1, "odd kernel keeps length exact"
        pad = dilation * (kernel - 1) // 2          # exact 'same' for odd kernel
        self.depthwise = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch)
        self.pointwise = nn.Conv1d(ch, ch, 1)
        self.norm = nn.LayerNorm(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        y = self.pointwise(self.depthwise(x))            # (B, C, L)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)  # LayerNorm over C
        y = self.drop(self.act(y))
        return x + y


class MatcherEncoder(nn.Module):
    """Char embed -> dilated-conv tower -> per-position match embedding + salience.

    No positional encoding: convolutions are translation-equivariant, which is what
    late-interaction matching needs (an allele must match wherever it appears in s).
    """

    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.blocks = nn.ModuleList([
            DilatedConvBlock(cfg.d_model, cfg.kernel,
                             cfg.dilations[i % len(cfg.dilations)], cfg.dropout)
            for i in range(cfg.n_layers)
        ])
        self.match_proj = nn.Linear(cfg.d_model, cfg.d_match)
        self.sal_proj = nn.Linear(cfg.d_model, 1)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        x = self.embed(ids).transpose(1, 2)      # (B, d_model, L)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)                     # (B, L, d_model)
        emb = F.normalize(self.match_proj(x), dim=-1)      # (B, L, d_match)
        sal = F.softplus(self.sal_proj(x)).squeeze(-1)     # (B, L)
        return emb, sal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_encoder.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/matcher/encoder.py tests/alignair/matcher/test_encoder.py
git commit -m "feat(matcher): length-preserving dilated-conv encoder with salience head"
```

---

## Task 3: Scoring — salience-weighted MaxSim + strand pick (pure)

**Files:**
- Create: `src/alignair/matcher/scoring.py`
- Test: `tests/alignair/matcher/test_scoring.py`

**Interfaces:**
- Produces:
  - `masked_maxsim(read_emb[Ls,d], read_mask[Ls], cand_emb[C,Lc,d], cand_sal[C,Lc], cand_mask[C,Lc]) -> scores[C]`
  - `pick_strand(v_scores_fwd[C], v_scores_rc[C]) -> int` (0 forward, 1 revcomp; by best candidate).

- [ ] **Step 1: Write the failing test**

`tests/alignair/matcher/test_scoring.py`:
```python
import torch
from alignair.matcher.scoring import masked_maxsim, pick_strand


def test_identical_candidate_scores_highest():
    # read positions = 3 orthonormal directions; candidate A equals read subseq, B is unrelated.
    d = 4
    read = torch.eye(3, d)                      # [3,4]
    read_mask = torch.ones(3, dtype=torch.bool)
    cand_A = torch.stack([torch.eye(3, d)[0], torch.eye(3, d)[1]])   # matches read pos 0,1
    cand_B = torch.stack([torch.eye(3, d)[3 - 1] * 0 + torch.tensor([0., 0., 0., 1.])] * 2)
    cand = torch.stack([cand_A, cand_B])        # [2,2,4]
    sal = torch.ones(2, 2)
    cmask = torch.ones(2, 2, dtype=torch.bool)
    scores = masked_maxsim(read, read_mask, cand, sal, cmask)
    assert scores[0] > scores[1]                # matching allele wins


def test_padding_positions_are_ignored():
    d = 2
    read = torch.tensor([[1., 0.], [0., 1.]])
    read_mask = torch.tensor([True, False])     # 2nd read pos is padding
    cand = torch.tensor([[[0., 1.]]])           # [1,1,2] best-matches the MASKED pos
    sal = torch.ones(1, 1)
    cmask = torch.ones(1, 1, dtype=torch.bool)
    s = masked_maxsim(read, read_mask, cand, sal, cmask)
    assert torch.allclose(s, torch.tensor([0.0]), atol=1e-3)  # forced to match pos0 -> sim 0


def test_salience_weights_the_sum():
    d = 2
    read = torch.tensor([[1., 0.], [0., 1.]])
    read_mask = torch.ones(2, dtype=torch.bool)
    cand = torch.tensor([[[1., 0.], [0., 1.]]])  # pos0 sim=1, pos1 sim=1
    cmask = torch.ones(1, 2, dtype=torch.bool)
    hi_first = masked_maxsim(read, read_mask, cand, torch.tensor([[10., 0.]]), cmask)
    assert torch.allclose(hi_first, torch.tensor([1.0]), atol=1e-3)  # weighted avg of two 1.0s


def test_pick_strand_prefers_better_v_match():
    assert pick_strand(torch.tensor([0.9, 0.1]), torch.tensor([0.2, 0.3])) == 0
    assert pick_strand(torch.tensor([0.2, 0.3]), torch.tensor([0.9, 0.1])) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: ...scoring`.

- [ ] **Step 3: Write minimal implementation**

`src/alignair/matcher/scoring.py`:
```python
import torch

_NEG = -1e4


def masked_maxsim(read_emb, read_mask, cand_emb, cand_sal, cand_mask):
    """Salience-weighted MaxSim of each candidate against one read.

    read_emb [Ls,d], read_mask [Ls] bool; cand_emb [C,Lc,d], cand_sal [C,Lc],
    cand_mask [C,Lc] bool. Returns scores [C].
    """
    sim = cand_emb @ read_emb.t()                       # [C,Lc,Ls]
    sim = sim.masked_fill(~read_mask.view(1, 1, -1), _NEG)
    best = sim.max(dim=-1).values                       # [C,Lc]  best read match per cand pos
    best = best.masked_fill(~cand_mask, 0.0)
    w = cand_sal.masked_fill(~cand_mask, 0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # normalize salience per candidate
    return (w * best).sum(dim=-1)                        # [C]


def pick_strand(v_scores_fwd, v_scores_rc) -> int:
    """0 if forward's best V candidate beats revcomp's, else 1."""
    return 0 if v_scores_fwd.max() >= v_scores_rc.max() else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_scoring.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/matcher/scoring.py tests/alignair/matcher/test_scoring.py
git commit -m "feat(matcher): salience-weighted MaxSim scoring and strand pick"
```

---

## Task 4: Loss — additive-margin, set-aware selection CE

**Files:**
- Create: `src/alignair/matcher/loss.py`
- Test: `tests/alignair/matcher/test_loss.py`

**Interfaces:**
- Produces: `selection_ce(scores[C], target_mask[C] bool, tau: float, margin: float) -> scalar loss`. Set-aware (≥1 True in target_mask). Margin applied in score space before τ.

- [ ] **Step 1: Write the failing test**

`tests/alignair/matcher/test_loss.py`:
```python
import torch
from alignair.matcher.loss import selection_ce


def test_lower_loss_when_true_scores_higher():
    tgt = torch.tensor([True, False, False])
    good = selection_ce(torch.tensor([2.0, 0.0, 0.0]), tgt, tau=0.5, margin=0.0)
    bad = selection_ce(torch.tensor([0.0, 2.0, 0.0]), tgt, tau=0.5, margin=0.0)
    assert good < bad


def test_margin_acts_in_score_space_not_scaled_by_tau():
    # With margin applied BEFORE the tau divide, the true logit drops by margin/tau.
    scores = torch.tensor([1.0, 0.0, 0.0])
    tgt = torch.tensor([True, False, False])
    tau, m = 0.05, 0.2
    loss_m = selection_ce(scores, tgt, tau=tau, margin=m)
    # equivalent hand computation: logits = (scores - m*tgt)/tau
    logits = (scores - m * tgt.float()) / tau
    expected = torch.logsumexp(logits, 0) - logits[0]
    assert torch.allclose(loss_m, expected, atol=1e-5)
    # sanity: the effective margin is m (=0.2 in score space), NOT m*tau (=0.01).
    loss_no_m = selection_ce(scores, tgt, tau=tau, margin=0.0)
    assert (loss_m - loss_no_m) > 1.0           # a 0.2 score-space margin at tau=0.05 is large


def test_set_aware_any_true_target_is_low_loss():
    # two true sequences; either being top should give low loss.
    tgt = torch.tensor([True, True, False])
    loss = selection_ce(torch.tensor([3.0, 2.9, 0.0]), tgt, tau=0.5, margin=0.0)
    assert loss < 0.1


def test_gradient_flows():
    scores = torch.tensor([1.0, 0.5, 0.2], requires_grad=True)
    loss = selection_ce(scores, torch.tensor([True, False, False]), tau=0.5, margin=0.1)
    loss.backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_loss.py -q`
Expected: FAIL — `ModuleNotFoundError: ...loss`.

- [ ] **Step 3: Write minimal implementation**

`src/alignair/matcher/loss.py`:
```python
import torch


def selection_ce(scores, target_mask, tau: float, margin: float):
    """Additive-margin, set-aware selection cross-entropy over one gene's unique sequences.

    scores [C], target_mask [C] bool (>=1 True). Margin is subtracted from the true
    logit(s) in SCORE space, before the tau scale: logits = (scores - m*mask)/tau.
    Numerator = logsumexp over the true set (any-correct).
    """
    logits = (scores - margin * target_mask.float()) / tau
    denom = torch.logsumexp(logits, dim=-1)
    num = torch.logsumexp(logits.masked_fill(~target_mask, -1e9), dim=-1)
    return denom - num
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_loss.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/matcher/loss.py tests/alignair/matcher/test_loss.py
git commit -m "feat(matcher): additive-margin set-aware selection cross-entropy"
```

---

## Task 5: Data bridge — sequence groups, targets, batching, gym stream

**Files:**
- Create: `src/alignair/matcher/data.py`
- Test: `tests/alignair/matcher/test_data.py`

**Interfaces:**
- Consumes: `alignair.reference.reference_set.GeneReference/ReferenceSet`, `alignair.matcher.tokenizer`.
- Produces:
  - `build_sequence_groups(gene_ref) -> SequenceGroups` with `.unique_seqs: List[str]`, `.group_names: List[List[str]]`, `.name_to_group: Dict[str,int]`, `.n: int`.
  - `target_mask(truth_names: List[str], groups: SequenceGroups) -> BoolTensor[n]`.
  - `encode_batch(seqs: List[str], max_len: int) -> (ids[B,Lmax], mask[B,Lmax])`.
  - `stream_batches(dataconfig, cfg: MatcherConfig, params: dict, batch_size, seed, rc_prob=0.5) -> Iterator[Batch]` where `Batch` is the dict documented in Task 6 Interfaces. Sets `params["invert_d_prob"]=0.0`.

- [ ] **Step 1: Write the failing test** (pure parts — no gym needed)

`tests/alignair/matcher/test_data.py`:
```python
import torch
from alignair.reference.reference_set import GeneReference
from alignair.matcher.data import build_sequence_groups, target_mask, encode_batch


def _gene():
    # two names share an identical sequence -> must collapse to one group.
    names = ["V1*01", "V1*02", "V2*01"]
    seqs = ["ACGTACGT", "ACGTACGT", "TTTTGGGG"]
    index = {n: i for i, n in enumerate(names)}
    return GeneReference(names, seqs, index)


def test_sequence_identical_alleles_collapse():
    g = build_sequence_groups(_gene())
    assert g.n == 2                                  # 3 alleles -> 2 unique sequences
    assert g.name_to_group["V1*01"] == g.name_to_group["V1*02"]
    assert set(g.group_names[g.name_to_group["V1*01"]]) == {"V1*01", "V1*02"}


def test_target_mask_is_set_aware_over_groups():
    g = build_sequence_groups(_gene())
    m = target_mask(["V1*02", "V2*01"], g)           # comma-list truth
    assert m.dtype == torch.bool and m.sum().item() == 2
    assert m[g.name_to_group["V1*01"]]               # V1*02 lights up the shared group


def test_encode_batch_pads_and_masks():
    ids, mask = encode_batch(["ACG", "ACGTA"], max_len=8)
    assert ids.shape == (2, 5) and mask.shape == (2, 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert ids[0, 3:].sum().item() == 0              # PAD id = 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_data.py -q`
Expected: FAIL — `ModuleNotFoundError: ...data`.

- [ ] **Step 3: Write minimal implementation**

`src/alignair/matcher/data.py`:
```python
"""Data bridge: sequence-group collapse, set-aware targets, batching, gym streaming."""
from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch

from .config import MatcherConfig
from . import tokenizer as tk

GENES = ("V", "D", "J")


@dataclass
class SequenceGroups:
    unique_seqs: List[str]
    group_names: List[List[str]]
    name_to_group: Dict[str, int]

    @property
    def n(self) -> int:
        return len(self.unique_seqs)


def build_sequence_groups(gene_ref) -> SequenceGroups:
    seq_to_gid: Dict[str, int] = {}
    unique, group_names, name_to_group = [], [], {}
    for name, seq in zip(gene_ref.names, gene_ref.sequences):
        gid = seq_to_gid.get(seq)
        if gid is None:
            gid = len(unique)
            seq_to_gid[seq] = gid
            unique.append(seq)
            group_names.append([])
        name_to_group[name] = gid
        group_names[gid].append(name)
    return SequenceGroups(unique, group_names, name_to_group)


def target_mask(truth_names: List[str], groups: SequenceGroups) -> torch.Tensor:
    m = torch.zeros(groups.n, dtype=torch.bool)
    for nm in truth_names:
        gid = groups.name_to_group.get(nm)
        if gid is not None:
            m[gid] = True
    return m


def encode_batch(seqs: List[str], max_len: int):
    L = min(max(len(s) for s in seqs), max_len)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        e = tk.encode(s)[:L]
        ids[i, : e.numel()] = e
        mask[i, : e.numel()] = True
    return ids, mask


def _split_calls(value) -> List[str]:
    return [x for x in str(value or "").split(",") if x]


def stream_batches(dataconfig, cfg: MatcherConfig, params: dict, batch_size: int,
                   seed: int, rc_prob: float = 0.5) -> Iterator[dict]:
    """Yield config-homogeneous training batches from a single dataconfig.

    Reference R is built once from this dataconfig; inverted-D is disabled; reads are
    forward/revcomp-augmented. See Task 6 for the emitted batch dict schema.
    """
    import random
    from alignair.reference.reference_set import ReferenceSet
    from alignair.gym.experiment import build_experiment

    ref = ReferenceSet.from_dataconfigs(dataconfig)
    present = [g for g in GENES if g in ref.genes]
    groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
    cand = {}
    for g in present:
        cand[g] = encode_batch(groups[g].unique_seqs, cfg.max_len)  # (ids, mask), cached

    p = dict(params)
    p["invert_d_prob"] = 0.0                       # disable inverted-D in v1
    exp = build_experiment(dataconfig, p, allow_curatable=True)
    rng = random.Random(seed)

    buf: List[dict] = []
    for rec in exp.stream_records(n=None, seed=seed):
        seq = str(rec["sequence"]).upper()
        truth = {g: _split_calls(rec.get(f"{g.lower()}_call")) for g in present}
        if rc_prob and rng.random() < rc_prob:
            seq = tk.revcomp(seq)
        buf.append({"seq": seq, "truth": truth})
        if len(buf) < batch_size:
            continue
        read_ids, read_mask = encode_batch([b["seq"] for b in buf], cfg.max_len)
        read_rc_ids, read_rc_mask = encode_batch([tk.revcomp(b["seq"]) for b in buf], cfg.max_len)
        genes = {}
        for g in present:
            tgt = torch.stack([target_mask(b["truth"][g], groups[g]) for b in buf])  # [B, n_g]
            genes[g] = {"cand_ids": cand[g][0], "cand_mask": cand[g][1], "target": tgt,
                        "groups": groups[g]}
        yield {"read_ids": read_ids, "read_mask": read_mask,
               "read_rc_ids": read_rc_ids, "read_rc_mask": read_rc_mask, "genes": genes}
        buf = []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_data.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Add a gym-streaming smoke test**

Append to `tests/alignair/matcher/test_data.py`:
```python
def test_stream_batches_smoke_igh():
    import GenAIRR.data as gd
    from alignair.gym.curriculum import Curriculum
    from alignair.matcher.config import MatcherConfig
    from alignair.matcher.data import stream_batches
    cfg = MatcherConfig(max_len=576)
    it = stream_batches(gd.HUMAN_IGH_OGRDB, cfg, dict(Curriculum().params(0.2)),
                        batch_size=4, seed=1)
    batch = next(iter(it))
    assert batch["read_ids"].shape[0] == 4
    assert set(batch["genes"]) == {"V", "D", "J"}
    assert batch["genes"]["V"]["target"].shape[0] == 4
    assert batch["genes"]["V"]["target"].any()          # some read has a true V group
```

- [ ] **Step 6: Run and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_data.py -q`
Expected: PASS (4 passed).

```bash
git add src/alignair/matcher/data.py tests/alignair/matcher/test_data.py
git commit -m "feat(matcher): data bridge — sequence groups, set-aware targets, gym stream"
```

---

## Task 6: Model — AIRRMatch (train forward + predict)

**Files:**
- Create: `src/alignair/matcher/model.py`
- Create: `src/alignair/matcher/infer.py`
- Test: `tests/alignair/matcher/test_model.py`

**Interfaces:**
- Consumes: `MatcherEncoder`, `masked_maxsim`, `pick_strand`, `selection_ce`, data-bridge `Batch` dict, `build_sequence_groups`, `encode_batch`, `ReferenceSet`.
- Batch dict schema (from Task 5): `{read_ids[B,Ls], read_mask[B,Ls], read_rc_ids[B,Ls], read_rc_mask[B,Ls], genes: {G: {cand_ids[C_G,Lc], cand_mask[C_G,Lc], target[B,C_G] bool, groups: SequenceGroups}}}`.
- Produces:
  - `AIRRMatch(cfg)` with `training_loss(batch) -> scalar`, and `predict(ref: ReferenceSet, seq: str) -> Dict[str, dict]` returning per present gene `{"names": [...], "seq": str}`.
  - `infer.called_names(gid: int, groups) -> List[str]`.

- [ ] **Step 1: Write the failing test**

`tests/alignair/matcher/test_model.py`:
```python
import torch
from alignair.reference.reference_set import ReferenceSet, GeneReference
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch


def _toy_ref():
    def gr(names, seqs):
        return GeneReference(names, seqs, {n: i for i, n in enumerate(names)})
    genes = {
        "V": gr(["V1*01", "V2*01"], ["ACGTACGTACGTACGT", "TTTTGGGGCCCCAAAA"]),
        "D": gr(["D1*01", "D2*01"], ["GGGGTTTT", "CCCCAAAA"]),
        "J": gr(["J1*01", "J2*01"], ["TTTTAAAACCCC", "GGGGCCCCAAAA"]),
    }
    return ReferenceSet(genes, has_d=True)


def test_predict_returns_names_and_sequences_for_each_gene():
    ref = _toy_ref()
    m = AIRRMatch(MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2))).eval()
    read = ref.gene("V").sequences[0] + ref.gene("D").sequences[0] + ref.gene("J").sequences[0]
    out = m.predict(ref, read)
    assert set(out) == {"V", "D", "J"}
    assert out["V"]["seq"] in ref.gene("V").sequences
    assert isinstance(out["V"]["names"], list) and out["V"]["names"]


def test_training_loss_is_scalar_and_differentiable():
    from alignair.matcher.data import build_sequence_groups, encode_batch, target_mask
    ref = _toy_ref()
    m = AIRRMatch(MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2)))
    groups = {g: build_sequence_groups(ref.gene(g)) for g in "VDJ"}
    reads = [ref.gene("V").sequences[0] + ref.gene("D").sequences[0] + ref.gene("J").sequences[0]]
    rid, rmask = encode_batch(reads, 64)
    genes = {}
    for g in "VDJ":
        cid, cmask = encode_batch(groups[g].unique_seqs, 64)
        genes[g] = {"cand_ids": cid, "cand_mask": cmask,
                    "target": torch.stack([target_mask([ref.gene(g).names[0]], groups[g])]),
                    "groups": groups[g]}
    batch = {"read_ids": rid, "read_mask": rmask, "read_rc_ids": rid, "read_rc_mask": rmask,
             "genes": genes}
    loss = m.training_loss(batch)
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_model.py -q`
Expected: FAIL — `ModuleNotFoundError: ...model`.

- [ ] **Step 3: Write minimal implementation**

`src/alignair/matcher/infer.py`:
```python
def called_names(gid: int, groups) -> list:
    """All allele names in the winning unique-sequence group."""
    return list(groups.group_names[gid])
```

`src/alignair/matcher/model.py`:
```python
import torch
import torch.nn as nn

from .config import MatcherConfig
from .encoder import MatcherEncoder
from .scoring import masked_maxsim, pick_strand
from .loss import selection_ce
from .data import build_sequence_groups, encode_batch, GENES
from . import tokenizer as tk
from . import infer


class AIRRMatch(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MatcherEncoder(cfg)

    # ---- training ----
    def training_loss(self, batch) -> torch.Tensor:
        r_emb, r_sal = self.encoder(batch["read_ids"], batch["read_mask"])
        rc_emb, rc_sal = self.encoder(batch["read_rc_ids"], batch["read_rc_mask"])
        losses = []
        for g, gb in batch["genes"].items():
            c_emb, c_sal = self.encoder(gb["cand_ids"], gb["cand_mask"])
            for b in range(r_emb.shape[0]):
                fwd = masked_maxsim(r_emb[b], batch["read_mask"][b], c_emb, c_sal, gb["cand_mask"])
                rc = masked_maxsim(rc_emb[b], batch["read_rc_mask"][b], c_emb, c_sal, gb["cand_mask"])
                scores = torch.maximum(fwd, rc)                # strand-invariant training signal
                if gb["target"][b].any():
                    losses.append(selection_ce(scores, gb["target"][b], self.cfg.tau, self.cfg.margin))
        return torch.stack(losses).mean()

    # ---- inference ----
    @torch.no_grad()
    def predict(self, ref, seq: str):
        present = [g for g in GENES if g in ref.genes]
        groups = {g: build_sequence_groups(ref.gene(g)) for g in present}
        r_ids, r_mask = encode_batch([seq], self.cfg.max_len)
        rc_ids, rc_mask = encode_batch([tk.revcomp(seq)], self.cfg.max_len)
        r_emb, _ = self.encoder(r_ids, r_mask)
        rc_emb, _ = self.encoder(rc_ids, rc_mask)
        cand = {}
        for g in present:
            cids, cmask = encode_batch(groups[g].unique_seqs, self.cfg.max_len)
            cemb, csal = self.encoder(cids, cmask)
            cand[g] = (cemb, csal, cmask)
        # strand pick anchored by V
        vfwd = masked_maxsim(r_emb[0], r_mask[0], *cand["V"])
        vrc = masked_maxsim(rc_emb[0], rc_mask[0], *cand["V"])
        strand = pick_strand(vfwd, vrc)
        emb, mask = (r_emb[0], r_mask[0]) if strand == 0 else (rc_emb[0], rc_mask[0])
        out = {}
        for g in present:
            scores = masked_maxsim(emb, mask, *cand[g])
            gid = int(scores.argmax())
            out[g] = {"names": infer.called_names(gid, groups[g]),
                      "seq": groups[g].unique_seqs[gid]}
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_model.py -q`
Expected: PASS (2 passed). (The toy sequences are distinct enough that an untrained net still returns valid shapes; `predict` correctness under training is exercised in Task 7's overfit test.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/matcher/model.py src/alignair/matcher/infer.py tests/alignair/matcher/test_model.py
git commit -m "feat(matcher): AIRRMatch model — training_loss and predict"
```

---

## Task 7: Training script + overfit sanity

**Files:**
- Create: `scripts/train_matcher.py`
- Test: `tests/alignair/matcher/test_e2e.py`

**Interfaces:**
- Consumes: `AIRRMatch`, `stream_batches`, `MatcherConfig`.
- Produces: `train(configs, cfg, steps, lr, batch_size, held_out, out_path, device, eval_every)` reusable from the test; CLI wrapper.

- [ ] **Step 1: Write the failing overfit test**

`tests/alignair/matcher/test_e2e.py`:
```python
import torch
from alignair.reference.reference_set import ReferenceSet, GeneReference
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch


def _toy_ref():
    def gr(names, seqs):
        return GeneReference(names, seqs, {n: i for i, n in enumerate(names)})
    return ReferenceSet({
        "V": gr(["V1*01", "V2*01"], ["ACGTACGTACGTACGTACGT", "TTTTGGGGCCCCAAAATTTT"]),
        "D": gr(["D1*01", "D2*01"], ["GGGGTTTTGGGG", "CCCCAAAACCCC"]),
        "J": gr(["J1*01", "J2*01"], ["TTTTAAAACCCCGGGG", "GGGGCCCCAAAATTTT"]),
    }, has_d=True)


def test_overfits_tiny_set():
    """Learning sanity: the model must fit a handful of exact-concat reads."""
    torch.manual_seed(0)
    from alignair.matcher.data import build_sequence_groups, encode_batch, target_mask
    ref = _toy_ref()
    groups = {g: build_sequence_groups(ref.gene(g)) for g in "VDJ"}
    # two examples: (V1,D1,J1) and (V2,D2,J2)
    picks = [(0, 0, 0), (1, 1, 1)]
    reads = ["".join([ref.gene(g).sequences[i] for g, i in zip("VDJ", p)]) for p in picks]
    rid, rmask = encode_batch(reads, 128)
    genes = {}
    for gi, g in enumerate("VDJ"):
        cid, cmask = encode_batch(groups[g].unique_seqs, 128)
        tgt = torch.stack([target_mask([ref.gene(g).names[p[gi]]], groups[g]) for p in picks])
        genes[g] = {"cand_ids": cid, "cand_mask": cmask, "target": tgt, "groups": groups[g]}
    batch = {"read_ids": rid, "read_mask": rmask, "read_rc_ids": rid, "read_rc_mask": rmask,
             "genes": genes}
    m = AIRRMatch(MatcherConfig(d_model=48, d_match=24, n_layers=4, dilations=(1, 2)))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad(); loss = m.training_loss(batch); loss.backward(); opt.step()
    assert loss.item() < 0.05
    # and predict picks the right V for read 0
    out = m.predict(ref, reads[0])
    assert "V1*01" in out["V"]["names"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_e2e.py -q`
Expected: FAIL — either the model doesn't converge yet or (if run before Task 6) import error. If it fails to reach `<0.05`, that is a real signal to debug the encoder/scoring before proceeding — do NOT weaken the threshold.

- [ ] **Step 3: Write the training script**

`scripts/train_matcher.py`:
```python
"""Train AIRRMatch on GenAIRR's multi-locus/species stream; hold out configs for the exam."""
import argparse
import itertools

import torch

from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.data import stream_batches
from alignair.gym.curriculum import Curriculum


def _round_robin(configs, cfg, params, batch_size, seed):
    streams = [stream_batches(dc, cfg, params, batch_size, seed + i) for i, dc in enumerate(configs)]
    for batch in itertools.chain.from_iterable(zip(*[iter(s) for s in streams])):
        yield batch


def train(configs, cfg, steps, lr, batch_size, out_path, device="cuda",
          progress=0.3, seed=0, log_every=50):
    model = AIRRMatch(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    params = dict(Curriculum().params(progress))
    stream = _round_robin(configs, cfg, params, batch_size, seed)
    model.train()
    for step, batch in zip(range(1, steps + 1), stream):
        batch = _to_device(batch, device)
        loss = model.training_loss(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"[{step}/{steps}] loss {loss.item():.4f}", flush=True)
    torch.save({"config": vars(cfg), "model": model.state_dict()}, out_path)
    return model


def _to_device(batch, device):
    out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "genes"}
    out["genes"] = {}
    for g, gb in batch["genes"].items():
        out["genes"][g] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in gb.items()}
    return out


def _resolve_configs(names):
    import GenAIRR.data as gd
    return [getattr(gd, n) for n in names]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["HUMAN_IGH_OGRDB"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=".private/models/airrmatch_v1.pt")
    a = ap.parse_args()
    train(_resolve_configs(a.configs), MatcherConfig(), a.steps, a.lr, a.batch_size,
          a.out, device=a.device, progress=a.progress)
```

- [ ] **Step 4: Run overfit test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_e2e.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Real smoke run (short, GPU)**

Run: `PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_matcher.py --configs HUMAN_IGH_OGRDB --steps 200 --batch-size 16 --out /tmp/airrmatch_smoke.pt`
Expected: prints decreasing `loss` lines; writes checkpoint; no crash.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_matcher.py tests/alignair/matcher/test_e2e.py
git commit -m "feat(matcher): training loop + overfit sanity test"
```

---

## Task 8: Held-out evaluation script

**Files:**
- Create: `scripts/eval_matcher.py`
- Test: `tests/alignair/matcher/test_eval.py`

**Interfaces:**
- Consumes: `AIRRMatch`, `ReferenceSet`, `stream_batches`/gym records, `build_sequence_groups`.
- Produces: `evaluate(model, dataconfig, cfg, n, progress, seed) -> Dict[str, float]` — set-aware top-1-in-set accuracy per gene `{"V": .., "D": .., "J": ..}` on records from a (potentially held-out) dataconfig.

- [ ] **Step 1: Write the failing test**

`tests/alignair/matcher/test_eval.py`:
```python
import os
import sys

import GenAIRR.data as gd
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch

# scripts/ is not an importable package — add it to sys.path to import the eval entry point.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
from eval_matcher import evaluate  # noqa: E402


def test_evaluate_returns_per_gene_accuracy_keys():
    cfg = MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2))
    model = AIRRMatch(cfg).eval()
    acc = evaluate(model, gd.HUMAN_IGH_OGRDB, cfg, n=8, progress=0.0, seed=3, device="cpu")
    assert set(acc) == {"V", "D", "J"}
    assert all(0.0 <= v <= 1.0 for v in acc.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: eval_matcher`.

- [ ] **Step 3: Write the eval script**

`scripts/eval_matcher.py`:
```python
"""Evaluate AIRRMatch: set-aware top-1-in-set allele accuracy on a (held-out) dataconfig."""
import argparse

import torch

from alignair.reference.reference_set import ReferenceSet
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.data import GENES, _split_calls
from alignair.gym.experiment import build_experiment
from alignair.gym.curriculum import Curriculum


def evaluate(model, dataconfig, cfg: MatcherConfig, n=200, progress=0.3, seed=0, device="cpu"):
    ref = ReferenceSet.from_dataconfigs(dataconfig)
    present = [g for g in GENES if g in ref.genes]
    hit = {g: 0 for g in present}
    tot = {g: 0 for g in present}
    params = dict(Curriculum().params(progress)); params["invert_d_prob"] = 0.0
    exp = build_experiment(dataconfig, params, allow_curatable=True)
    model = model.to(device).eval()
    for rec in exp.stream_records(n=n, seed=seed):
        out = model.predict(ref, str(rec["sequence"]).upper())
        for g in present:
            truth = set(_split_calls(rec.get(f"{g.lower()}_call")))
            if not truth:
                continue
            tot[g] += 1
            if truth & set(out[g]["names"]):        # set-aware: any true name in the called group
                hit[g] += 1
    return {g: (hit[g] / tot[g] if tot[g] else 0.0) for g in present}


def _resolve(name):
    import GenAIRR.data as gd
    return getattr(gd, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="HUMAN_IGH_OGRDB", help="held-out dataconfig name")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location=a.device)
    cfg = MatcherConfig(**ck["config"])
    model = AIRRMatch(cfg); model.load_state_dict(ck["model"])
    acc = evaluate(model, _resolve(a.config), cfg, n=a.n, progress=a.progress, device=a.device)
    print("set-aware allele accuracy:", {g: round(v, 3) for g, v in acc.items()})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/test_eval.py -q`
Expected: PASS (1 passed) — untrained accuracy will be low but keys/ranges valid.

- [ ] **Step 5: Full-suite regression check**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/matcher/ -q`
Expected: all matcher tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_matcher.py tests/alignair/matcher/test_eval.py
git commit -m "feat(matcher): held-out set-aware allele-accuracy eval"
```

---

## After the plan: first real experiment (not a code task)

Once all 8 tasks pass, run the first honest experiment (drives the A→B/C decision gate from the spec):

1. Train multi-config with a held-out species, e.g.
   `--configs HUMAN_IGH_OGRDB HUMAN_IGK_OGRDB HUMAN_IGL_OGRDB MOUSE_IGH_OGRDB` (reserve one, e.g. a non-human config, for the exam).
2. `eval_matcher.py --config <held-out>` → read the per-gene set-aware accuracy.
3. Compare V/D/J held-out vs in-distribution and vs IgBLAST (`scripts/vs_igblast.py`).
4. **Decision gate (from spec):** if held-out J or D top-1-in-set trails IgBLAST by >5 pp after ~50k steps, or ID→held-out drop >10 pp on any gene → design Approach B (learned rerank) or C (classical rerank). Otherwise A stands.

## Notes for the implementer

- **D is expected weakest** (short, trimmed, spuriously-matching). A low held-out D number is anticipated, not a bug — it is the signal the spec's decision gate watches.
- **Do not add** coordinates, segmentation, a reranker, curriculum, or novel-allele rejection — all are explicit v1 non-goals (see spec).
- **`stream_records(n=None, ...)`** streams indefinitely; the training loop bounds it with `zip(range(steps), stream)`.
- **GPU etiquette:** the user runs Terraria on the same GPU — use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, keep batch sizes modest, and never kill other GPU processes.
