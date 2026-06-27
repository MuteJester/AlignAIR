# XAttnAligner Full-Model Forward Implementation Plan (LLM-Encoder Aligner — Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `XAttnAligner` — a model whose `forward(tokens, mask, ref_emb)` runs encoder → orientation + region/boundary (query-span) heads → per-gene retrieval ∪ k-mer-seed candidate pool → `xattn_match` → the four head outputs (orientation, query span, germline span, allele) end to end.

**Architecture:** Pure composition of existing modules (`SharedNucleotideEncoder`, `OrientationHead`, `RegionMaskSpanDecoder`, `AlleleMatchingHead`) plus the new `CrossAttnMatcher`/`xattn_match`. No new training or losses (Plan 4); no derivations (Plan 5). Tests lock output shapes + dynamic-genotype masking on a real tiny reference; learned behavior is a later eval gate.

**Tech Stack:** Python 3.12, PyTorch, GenAIRR (for a tiny real reference), `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- Reuse, do not re-implement: `SharedNucleotideEncoder` (`forward_positions(tokens, mask, token_type)`, `.proj`, `READ`/`GERMLINE`), `OrientationHead(d).forward(tokens, mask)->(B,4)`, `apply_orientation(tokens, mask, ids)`, `RegionMaskSpanDecoder(d_model, nhead).forward(reps, mask)->{region_logits,start_logits,end_logits}`, `AlleleMatchingHead().forward(query (B,d), candidates (K,d), candidate_mask=None)->(B,K)`, `extract_segment(reps, mask, region_labels, gene)->(seg,seg_mask)`, `CrossAttnMatcher`, `xattn_match` (Plans 1–2).
- `ref_emb = model.encode_reference(rs)` → per gene `{embeddings (K,d) normalized, pos_reps (K,Lg,d), pos_mask (K,Lg), pos_tok (K,Lg)}`.
- Genes: `["V","J"] + (["D"] if "D" in ref_emb else [])`. Region label ids come from `REGION_INDEX` in `core/dnalignair.py`.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Create `src/alignair/core/xattn_aligner.py` — `XAttnAligner(nn.Module)`.
- Create `tests/alignair/core/test_xattn_aligner.py` — forward shape + genotype-mask tests.

---

### Task 1: `XAttnAligner` — init, encode_reference, forward

**Files:**
- Create: `src/alignair/core/xattn_aligner.py`
- Test: `tests/alignair/core/test_xattn_aligner.py`

**Interfaces:**
- Produces: `XAttnAligner(config)` where `config` has `d_model, n_layers, nhead` (a `DNAlignAIRConfig` works). Methods:
  - `encode_reference(reference_set) -> dict` (per-gene embeddings/pos_reps/pos_mask).
  - `forward(tokens, mask, ref_emb, orientation_ids=None, candidate_masks=None, topk=8) -> dict` with keys: `orientation_logits (B,4)`, `region_logits (B,L,R)`, `boundary {start,end}` (each `{gene:(B,L)}`), `reps (B,L,d)`, and per gene `match[gene]` = the `xattn_match` dict (`allele_logits (B,C)`, `best_global_idx (B,)`, `germ_start (B,)`, `germ_end (B,)`, ...).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/core/test_xattn_aligner.py
import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.reference.reference_set import ReferenceSet
from alignair.core.xattn_aligner import XAttnAligner
from alignair.data.tokenizer import pad_tokenize


def _model_and_ref():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64)
    model = XAttnAligner(cfg).eval()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)   # V/J only -> small
    ref_emb = model.encode_reference(rs)
    return model, rs, ref_emb


def test_forward_produces_four_heads_with_shapes():
    model, rs, ref_emb = _model_and_ref()
    reads = ["ACGTACGT" * 12, "TTGCAACGTACG" * 8]
    tok, msk = pad_tokenize(reads)
    with torch.no_grad():
        out = model(tok, msk, ref_emb, topk=8)
    B = len(reads)
    assert out["orientation_logits"].shape[0] == B
    assert out["region_logits"].shape[0] == B
    for g in ("V", "J"):
        m = out["match"][g]
        assert m["allele_logits"].shape[0] == B and m["allele_logits"].shape[1] == 8
        assert m["best_global_idx"].shape == (B,)
        assert m["germ_start"].shape == (B,) and m["germ_end"].shape == (B,)
        assert int(m["best_global_idx"].max()) < len(rs.gene(g).names)


def test_genotype_mask_restricts_candidate_pool():
    model, rs, ref_emb = _model_and_ref()
    reads = ["ACGTACGT" * 12]
    tok, msk = pad_tokenize(reads)
    K = len(rs.gene("V").names)
    allowed = torch.zeros(K, dtype=torch.bool); allowed[:3] = True       # only first 3 V alleles
    with torch.no_grad():
        out = model(tok, msk, ref_emb, candidate_masks={"V": allowed}, topk=8)
    assert int(out["match"]["V"]["best_global_idx"][0]) in (0, 1, 2)     # call stays in genotype
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_xattn_aligner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.core.xattn_aligner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/core/xattn_aligner.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.encoder.shared import SharedNucleotideEncoder
from ..nn.heads.orientation import OrientationHead, apply_orientation
from ..nn.heads.region_decoder import RegionMaskSpanDecoder
from ..nn.heads.matching import AlleleMatchingHead
from ..nn.heads.cross_attn_matcher import CrossAttnMatcher, xattn_match
from ..core.dnalignair import extract_segment
from ..data.tokenizer import pad_tokenize


def _masked_mean(reps, mask):
    m = mask.unsqueeze(-1).to(reps.dtype)
    return (reps * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class XAttnAligner(nn.Module):
    """LLM-encoder aligner: shared encoder + orientation/region heads + retrieval-top-k candidate
    pool + token-level cross-attention matcher. Reference is an input (encoded + cached); the four
    head outputs (orientation, query span, germline span, allele) come from one forward pass."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.orientation_head = OrientationHead(d=getattr(config, "orientation_dim", 64))
        self.backbone = SharedNucleotideEncoder(d_model=d, n_layers=config.n_layers, nhead=config.nhead)
        self.region_tagger = RegionMaskSpanDecoder(d_model=d, nhead=config.nhead)
        self.matching = AlleleMatchingHead()
        self.matcher = CrossAttnMatcher(d_model=d, nhead=config.nhead)

    def encode_reference(self, reference_set) -> dict:
        device = next(self.parameters()).device
        out = {}
        for gene, ref in reference_set.genes.items():
            tok, msk = pad_tokenize(ref.sequences)
            tok, msk = tok.to(device), msk.to(device)
            pooled = self.backbone(tok, msk, token_type=SharedNucleotideEncoder.GERMLINE)   # (K,d) norm
            pos = self.backbone.forward_positions(tok, msk, SharedNucleotideEncoder.GERMLINE)  # (K,Lg,d)
            out[gene] = {"embeddings": pooled, "pos_reps": pos, "pos_mask": msk, "pos_tok": tok}
        return out

    def forward(self, tokens, mask, ref_emb, orientation_ids=None,
                candidate_masks=None, topk: int = 8) -> dict:
        orientation_logits = self.orientation_head(tokens, mask)
        t = orientation_ids if orientation_ids is not None else orientation_logits.argmax(dim=-1)
        canon = apply_orientation(tokens, mask, t)
        reps = self.backbone.forward_positions(canon, mask)
        rdec = self.region_tagger(reps, mask)
        region_labels = rdec["region_logits"].argmax(dim=-1)
        genes = ["V", "J"] + (["D"] if "D" in ref_emb else [])
        match = {}
        for g in genes:
            emb = ref_emb[g]
            seg, seg_mask = extract_segment(reps, mask, region_labels, g)
            query = F.normalize(self.backbone.proj(_masked_mean(seg, seg_mask)), dim=-1)    # (B,d)
            cm = candidate_masks.get(g) if candidate_masks else None
            cm = cm.to(query.device) if cm is not None else None
            scores = self.matching(query, emb["embeddings"], candidate_mask=cm)             # (B,K)
            k = min(topk, scores.shape[-1])
            cand_idx = scores.topk(k, dim=-1).indices                                       # (B,k)
            match[g] = xattn_match(self.matcher, seg, seg_mask,
                                   emb["pos_reps"], emb["pos_mask"], cand_idx)
        return {"orientation_logits": orientation_logits,
                "region_logits": rdec["region_logits"],
                "boundary": {"start": rdec["start_logits"], "end": rdec["end_logits"]},
                "reps": reps, "canon_tokens": canon, "match": match}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_xattn_aligner.py -q`
Expected: both PASS. (If `RegionMaskSpanDecoder` returns different key names, adjust `rdec[...]` to match its actual return dict — verify against `src/alignair/nn/heads/region_decoder.py:55`.)

- [ ] **Step 5: Commit**

```bash
git add src/alignair/core/xattn_aligner.py tests/alignair/core/test_xattn_aligner.py
git commit -m "core: XAttnAligner full forward (encoder + heads + retrieval-topk -> xattn_match)"
```

---

### Task 2: K-mer seed admission into the candidate pool

**Files:**
- Modify: `src/alignair/core/xattn_aligner.py`
- Test: `tests/alignair/core/test_xattn_aligner.py`

**Interfaces:**
- Consumes: `align/seed_prefilter.py` `SeedPrefilter(reference_set, k).candidates(segment_str, gene, m, allowed)`. Adds the non-learned seed pool to `cand_idx` so a divergent/novel allele the retrieval misranks can still enter the matcher.
- Produces: `forward(..., seed_m: int = 4, reference_set=None)` — when `reference_set` is given, the per-gene pool is `retrieval_topk ∪ seed_topm`, deduped and padded back to a fixed width; otherwise retrieval-only (Task 1 behavior unchanged).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/alignair/core/test_xattn_aligner.py
def test_seed_admission_includes_a_retrieval_missed_allele():
    # feed a read that IS V allele `true_idx`; force retrieval to miss it (tiny untrained encoder),
    # and confirm the seed path admits it into the pool so best_global_idx CAN be it.
    model, rs, ref_emb = _model_and_ref()
    vg = rs.gene("V")
    true_idx = 5
    read = vg.sequences[true_idx]
    tok, msk = pad_tokenize([read])
    with torch.no_grad():
        out = model(tok, msk, ref_emb, topk=4, seed_m=8, reference_set=rs)
    pool = out["match"]["V"]["pool_idx"][0].tolist() if torch.is_tensor(
        out["match"]["V"]["pool_idx"]) else out["match"]["V"]["pool_idx"][0]
    assert true_idx in pool                                  # admitted by the seed prefilter
```

Note: `xattn_match` must surface `pool_idx`. It already returns the matcher outputs; add `pool_idx=cand_idx` to its return dict (one line in `cross_attn_matcher.py`) and assert it here.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_xattn_aligner.py::test_seed_admission_includes_a_retrieval_missed_allele -q`
Expected: FAIL (seed pool not yet wired; `pool_idx` may be missing).

- [ ] **Step 3: Write minimal implementation**

Add `pool_idx` to `xattn_match`'s return (in `cross_attn_matcher.py`): `"pool_idx": cand_idx,`.

Then in `XAttnAligner.forward`, accept `seed_m: int = 4, reference_set=None`, build/cache a `SeedPrefilter`, and union its candidates into `cand_idx`:

```python
        # in __init__: self._seed = None
        # at top of forward, after region_labels:
        if reference_set is not None and self._seed is None:
            from ..align.seed_prefilter import SeedPrefilter
            self._seed = SeedPrefilter(reference_set, k=11)
        # inside the per-gene loop, after computing retrieval cand_idx (B,k):
        if reference_set is not None:
            from ..core.dnalignair import extract_segment_tokens
            from ..data.tokenizer import TOKEN_DICT
            inv = {v: k_ for k_, v in TOKEN_DICT.items()}
            seg_tok, seg_tm = extract_segment_tokens(canon, mask, region_labels, g)
            allowed = set(int(i) for i in cm.nonzero().flatten().tolist()) if cm is not None else None
            rows = []
            for b in range(cand_idx.shape[0]):
                s = "".join(inv.get(int(x), "N") for x, mok in zip(seg_tok[b], seg_tm[b]) if mok)
                seeds = self._seed.candidates(s, g, seed_m, allowed=allowed)
                base = cand_idx[b].tolist()
                merged = list(dict.fromkeys(base + seeds))[:len(base) + seed_m]
                while len(merged) < len(base) + seed_m:
                    merged.append(merged[-1] if merged else 0)
                rows.append(merged)
            cand_idx = torch.tensor(rows, dtype=torch.long, device=cand_idx.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_xattn_aligner.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/core/xattn_aligner.py src/alignair/nn/heads/cross_attn_matcher.py tests/alignair/core/test_xattn_aligner.py
git commit -m "core: union k-mer seed candidates into the XAttnAligner pool (novel-allele admission)"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/core/test_xattn_aligner.py tests/alignair/nn/ -q` is green.
- `XAttnAligner.forward` returns the four head outputs at correct shapes on a real tiny IGK reference; genotype masks restrict the call; the k-mer seed path admits a retrieval-missed allele.

## Follow-on plans (not this plan)

1. **Derivations module** — `match` + boundary → full GenAIRR record (vectorized).
2. **Training loop + losses** — set-NCE on `allele_logits`, CE on spans + germline pointers + orientation, on the gym with embargo'd alleles.
3. **Eval/gates** — assay, IgBLAST head-to-head, embargo, throughput.
