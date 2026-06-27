# Cross-Attention Match Integration Implementation Plan (LLM-Encoder Aligner — Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `xattn_match` — the function that gathers a per-read candidate pool's germline reps from `encode_reference` output, runs the `CrossAttnMatcher`, and decodes per-gene allele logits + germline start/end coordinates + the chosen global allele index.

**Architecture:** Pure tensor glue between the cached reference embeddings and the matcher. Given segment reps + a candidate-index tensor `(B,C)`, it index-gathers `(B,C,Lg,d)` germline reps, calls the matcher, picks the best candidate, and reads that candidate's germline-coordinate argmaxes. No encoding, retrieval, or training here — those are Plan 3 (full model assembly). Tests lock plumbing/shape/decode correctness; learned ranking is a Plan 5 eval gate.

**Tech Stack:** Python 3.12, PyTorch, `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run everything via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- Consumes `CrossAttnMatcher` (Plan 1, `src/alignair/nn/heads/cross_attn_matcher.py`) — `forward(seg_reps, seg_mask, cand_reps, cand_mask) -> (match (B,C), gstart (B,C,Lg), gend (B,C,Lg))`.
- Reference embeddings come from `model.encode_reference(rs)[gene]` → `pos_reps (K,Lg,d)`, `pos_mask (K,Lg)` (bool, True=valid). `K` = number of alleles for the gene.
- `cand_idx (B,C)` long: per-read candidate allele indices into `[0,K)` (the retrieval ∪ seed pool; supplied by the caller — Plan 3 builds it). Padding within a row is allowed by repeating an index; the matcher scores each independently.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Modify `src/alignair/nn/heads/cross_attn_matcher.py` — add module-level `xattn_match(...)` next to `CrossAttnMatcher` (they are always used together).
- Create `tests/alignair/nn/test_xattn_match.py` — plumbing/shape/decode tests.

---

### Task 1: `xattn_match` gather + matcher + decode

**Files:**
- Modify: `src/alignair/nn/heads/cross_attn_matcher.py`
- Test: `tests/alignair/nn/test_xattn_match.py`

**Interfaces:**
- Produces: `xattn_match(matcher, seg_reps, seg_mask, pos_reps, pos_mask, cand_idx) -> dict` with keys:
  - `allele_logits (B,C)` — the matcher's per-candidate match scores
  - `best_idx (B,)` — argmax candidate *within the pool*
  - `best_global_idx (B,)` — the global allele index `cand_idx[b, best_idx[b]]`
  - `germ_start (B,)`, `germ_end (B,)` — argmax germline position of the chosen candidate's pointers
  - `gstart_logits (B,C,Lg)`, `gend_logits (B,C,Lg)` — full pointer logits (for the training loss)

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/nn/test_xattn_match.py
import torch
from alignair.nn.heads.cross_attn_matcher import CrossAttnMatcher, xattn_match


def _setup(B=2, C=3, S=5, K=6, Lg=7, d=16):
    torch.manual_seed(0)
    matcher = CrossAttnMatcher(d_model=d, nhead=4)
    seg = torch.randn(B, S, d)
    sm = torch.ones(B, S, dtype=torch.bool)
    pos_reps = torch.randn(K, Lg, d)
    pos_mask = torch.ones(K, Lg, dtype=torch.bool)
    cand_idx = torch.tensor([[0, 3, 5], [1, 2, 4]])[:B, :C]
    return matcher, seg, sm, pos_reps, pos_mask, cand_idx


def test_output_keys_and_shapes():
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    assert out["allele_logits"].shape == (2, 3)
    assert out["best_idx"].shape == (2,) and out["best_global_idx"].shape == (2,)
    assert out["germ_start"].shape == (2,) and out["germ_end"].shape == (2,)
    assert out["gstart_logits"].shape == (2, 3, 7)


def test_best_global_idx_maps_through_cand_idx():
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    bi = out["best_idx"]
    expected_global = ci[torch.arange(2), bi]
    assert torch.equal(out["best_global_idx"], expected_global)


def test_germ_coords_are_argmax_of_chosen_candidate_pointers():
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    bi = out["best_idx"]
    chosen_gs = out["gstart_logits"][torch.arange(2), bi]      # (B,Lg)
    assert torch.equal(out["germ_start"], chosen_gs.argmax(-1))


def test_gathered_candidate_reps_match_reference():
    # the gather must pull pos_reps[cand_idx]; verify via a matcher-independent identity:
    # building cand_reps directly and calling the matcher gives the same logits.
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    cand_reps = pr[ci]                                         # (B,C,Lg,d)
    cand_mask = pm[ci]
    match, _, _ = matcher(seg, sm, cand_reps, cand_mask)
    assert torch.allclose(out["allele_logits"], match)


def test_differentiable_through_seg_reps():
    matcher, seg, sm, pr, pm, ci = _setup()
    seg.requires_grad_(True)
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    out["allele_logits"].sum().backward()
    assert seg.grad is not None and torch.isfinite(seg.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_xattn_match.py -q`
Expected: FAIL with `ImportError: cannot import name 'xattn_match'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/alignair/nn/heads/cross_attn_matcher.py
def xattn_match(matcher, seg_reps, seg_mask, pos_reps, pos_mask, cand_idx):
    """Gather a per-read candidate pool's germline reps, run the CrossAttnMatcher, and decode
    allele logits + germline coords + the chosen global allele index.

    seg_reps (B,S,d), seg_mask (B,S); pos_reps (K,Lg,d), pos_mask (K,Lg); cand_idx (B,C) long.
    Returns dict(allele_logits, best_idx, best_global_idx, germ_start, germ_end,
                 gstart_logits, gend_logits)."""
    B, C = cand_idx.shape
    cand_reps = pos_reps[cand_idx]                                # (B,C,Lg,d)
    cand_mask = pos_mask[cand_idx]                                # (B,C,Lg)
    match, gstart, gend = matcher(seg_reps, seg_mask, cand_reps, cand_mask)
    bi = torch.arange(B, device=cand_idx.device)
    best = match.argmax(dim=1)                                    # (B,)
    return {
        "allele_logits": match,
        "best_idx": best,
        "best_global_idx": cand_idx[bi, best],
        "germ_start": gstart[bi, best].argmax(dim=-1),
        "germ_end": gend[bi, best].argmax(dim=-1),
        "gstart_logits": gstart,
        "gend_logits": gend,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_xattn_match.py -q`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/nn/heads/cross_attn_matcher.py tests/alignair/nn/test_xattn_match.py
git commit -m "nn: xattn_match — gather candidate germline reps, run matcher, decode allele + coords"
```

---

### Task 2: Real-encoder integration smoke (shapes end-to-end)

**Files:**
- Test: `tests/alignair/nn/test_xattn_match.py`

**Interfaces:**
- Consumes: `SharedNucleotideEncoder` (`src/alignair/nn/encoder/shared.py`: `forward_positions(tokens, mask, token_type)`, with `READ`/`GERMLINE` class attrs), `pad_tokenize` (`src/alignair/data/tokenizer.py`), `xattn_match`.

- [ ] **Step 1: Write the test**

```python
# append to tests/alignair/nn/test_xattn_match.py
def test_real_encoder_end_to_end_shapes():
    # encode a few germlines (GERMLINE type) + a read (READ type) with the real encoder, then run
    # xattn_match on a constructed candidate pool. Locks that the real reps flow through cleanly.
    from alignair.nn.encoder.shared import SharedNucleotideEncoder
    from alignair.data.tokenizer import pad_tokenize
    torch.manual_seed(0)
    d = 32
    enc = SharedNucleotideEncoder(d_model=d, n_layers=1, nhead=4).eval()
    germlines = ["ACGTACGTACGTACGT", "ACGTACGTACGTACGA", "TTTTGGGGCCCCAAAA"]
    gtok, gmsk = pad_tokenize(germlines)
    with torch.no_grad():
        pos_reps = enc.forward_positions(gtok, gmsk, SharedNucleotideEncoder.GERMLINE)  # (3,Lg,d)
        rtok, rmsk = pad_tokenize([germlines[0]])                                        # read == allele 0
        seg = enc.forward_positions(rtok, rmsk, SharedNucleotideEncoder.READ)           # (1,S,d)
    matcher = CrossAttnMatcher(d_model=d, nhead=4)
    cand_idx = torch.tensor([[1, 0, 2]])                                                 # pool incl. true (0)
    out = xattn_match(matcher, seg, rmsk, pos_reps, gmsk, cand_idx)
    assert out["allele_logits"].shape == (1, 3)
    assert int(out["best_global_idx"][0]) in (0, 1, 2)
    assert 0 <= int(out["germ_start"][0]) < pos_reps.shape[1]
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/test_xattn_match.py::test_real_encoder_end_to_end_shapes -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/alignair/nn/test_xattn_match.py
git commit -m "nn: real-encoder end-to-end shape smoke for xattn_match"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/nn/ -q` is green.
- `xattn_match` gathers `pos_reps[cand_idx]`, runs the matcher, and returns allele logits + correctly-mapped global index + germline-coordinate argmaxes; outputs are differentiable; real encoder reps flow through.

## Follow-on plans (not this plan)

1. **Full model forward** — a model class that runs encoder → segmentation/orientation heads →
   `extract_segment` → learned retrieval ∪ `align/seed_prefilter.py` to build `cand_idx` → `xattn_match`
   per gene, returning the four head outputs end to end.
2. **Derivations module**, **training loop + losses**, **eval/gates** — as in the spec.
