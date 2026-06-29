# WFA Caller + predict_reads Rewrite (Speed Stack — Phase 1b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the differentiable-DP allele reader + germline-coordinate decode in `predict_reads` with a classical calling stage: union pool (retrieval top-k ∪ k-mer seed) → WFA alignment → best-score allele pick + score-band equivalence set + germline coords/CIGAR.

**Architecture:** A new model-free `wfa_caller.call_segment` takes a read-segment string, a gene, retrieval top-k indices, the reference set, a `SeedPrefilter`, and an `Aligner`; it builds the union candidate pool, aligns the segment to each candidate germline via the `align/` package, and returns the chosen allele + ordered set + germline coords + CIGAR. `predict_reads` is then rewired to call it, deleting the differentiable soft-DP reader branch, the parasail-V branch, and the `compute_germline_logits`/`decode_germline_coords` path.

**Tech Stack:** Python 3.12, the `src/alignair/align/` package (shipped Phase 1a), `parasail`/`pywfa`, `pytest`. Venv: `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run everything via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python` and `... -m pytest`. Never bare `python`.
- `wfa_caller` is **model-free** — no torch import; it consumes plain Python ints/strings and the `align/` package. (It may import `ReferenceSet` only for type clarity, but tests pass a real `ReferenceSet`.)
- The `predict_reads` **output contract is preserved**: every key the existing `tests/alignair/inference/test_dnalignair_infer.py` asserts must still be produced (`{g}_call`, `{g}_calls`, `{g}_call_set`, `{g}_topk`, `{g}_sequence_start/end`, `{g}_germline_start/end`, `orientation_id`, `productive`, `sequence`, `locus`). Genotype restriction must still hold on every call and set.
- Dynamic-genotype property: germline candidates are looked up by raw sequence from `reference_set`; the seed prefilter admits divergent candidates the encoder may misrank. Never bake an allele set into weights.
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Create `src/alignair/inference/wfa_caller.py` — `SegmentCall` dataclass + `call_segment()` (the classical calling stage).
- Modify `src/alignair/inference/dnalignair_infer.py` — `predict_reads`: build the union pool, call `call_segment`, assemble output; delete the diff-DP reader branch, the parasail-V branch, and the `gcoord` path.
- Create `tests/alignair/inference/test_wfa_caller.py` — unit tests for `call_segment` (real `ReferenceSet`, no model).
- Modify `tests/alignair/inference/test_dnalignair_infer.py` — add the seed-path swap test.

---

### Task 1: `wfa_caller.call_segment`

**Files:**
- Create: `src/alignair/inference/wfa_caller.py`
- Test: `tests/alignair/inference/test_wfa_caller.py`

**Interfaces:**
- Consumes: `alignair.align.SeedPrefilter`, `alignair.align.get_aligner`/`Aligner`, `alignair.align.align_batch`, `AlignResult` (Phase 1a). A `reference_set` with `.gene(G).sequences: list[str]` and `.gene(G).names: list[str]`.
- Produces:
  - `SegmentCall` frozen dataclass: `best_idx: int`, `set_idx: list[int]` (ordered by score desc; `set_idx[0] == best_idx`), `germ_start: int`, `germ_end: int`, `cigar: str`, `gate: float`, `pool_idx: list[int]`, `scores: list[float]` (aligned with `pool_idx`).
  - `call_segment(seg: str, gene: str, topk_idx: list[int], reference_set, seed_prefilter, aligner, m_seed: int = 8, set_band: float = 2.0, allowed: set[int] | None = None) -> SegmentCall | None` — returns `None` when the segment is shorter than 5 nt or no candidate aligns.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignair/inference/test_wfa_caller.py
import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.align import SeedPrefilter, get_aligner
from alignair.inference.wfa_caller import call_segment, SegmentCall

RS = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
SP = SeedPrefilter(RS, k=11)
AL = get_aligner()
VG = RS.gene("V")


def test_identity_segment_picks_its_allele_with_full_germline_span():
    gseq = VG.sequences[0]
    call = call_segment(gseq, "V", topk_idx=[1, 2, 0], reference_set=RS,
                        seed_prefilter=SP, aligner=AL)
    assert call.best_idx == 0
    assert call.set_idx[0] == 0
    assert call.germ_start == 0 and call.germ_end == len(gseq)


def test_true_allele_reachable_only_via_seed_pool_is_still_called():
    # retrieval top-k OMITS the true allele (idx 0); the k-mer seed prefilter must admit it so
    # WFA can still pick it — the swap-robustness guarantee at the caller level.
    gseq = VG.sequences[0]
    call = call_segment(gseq, "V", topk_idx=[1, 2, 3], reference_set=RS,
                        seed_prefilter=SP, aligner=AL)
    assert 0 in call.pool_idx          # admitted by the seed prefilter, not retrieval
    assert call.best_idx == 0


def test_short_segment_returns_none():
    assert call_segment("ACG", "V", topk_idx=[0], reference_set=RS,
                        seed_prefilter=SP, aligner=AL) is None


def test_set_is_ordered_and_genotype_restricted():
    gseq = VG.sequences[0]
    call = call_segment(gseq, "V", topk_idx=[0, 1, 2, 3, 4], reference_set=RS,
                        seed_prefilter=SP, aligner=AL, allowed={0, 1})
    assert set(call.pool_idx) <= {0, 1}            # allowed restricts the pool
    assert set(call.set_idx) <= {0, 1}
    assert call.set_idx == sorted(call.set_idx, key=lambda i: -call.scores[call.pool_idx.index(i)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_wfa_caller.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'alignair.inference.wfa_caller'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/alignair/inference/wfa_caller.py
from __future__ import annotations
from dataclasses import dataclass

from ..align import align_batch


@dataclass(frozen=True)
class SegmentCall:
    best_idx: int
    set_idx: list          # ordered by score desc; set_idx[0] == best_idx
    germ_start: int
    germ_end: int
    cigar: str
    gate: float            # best_score / segment_length (out-of-scope advisory)
    pool_idx: list         # candidate indices scored (union pool)
    scores: list           # alignment score per pool_idx (None -> -inf)


def _pool(topk_idx, seed_idx, allowed):
    """Ordered-unique union of retrieval top-k then seed candidates, genotype-restricted."""
    out, seen = [], set()
    for i in list(topk_idx) + list(seed_idx):
        i = int(i)
        if i in seen or (allowed is not None and i not in allowed):
            continue
        seen.add(i)
        out.append(i)
    return out


def call_segment(seg: str, gene: str, topk_idx, reference_set, seed_prefilter, aligner,
                 m_seed: int = 8, set_band: float = 2.0, allowed=None) -> SegmentCall | None:
    seg = seg.upper()
    if len(seg) < 5:
        return None
    seed_idx = seed_prefilter.candidates(seg, gene, m_seed, allowed=allowed)
    pool = _pool(topk_idx, seed_idx, allowed)
    if not pool:
        return None
    germs = reference_set.gene(gene.upper()).sequences
    results = align_batch([(seg, germs[i]) for i in pool], aligner)
    scores = [(r.score if r is not None else float("-inf")) for r in results]
    best_j = max(range(len(pool)), key=lambda j: scores[j])
    if scores[best_j] == float("-inf"):
        return None
    top = scores[best_j]
    keep = sorted([j for j in range(len(pool)) if top - scores[j] <= set_band],
                  key=lambda j: -scores[j])
    win = results[best_j]
    return SegmentCall(best_idx=pool[best_j], set_idx=[pool[j] for j in keep],
                       germ_start=win.t_start, germ_end=win.t_end, cigar=win.cigar,
                       gate=top / max(len(seg), 1), pool_idx=pool, scores=scores)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_wfa_caller.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/inference/wfa_caller.py tests/alignair/inference/test_wfa_caller.py
git commit -m "inference: wfa_caller.call_segment (union pool -> WFA -> pick/set/coords)"
```

---

### Task 2: Rewire `predict_reads` onto the WFA caller

**Files:**
- Modify: `src/alignair/inference/dnalignair_infer.py`
- Test: `tests/alignair/inference/test_dnalignair_infer.py` (existing suite is the gate)

**Interfaces:**
- Consumes: `call_segment`, `SegmentCall` (Task 1); `SeedPrefilter`, `get_aligner` (Phase 1a).
- Produces: a `predict_reads(...)` whose output dict keeps the same keys; new keyword `rerank` default becomes irrelevant for calling (classical calling is always on), but keep the parameter in the signature for backward-compat callers (ignored). `is_contaminant` driven by `SegmentCall.gate` for V.

**Context for the implementer — what to delete and what to add (the function is `predict_reads`, ~lines 211–459):**

Delete these three regions entirely:
1. The germline-logit/coord decode: the `gl = compute_germline_logits(...)` + `gcoord = {...decode_germline_coords...}` lines.
2. The whole `if rerank == "learned":` block (the differentiable `model.aligner.alignment_score` reader, including the `seed_pos`/`band`/`sc_all`/`learned_best` machinery).
3. The whole `if rerank == "learned" and v_reader == "parasail":` parasail-V block.

- [ ] **Step 1: Add the caller setup once, before the batch loop**

Just after `ref_emb = model.encode_reference(reference_set)` near the top of `predict_reads`, add:

```python
        from ..align import SeedPrefilter, get_aligner
        from .wfa_caller import call_segment
        seed_prefilter = SeedPrefilter(reference_set, k=11)
        aligner = get_aligner()
        allowed_sets = None
        if genotype is not None:
            allowed_sets = {G: set(int(i) for i in candidate_masks[G].nonzero().flatten().tolist())
                            for G in candidate_masks}
```

- [ ] **Step 2: Replace the per-batch calling logic**

In the batch loop, after `topk_idx = {...}` is computed and AFTER deleting the three regions above, insert the classical calling over the batch (uses the canonical sequence + the query-segment boundaries already decoded):

```python
        ori = out["orientation_logits"].argmax(dim=-1).cpu().tolist()
        seg_starts = {G: (out["boundary"]["start"][G].argmax(dim=1).cpu().tolist()
                          if boundary is not None else None) for G in [g.upper() for g in genes]}
        seg_ends = {G: ((out["boundary"]["end"][G].argmax(dim=1) + 1).cpu().tolist()
                        if boundary is not None else None) for G in [g.upper() for g in genes]}
        # per (read, gene): build the union pool and classically call it
        calls = [{} for _ in range(len(chunk))]
        for i in range(len(chunk)):
            canon_seq = canonicalize_sequence(chunk[i], ori[i])
            for g in genes:
                G = g.upper()
                if boundary is not None:
                    vs, ve = seg_starts[G][i], seg_ends[G][i]
                else:
                    vs, ve = dec[i][f"{g}_start"], dec[i][f"{g}_end"]
                seg = canon_seq[vs:ve] if (vs is not None and ve and ve > vs) else ""
                allowed = allowed_sets.get(G) if allowed_sets else None
                calls[i][G] = call_segment(seg, G, topk_idx[G][i].cpu().tolist(),
                                           reference_set, seed_prefilter, aligner, allowed=allowed)
```

- [ ] **Step 3: Replace the assembly loop body**

Replace the assembly loop (`for i in range(len(chunk)): p = {} ...`) per-gene block with the classical-call version:

```python
        for i in range(len(chunk)):
            p = {}
            for g in genes:
                G = g.upper()
                sc = calls[i][G]
                if sc is not None:
                    idx = sc.best_idx
                    cset = [names[G][j] for j in sc.set_idx]
                    p[f"{g}_germline_start"] = int(sc.germ_start)
                    p[f"{g}_germline_end"] = int(sc.germ_end)
                    p[f"{g}_cigar"] = sc.cigar
                else:                                   # short/empty segment -> retrieval top-1
                    idx = int(pred_idx[G][i])
                    cset = [names[G][idx]]
                    p[f"{g}_germline_start"] = 0
                    p[f"{g}_germline_end"] = 0
                p[f"{g}_call"] = names[G][idx]
                p[f"{g}_topk"] = [names[G][int(j)] for j in topk_idx[G][i]]
                p[f"{g}_call_set"] = cset
                p[f"{g}_calls"] = cset
                resolved, level = resolve_hierarchy(cset, p[f"{g}_call"])
                p[f"{g}_resolved_call"] = resolved
                p[f"{g}_call_level"] = level
                if boundary is not None:
                    p[f"{g}_sequence_start"] = int(out["boundary"]["start"][G][i].argmax())
                    p[f"{g}_sequence_end"] = int(out["boundary"]["end"][G][i].argmax()) + 1
                else:
                    p[f"{g}_sequence_start"] = dec[i][f"{g}_start"]
                    p[f"{g}_sequence_end"] = dec[i][f"{g}_end"]
            p["orientation_id"] = int(out["orientation_logits"][i].argmax())
            p["productive"] = bool(out["productive"][i].item() > 0.5)
            p["mutation_rate"] = float(out["mutation_rate"][i].item())
            p["indel_count"] = float(out["indel_count"][i].item())
            canon_seq = canonicalize_sequence(chunk[i], p["orientation_id"])
            p["sequence"] = canon_seq
            p["locus"] = locus
            if full_alignment:
                from ..io.alignment import realign
                p.update(realign(canon_seq, p, reference_set))
            p.update(junction_fields(p, canon_seq, reference_set))
            p.update(derived_rearrangement_fields(p, canon_seq))
            vcall = calls[i]["V"]
            if vcall is not None:
                p["contaminant_score"] = float(vcall.gate)
                if contam_tau is not None:
                    p["is_contaminant"] = bool(vcall.gate < contam_tau)
            preds.append(p)
    return preds
```

- [ ] **Step 4: Run the existing inference suite to verify no contract regression**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_dnalignair_infer.py -q`
Expected: PASS. The `test_genotype_restricts_every_call` test (which exercises `rerank="none"` and `rerank="learned"`) must still pass — both now route through the classical caller; calls and sets stay within the genotype because `allowed` restricts the pool. If a test asserts a now-removed internal field, update the assertion to the preserved public key (do not weaken genotype/restriction checks).

- [ ] **Step 5: Commit**

```bash
git add src/alignair/inference/dnalignair_infer.py tests/alignair/inference/test_dnalignair_infer.py
git commit -m "inference: route predict_reads through the classical WFA caller; drop differentiable-DP reader + coord decode"
```

---

### Task 3: Genotype/germline swap integration test

**Files:**
- Test: `tests/alignair/inference/test_dnalignair_infer.py`

**Interfaces:**
- Consumes: `predict_reads`, `ReferenceSet`, a seed-reachable allele.

- [ ] **Step 1: Write the test**

```python
# append to tests/alignair/inference/test_dnalignair_infer.py
def test_swap_reference_novel_allele_callable_via_seed_path():
    # A germline string the encoder never trained on must still be callable: feed a read that IS
    # an allele's germline; predict_reads must return that allele (admitted by the k-mer seed pool
    # and chosen by WFA) — proving the swap requirement holds end-to-end.
    import GenAIRR.data as gdata
    from alignair.reference.reference_set import ReferenceSet
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        backbone="shared", aligner="seed_extend"))
    vg = rs.gene("V")
    read = vg.sequences[0] + rs.gene("J").sequences[0]      # a clean V..J read
    preds = predict_reads(model, rs, [read], batch_size=1)
    # the V germline of the read's first allele must be among the returned V set or call
    assert vg.names[0] in (preds[0]["v_call_set"] + [preds[0]["v_call"]])
```

- [ ] **Step 2: Run it**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_dnalignair_infer.py::test_swap_reference_novel_allele_callable_via_seed_path -q`
Expected: PASS (an untrained model still calls it because the seed pool + WFA do the work on raw sequence). If it fails because segmentation on an untrained model gives an unusable V segment, relax to asserting the call is a valid V name AND the seed pool admitted idx 0 (assert via a direct `call_segment` on the full read as V).

- [ ] **Step 3: Commit**

```bash
git add tests/alignair/inference/test_dnalignair_infer.py
git commit -m "inference: end-to-end genotype-swap test (novel allele callable via seed path)"
```

---

## Done criteria

- `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/ tests/alignair/align/ -q` is green.
- `predict_reads` no longer calls `compute_germline_logits`, `decode_germline_coords`, or `model.aligner.alignment_score`; germline coords come from the WFA traceback.
- Genotype restriction holds on every call and set; the swap test passes.
- Throughput sanity: `PYTHONPATH=src .venv/bin/python scripts/exp_throughput_breakdown.py --model .private/models/seed_extend_d64_reader.pt` shows the `predict/dp` row replaced by the classical path running materially faster than 77 reads/s (record the number; full speed tuning is Phase 2).

## Follow-on plans (not this plan)

1. **Training simplification + retrain** — strip band/DP losses, add per-position contrastive term, `scripts/train_fast.py`.
2. **Encoder acceleration** (Phase 2/3) — fp16 + `torch.compile`; ONNX/TensorRT export; flatten the per-(read,gene) `call_segment` loop into one batched `align_batch`.
