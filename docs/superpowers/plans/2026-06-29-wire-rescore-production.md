# Wire Classical Rescore into XAttnAligner Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `predict_reads_xattn` use classical raw-base rescore (the `align/` package) of the neural top-k for the allele call + germline coords, and derive query coords from the trained `region_logits` — banking the heavy-SHM-V 0.28→0.62 win and fixing the query-coordinate bug.

**Architecture:** Per read/gene: keep the XAttnAligner forward (segmentation + neural retrieval pool), but replace the matcher's pick with `call_segment` over the neural pool (`m_seed=0`), and replace the untrained-boundary-head query coords with `decode_boundaries(region_logits)`. Germline coords + CIGAR come from `call_segment`'s WFA/parasail traceback. A `rescore` flag (default True) preserves the pure-neural path for ablation.

**Tech Stack:** Python 3.12, PyTorch, GenAIRR, the `align/` package, `pytest`. Venv: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`.

## Global Constraints

- Run via `PYTHONPATH=src /home/thomas/Desktop/AlignAIR/.venv/bin/python`; prefix GPU with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Never bare `python`.
- Reuse: `call_segment(seg, gene, topk_idx, reference_set, seed_prefilter, aligner, m_seed=0, set_band=)` → `SegmentCall(best_idx[global], set_idx[global, ordered], germ_start, germ_end, cigar, ...)`; `decode_boundaries(region_logits, mask, has_d)` → per-row `{v_start, v_end, d_start, d_end, j_start, j_end}`; `get_aligner()`, `SeedPrefilter` (`align/`); existing `canonicalize_sequence`, `junction_fields`, `derived_rearrangement_fields`, `resolve_hierarchy`.
- `call_segment`'s germline end is the last inclusive position → AIRR exclusive end = `germ_end + 1`.
- Output contract unchanged (same keys as the current `predict_reads_xattn`).
- Git commit messages: **never** include Co-Authored-By or Claude/AI attribution.
- Do not modify `.claude/settings.json` or `docs/architecture/adoption_roadmap.md`.

---

## File Structure

- Modify `src/alignair/inference/xattn_infer.py` — rewrite `predict_reads_xattn`.
- Modify `tests/alignair/inference/test_xattn_infer.py` — add a query-coord-sanity assertion (no negative spans).

---

### Task 1: Rewrite `predict_reads_xattn` (rescore + region-based query coords)

**Files:**
- Modify: `src/alignair/inference/xattn_infer.py`
- Test: `tests/alignair/inference/test_xattn_infer.py`

**Interfaces:**
- Produces: `predict_reads_xattn(..., rescore: bool = True)`. With `rescore=True` (default): allele + germline coords + CIGAR from `call_segment` over the neural pool; query coords from `decode_boundaries`. Output keys unchanged.

- [ ] **Step 1: Add the query-coord-sanity test**

```python
# append to tests/alignair/inference/test_xattn_infer.py
def test_rescore_query_coords_are_non_negative_spans():
    # the rewrite must derive query coords from region_logits (decode_boundaries), not the untrained
    # boundary head, so sequence_end >= sequence_start (no negative spans) for any gene reported.
    import torch
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    torch.manual_seed(0)
    model = XAttnAligner(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64))
    reads = ["ACGTACGT" * 30, "TTGCAACGTACG" * 20]
    preds = predict_reads_xattn(model, rs, reads, batch_size=2, rescore=True)
    for p in preds:
        for g in ("v", "d", "j"):
            assert p[f"{g}_sequence_end"] >= p[f"{g}_sequence_start"] >= 0
```

- [ ] **Step 2: Run it (fails on the current code — boundary-head coords go negative)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_xattn_infer.py::test_rescore_query_coords_are_non_negative_spans -q`
Expected: the function signature lacks `rescore=` → FAIL (TypeError) until Step 3.

- [ ] **Step 3: Rewrite `predict_reads_xattn`**

```python
# src/alignair/inference/xattn_infer.py  (replace the whole module body below the docstring)
from __future__ import annotations
import torch

from ..data.tokenizer import pad_tokenize
from ..nn.heads.region import decode_boundaries
from .dnalignair_infer import (canonicalize_sequence, junction_fields,
                               derived_rearrangement_fields, resolve_hierarchy)


@torch.no_grad()
def predict_reads_xattn(model, reference_set, reads, device=None, batch_size: int = 64,
                        topk: int = 16, seed_m: int = 0, set_band: float = 2.0,
                        cand_chunk: int = 4, locus: str = "IGH", rescore: bool = True) -> list:
    device = device or next(model.parameters()).device
    model.eval()
    ref_emb = model.encode_reference(reference_set)
    has_d = reference_set.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    names = {g.upper(): reference_set.gene(g.upper()).names for g in genes}
    if rescore:
        from ..align import SeedPrefilter, get_aligner
        from .wfa_caller import call_segment
        sp, al = SeedPrefilter(reference_set, k=11), get_aligner()
    preds = []
    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]
        tok, mask = pad_tokenize(chunk)
        tok, mask = tok.to(device), mask.to(device)
        out = model(tok, mask, ref_emb, topk=topk, seed_m=seed_m,
                    reference_set=(reference_set if seed_m > 0 else None), cand_chunk=cand_chunk)
        ori = out["orientation_logits"].argmax(-1).cpu().tolist()
        boundary = out["boundary"]
        dec = decode_boundaries(out["region_logits"], mask, has_d=has_d) if rescore else None
        for i in range(len(chunk)):
            p = {}
            canon = canonicalize_sequence(chunk[i], ori[i])
            for g in genes:
                G = g.upper()
                mg = out["match"][G]
                pool = mg["pool_idx"][i].tolist()
                if rescore:
                    qs, qe = dec[i][f"{g}_start"], dec[i][f"{g}_end"]
                else:
                    qs = int(boundary["start"][G][i].argmax())
                    qe = int(boundary["end"][G][i].argmax()) + 1
                p[f"{g}_sequence_start"] = int(qs) if qs is not None else 0
                p[f"{g}_sequence_end"] = int(qe) if (qe is not None and qe >= (qs or 0)) else int(qs or 0)
                sc = None
                if rescore:
                    seg = canon[qs:qe] if (qs is not None and qe and qe > qs) else ""
                    sc = call_segment(seg, G, pool, reference_set, sp, al, m_seed=0, set_band=set_band)
                if sc is not None:                                 # classical rescore of the neural pool
                    idx = sc.best_idx
                    cset = [names[G][j] for j in sc.set_idx]
                    p[f"{g}_germline_start"] = int(sc.germ_start)
                    p[f"{g}_germline_end"] = int(sc.germ_end) + 1
                    p[f"{g}_cigar"] = sc.cigar
                else:                                              # neural fallback (short seg / rescore off)
                    idx = int(mg["best_global_idx"][i])
                    logits = mg["allele_logits"][i].tolist()
                    top = max(logits)
                    keep = sorted([(int(pool[j]), logits[j]) for j in range(len(pool))
                                   if top - logits[j] <= set_band], key=lambda x: -x[1])
                    cset, seen = [], set()
                    for gi, _ in keep:
                        nm = names[G][gi]
                        if nm not in seen:
                            seen.add(nm); cset.append(nm)
                    if not cset:
                        cset = [names[G][idx]]
                    p[f"{g}_germline_start"] = int(mg["germ_start"][i])
                    p[f"{g}_germline_end"] = int(mg["germ_end"][i]) + 1
                p[f"{g}_call"] = names[G][idx]
                p[f"{g}_call_set"] = cset
                p[f"{g}_calls"] = cset
                resolved, level = resolve_hierarchy(cset, p[f"{g}_call"])
                p[f"{g}_resolved_call"] = resolved
                p[f"{g}_call_level"] = level
            p["orientation_id"] = int(ori[i])
            p["sequence"] = canon
            p["locus"] = locus
            p.update(junction_fields(p, canon, reference_set))
            p.update(derived_rearrangement_fields(p, canon))
            if "vj_in_frame" in p and "stop_codon" in p:
                p["productive"] = bool(p["vj_in_frame"] and not p["stop_codon"])
            preds.append(p)
    return preds
```

- [ ] **Step 4: Run the inference tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/alignair/inference/test_xattn_infer.py -q`
Expected: PASS — the existing AIRR-validity test and the new non-negative-span test both green.

- [ ] **Step 5: Commit**

```bash
git add src/alignair/inference/xattn_infer.py tests/alignair/inference/test_xattn_infer.py
git commit -m "inference: classical rescore of the neural top-k in predict_reads_xattn + region-based query coords"
```

---

### Task 2: Confirm the win at benchmark scale

**Files:** (none — a measurement run using the committed `bench_xattn.py`)

**Interfaces:** consumes `scripts/bench_xattn.py` (its `xattn_predictor` now routes through the rescored `predict_reads_xattn`).

- [ ] **Step 1: Run a moderate assay (rescore on by default) and read the per-stratum V + coords**

Run:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src .venv/bin/python scripts/bench_xattn.py \
  --model .private/models/xattn_igh.pt --n-per-stratum 2000 --n-per-focus 2000 \
  --out .private/bench/xattn_igh_rescore.json
```
Expected: ~46k cases. Confirm (a) **heavy-SHM-V `top1_in_set` ≈ 0.6** (up from 0.32), (b) clean/indel unchanged ≈ 0.97, (c) **query-coord `ss_mae` now small** (the boundary-head bug is fixed; was ~125), and the sibling `same_gene_wrong_allele` confusions drop.

- [ ] **Step 2: Commit the bench artifact pointer (if you keep it)**

```bash
git add -A && git commit -m "bench: xattn rescore assay (~46k cases) confirming heavy-SHM V lift + coord fix" || echo "nothing to commit"
```

---

## Done criteria

- `tests/alignair/inference/test_xattn_infer.py` green (AIRR-valid + non-negative query spans).
- The moderate assay shows heavy-SHM-V ≈ 0.6 (vs 0.32), no clean/indel regression, and sane query-coordinate MAE.

## Follow-on (not this plan)

- Spec + run the **retrieval-focused gym retrain** (embargo, genotype augmentation, heavy-SHM oversampling, sibling-pool fix) to lift `pool_recall` (the new ceiling, ~0.63 on heavy-SHM) — which lifts the rescored accuracy further.
