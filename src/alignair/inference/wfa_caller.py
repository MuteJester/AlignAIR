from __future__ import annotations
import math
from dataclasses import dataclass

from ..align import align_batch

_RC = str.maketrans("ACGTN", "TGCAN")
_DNA_MAT = None


@dataclass(frozen=True)
class DCall:
    idx: int               # best D germline index (global)
    inverted: bool         # True if the read's D matched the reverse-complement germline
    t_start: int           # D start, window-relative
    t_end: int             # D end, window-relative
    germ_start: int        # forward-germline coords of the matched core
    germ_end: int
    score: float
    set_idx: list          # near-best D germlines (sibling equivalence set), best first


def call_d_in_window(window: str, d_names, d_seqs, *, min_score: float = 16.0,
                     set_band: float = 2.0, inv_margin: float = 6.0) -> "DCall | None":
    """Call D by LOCAL-aligning every D germline — forward AND reverse-complement — inside the whole
    junction window [V_end:J_start]. Returns the best D with window-relative coords, or None when no
    germline clears ``min_score`` (a genuine no-D call). ``inverted`` is set only when the best RC
    germline beats the best forward germline by more than ``inv_margin`` — a deliberately conservative
    test, because the caller uses ``inverted`` to override the (clean-accurate) forward rescore, so a
    false positive costs a clean miss. Smith-Waterman (parasail sw): the observed D is a trimmed
    substring of the germline embedded in random N-region bases — both ends free on both sequences."""
    global _DNA_MAT
    import parasail
    if _DNA_MAT is None:
        _DNA_MAT = parasail.matrix_create("ACGTN", 2, -1)
    window = window.upper()
    if len(window) < 4:
        return None
    scored = []                                            # (score, idx, inverted, t_start, t_end, gs, ge)
    for idx, seq in enumerate(d_seqs):
        L = len(seq)
        for inverted, q in ((False, seq), (True, seq.translate(_RC)[::-1])):
            if len(q) < 4:
                continue
            r = parasail.sw_trace_striped_16(q, window, 3, 1, _DNA_MAT)
            ref_used = sum(1 for c in r.traceback.ref if c != "-")
            q_used = sum(1 for c in r.traceback.query if c != "-")
            t_end = int(r.end_ref) + 1
            t_start = t_end - ref_used
            qe = int(r.end_query) + 1
            qs = qe - q_used
            gs, ge = (L - qe, L - qs) if inverted else (qs, qe)   # map RC coords to forward germline
            scored.append((float(r.score), idx, inverted, t_start, t_end, gs, ge))
    if not scored:
        return None
    best_fwd = max((s for s in scored if not s[2]), key=lambda x: x[0], default=None)
    best_rc = max((s for s in scored if s[2]), key=lambda x: x[0], default=None)
    # inverted only when RC clearly beats forward (conservative — see docstring)
    if best_rc is not None and (best_fwd is None or best_rc[0] > best_fwd[0] + inv_margin):
        best, inverted = best_rc, True
    else:
        best, inverted = best_fwd, False
    if best is None or best[0] < min_score:
        return None
    set_idx, seen = [], set()                              # sibling set within band, same orientation
    for sc, idx, inv, *_ in sorted(scored, key=lambda x: -x[0]):
        if inv != inverted or best[0] - sc > set_band:
            continue
        if idx not in seen:
            seen.add(idx)
            set_idx.append(idx)
    return DCall(idx=best[1], inverted=inverted, t_start=best[3], t_end=best[4],
                 germ_start=best[5], germ_end=best[6], score=best[0], set_idx=set_idx)


@dataclass(frozen=True)
class SegmentCall:
    best_idx: int
    set_idx: list          # ordered by score desc; set_idx[0] == best_idx
    germ_start: int
    germ_end: int
    cigar: str
    gate: float            # best_score / segment_length (out-of-scope advisory)
    confidence: float      # softmax-posterior mass inside the equivalence set (0..1)
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
                 m_seed: int = 8, set_band: float = 2.0, allowed=None,
                 workers: int = 1) -> SegmentCall | None:
    # workers=1 (serial) by default: the per-segment pool is tiny (~tens of microsecond-scale
    # alignments), so a thread pool per call costs more than it saves. Batch-level threading
    # across the whole read batch is the Phase 2 speed lever (one align_batch for all pairs).
    seg = seg.upper()
    if len(seg) < 5:
        return None
    seed_idx = seed_prefilter.candidates(seg, gene, m_seed, allowed=allowed)
    pool = _pool(topk_idx, seed_idx, allowed)
    if not pool:
        return None
    germs = reference_set.gene(gene.upper()).sequences
    results = align_batch([(seg, germs[i]) for i in pool], aligner, workers=workers)
    scores = [(r.score if r is not None else float("-inf")) for r in results]
    best_j = max(range(len(pool)), key=lambda j: scores[j])
    if scores[best_j] == float("-inf"):
        return None
    top = scores[best_j]
    keep = sorted([j for j in range(len(pool)) if top - scores[j] <= set_band],
                  key=lambda j: -scores[j])
    # posterior mass inside the kept set (softmax over finite pool scores, anchored at the top)
    z = sum(math.exp(s - top) for s in scores if s > float("-inf")) or 1.0
    confidence = sum(math.exp(scores[j] - top) for j in keep) / z
    win = results[best_j]
    return SegmentCall(best_idx=pool[best_j], set_idx=[pool[j] for j in keep],
                       germ_start=win.t_start, germ_end=win.t_end, cigar=win.cigar,
                       gate=top / max(len(seg), 1), confidence=confidence,
                       pool_idx=pool, scores=scores)


def call_segments_batched(items, reference_set, seed_prefilter, aligner,
                          m_seed: int = 8, set_band: float = 2.0, workers: int = 8):
    """Batched call_segment: align ALL (segment, candidate-germline) pairs across a read batch in ONE
    threaded align_batch — vs a serial align_batch per (read,gene). With a GIL-releasing aligner
    (parasail) this parallelizes the alignment (the rescore bottleneck) across cores.

    items: list of (segment_str, gene, neural_pool_list, allowed_set_or_None).
    Returns a list of SegmentCall | None, aligned with items."""
    pools, spans, pairs = [], [], []
    for seg, gene, topk, allowed in items:
        if len(seg) < 5:
            pools.append(None); spans.append((0, 0)); continue
        seed_idx = seed_prefilter.candidates(seg, gene, m_seed, allowed=allowed)
        pool = _pool(topk, seed_idx, allowed)
        germs = reference_set.gene(gene.upper()).sequences
        start = len(pairs)
        pairs.extend((seg, germs[i]) for i in pool)
        pools.append(pool); spans.append((start, len(pairs)))
    results = align_batch(pairs, aligner, workers=workers)
    out = []
    for (seg, gene, topk, allowed), pool, (a, b) in zip(items, pools, spans):
        if pool is None or a == b:
            out.append(None); continue
        res = results[a:b]
        scores = [(r.score if r is not None else float("-inf")) for r in res]
        best_j = max(range(len(pool)), key=lambda j: scores[j])
        if scores[best_j] == float("-inf"):
            out.append(None); continue
        top = scores[best_j]
        keep = sorted([j for j in range(len(pool)) if top - scores[j] <= set_band],
                      key=lambda j: -scores[j])
        z = sum(math.exp(s - top) for s in scores if s > float("-inf")) or 1.0
        conf = sum(math.exp(scores[j] - top) for j in keep) / z
        win = res[best_j]
        out.append(SegmentCall(best_idx=pool[best_j], set_idx=[pool[j] for j in keep],
                               germ_start=win.t_start, germ_end=win.t_end, cigar=win.cigar,
                               gate=top / max(len(seg), 1), confidence=conf,
                               pool_idx=pool, scores=scores))
    return out
