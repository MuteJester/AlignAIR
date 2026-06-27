from __future__ import annotations
import math
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
    # posterior mass inside the kept set (softmax over finite pool scores, anchored at the top)
    z = sum(math.exp(s - top) for s in scores if s > float("-inf")) or 1.0
    confidence = sum(math.exp(scores[j] - top) for j in keep) / z
    win = results[best_j]
    return SegmentCall(best_idx=pool[best_j], set_idx=[pool[j] for j in keep],
                       germ_start=win.t_start, germ_end=win.t_end, cigar=win.cigar,
                       gate=top / max(len(seg), 1), confidence=confidence,
                       pool_idx=pool, scores=scores)
