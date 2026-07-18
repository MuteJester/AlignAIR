from __future__ import annotations
from .backend import AlignResult
from .parasail import _ops

_ALIGNER = None


def _bio_aligner():
    """A cached PairwiseAligner: query-global / germline(target)-ends-free, gap-affine."""
    global _ALIGNER
    if _ALIGNER is None:
        from Bio import Align

        a = Align.PairwiseAligner()
        a.mode = "global"
        a.match_score = 2
        a.mismatch_score = -1
        a.open_gap_score = -3
        a.extend_gap_score = -1
        # Free the germline (target) end gaps so 5'/3' germline trim is unpenalized; the query stays
        # global. Biopython 1.87 renamed target_end_gap_score -> end_insertion_score. Probe with
        # dir() rather than hasattr: reading the aggregate gap-score getter before it is set raises
        # ValueError ("ambiguous"), which hasattr would propagate.
        if "end_insertion_score" in dir(a):  # Biopython >= 1.87
            a.end_insertion_score = 0.0
        else:  # Biopython < 1.87
            a.target_end_gap_score = 0.0
        _ALIGNER = a
    return _ALIGNER


def bio_available() -> bool:
    try:
        _bio_aligner()
        return True
    except Exception:
        return False


class BioAligner:
    """Pure-Biopython fallback aligner with the same AlignResult contract as ParasailAligner
    (query-global / germline-ends-free, core M/I/D CIGAR). Biopython is a core dependency, so this
    backend is always available - it is what a default Apple Silicon (arm64-macOS) install uses,
    since parasail ships no wheel there, and any environment with neither pywfa nor parasail."""

    def align_many(self, query: str, targets) -> list:
        return [self.align(query, t) for t in targets]

    def align(self, query: str, target: str) -> AlignResult | None:
        if len(query) < 1 or len(target) < 1:
            return None
        aligner = _bio_aligner()
        try:
            alignment = aligner.align(target, query)[0]  # (target=germline, query=read)
        except Exception:
            return None
        r = str(alignment[0])  # germline (target), gapped
        q = str(alignment[1])  # read (query), gapped
        cols = [i for i, c in enumerate(q) if c != "-"]
        if not cols:
            return None
        a, b = cols[0], cols[-1] + 1
        t_start = sum(1 for c in r[:a] if c != "-")
        cq, cr = q[a:b], r[a:b]
        t_end = t_start + sum(1 for c in cr if c != "-")
        return AlignResult(score=float(alignment.score), cigar=_ops(cq, cr),
                           q_start=0, q_end=len(query), t_start=t_start, t_end=t_end)
