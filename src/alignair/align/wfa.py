from __future__ import annotations
from .backend import AlignResult
from .parasail import _ops

_OK = None


def wfa_available() -> bool:
    global _OK
    if _OK is None:
        try:
            import pywfa  # noqa: F401
            _OK = True
        except Exception:
            _OK = False
    return _OK


class WFAAligner:
    """WFA2 (pywfa) backend: query (pattern) global, germline (text) ends free so 5'/3' germline
    trim is unpenalized. Same AlignResult contract as ParasailAligner.

    pywfa cigartuples op codes are relative to (pattern=query, text=germline) and are REVERSED
    from this package's convention: op 1 (pywfa "I") consumes germline only, op 2 (pywfa "D")
    consumes query only. We reconstruct the core gapped strings and reuse parasail's `_ops` so the
    emitted CIGAR uses the same convention everywhere (M; D=germline-only; I=query-only). Germline
    coords come from text_start/text_end; raw score is a penalty (higher = better is preserved)."""

    def __init__(self, match: int = 0, mismatch: int = 4, gap_open: int = 6, gap_ext: int = 2):
        # WFA2 is a PENALTY model: match must be <= 0, mismatch/gap penalties > 0; it minimizes
        # cost, so res.score is <= 0 and higher (closer to 0) = better (the AlignResult contract).
        self._match, self._mismatch = match, mismatch
        self._gap_open, self._gap_ext = gap_open, gap_ext

    def align(self, query: str, target: str) -> AlignResult | None:
        if len(query) < 1 or len(target) < 1:
            return None
        from pywfa import WavefrontAligner
        aligner = WavefrontAligner(
            query, span="ends-free", pattern_begin_free=0, pattern_end_free=0,
            text_begin_free=len(target), text_end_free=len(target),
            match=self._match, mismatch=self._mismatch,
            gap_opening=self._gap_open, gap_extension=self._gap_ext)
        res = aligner(target)
        ct = res.cigartuples
        if not ct:
            return None
        t_start, t_end = int(res.text_start), int(res.text_end)
        qi, ti = 0, 0
        gq, gt = [], []
        for op, length in ct:
            if op in (0, 7, 8):                      # M / = / X -> aligned columns
                gq.append(query[qi:qi + length]); gt.append(target[ti:ti + length])
                qi += length; ti += length
            elif op == 1:                            # germline-only (free overhang or true deletion)
                if t_start <= ti < t_end:            # emit only inside the aligned core
                    gq.append("-" * length); gt.append(target[ti:ti + length])
                ti += length
            elif op == 2:                            # query-only (insertion)
                gq.append(query[qi:qi + length]); gt.append("-" * length)
                qi += length
            elif op == 4:                            # soft-clip (text overhang) -> skip, advance text
                ti += length
        cigar = _ops("".join(gq), "".join(gt))
        if not cigar:
            return None
        return AlignResult(score=float(res.score), cigar=cigar,
                           q_start=0, q_end=len(query), t_start=t_start, t_end=t_end)
