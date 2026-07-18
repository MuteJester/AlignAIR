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

    def _make(self, query: str, free: int):
        # pywfa requires text_*_free <= |target|; `free` = min target length in the pool is <= every
        # target and >= any realistic germline trim, so ONE aligner is reusable across all targets
        # with identical results (the query is the cached pattern).
        from pywfa import WavefrontAligner
        return WavefrontAligner(
            query, span="ends-free", pattern_begin_free=0, pattern_end_free=0,
            text_begin_free=free, text_end_free=free,
            match=self._match, mismatch=self._mismatch,
            gap_opening=self._gap_open, gap_extension=self._gap_ext)

    def align_many(self, query: str, targets) -> list:
        """Align one query against many germline targets, REUSING the WavefrontAligner (the query
        is the cached pattern) — eliminates the per-pair construction that dominated rescore time."""
        if len(query) < 1:
            return [None] * len(targets)
        valid = [len(t) for t in targets if len(t) >= 1]
        if not valid:
            return [None] * len(targets)
        aligner = self._make(query, min(valid))
        return [self._extract(aligner(t), query) if len(t) >= 1 else None for t in targets]

    def align(self, query: str, target: str) -> AlignResult | None:
        if len(query) < 1 or len(target) < 1:
            return None
        return self._extract(self._make(query, len(target))(target), query)

    def _extract(self, res, query: str) -> AlignResult | None:
        ct = res.cigartuples
        if not ct:
            return None
        target = res.text                                        # the aligned germline string
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
