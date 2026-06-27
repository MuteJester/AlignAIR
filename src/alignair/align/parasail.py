from __future__ import annotations
from .backend import AlignResult

_PARASAIL = None


def _parasail():
    global _PARASAIL
    if _PARASAIL is None:
        import parasail
        _PARASAIL = (parasail, parasail.matrix_create("ACGTN", 2, -1))
    return _PARASAIL


def parasail_available() -> bool:
    try:
        _parasail()
        return True
    except Exception:
        return False


def _ops(gapped_q: str, gapped_r: str) -> str:
    """Core CIGAR: both bases -> M, query gap -> D (germline-only), ref gap -> I (query-only)."""
    out, run_op, run_len = [], None, 0
    for q, r in zip(gapped_q, gapped_r):
        op = "D" if q == "-" else ("I" if r == "-" else "M")
        if op == run_op:
            run_len += 1
        else:
            if run_op:
                out.append(f"{run_len}{run_op}")
            run_op, run_len = op, 1
    if run_op:
        out.append(f"{run_len}{run_op}")
    return "".join(out)


class ParasailAligner:
    """Query-global / germline-ends-free gap-affine alignment (parasail sg_dx)."""

    def align(self, query: str, target: str) -> AlignResult | None:
        if len(query) < 1 or len(target) < 1:
            return None
        par, mat = _parasail()
        res = par.sg_dx_trace_striped_16(query, target, 3, 1, mat)
        q, r = res.traceback.query, res.traceback.ref
        cols = [i for i, c in enumerate(q) if c != "-"]
        if not cols:
            return None
        a, b = cols[0], cols[-1] + 1
        t_start = sum(1 for c in r[:a] if c != "-")
        cq, cr = q[a:b], r[a:b]
        t_end = t_start + sum(1 for c in cr if c != "-")
        return AlignResult(score=float(res.score), cigar=_ops(cq, cr),
                           q_start=0, q_end=len(query), t_start=t_start, t_end=t_end)
