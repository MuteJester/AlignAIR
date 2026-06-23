"""Proper gapped alignment of predicted V/D/J segments to their called germlines, producing the
AIRR alignment-representation fields from a REAL alignment (parasail) rather than from coordinates:
exact per-gene CIGAR, alignment-derived germline start/end, fractional identity, and a stitched
gapped `sequence_alignment` / `germline_alignment` pair (non-templated N regions are 'N').

parasail is an optional dependency ([reader]); `realign` returns {} when it is unavailable, so
callers fall back to the coordinate-derived approximation.
"""
from __future__ import annotations

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
    """CIGAR ops (M/I/D) over the aligned core: both bases -> M, query gap -> D (germline-only),
    ref gap -> I (query-only). Consecutive ops are run-length encoded."""
    out = []
    run_op = None
    run_len = 0
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


def _align_segment(seg: str, gseq: str):
    """Align a read segment (query, fully consumed) to a germline (free germline end gaps) and
    return (germline_start, germline_end, gapped_query, gapped_ref, identity) or None."""
    par, mat = _parasail()
    res = par.sg_dx_trace_striped_16(seg, gseq, 3, 1, mat)   # query global, germline ends free
    q, r = res.traceback.query, res.traceback.ref
    cols = [i for i, c in enumerate(q) if c != "-"]          # columns the query occupies
    if not cols:
        return None
    a, b = cols[0], cols[-1] + 1                             # core = first..last query base
    germ_start = sum(1 for c in r[:a] if c != "-")           # germline consumed before the core
    cq, cr = q[a:b], r[a:b]
    germ_end = germ_start + sum(1 for c in cr if c != "-")
    matches = sum(1 for x, y in zip(cq, cr) if x == y and x != "-")
    identity = round(matches / max(len(cq), 1), 4)
    return germ_start, germ_end, cq, cr, identity


def realign(seq: str, p: dict, reference_set, genes=("v", "d", "j")) -> dict:
    """Return AIRR alignment fields for one prediction by aligning each predicted segment to its
    called germline. {} if parasail is unavailable. Overrides germline coords with the (more
    accurate) alignment-derived ones for self-consistency with the cigar."""
    if not parasail_available():
        return {}
    seq = seq.upper()
    per = {}
    for g in genes:
        nm = p.get(f"{g}_call")
        ss, se = p.get(f"{g}_sequence_start"), p.get(f"{g}_sequence_end")
        if not nm or ss is None or not se or se <= ss:
            continue
        try:
            ref = reference_set.gene(g.upper())
            gseq = ref.sequences[ref.index[nm]]
        except (KeyError, Exception):
            continue
        if len(seq[ss:se]) < 5 or not gseq:
            continue
        r = _align_segment(seq[ss:se], gseq)
        if r is None:
            continue
        gstart, gend, cq, cr, ident = r
        tail = len(seq) - se
        cigar = (f"{ss}S" if ss > 0 else "") + (f"{gstart}N" if gstart else "") \
            + _ops(cq, cr) + (f"{tail}S" if tail > 0 else "")
        per[g] = {"germ_start": gstart, "germ_end": gend, "cq": cq, "cr": cr,
                  "identity": ident, "cigar": cigar, "ss": ss, "se": se}

    if not per:
        return {}
    out = {}
    for g, d in per.items():
        out[f"{g}_cigar"] = d["cigar"]
        out[f"{g}_identity"] = d["identity"]
        out[f"{g}_germline_start"] = d["germ_start"]      # alignment-derived (overrides soft-DP)
        out[f"{g}_germline_end"] = d["germ_end"]

    # stitch a gapped sequence_alignment / germline_alignment over V..J (N regions -> 'N')
    ordered = sorted(per.values(), key=lambda d: d["ss"])
    sa, ga = [], []
    prev_se = None
    for d in ordered:
        if prev_se is not None:
            gap = max(0, d["ss"] - prev_se)                # non-templated N region between segments
            sa.append(seq[prev_se:prev_se + gap]); ga.append("N" * gap)
        sa.append(d["cq"]); ga.append(d["cr"])
        prev_se = d["se"]
    out["sequence_alignment"] = "".join(sa)
    out["germline_alignment"] = "".join(ga)
    return out
