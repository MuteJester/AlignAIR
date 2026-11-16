"""CIGARs: the germline reader writes them with M (a read base on a germline base), I (a read base on no germline
base) and D (a germline base on no read base); ``derive_cigar`` derives one for a read segment on its located germline
window, and the readers here accept every SAM operation letter."""
from __future__ import annotations

import functools
import re

import numpy as np

# The CIGAR's banded alignment: a mismatch costs MISMATCH_COST and an indel GAP_COST, so substitutions (SHM) are
# favoured over indels; the band is the net length difference plus the indel allowance plus BAND_MARGIN. The compiled
# kernel's defaults are these (tests/alignair/predict/germline/test_locate.py holds them equal).
MISMATCH_COST, GAP_COST, BAND_MARGIN = 1, 2, 2


def _run_length_encode(ops: str) -> str:
    """'MMMIMM' -> '3M1I2M' (M = aligned column, I = read-only, D = germline-only)."""
    if not ops:
        return ""
    out, prev, count = [], ops[0], 1
    for op in ops[1:]:
        if op == prev:
            count += 1
        else:
            out.append(f"{count}{prev}")
            prev, count = op, 1
    out.append(f"{count}{prev}")
    return "".join(out)


def _derive_cigar_py(read: str, germ: str, indels: int) -> str:
    """Pure-Python banded global alignment of a *pinned* read/germline window -> M/I/D CIGAR string.

    An equal-length window whose mismatches cost less than an insertion-deletion pair is a single ``M``
    run (no DP): every gapped path through it costs at least that pair, so it is the DP's own optimum,
    whatever the predicted indel count. Otherwise a global alignment restricted to a narrow band
    (width = net length difference + predicted indel count + ``BAND_MARGIN``) places the I/D ops;
    ``MISMATCH_COST``/``GAP_COST`` favour substitutions (SHM) over indels. This is the reference implementation
    and the fallback when the compiled :mod:`_derive_cy` kernel is unavailable — byte-identical output.
    """
    m, n = len(read), len(germ)
    if m == n and sum(a != b for a, b in zip(read, germ)) * MISMATCH_COST < 2 * GAP_COST:
        return f"{m}M" if m else ""
    if m == 0:
        return f"{n}D" if n else ""
    if n == 0:
        return f"{m}I" if m else ""

    return _traceback(_band_dp(read, germ, abs(m - n) + indels + BAND_MARGIN), m, n)


def _band_dp(read: str, germ: str, band: int) -> dict[tuple[int, int], str]:
    """The banded global alignment's back pointers: for each reachable cell ``(i, j)`` (read bases, germline bases)
    within ``band`` diagonals of the net length difference, the operation (M, I or D) its best path ends with."""
    m, n = len(read), len(germ)
    lo_off, hi_off = min(0, m - n) - band, max(0, m - n) + band   # allowed range of (i - j)
    INF = float("inf")
    cost = {(0, 0): 0.0}
    back: dict = {}
    for i in range(m + 1):
        j0 = max(0, i - hi_off)
        j1 = min(n, i - lo_off)
        for j in range(j0, j1 + 1):
            if i == 0 and j == 0:
                continue
            best, op = INF, None
            if i and j:                                     # M (diagonal): match or mismatch
                c = cost.get((i - 1, j - 1), INF) + (0 if read[i - 1] == germ[j - 1] else MISMATCH_COST)
                if c < best:
                    best, op = c, "M"
            if i:                                           # I (read-only base)
                c = cost.get((i - 1, j), INF) + GAP_COST
                if c < best:
                    best, op = c, "I"
            if j:                                           # D (germline-only base)
                c = cost.get((i, j - 1), INF) + GAP_COST
                if c < best:
                    best, op = c, "D"
            if op is not None:
                cost[(i, j)] = best
                back[(i, j)] = op
    return back


def _traceback(back: dict[tuple[int, int], str], m: int, n: int) -> str:
    """The CIGAR of the best path from ``(m, n)`` back to ``(0, 0)`` along ``back`` (``_band_dp``)."""
    ops, i, j = [], m, n
    while (i, j) != (0, 0):
        op = back.get((i, j))
        if op is None:                                      # off-band fallback (shouldn't happen)
            break
        ops.append(op)
        if op == "M":
            i, j = i - 1, j - 1
        elif op == "I":
            i -= 1
        else:
            j -= 1
    return _run_length_encode("".join(reversed(ops)))


try:                                                        # optional C-speed kernel (identical output)
    from ._derive_cy import derive_cigar as _cy_derive_cigar

    def _derive_cigar(read: str, germ: str, indels: int) -> str:
        return _cy_derive_cigar(read.encode("ascii"), germ.encode("ascii"), indels, MISMATCH_COST, GAP_COST,
                                BAND_MARGIN)

    DERIVE_BACKEND = "cython"
except (ImportError, OSError):                              # pragma: no cover - optional compiled kernel
    _derive_cigar = _derive_cigar_py
    DERIVE_BACKEND = "python"


def derive_cigar(read_window: str, germline_window: str, indel_count=0) -> str:
    """The M/I/D CIGAR of a read segment against its (already located) germline window.

    The window is *pinned* by ``locate.GermlineLocator`` (both ends fixed), so it can't drift
    the way an ends-free aligner does. Uses the compiled :mod:`_derive_cy` kernel when present, else
    the identical pure-Python fallback (:data:`DERIVE_BACKEND` reports which)."""
    return _derive_cigar(read_window, germline_window, indel_allowance(indel_count))


def joined_cigar(head, tail) -> str:
    """The CIGAR of ``head`` then ``tail`` ops, equal neighbouring ops merged."""
    out: list[list] = []
    for n, op in (*head, *tail):
        if n <= 0:
            continue
        if out and out[-1][1] == op:
            out[-1][0] += n
        else:
            out.append([n, op])
    return "".join(f"{n}{op}" for n, op in out)


def _scalar(x) -> float:
    """A possibly-numpy / 0-d indel count as a Python float."""
    return float(np.asarray(x).squeeze())


def indel_allowance(indel_count) -> int:
    """The whole number of indels a predicted (float) indel count allows: rounded, never negative."""
    return max(0, round(_scalar(indel_count)))


_OP = re.compile(r"(\d+)([MIDNSHP=X])")
ALIGNED = frozenset("M=X")        # the operations that put a read base on a germline base


@functools.lru_cache(maxsize=4096)
def cigar_ops(cigar: str | None) -> tuple[tuple[int, str], ...]:
    """``cigar``'s ``(count, operation)`` pairs in order; none for None or an empty CIGAR."""
    return tuple((int(count), op) for count, op in _OP.findall(cigar or ""))


def aligned_runs(cigar: str | None, read_start: int = 0, germ_start: int = 0):
    """``(read position, germline position, length)`` of each aligned run (M, = or X) of ``cigar``, walked from
    ``read_start`` and ``germ_start``: an I moves the read on, a D the germline, and the other operations neither."""
    r, g = read_start, germ_start
    for count, op in cigar_ops(cigar):
        if op in ALIGNED:
            yield r, g, count
            r, g = r + count, g + count
        elif op == "I":
            r += count
        elif op == "D":
            g += count


def aligned_columns(cigar: str | None, read_start: int = 0, germ_start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The read and germline positions (int64) of ``cigar``'s aligned columns (``aligned_runs``)."""
    reads, germs = [], []
    for r, g, count in aligned_runs(cigar, read_start, germ_start):
        reads.extend(range(r, r + count))
        germs.extend(range(g, g + count))
    return np.array(reads, dtype=np.int64), np.array(germs, dtype=np.int64)


def matched_bases(read: str, germ: str, cigar: str | None, read_start: int = 0, germ_start: int = 0) -> int:
    """How many read bases ``cigar`` puts on an equal germline base (``aligned_runs``)."""
    return sum(sum(a == b for a, b in zip(read[r:r + count], germ[g:g + count]))
               for r, g, count in aligned_runs(cigar, read_start, germ_start))
