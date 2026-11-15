"""CIGARs: the germline reader writes them with M (a read base on a germline base), I (a read base on no germline
base) and D (a germline base on no read base). The readers here accept every SAM operation letter."""
from __future__ import annotations

import functools
import re

import numpy as np

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
