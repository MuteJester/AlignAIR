"""CIGAR → per-germline-position observation.

Walk a read's CIGAR over its germline window to record, for each germline position, the read base
aligned to it (``None`` for a deletion). Positions not walked are simply absent (uncovered). This is
the baseline (any-model) polymorphism signal; the state head refines it when available.
"""
from __future__ import annotations

import re

_CIGAR = re.compile(r"(\d+)([MIDNSX=])")


def germline_observations(read_seq: str, cigar: str, seq_start: int, germ_start: int) -> dict:
    """{germline_position: observed_read_base | None} over the aligned window."""
    obs: dict[int, str | None] = {}
    r, g = seq_start, germ_start
    for count, op in _CIGAR.findall(cigar or ""):
        for _ in range(int(count)):
            if op in "M=X":
                obs[g] = read_seq[r] if 0 <= r < len(read_seq) else None
                r += 1
                g += 1
            elif op == "D":
                obs[g] = None                      # germline base deleted in the read (still observed)
                g += 1
            elif op in "IS":
                r += 1                             # read-only base: no germline position
    return obs
