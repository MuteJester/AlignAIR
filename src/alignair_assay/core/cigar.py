"""CIGAR parsing/formatting and read<->germline column walking."""
from __future__ import annotations

import re
from dataclasses import dataclass

Op = tuple[int, str]

_TOKEN = re.compile(r"(\d+)([MIDNSHP=X])")
_BOTH = frozenset("M=X")


def parse_cigar(s: str) -> list[Op]:
    """Parse a CIGAR string into ``[(length, op), ...]``."""
    ops = [(int(n), op) for n, op in _TOKEN.findall(s)]
    if "".join(f"{n}{op}" for n, op in ops) != s:
        raise ValueError(f"malformed CIGAR: {s!r}")
    return ops


def format_cigar(ops: list[Op]) -> str:
    """Render ops as a CIGAR string, dropping zero-length ops and merging adjacent equal ops."""
    merged: list[Op] = []
    for n, op in ops:
        if n <= 0:
            continue
        if merged and merged[-1][1] == op:
            merged[-1] = (merged[-1][0] + n, op)
        else:
            merged.append((n, op))
    return "".join(f"{n}{op}" for n, op in merged)


@dataclass(frozen=True)
class Column:
    """One alignment column; ``None`` marks a gap on that side."""
    read_pos: int | None
    germ_pos: int | None


def walk(ops: list[Op], seq_start: int, germ_start: int) -> list[Column]:
    """Expand ops into per-column (read, germline) positions.

    ``M``/``=``/``X`` consume both, ``I`` the read only, ``D`` the germline only. ``S``/``N``/``H``
    are rejected: callers normalize them away first.
    """
    cols: list[Column] = []
    r, g = seq_start, germ_start
    for n, op in ops:
        if op in _BOTH:
            cols.extend(Column(r + i, g + i) for i in range(n))
            r += n
            g += n
        elif op == "I":
            cols.extend(Column(r + i, None) for i in range(n))
            r += n
        elif op == "D":
            cols.extend(Column(None, g + i) for i in range(n))
            g += n
        else:
            raise ValueError(f"unsupported CIGAR op {op!r} in walk()")
    return cols
