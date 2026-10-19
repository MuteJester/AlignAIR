"""Boundary equivalence intervals for segment edges.

A perfect aligner cannot recover the generative boundary when an N-region base equals the next germline
base, or a mutation sits on the last germline base. Truth for each edge is therefore an inclusive
interval of equally defensible positions: shrinking inward over mismatching edge columns, and extending
outward over neighbouring read bases that equal the next germline base. Read and germline move in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass

from alignair_assay.cigar import parse_cigar, walk
from alignair_assay.truth.segment import ObservedSegment

Interval = tuple[int, int]  # inclusive (lo, hi)


@dataclass(frozen=True)
class BoundaryTruth:
    seq_start: Interval
    seq_end: Interval
    germ_start: Interval
    germ_end: Interval


def _mismatch_run(flags: list[bool | None]) -> int:
    """Length of the leading run of ``True`` (mismatching ``M`` columns); gaps (None) and matches stop it."""
    n = 0
    for f in flags:
        if f is not True:
            break
        n += 1
    return n


def boundary_intervals(read: str, allele_seq: str, seg: ObservedSegment,
                       left_limit: int, right_limit: int) -> BoundaryTruth:
    """Equivalence intervals for the four edges of ``seg``.

    ``left_limit``/``right_limit`` are read positions bounding the neighbouring non-segment bases
    (end of the previous observed segment / start of the next, or 0 / ``len(read)``).
    """
    read = read.upper()
    allele_seq = allele_seq.upper()

    # Per column: True = mismatching M (an N read base counts), False = matching M, None = gap.
    flags: list[bool | None] = []
    for col in walk(parse_cigar(seg.cigar), seg.seq_start, seg.germ_start):
        if col.read_pos is None or col.germ_pos is None:
            flags.append(None)
        else:
            b = read[col.read_pos]
            flags.append(b == "N" or b != allele_seq[col.germ_pos])

    n_aligned = sum(f is not None for f in flags)
    shrink_r = _mismatch_run(flags[::-1])
    shrink_l = _mismatch_run(flags)
    # Keep at least one aligned column.
    cap = max(n_aligned - 1, 0)
    shrink_r = min(shrink_r, cap)
    shrink_l = min(shrink_l, cap - shrink_r)

    ext_r = 0
    while (seg.seq_end + ext_r < right_limit and seg.germ_end + ext_r < len(allele_seq)
           and _equal(read[seg.seq_end + ext_r], allele_seq[seg.germ_end + ext_r])):
        ext_r += 1
    ext_l = 0
    while (seg.seq_start - 1 - ext_l >= left_limit and seg.germ_start - 1 - ext_l >= 0
           and _equal(read[seg.seq_start - 1 - ext_l], allele_seq[seg.germ_start - 1 - ext_l])):
        ext_l += 1

    return BoundaryTruth(
        seq_start=(seg.seq_start - ext_l, seg.seq_start + shrink_l),
        seq_end=(seg.seq_end - shrink_r, seg.seq_end + ext_r),
        germ_start=(seg.germ_start - ext_l, seg.germ_start + shrink_l),
        germ_end=(seg.germ_end - shrink_r, seg.germ_end + ext_r),
    )


def _equal(read_base: str, germ_base: str) -> bool:
    return read_base == "N" or read_base == germ_base
