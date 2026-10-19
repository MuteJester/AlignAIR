"""Observed segment alignment: what is actually visible in the read.

GenAIRR encodes bases removed by observation-stage end loss as CIGAR ``D`` ops. An alignment can never
begin or end with a gap, and a clipped or edge-deleted germline base is unobservable (indistinguishable
from extra trimming), so edge ``D`` and ``I`` runs are stripped.
"""
from __future__ import annotations

from dataclasses import dataclass

from alignair_assay.cigar import Op, format_cigar, parse_cigar, walk


@dataclass(frozen=True)
class ObservedSegment:
    allele: str
    seq_start: int
    seq_end: int
    germ_start: int
    germ_end: int
    cigar: str
    n_mismatch: int
    n_aligned: int
    observed_germline: str  # allele_seq[germ_start:germ_end]


def observe_segment(read: str, allele_seq: str, allele: str, seq_start: int,
                    germ_start: int, cigar: str) -> ObservedSegment | None:
    """Strip edge D/I runs from ``cigar``; return None if no M column remains."""
    ops: list[Op] = [(n, op) for n, op in parse_cigar(cigar) if n > 0]
    # Drop leading/trailing gap ops until both ends are aligned columns.
    while ops and ops[0][1] in "DI":
        n, op = ops.pop(0)
        if op == "D":
            germ_start += n
        else:
            seq_start += n
    while ops and ops[-1][1] in "DI":
        ops.pop()
    if not any(op in "M=X" for _, op in ops):
        return None

    read = read.upper()
    allele_seq = allele_seq.upper()
    n_read = sum(n for n, op in ops if op in "M=XI")
    n_germ = sum(n for n, op in ops if op in "M=XD")
    n_aligned = n_mismatch = 0
    for col in walk(ops, seq_start, germ_start):
        if col.read_pos is None or col.germ_pos is None:
            continue
        n_aligned += 1
        base = read[col.read_pos]
        if base != "N" and base != allele_seq[col.germ_pos]:
            n_mismatch += 1
    germ_end = germ_start + n_germ
    return ObservedSegment(
        allele=allele, seq_start=seq_start, seq_end=seq_start + n_read,
        germ_start=germ_start, germ_end=germ_end, cigar=format_cigar(ops),
        n_mismatch=n_mismatch, n_aligned=n_aligned,
        observed_germline=allele_seq[germ_start:germ_end],
    )


def check_invariants(read: str, allele_seq: str, seg: ObservedSegment) -> list[str]:
    """Return violated-invariant messages (empty when the segment is self-consistent)."""
    ops = parse_cigar(seg.cigar)
    m = sum(n for n, op in ops if op in "M=X")
    i = sum(n for n, op in ops if op == "I")
    d = sum(n for n, op in ops if op == "D")
    problems: list[str] = []
    if ops and (ops[0][1] in "DI" or ops[-1][1] in "DI"):
        problems.append(f"alignment begins or ends with a gap: {seg.cigar}")
    if m + i != seg.seq_end - seg.seq_start:
        problems.append(f"M+I={m + i} != read span {seg.seq_end - seg.seq_start}")
    if m + d != seg.germ_end - seg.germ_start:
        problems.append(f"M+D={m + d} != germline span {seg.germ_end - seg.germ_start}")
    for col in walk(ops, seg.seq_start, seg.germ_start):
        if col.read_pos is None or col.germ_pos is None:
            continue
        if not 0 <= col.read_pos < len(read):
            problems.append(f"read index {col.read_pos} out of range")
            break
        if not 0 <= col.germ_pos < len(allele_seq):
            problems.append(f"germline index {col.germ_pos} out of range")
            break
    return problems
