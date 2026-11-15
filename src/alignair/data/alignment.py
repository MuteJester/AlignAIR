"""Validated segment-local CIGAR paths shared by training target and residue projection code."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re

from ..errors import TrainingDataError
from .coordinates import parse_coordinate, segment_spans

_CIGAR = re.compile(r"([1-9][0-9]*)([MIDNSHP=X])")


def parse_cigar(cigar) -> tuple[tuple[int, str], ...]:
    """Parse a complete SAM operation string; missing/unknown alignments have no operations."""
    if cigar is None:
        return ()
    if not isinstance(cigar, str):
        raise TrainingDataError("CIGAR must be a string", context={"field": "cigar"})
    if cigar in ("", "*"):
        return ()
    operations, end = [], 0
    for match in _CIGAR.finditer(cigar):
        if match.start() != end:
            raise TrainingDataError("malformed CIGAR operation string", context={"field": "cigar"})
        operations.append((int(match[1]), match[2]))
        end = match.end()
    if end != len(cigar):
        raise TrainingDataError("malformed CIGAR operation string", context={"field": "cigar"})
    return tuple(operations)


@dataclass(frozen=True)
class QueryReferenceMap:
    query_start: int
    positions: tuple[int | None, ...]
    reason: str | None = None


def query_reference_map(cigar, query_span, reference_span) -> QueryReferenceMap:
    """Map each segment-local query base to its reference base; insertions map to None.

    A missing path or disagreement with the declared spans is explicitly unavailable. It is never
    replaced by an offset-only alignment, even when the two spans happen to have equal lengths.
    """
    try:
        qs, qe = map(parse_coordinate, query_span)
        rs, re_ = map(parse_coordinate, reference_span)
        if not 0 <= qs <= qe or not 0 <= rs <= re_:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingDataError("invalid alignment coordinate spans") from exc
    operations = parse_cigar(cigar)
    if not operations:
        return QueryReferenceMap(qs, (), "missing_cigar")
    query_consumed = sum(count for count, op in operations if op in "MIS=X")
    reference_consumed = sum(count for count, op in operations if op in "MDN=X")
    if query_consumed != qe - qs or reference_consumed != re_ - rs:
        return QueryReferenceMap(qs, (), "alignment_span_mismatch")
    positions, reference = [], rs
    for count, op in operations:
        if op in "M=X":
            positions.extend(range(reference, reference + count))
            reference += count
        elif op in "IS":
            positions.extend([None] * count)
        elif op in "DN":
            reference += count
    return QueryReferenceMap(qs, tuple(positions))


@dataclass(frozen=True)
class RecordAlignment:
    """A training record's alignment, read once: its read (uppercase) and each gene's read and germline spans
    (``segment_spans``, the genes the read shows); each gene's read-to-germline map (``query_reference_map`` of its
    CIGAR) is computed on first use and kept."""
    record: Mapping
    sequence: str
    query_spans: dict[str, tuple[int, int]]
    germline_spans: dict[str, tuple[int, int]]
    _paths: dict[str, QueryReferenceMap] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def of(cls, record: Mapping) -> "RecordAlignment":
        sequence = str(record["sequence"]).upper()
        query, germline = segment_spans(record)
        return cls(record, sequence, query, germline)

    @property
    def d_inverted(self) -> bool:
        """Whether the read's D is inverted (it reads as the reverse-complemented germline)."""
        return bool(self.record.get("d_inverted", False))

    def primary(self, gene: str) -> str:
        """The allele ``gene``'s germline coordinates belong to (GenAIRR lists it first in the call)."""
        return str(self.record[f"{gene}_call"]).split(",")[0]

    def path(self, gene: str) -> QueryReferenceMap:
        """``gene``'s read-to-germline map (a gene the read shows)."""
        if gene not in self._paths:
            self._paths[gene] = query_reference_map(self.record.get(f"{gene}_cigar"), self.query_spans[gene],
                                                    self.germline_spans[gene])
        return self._paths[gene]
