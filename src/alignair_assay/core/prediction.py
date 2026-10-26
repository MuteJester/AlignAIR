"""Normalized prediction records: one tool's annotation of one read, in the form every scorer reads."""
from __future__ import annotations

from dataclasses import dataclass, field

GENES = ("v", "d", "j")


@dataclass(frozen=True)
class PredSegment:
    """One gene of a prediction: the top-1 ``call``, the asserted ``call_set``, the read span ``seq_start``/
    ``seq_end`` and germline span ``germ_start``/``germ_end`` (0-based, half-open; germline coordinates in the
    called allele's frame) and a CIGAR without clip or skip ops. Any field may be None when the tool did not
    report it or reported it malformed."""

    call: str | None = None
    call_set: tuple[str, ...] = ()
    seq_start: int | None = None
    seq_end: int | None = None
    germ_start: int | None = None
    germ_end: int | None = None
    cigar: str | None = None


@dataclass(frozen=True)
class Prediction:
    """One tool's row for one read, matched to truth by ``sequence_id``. Junction coordinates are 0-based, half-open
    and in the canonical (forward) frame; ``invalid`` lists the malformed values that were dropped to None."""

    sequence_id: str
    sequence: str | None = None
    rev_comp: bool | None = None
    segments: dict[str, PredSegment] = field(default_factory=dict)
    junction_nt: str | None = None
    junction_aa: str | None = None
    junction_start: int | None = None
    junction_end: int | None = None
    productive: bool | None = None
    vj_in_frame: bool | None = None
    stop_codon: bool | None = None
    invalid: tuple[str, ...] = ()
