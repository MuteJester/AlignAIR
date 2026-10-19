"""Tool-agnostic reader for AIRR rearrangement TSVs.

Turns any tool's AIRR output into normalized ``Prediction`` records: 0-based half-open
coordinates, CIGARs stripped of leading/trailing ``S``/``N``/``H``, and malformed values
mapped to ``None`` with a message in ``Prediction.invalid`` (never an exception). Junction
coordinates a tool does not report are derived from its CDR3 coordinates (the junction is the
CDR3 plus the two anchor codons).
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field

from alignair_assay.cigar import format_cigar, parse_cigar

GENES = ("v", "d", "j")
_NULLS = frozenset({"", "none", "na", "nan"})
_TRUE = frozenset({"t", "true", "1"})
_FALSE = frozenset({"f", "false", "0"})
_CLIP = frozenset("SNH")


@dataclass(frozen=True)
class PredSegment:
    call: str | None = None
    call_set: tuple[str, ...] = ()
    seq_start: int | None = None
    seq_end: int | None = None
    germ_start: int | None = None
    germ_end: int | None = None
    cigar: str | None = None


@dataclass(frozen=True)
class Prediction:
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


def _text(row: dict, key: str) -> str | None:
    val = row.get(key)
    if val is None:
        return None
    val = val.strip()
    return None if val.lower() in _NULLS else val


def _bool(row: dict, key: str, invalid: list[str]) -> bool | None:
    val = _text(row, key)
    if val is None:
        return None
    low = val.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    invalid.append(f"{key}: unparseable boolean {val!r}")
    return None


def _coord(row: dict, key: str, shift: int, limit: int | None, invalid: list[str]) -> int | None:
    """Parse an AIRR coordinate; ``shift`` is subtracted (1 for starts, 0 for ends)."""
    val = _text(row, key)
    if val is None:
        return None
    num = _integer(val)
    if num is None:
        invalid.append(f"{key}: unparseable coordinate {val!r}")
        return None
    num -= shift
    if num < 0:
        invalid.append(f"{key}: negative coordinate {num}")
        return None
    if limit is not None and num > limit:
        invalid.append(f"{key}: coordinate {num} exceeds sequence length {limit}")
        return None
    return num


def _integer(val: str) -> int | None:
    """An integer, also written as an integral float (``"12.0"``), else None."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        num = float(val)
    except ValueError:
        return None
    return int(num) if num.is_integer() else None


def _junction_span(row: dict, seq_len: int | None, invalid: list[str]) -> tuple[int | None, int | None]:
    """Junction start/end; a coordinate the row leaves empty is taken from the CDR3 one (start - 3, end + 3) when
    that stays inside the read."""
    span = []
    for key, cdr3_key, shift, pad in (("junction_start", "cdr3_start", 1, -3), ("junction_end", "cdr3_end", 0, 3)):
        pos = _coord(row, key, shift, seq_len, invalid)
        if _text(row, key) is None:
            cdr3 = _coord(row, cdr3_key, shift, seq_len, invalid)
            if cdr3 is not None and 0 <= cdr3 + pad and (seq_len is None or cdr3 + pad <= seq_len):
                pos = cdr3 + pad
        span.append(pos)
    start, end = span
    if start is not None and end is not None and end < start:
        invalid.append("junction: end before start")
        return None, None
    return start, end


def _entries(val: str | None) -> tuple[str, ...]:
    if val is None:
        return ()
    return tuple(x.strip() for x in val.split(",") if x.strip())


def _normalize_cigar(raw: str, gene: str, seq_start, seq_end, germ_start, germ_end,
                     invalid: list[str]) -> str | None:
    try:
        ops = parse_cigar(raw)
    except ValueError:
        invalid.append(f"{gene}_cigar: malformed CIGAR {raw!r}")
        return None
    while ops and ops[0][1] in _CLIP:
        ops.pop(0)
    while ops and ops[-1][1] in _CLIP:
        ops.pop()
    if any(op in _CLIP or op == "P" for _, op in ops):
        invalid.append(f"{gene}_cigar: interior clip/skip op in {raw!r}")
        return None
    if None in (seq_start, seq_end, germ_start, germ_end):
        invalid.append(f"{gene}_cigar: dropped, coordinates unavailable to validate {raw!r}")
        return None
    read_len = sum(n for n, op in ops if op in "MI=X")
    germ_len = sum(n for n, op in ops if op in "MD=X")
    if read_len != seq_end - seq_start or germ_len != germ_end - germ_start:
        invalid.append(f"{gene}_cigar: {raw!r} disagrees with coordinates")
        return None
    return format_cigar(ops) or None


def _segment(row: dict, gene: str, seq_len: int | None, invalid: list[str]) -> PredSegment:
    calls = _entries(_text(row, f"{gene}_call"))
    call_set = _entries(_text(row, f"{gene}_call_set")) or calls
    seq_start = _coord(row, f"{gene}_sequence_start", 1, seq_len, invalid)
    seq_end = _coord(row, f"{gene}_sequence_end", 0, seq_len, invalid)
    germ_start = _coord(row, f"{gene}_germline_start", 1, None, invalid)
    germ_end = _coord(row, f"{gene}_germline_end", 0, None, invalid)
    if seq_start is not None and seq_end is not None and seq_end < seq_start:
        invalid.append(f"{gene}: sequence end before start")
        seq_start = seq_end = None
    if germ_start is not None and germ_end is not None and germ_end < germ_start:
        invalid.append(f"{gene}: germline end before start")
        germ_start = germ_end = None
    cigar = None
    raw = _text(row, f"{gene}_cigar")
    if raw is not None:
        cigar = _normalize_cigar(raw, gene, seq_start, seq_end, germ_start, germ_end, invalid)
    return PredSegment(calls[0] if calls else None, call_set,
                       seq_start, seq_end, germ_start, germ_end, cigar)


def _parse_row(row: dict, sequence_id: str) -> Prediction:
    invalid: list[str] = []
    sequence = _text(row, "sequence")
    seq_len = len(sequence) if sequence else None
    segments = {g: _segment(row, g, seq_len, invalid) for g in GENES}
    junction = _text(row, "junction")
    junction_start, junction_end = _junction_span(row, seq_len, invalid)
    return Prediction(
        sequence_id=sequence_id,
        sequence=sequence,
        rev_comp=_bool(row, "rev_comp", invalid),
        segments=segments,
        junction_nt=junction.upper() if junction else None,
        junction_aa=_text(row, "junction_aa"),
        junction_start=junction_start,
        junction_end=junction_end,
        productive=_bool(row, "productive", invalid),
        vj_in_frame=_bool(row, "vj_in_frame", invalid),
        stop_codon=_bool(row, "stop_codon", invalid),
        invalid=tuple(invalid),
    )


def load_airr_tsv(path) -> tuple[dict[str, Prediction], dict]:
    """Read an AIRR TSV into ``{sequence_id: Prediction}`` plus ``{"rows", "duplicates", "invalid_rows"}``.

    The first row wins on duplicate ``sequence_id``; rows without an id are skipped and counted
    as invalid.
    """
    csv.field_size_limit(sys.maxsize)
    preds: dict[str, Prediction] = {}
    stats = {"rows": 0, "duplicates": 0, "invalid_rows": 0}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            stats["rows"] += 1
            sid = (row.get("sequence_id") or "").strip()
            if not sid:
                stats["invalid_rows"] += 1
                continue
            if sid in preds:
                stats["duplicates"] += 1
                continue
            pred = _parse_row({k: v for k, v in row.items() if k and isinstance(v, str)}, sid)
            if pred.invalid:
                stats["invalid_rows"] += 1
            preds[sid] = pred
    return preds, stats
