"""Validate an AIRR rearrangement TSV for structural + coordinate soundness (a fast, dependency-free
check used by ``alignair validate-airr`` and release gating).

This enforces the invariants that matter for downstream tools: required columns present,
non-empty sequence, in-bounds 1-based coordinates, and —
critically — that each per-gene CIGAR never consumes more query bases than the emitted ``sequence``.
"""
from __future__ import annotations

import csv

_REQUIRED = ("sequence_id", "sequence", "v_call", "j_call")
_GENES = ("v", "d", "j")
_QUERY_OPS = set("MIS=X")          # CIGAR ops that consume query bases (D/N/H/P do not)


def cigar_query_length(cigar: str) -> int:
    """Number of query (read) bases a CIGAR consumes: the sum of M/I/S/=/X op lengths."""
    if not cigar:
        return 0
    total, num = 0, ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            if num and ch in _QUERY_OPS:
                total += int(num)
            num = ""
    return total


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _bool(v):
    """AIRR logical field -> True/False/None (blank/unknown)."""
    if v is None or v == "":
        return None
    return str(v).strip().lower() in ("t", "true", "1", "yes")


def validate_airr_file(path: str) -> dict:
    """Validate an AIRR TSV. Returns ``{n_rows, missing_columns, errors}`` where ``errors`` is a list
    of ``(sequence_id, message)`` — empty means the file passed."""
    errors: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        delim = "\t" if "\t" in f.readline() else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        fields = reader.fieldnames or []
        missing = [c for c in _REQUIRED if c not in fields]
        n = 0
        if missing:
            return {"n_rows": 0, "missing_columns": missing, "errors": errors}
        for row in reader:
            n += 1
            sid = row.get("sequence_id") or f"row{n}"
            seq = row.get("sequence") or ""
            if not seq:
                errors.append((sid, "empty sequence"))
                continue
            L = len(seq)
            for g in _GENES:
                cig = row.get(f"{g}_cigar")
                if cig:
                    q = cigar_query_length(cig)
                    if q > L:
                        errors.append((sid, f"{g}_cigar consumes {q} query bases > sequence length {L}"))
                s, e = _int(row.get(f"{g}_sequence_start")), _int(row.get(f"{g}_sequence_end"))
                if s is not None and e is not None:
                    if not (1 <= s <= e <= L):
                        errors.append((sid, f"{g} coordinates out of bounds: start={s} end={e} len={L}"))
            # productivity cross-field invariant: a productive rearrangement must be in-frame and have
            # no stop codon (when those derived fields are present).
            prod, inframe, stop = (_bool(row.get("productive")), _bool(row.get("vj_in_frame")),
                                   _bool(row.get("stop_codon")))
            if prod is True:
                if inframe is False:
                    errors.append((sid, "productive=T but vj_in_frame=F"))
                if stop is True:
                    errors.append((sid, "productive=T but stop_codon=T"))
    return {"n_rows": n, "missing_columns": [], "errors": errors}
