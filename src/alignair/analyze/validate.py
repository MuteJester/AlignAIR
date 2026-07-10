"""Lightweight AIRR rearrangement validation for an emitted TSV."""
from __future__ import annotations

_REQUIRED = ["sequence_id", "sequence", "v_call", "j_call", "junction", "junction_aa",
             "productive", "vj_in_frame", "stop_codon", "v_cigar", "j_cigar"]


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def validate_airr(rows: list[dict], columns: list[str]) -> dict:
    """Check required columns are present and per-row coordinate/junction-length sanity."""
    missing = [f for f in _REQUIRED if f not in (columns or [])]
    coord_bad = length_bad = 0
    for r in rows:
        for g in ("v", "d", "j"):
            ss, se = _num(r.get(f"{g}_sequence_start")), _num(r.get(f"{g}_sequence_end"))
            if ss is not None and se is not None and se < ss:
                coord_bad += 1
        jl, jj = _num(r.get("junction_length")), r.get("junction")
        if jl is not None and jj and int(jl) != len(str(jj)):
            length_bad += 1
    return {"missing_required_columns": missing, "coord_violations": coord_bad,
            "junction_length_violations": length_bad,
            "valid": not missing and coord_bad == 0 and length_bad == 0}
