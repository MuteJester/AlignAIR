from __future__ import annotations

import re
from typing import Any

from ..performance import PERFORMANCE_PREDICTION_FIELD_KEYS
from ...core.schema import GENES


_CALL_SPLIT_RE = re.compile(r"[,;]")
_TRUE_STRINGS = {
    "1",
    "t",
    "true",
    "yes",
    "y",
    "productive",
    "in_frame",
    "in-frame",
    "inverted",
}
_FALSE_STRINGS = {
    "0",
    "f",
    "false",
    "no",
    "n",
    "nonproductive",
    "out_of_frame",
    "out-of-frame",
    "forward",
}


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _split_calls(value: Any) -> list[str]:
    if _missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in _CALL_SPLIT_RE.split(str(value)) if part.strip()]


def _coerce_bool(value: Any) -> bool | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return None


def _coerce_number(value: Any, *, delta: int = 0) -> int | float | None:
    if _missing(value):
        return None
    try:
        out = float(value) + delta
    except (TypeError, ValueError):
        return None
    return int(out) if out.is_integer() else out


def igblast_airr_to_prediction(row: dict[str, Any] | None) -> dict[str, Any]:
    """Convert one IgBLAST AIRR row to the benchmark prediction convention.

    AIRR starts are 1-based; benchmark starts are 0-based, ends are position-style.
    """

    pred: dict[str, Any] = {}
    row = row or {}
    for g in GENES:
        calls = _split_calls(row.get(f"{g}_call"))
        pred[f"{g}_call"] = calls[0] if calls else None
        if calls:
            pred[f"{g}_calls"] = calls
        for src, dst, delta in (
            (f"{g}_sequence_start", f"{g}_sequence_start", -1),
            (f"{g}_sequence_end", f"{g}_sequence_end", 0),
            (f"{g}_germline_start", f"{g}_germline_start", -1),
            (f"{g}_germline_end", f"{g}_germline_end", 0),
        ):
            pred[dst] = _coerce_number(row.get(src), delta=delta)
        for key in (
            f"{g}_cigar",
            f"{g}_identity",
            f"{g}_support",
            f"{g}_score",
            f"{g}_trim_5",
            f"{g}_trim_3",
        ):
            if key in row:
                pred[key] = row.get(key)
    for key in (
        "sequence_id",
        "sequence",
        "locus",
        "productive",
        "vj_in_frame",
        "stop_codon",
        "junction",
        "junction_aa",
        "junction_length",
        "np1",
        "np2",
        "np1_length",
        "np2_length",
        "c_call",
        "read_layout",
    ):
        if key in row:
            pred[key] = row.get(key)
    for key in PERFORMANCE_PREDICTION_FIELD_KEYS:
        if key in row:
            pred[key] = row.get(key)
    for key in (
        "productive",
        "vj_in_frame",
        "stop_codon",
        "d_inverted",
        "is_contaminant",
        "receptor_revision_applied",
    ):
        if key in row:
            coerced = _coerce_bool(row.get(key))
            pred[key] = coerced if coerced is not None else row.get(key)
    for src, dst, delta in (
        ("junction_start", "junction_start", -1),
        ("cdr3_start", "junction_start", -1),
        ("junction_end", "junction_end", 0),
        ("cdr3_end", "junction_end", 0),
    ):
        if src in row and (dst not in pred or pred.get(dst) is None):
            pred[dst] = _coerce_number(row.get(src), delta=delta)
    if "rev_comp" in row:
        rev_comp = _coerce_bool(row.get("rev_comp"))
        if rev_comp is not None:
            pred["rev_comp"] = rev_comp
            pred["orientation_id"] = 1 if rev_comp else 0
    return pred
