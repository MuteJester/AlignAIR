"""Prediction adapters and helpers.

Benchmark metrics accept a simple prediction dict with keys like ``v_call`` or
``v_calls`` plus optional coordinates. This module provides adapters for common
formats and perfect-prediction helpers used by tests and smoke checks.
"""
from __future__ import annotations

import re
from typing import Any

from ..core.schema import BenchmarkCase, GENES


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


def case_to_prediction(case: BenchmarkCase, frame: str = "canonical", set_calls: bool = True) -> dict:
    """Return a perfect prediction dict for a benchmark case."""

    pred: dict[str, Any] = {}
    for g, truth in case.truth(frame).items():
        if not truth.present:
            continue
        if set_calls:
            pred[f"{g}_calls"] = list(truth.calls)
        pred[f"{g}_call"] = truth.primary or (truth.calls[0] if truth.calls else None)
        pred[f"{g}_sequence_start"] = truth.sequence_start
        pred[f"{g}_sequence_end"] = truth.sequence_end
        pred[f"{g}_germline_start"] = truth.germline_start
        pred[f"{g}_germline_end"] = truth.germline_end
        for key in (
            f"{g}_cigar",
            f"{g}_identity",
            f"{g}_score",
            f"{g}_support",
            f"{g}_trim_5",
            f"{g}_trim_3",
        ):
            if key in case.record:
                pred[key] = case.record[key]
    pred["orientation_id"] = case.orientation_id
    pred["region_labels"] = case.labels("region", frame)
    pred["state_labels"] = case.labels("state", frame)
    pred.update(case.scalars)
    pred["sequence_id"] = case.case_id
    pred["sequence"] = case.canonical_sequence if frame == "canonical" else case.sequence
    pred["rev_comp"] = case.orientation_id == 1
    for key in (
        "locus",
        "c_call",
        "junction",
        "junction_aa",
        "junction_start",
        "junction_end",
        "junction_length",
        "np1",
        "np2",
        "np1_aa",
        "np2_aa",
        "np1_length",
        "np2_length",
        "p_v_3_length",
        "p_d_5_length",
        "p_d_3_length",
        "p_j_5_length",
        "vj_in_frame",
        "stop_codon",
        "d_inverted",
        "read_layout",
        "is_contaminant",
        "receptor_revision_applied",
        "original_v_call",
        "n_fwr1_mutations",
        "n_cdr1_mutations",
        "n_fwr2_mutations",
        "n_cdr2_mutations",
        "n_fwr3_mutations",
        "n_mutations",
        "n_v_mutations",
        "n_d_mutations",
        "n_j_mutations",
        "n_indels",
        "n_v_indels",
        "n_d_indels",
        "n_j_indels",
    ):
        if key in case.record:
            pred[key] = case.record[key]
    if frame == "presented" and case.orientation_id in (1, 3):
        start = case.record.get("junction_start")
        end = case.record.get("junction_end")
        if start is not None and end is not None:
            length = len(case.canonical_sequence)
            pred["junction_start"] = length - end
            pred["junction_end"] = length - start
    return pred


def normalize_call_set(pred: dict[str, Any], gene: str) -> tuple[str, ...]:
    """Extract a predicted allele set for one gene from flexible prediction keys."""

    calls = pred.get(f"{gene}_calls")
    if calls is None:
        call = pred.get(f"{gene}_call")
        if call is None:
            return ()
        calls = call.split(",") if isinstance(call, str) else [call]
    if isinstance(calls, str):
        calls = calls.split(",")
    return tuple(str(c).strip() for c in calls if str(c).strip())
