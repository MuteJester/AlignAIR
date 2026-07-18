"""Shared scoring primitives."""
from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any

from ...core.schema import BenchmarkCase

_CIGAR_RE = re.compile(r"([0-9]+)([A-Za-z=])")


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else float("nan")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "t", "true", "yes", "y", "productive", "in_frame", "inverted"}:
        return True
    if text in {"0", "f", "false", "no", "n", "nonproductive", "out_of_frame", "forward"}:
        return False
    return None


def pred_value(pred: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in pred and not is_missing(pred[key]):
            return pred[key]
    return None


def field_presence(pred: dict[str, Any], keys: tuple[str, ...]) -> float:
    if not keys:
        return float("nan")
    present = sum(1 for key in keys if key in pred and not is_missing(pred[key]))
    return present / len(keys)


def coord(pred: dict[str, Any], gene: str, suffix: str) -> float | None:
    keys = {
        "ss": (f"{gene}_sequence_start", f"{gene}_start"),
        "se": (f"{gene}_sequence_end", f"{gene}_end"),
        "gs": (f"{gene}_germline_start",),
        "ge": (f"{gene}_germline_end",),
    }[suffix]
    return as_float(pred_value(pred, *keys))


def string_exact(pred_value: Any, truth_value: Any, *, case_sensitive: bool = False) -> float | None:
    if pred_value is None or truth_value is None:
        return None
    pred_text = str(pred_value)
    truth_text = str(truth_value)
    if not case_sensitive:
        pred_text = pred_text.upper()
        truth_text = truth_text.upper()
    return 1.0 if pred_text == truth_text else 0.0


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cigar_has_gap(cigar: Any) -> bool | None:
    if is_missing(cigar):
        return None
    return any(op in {"I", "D", "N"} for _, op in _CIGAR_RE.findall(str(cigar)))


def interval_iou(
    pred_start: float | None,
    pred_end: float | None,
    truth_start: float | None,
    truth_end: float | None,
) -> float | None:
    if None in (pred_start, pred_end, truth_start, truth_end):
        return None
    inter = max(0.0, min(pred_end, truth_end) - max(pred_start, truth_start))
    union = max(pred_end, truth_end) - min(pred_start, truth_start)
    return inter / union if union > 0 else 0.0


def oriented_interval(case: BenchmarkCase, start_key: str, end_key: str, frame: str) -> tuple[float | None, float | None]:
    start = as_float(case.record.get(start_key))
    end = as_float(case.record.get(end_key))
    if start is None or end is None:
        return None, None
    if frame == "presented" and case.orientation_id in (1, 3):
        length = len(case.canonical_sequence)
        return float(length) - end, float(length) - start
    return start, end
