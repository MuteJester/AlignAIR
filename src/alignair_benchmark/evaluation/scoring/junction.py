"""Junction and recombination-region scoring."""
from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase
from .primitives import as_float, interval_iou, oriented_interval, pred_value, string_exact


def score_junction(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    record = case.record
    out: dict[str, float] = {}
    truth_start, truth_end = oriented_interval(case, "junction_start", "junction_end", frame)
    pred_start = as_float(pred_value(pred, "junction_start", "cdr3_start"))
    pred_end = as_float(pred_value(pred, "junction_end", "cdr3_end"))
    if pred_start is not None and truth_start is not None:
        out["junction_start_mae"] = abs(pred_start - truth_start)
    if pred_end is not None and truth_end is not None:
        out["junction_end_mae"] = abs(pred_end - truth_end)
    iou = interval_iou(pred_start, pred_end, truth_start, truth_end)
    if iou is not None:
        out["cdr3_overlap_iou"] = iou

    pred_junction = pred_value(pred, "junction", "cdr3")
    exact = string_exact(pred_junction, record.get("junction"))
    if exact is not None:
        out["junction_nt_exact"] = exact
    pred_junction_aa = pred_value(pred, "junction_aa", "cdr3_aa")
    exact_aa = string_exact(pred_junction_aa, record.get("junction_aa"))
    if exact_aa is not None:
        out["junction_aa_exact"] = exact_aa

    truth_len = as_float(record.get("junction_length"))
    pred_len = as_float(pred_value(pred, "junction_length", "cdr3_length"))
    if pred_len is None and pred_junction is not None:
        pred_len = float(len(str(pred_junction)))
    if pred_len is not None and truth_len is not None:
        out["junction_length_mae"] = abs(pred_len - truth_len)
    if pred_junction_aa is not None and record.get("junction_aa") is not None:
        out["junction_aa_length_mae"] = abs(len(str(pred_junction_aa)) - len(str(record.get("junction_aa"))))

    for region, metric in (("np1", "n1"), ("np2", "n2")):
        pred_seq = pred_value(pred, region, metric)
        exact_region = string_exact(pred_seq, record.get(region))
        if exact_region is not None:
            out[f"{region}_exact"] = exact_region
        pred_region_len = as_float(pred_value(pred, f"{region}_length", f"{metric}_length"))
        if pred_region_len is None and pred_seq is not None:
            pred_region_len = float(len(str(pred_seq)))
        truth_region_len = as_float(record.get(f"{region}_length"))
        if pred_region_len is not None and truth_region_len is not None:
            out[f"{metric}_length_mae"] = abs(pred_region_len - truth_region_len)
            out[f"{region}_length_mae"] = abs(pred_region_len - truth_region_len)

    p_truth_parts = [
        as_float(record.get(key)) or 0.0
        for key in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")
    ]
    p_truth = sum(p_truth_parts)
    p_pred = as_float(pred_value(pred, "p_region_length"))
    if p_pred is None:
        p_pred_parts = [
            as_float(pred_value(pred, key)) or 0.0
            for key in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")
        ]
        if any(key in pred for key in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")):
            p_pred = sum(p_pred_parts)
    if p_pred is not None:
        out["p_region_length_mae"] = abs(p_pred - p_truth)
    return out
