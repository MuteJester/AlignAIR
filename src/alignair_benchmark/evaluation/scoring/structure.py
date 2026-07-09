"""Segment ordering and span-consistency scoring."""
from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase, GENES
from .primitives import coord


def score_segment_order(pred: dict[str, Any], case: BenchmarkCase, frame: str) -> dict[str, float]:
    truth = case.truth(frame)
    present_genes = [g for g in GENES if truth.get(g) and truth[g].present]
    if not present_genes:
        return {}
    spans = []
    missing = False
    negative = 0
    for gene in present_genes:
        start = coord(pred, gene, "ss")
        end = coord(pred, gene, "se")
        if start is None or end is None:
            missing = True
            continue
        negative += int(end < start)
        spans.append((gene, start, end))
    if not spans:
        return {"vdj_order_valid": 0.0, "overlap_rate": 1.0, "negative_span_rate": 1.0}
    ordered = True
    overlaps = 0
    pair_count = 0
    for (_, _, prev_end), (_, next_start, _) in zip(spans, spans[1:]):
        pair_count += 1
        if next_start < prev_end:
            overlaps += 1
            ordered = False
    if missing or negative:
        ordered = False
    return {
        "vdj_order_valid": 1.0 if ordered else 0.0,
        "overlap_rate": overlaps / pair_count if pair_count else 0.0,
        "negative_span_rate": negative / len(present_genes),
    }
