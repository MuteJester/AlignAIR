"""Segment-edge metrics: predicted coordinates against the truth equivalence intervals.

``*_exact`` is 1.0 when the predicted coordinate lies inside the inclusive truth interval (0.0 when missing);
``*_err`` is the distance to the interval (0 inside, ``nan`` when missing).
"""
from __future__ import annotations

from alignair_assay.ingest.airr import PredSegment
from alignair_assay.metrics import CaseScores
from alignair_assay.reference import Reference
from alignair_assay.truth.boundaries import BoundaryTruth, Interval
from alignair_assay.truth.build import SegmentTruth

NAN = float("nan")


def interval_distance(x: int, interval: Interval) -> int:
    lo, hi = interval
    return max(lo - x, x - hi, 0)


def _edge(prefix: str, edge: str, value: int | None, interval: Interval) -> CaseScores:
    if value is None:
        return {f"{prefix}.{edge}_exact": 0.0, f"{prefix}.{edge}_err": NAN}
    err = interval_distance(value, interval)
    return {f"{prefix}.{edge}_exact": float(err == 0), f"{prefix}.{edge}_err": float(err)}


def read_boundary_metrics(prefix: str, truth: BoundaryTruth, pred: PredSegment) -> CaseScores:
    """Read-frame edges; scored for every observed segment, whatever the call."""
    return {**_edge(prefix, "seq_start", pred.seq_start, truth.seq_start),
            **_edge(prefix, "seq_end", pred.seq_end, truth.seq_end)}


def germline_boundary_metrics(prefix: str, truth: SegmentTruth, pred: PredSegment,
                              reference: Reference) -> CaseScores:
    """Germline-frame edges; scored only when the top-1 call is in the truth set, never for an inverted D.

    The predicted coordinates are in the called allele's frame and are moved into the truth (primary allele)
    frame by the offset of the observed germline inside the called allele.
    """
    if truth.inverted or pred.call not in truth.alleles.alleles:
        return {}
    shift = _frame_shift(truth, pred.call, reference)
    if shift is None:
        return {}
    starts = None if pred.germ_start is None else pred.germ_start - shift
    ends = None if pred.germ_end is None else pred.germ_end - shift
    return {**_edge(prefix, "germ_start", starts, truth.boundaries.germ_start),
            **_edge(prefix, "germ_end", ends, truth.boundaries.germ_end)}


def _frame_shift(truth: SegmentTruth, call: str, reference: Reference) -> int | None:
    """Offset of the called allele's frame from the primary's, or None if the observed germline is absent.

    When the observed germline occurs more than once in the called allele, the occurrence nearest the truth
    position is taken (the primary allele's own frame is exact by construction).
    """
    obs = truth.observed
    if call == obs.allele:
        return 0
    if not reference.has(call):
        return None
    seq = reference.seq(call)
    hits = []
    pos = seq.find(obs.observed_germline)
    while pos != -1:
        hits.append(pos)
        pos = seq.find(obs.observed_germline, pos + 1)
    if not hits:
        return None
    return min(hits, key=lambda p: abs(p - obs.germ_start)) - obs.germ_start
