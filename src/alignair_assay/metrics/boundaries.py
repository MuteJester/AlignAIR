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
    """Germline-frame edges of an observed segment; an inverted D has none.

    The predicted coordinates are in the called allele's frame and are moved into the truth (primary allele)
    frame (see ``_frame_shift``). Without a top-1 call in the truth set the frames are not comparable: the metrics
    still apply but are ``nan`` (not covered).
    """
    if truth.inverted:
        return {}
    shift = _frame_shift(truth, pred.call, reference) if pred.call in truth.alleles.alleles else None
    if shift is None:
        return {f"{prefix}.{edge}_{kind}": NAN for edge in ("germ_start", "germ_end") for kind in ("exact", "err")}
    starts = None if pred.germ_start is None else pred.germ_start - shift
    ends = None if pred.germ_end is None else pred.germ_end - shift
    return {**_edge(prefix, "germ_start", starts, truth.boundaries.germ_start),
            **_edge(prefix, "germ_end", ends, truth.boundaries.germ_end)}


def _frame_shift(truth: SegmentTruth, call: str, reference: Reference) -> int | None:
    """Offset of the called allele's frame from the primary's; None for a call not in the reference.

    The observed germline is placed in the called allele: at its occurrence nearest the truth position (the
    primary allele's own frame is exact by construction); when it does not occur (a simulator tie differing
    inside the observed span), at offset 0 for an allele as long as the primary, else at the best-Hamming offset.
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
    if hits:
        return min(hits, key=lambda p: abs(p - obs.germ_start)) - obs.germ_start
    if len(seq) == len(reference.seq(obs.allele)):
        return 0
    return _best_hamming_offset(obs.observed_germline, seq, obs.germ_start) - obs.germ_start


def _best_hamming_offset(s: str, seq: str, near: int) -> int:
    """Offset of ``s`` against ``seq`` with the fewest mismatches (overhanging bases count as mismatches); ties go
    to the offset nearest ``near``, then the lower one."""
    def cost(p: int) -> int:
        return sum(not 0 <= p + k < len(seq) or seq[p + k] != base for k, base in enumerate(s))

    return min(range(1 - len(s), len(seq)), key=lambda p: (cost(p), abs(p - near), p))
