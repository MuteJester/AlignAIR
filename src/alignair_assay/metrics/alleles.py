"""Allele-call and segment-presence metrics for one gene of one read.

The tool's top-1 is ``PredSegment.call``; its predicted set is ``call_set`` plus the top-1. A segment without
a top-1 call counts as no call.
"""
from __future__ import annotations

from alignair_assay.ingest.airr import PredSegment
from alignair_assay.metrics import CaseScores
from alignair_assay.reference import gene_name
from alignair_assay.truth.alleles import AlleleTruth

NAN = float("nan")


def predicted_set(pred: PredSegment) -> frozenset[str]:
    """Alleles the tool asserts for the segment; empty when it made no top-1 call."""
    if pred.call is None:
        return frozenset()
    return frozenset(pred.call_set) | {pred.call}


def call_metrics(prefix: str, truth: AlleleTruth, pred: PredSegment) -> CaseScores:
    """Allele metrics for an observed segment; a missing call is wrong, and its set statistics are ``nan``."""
    called = predicted_set(pred)
    hit = called & truth.alleles
    return {
        f"{prefix}.top1_in_set": float(pred.call in truth.alleles),
        f"{prefix}.gene_top1": float(pred.call is not None and gene_name(pred.call) in truth.genes),
        f"{prefix}.any_overlap": float(bool(hit)),
        f"{prefix}.set_precision": len(hit) / len(called) if called else NAN,
        f"{prefix}.set_size": float(len(called)) if called else NAN,
        f"{prefix}.no_call": float(pred.call is None),
    }


def short_call_metrics(prefix: str, truth: AlleleTruth, pred: PredSegment, *, missing: bool) -> CaseScores:
    """Metrics for a segment too short to call reliably: abstaining is as correct as a right call.

    ``missing`` (the tool emitted no row) is a failure, not an abstention.
    """
    return {
        f"{prefix}.top1_in_set": float(pred.call in truth.alleles),
        f"{prefix}.abstain_ok": 0.0 if missing else float(pred.call is None or pred.call in truth.alleles),
    }


def spurious_call(prefix: str, pred: PredSegment, *, missing: bool) -> CaseScores:
    """Presence metric for a segment absent from the read: 1.0 when the tool called it anyway; ``nan`` when the
    tool emitted no row (``missing``), which says nothing about presence."""
    return {f"{prefix}.spurious_call": NAN if missing else float(pred.call is not None)}
