"""Score one tool's prediction for one read against its observable truth."""
from __future__ import annotations

from alignair_assay.ingest.airr import GENES, PredSegment, Prediction
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.alleles import call_metrics, short_call_metrics, spurious_call
from alignair_assay.metrics.boundaries import germline_boundary_metrics, read_boundary_metrics
from alignair_assay.reference import Reference
from alignair_assay.truth.build import ObservableTruth

_NO_CALL = PredSegment()


def score_case(truth: ObservableTruth, pred: Prediction | None, reference: Reference, *,
               d_min_observed: int = 8) -> CaseScores:
    """Metric values for one read; ``{}`` for an excluded truth, ``pred=None`` when the tool emitted no row.

    Observed segments get allele and boundary metrics; a D with fewer than ``d_min_observed`` observed germline
    bases gets only the ``d_short`` call/abstention metrics. Segments of the locus absent from the read get a
    spurious-call metric. Genes whose truth could not be built (``truth.unscorable``) get no metric at all.
    """
    if truth.excluded is not None:
        return {}
    scores: CaseScores = {}
    for g in GENES:
        if g in truth.unscorable:
            continue
        p = pred.segments.get(g, _NO_CALL) if pred is not None else _NO_CALL
        seg = truth.segments.get(g)
        if seg is None:
            if reference.alleles.get(g):
                scores.update(spurious_call(g, p))
        elif g == "d" and seg.observed_length < d_min_observed:
            scores.update(short_call_metrics("d_short", seg.alleles, p))
        else:
            scores.update(call_metrics(g, seg.alleles, p))
            scores.update(read_boundary_metrics(g, seg.boundaries, p))
            scores.update(germline_boundary_metrics(g, seg, p, reference))
    return scores
