"""Score one tool's prediction for one read against its observable truth."""
from __future__ import annotations

from alignair_assay.core.prediction import GENES, PredSegment, Prediction
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.alleles import call_metrics, short_call_metrics, spurious_call
from alignair_assay.metrics.boundaries import germline_boundary_metrics, read_boundary_metrics
from alignair_assay.metrics.flags import flag_metrics
from alignair_assay.metrics.junction import clone_key_metric, junction_metrics
from alignair_assay.metrics.shm import germline_metrics, indel_metrics
from alignair_assay.truth.model import ObservableTruth

_NO_CALL = PredSegment()

# Default minimum observed germline bases per segment below which a call is too unreliable to require.
MIN_OBSERVED = {"v": 20, "d": 8, "j": 10}


def score_case(truth: ObservableTruth, pred: Prediction | None, reference: Reference, *,
               v_min_observed: int = MIN_OBSERVED["v"], d_min_observed: int = MIN_OBSERVED["d"],
               j_min_observed: int = MIN_OBSERVED["j"]) -> CaseScores:
    """Metric values for one read; ``{}`` for an excluded truth, ``pred=None`` when the tool emitted no row.

    Observed segments get allele and boundary metrics; a segment with fewer observed germline bases than its
    ``{g}_min_observed`` gets only the ``{g}_short`` call/abstention metrics instead of every ``{g}.*`` allele,
    boundary, indel and SHM metric. Segments of the locus absent from the read get a
    spurious-call metric. Genes whose truth could not be built (``truth.unscorable``) get no metric at all.
    Observed V and J also get indel metrics, and observed V mutation-count and germline-reconstruction metrics.
    A missing row fails every accuracy metric (abstention included) and leaves error and presence metrics
    uncovered.
    Junction, clone-key and productivity metrics apply when the truth defines them; orientation and AIRR validity
    (the row's sequence is the read and its junction coordinates reproduce its junction) are scored for every read.
    """
    if truth.excluded is not None:
        return {}
    scores: CaseScores = {}
    missing = pred is None
    min_observed = {"v": v_min_observed, "d": d_min_observed, "j": j_min_observed}
    for g in GENES:
        if g in truth.unscorable:
            continue
        p = pred.segments.get(g, _NO_CALL) if pred is not None else _NO_CALL
        seg = truth.segments.get(g)
        if seg is None:
            if reference.alleles.get(g):
                scores.update(spurious_call(g, p, missing=missing))
        elif seg.observed_length < min_observed[g]:
            scores.update(short_call_metrics(f"{g}_short", seg.alleles, p, missing=missing))
        else:
            scores.update(call_metrics(g, seg.alleles, p))
            scores.update(read_boundary_metrics(g, seg.boundaries, p))
            scores.update(germline_boundary_metrics(g, seg, p, reference))
            if g in ("v", "j"):
                scores.update(indel_metrics(g, seg, p))
            if g == "v":
                scores.update(germline_metrics(g, seg, truth.sequence, p, reference))
    if truth.junction is not None:
        scores.update(junction_metrics(truth.junction, truth.sequence, pred,
                                       frame_unambiguous=truth.frame_unambiguous))
    if truth.clone_key is not None:
        scores.update(clone_key_metric(truth.clone_key, pred))
    scores.update(flag_metrics(truth, pred))
    return scores
