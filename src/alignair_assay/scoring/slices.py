"""Slice dimensions: the labels a case is summarized under, derived from its truth."""
from __future__ import annotations

from alignair_assay.metrics import CaseScores
from alignair_assay.stats import by_slice
from alignair_assay.truth.model import ObservableTruth

SLICE_DIMENSIONS = ("stratum", "orientation", "d_len_bin", "d_inverted")
_D_BINS = ((5, "<5"), (8, "5-7"), (10, "8-9"), (12, "10-11"), (15, "12-14"))


def d_len_bin(length: int | None) -> str:
    """Slice label for an observed D length; ``absent`` when the read shows no D."""
    if length is None:
        return "absent"
    return next((label for upper, label in _D_BINS if length < upper), ">=15")


def slice_labels(truth: ObservableTruth) -> dict[str, str]:
    """The case's label in every slice dimension, derived from its truth."""
    d = truth.segments.get("d")
    return {"stratum": truth.stratum, "orientation": str(truth.orientation),
            "d_len_bin": d_len_bin(d.observed_length if d is not None else None),
            "d_inverted": "absent" if d is None else "yes" if d.inverted else "no"}


def slices(scores: dict[str, CaseScores], truths: list[ObservableTruth]) -> dict:
    """``{dimension: {label: {metric: MetricSummary}}}`` over the scored cases, for every slice dimension."""
    labels = {t.case_id: slice_labels(t) for t in truths if t.case_id in scores}
    return {dim: by_slice(scores, {cid: lab[dim] for cid, lab in labels.items()}) for dim in SLICE_DIMENSIONS}
