"""Slice dimensions: the labels a case is summarized under, derived from its truth."""
from __future__ import annotations

from alignair_assay.metrics import CaseScores
from alignair_assay.stats import by_slice
from alignair_assay.truth.model import ObservableTruth

SLICE_DIMENSIONS = ("stratum", "orientation", "v_len_bin", "d_len_bin", "d_inverted")
# (exclusive upper bound, label) per bin, then the label of every longer length.
_V_BINS = ((20, "<20"), (50, "20-49"), (100, "50-99"), (150, "100-149"), (200, "150-199"), (250, "200-249"))
_D_BINS = ((5, "<5"), (8, "5-7"), (10, "8-9"), (12, "10-11"), (15, "12-14"))


def v_len_bin(length: int | None) -> str:
    """Slice label for an observed V length in germline bases: ``<20``, ``20-49``, ``50-99``, ``100-149``,
    ``150-199``, ``200-249`` or ``>=250``; ``absent`` when the read shows no V."""
    return _length_bin(length, _V_BINS, ">=250")


def d_len_bin(length: int | None) -> str:
    """Slice label for an observed D length; ``absent`` when the read shows no D."""
    return _length_bin(length, _D_BINS, ">=15")


def _length_bin(length: int | None, bins: tuple[tuple[int, str], ...], longest: str) -> str:
    if length is None:
        return "absent"
    return next((label for upper, label in bins if length < upper), longest)


def slice_labels(truth: ObservableTruth) -> dict[str, str]:
    """The case's label in every slice dimension: the fixed dimensions derived from its truth plus every key of
    ``truth.labels``. A label key equal to a fixed dimension raises ValueError."""
    clash = sorted(set(truth.labels) & set(SLICE_DIMENSIONS))
    if clash:
        raise ValueError(f"{truth.case_id}: label {clash} collides with a fixed slice dimension")
    v, d = truth.segments.get("v"), truth.segments.get("d")
    fixed = {"stratum": truth.stratum, "orientation": str(truth.orientation),
             "v_len_bin": v_len_bin(v.observed_length if v is not None else None),
             "d_len_bin": d_len_bin(d.observed_length if d is not None else None),
             "d_inverted": "absent" if d is None else "yes" if d.inverted else "no"}
    return {**truth.labels, **fixed}


def slices(scores: dict[str, CaseScores], truths: list[ObservableTruth]) -> dict:
    """``{dimension: {label: {metric: MetricSummary}}}`` over the scored cases, for the fixed dimensions and every
    label key of the truths; a case without a label key is in the ``absent`` slice of that dimension."""
    labels = {t.case_id: slice_labels(t) for t in truths if t.case_id in scores}
    dimensions = SLICE_DIMENSIONS + tuple(sorted({k for lab in labels.values() for k in lab} - set(SLICE_DIMENSIONS)))
    return {dim: by_slice(scores, {cid: lab.get(dim, "absent") for cid, lab in labels.items()})
            for dim in dimensions}
