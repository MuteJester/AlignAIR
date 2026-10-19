"""Junction and clone-key metrics: the CDR3 region that defines a clone.

Predicted junction coordinates are the tool's ``junction_start``/``junction_end`` (canonical frame); a coordinate
the tool did not report is taken from the position of its junction string in the read, when that string occurs
exactly once.
``junction.start_err``/``junction.end_err`` are signed (predicted - truth) and informational; tool comparisons
use their ``*_abs_err`` twins.
"""
from __future__ import annotations

from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.core.reference import gene_name
from alignair_assay.core.seqops import translate
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.alleles import predicted_set
from alignair_assay.truth.model import JunctionTruth

NAN = float("nan")

_NO_CALL = PredSegment()


def junction_metrics(truth: JunctionTruth, sequence: str, pred: Prediction | None, *,
                     frame_unambiguous: bool = True) -> CaseScores:
    """Junction string, translation, length and edge metrics; ``sequence`` is the canonical read.

    The translation is scored only for an in-frame truth junction on a frame-unambiguous read (no V/J
    frameshift; see ``ObservableTruth.frame_unambiguous``).
    """
    nt = _pred_nt(pred)
    scores: CaseScores = {
        "junction.nt_exact": float(nt == truth.nt),
        "junction.length_exact": float(nt is not None and len(nt) == truth.length),
    }
    if truth.aa is not None and frame_unambiguous:
        scores["junction.aa_exact"] = float(_pred_aa(pred, nt) == truth.aa)
    start, end = _pred_span(pred, nt, sequence)
    scores.update(_signed_edge("start", start, truth.start))
    scores.update(_signed_edge("end", end, truth.end))
    return scores


def clone_key_metric(truth: tuple[frozenset[str], frozenset[str], int], pred: Prediction | None) -> CaseScores:
    """1.0 when the predicted V and J call sets each name a truth gene and the junction length is right."""
    v_genes, j_genes, length = truth
    nt = _pred_nt(pred)  # None whenever pred is None
    ok = (nt is not None and len(nt) == length
          and bool(_genes(pred, "v") & v_genes) and bool(_genes(pred, "j") & j_genes))
    return {"clone_key.exact": float(ok)}


def _genes(pred: Prediction, g: str) -> frozenset[str]:
    return frozenset(gene_name(a) for a in predicted_set(pred.segments.get(g, _NO_CALL)))


def _pred_nt(pred: Prediction | None) -> str | None:
    if pred is None or not pred.junction_nt:
        return None
    return pred.junction_nt.upper()


def _pred_aa(pred: Prediction | None, nt: str | None) -> str | None:
    """The tool's junction translation, or the translation of its junction string when it gave none."""
    if pred is not None and pred.junction_aa:
        return pred.junction_aa.upper()
    if nt is not None and len(nt) % 3 == 0:
        return translate(nt)
    return None


def _pred_span(pred: Prediction | None, nt: str | None, sequence: str) -> tuple[int | None, int | None]:
    if pred is None:
        return None, None
    start, end = pred.junction_start, pred.junction_end
    if (start is None or end is None) and nt is not None:
        found = _unique_find(sequence, nt)
        if found is not None:
            start = found if start is None else start
            end = found + len(nt) if end is None else end
    return start, end


def _unique_find(sequence: str, s: str) -> int | None:
    """Position of ``s`` in ``sequence`` when it occurs exactly once (overlapping occurrences count), else None."""
    first = sequence.find(s)
    if first == -1 or sequence.find(s, first + 1) != -1:
        return None
    return first


def _signed_edge(edge: str, value: int | None, truth: int) -> CaseScores:
    if value is None:
        return {f"junction.{edge}_err": NAN, f"junction.{edge}_abs_err": NAN}
    err = value - truth
    return {f"junction.{edge}_err": float(err), f"junction.{edge}_abs_err": float(abs(err))}
