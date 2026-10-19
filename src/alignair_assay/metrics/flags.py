"""Read-level flag metrics: productivity calls, orientation and AIRR validity.

A missing prediction or flag is wrong (0.0). Productivity flags are scored only where the truth defines them
(the truth junction exists).
"""
from __future__ import annotations

from alignair_assay.ingest.airr import Prediction
from alignair_assay.metrics import CaseScores
from alignair_assay.truth.build import ObservableTruth

_PRODUCTIVITY = ("productive", "vj_in_frame", "stop_codon")


def flag_metrics(truth: ObservableTruth, pred: Prediction | None) -> CaseScores:
    scores: CaseScores = {}
    for name in _PRODUCTIVITY:
        want = getattr(truth, name)
        if want is not None:
            scores[f"{name}.correct"] = float(pred is not None and getattr(pred, name) == want)
    rev_comp = pred.rev_comp if pred is not None else None
    scores["orientation.correct"] = float(rev_comp is not None and rev_comp == (truth.orientation == 1))
    scores["airr.valid"] = float(pred is not None and not pred.invalid)
    return scores
