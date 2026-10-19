"""Read-level flag metrics: productivity calls, orientation and AIRR validity.

A missing prediction or flag is wrong (0.0). Productivity flags are scored only where the truth defines them
(the truth junction exists); the stop-codon flag only on frame-unambiguous reads, where the frame does not depend
on how a tool resolves a frameshift.
"""
from __future__ import annotations

from alignair_assay.ingest.airr import Prediction
from alignair_assay.metrics import CaseScores
from alignair_assay.seqops import revcomp
from alignair_assay.truth.build import ObservableTruth

_PRODUCTIVITY = ("productive", "vj_in_frame", "stop_codon")


def flag_metrics(truth: ObservableTruth, pred: Prediction | None) -> CaseScores:
    scores: CaseScores = {}
    for name in _PRODUCTIVITY:
        want = getattr(truth, name)
        if name == "stop_codon" and not truth.frame_unambiguous:
            continue
        if want is not None:
            scores[f"{name}.correct"] = float(pred is not None and getattr(pred, name) == want)
    rev_comp = pred.rev_comp if pred is not None else None
    scores["orientation.correct"] = float(rev_comp is not None and rev_comp == (truth.orientation == 1))
    sequence_ok = pred is not None and _sequence_ok(truth, pred)
    scores["airr.sequence_ok"] = float(sequence_ok)
    scores["airr.valid"] = float(sequence_ok and not pred.invalid)
    return scores


def _sequence_ok(truth: ObservableTruth, pred: Prediction) -> bool:
    """The row's ``sequence`` is the read as presented to the tool and its junction coordinates, when given,
    reproduce its junction string in the canonical read.

    A row with ``rev_comp`` true may instead carry the reverse complement of the presented read, as IgBLAST
    does: that is the frame its coordinates refer to.
    """
    presented = revcomp(truth.sequence) if truth.orientation == 1 else truth.sequence
    sequence = (pred.sequence or "").upper()  # AIRR TSVs cannot tell an empty sequence from a missing one
    if sequence != presented and not (pred.rev_comp and sequence == revcomp(presented)):
        return False
    if pred.junction_nt and pred.junction_start is not None and pred.junction_end is not None:
        return truth.sequence[pred.junction_start:pred.junction_end] == pred.junction_nt.upper()
    return True
