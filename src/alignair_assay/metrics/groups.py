"""Group metrics: a case's score depends on the tool's predictions for the other cases of its group.

``clone.agree_modal`` asks whether a tool gives every member of a simulated clone the same clone assignment, the
"first call" grouping convention: a case's predicted key is (gene of its top-1 V call, gene of its top-1 J call,
junction length), and it agrees when that key is its clone's modal key. Set-overlap clone keys are scored per read
by ``clone_key.exact``. Members of a clone share their recombination, so the oracle's key (the generative
primary's genes and the truth junction length) is the same for every member and the oracle always agrees.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from alignair_assay.core.prediction import Prediction
from alignair_assay.core.reference import gene_name
from alignair_assay.metrics import CaseScores
from alignair_assay.metrics.junction import predicted_junction_nt
from alignair_assay.truth.model import ObservableTruth

PredictedKey = tuple[str, str, int]


def clone_agreement(truths: Iterable[ObservableTruth],
                    preds: Mapping[str, Prediction | None]) -> dict[str, CaseScores]:
    """``{"clone.agree_modal": 1.0 or 0.0}`` per eligible case id, in truth order.

    Eligible cases are scored (not excluded), carry ``groups["clone"]`` and a truth ``clone_key``, and belong to a
    clone with at least two such cases; every other case is absent. A case's predicted key is (gene of its top-1 V
    call, gene of its top-1 J call, length of its junction string) and is None when any part is missing, including
    a missing row (absent from ``preds`` or None). A clone's modal key is its most common non-None key; ties go to
    the key with the smallest ``repr``. A case scores 1.0 when its key is the modal key and 0.0 otherwise, a None
    key included (so a clone without any key scores 0.0 throughout).
    """
    clone_of = {truth.case_id: truth.groups["clone"] for truth in truths
                if truth.excluded is None and truth.clone_key is not None and "clone" in truth.groups}
    sizes = Counter(clone_of.values())
    keys = {case_id: predicted_key(preds.get(case_id)) for case_id, clone in clone_of.items() if sizes[clone] >= 2}
    members: dict[str, list[PredictedKey | None]] = {}
    for case_id, key in keys.items():
        members.setdefault(clone_of[case_id], []).append(key)
    modal = {clone: _modal(clone_keys) for clone, clone_keys in members.items()}
    return {case_id: {"clone.agree_modal": float(key is not None and key == modal[clone_of[case_id]])}
            for case_id, key in keys.items()}


def predicted_key(pred: Prediction | None) -> PredictedKey | None:
    """(gene of the top-1 V call, gene of the top-1 J call, junction length in nucleotides) of a prediction; None
    without a row, either call or a junction string."""
    if pred is None:
        return None
    v, j = pred.segments.get("v"), pred.segments.get("j")
    nt = predicted_junction_nt(pred)
    if v is None or v.call is None or j is None or j.call is None or nt is None:
        return None
    return gene_name(v.call), gene_name(j.call), len(nt)


def _modal(keys: Iterable[PredictedKey | None]) -> PredictedKey | None:
    """The most common non-None key, ties to the smallest ``repr``; None when every key is None."""
    counts = Counter(key for key in keys if key is not None)
    if not counts:
        return None
    top = max(counts.values())
    return min((key for key, count in counts.items() if count == top), key=repr)
