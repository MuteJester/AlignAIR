"""Group metrics: ``clone.agree_modal``, scored over the members of each simulated clone."""

import pytest
from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.metrics import is_informational, lower_is_better
from alignair_assay.metrics.groups import clone_agreement
from alignair_assay.truth.model import ObservableTruth

pytestmark = pytest.mark.component

TRUTH_KEY = (frozenset({"IGHV1-2"}), frozenset({"IGHJ4"}), 48)
K = ("IGHV1-2*02", "IGHJ4*02", 48)
L = ("IGHV3-23*01", "IGHJ4*02", 48)


def _truth(case_id, clone=None, clone_key=TRUTH_KEY, excluded=None):
    return ObservableTruth(
        case_id=case_id, stratum="lineages", sequence="ACGT", orientation=0, segments={}, unscorable=frozenset(),
        junction=None, vj_in_frame=None, stop_codon=None, productive=None, clone_key=clone_key, v_mismatches=None,
        v_frameshift=None, j_frameshift=None, excluded=excluded, warnings=(),
        groups={} if clone is None else {"clone": clone})


def _pred(case_id, key, *, v_set=(), j_set=()):
    """A prediction whose top-1 V and J calls and junction length are ``key``; ``None`` parts are left out."""
    v, j, length = key
    segments = {}
    if v is not None:
        segments["v"] = PredSegment(call=v, call_set=(v, *v_set))
    if j is not None:
        segments["j"] = PredSegment(call=j, call_set=(j, *j_set))
    return Prediction(sequence_id=case_id, segments=segments,
                      junction_nt=None if length is None else "A" * length)


def _scores(truths, keys):
    preds = {t.case_id: None if key is None else _pred(t.case_id, key) for t, key in zip(truths, keys)}
    return {cid: case["clone.agree_modal"] for cid, case in clone_agreement(truths, preds).items()}


def test_clone_agreement_hand():
    truths = [_truth(f"c{i}", clone="7") for i in range(4)]
    assert _scores(truths, [K, K, K, L]) == {"c0": 1.0, "c1": 1.0, "c2": 1.0, "c3": 0.0}
    # Every key missing (no row, no call, no junction): no modal key, so every member fails.
    none = [None, (None, "IGHJ4*02", 48), ("IGHV1-2*02", None, 48), ("IGHV1-2*02", "IGHJ4*02", None)]
    assert _scores(truths, none) == dict.fromkeys(("c0", "c1", "c2", "c3"), 0.0)
    # A missing key never wins the vote, and fails against the modal key.
    assert _scores(truths, [None, None, L, None]) == {"c0": 0.0, "c1": 0.0, "c2": 1.0, "c3": 0.0}

    # A clone with one eligible member is absent, as are cases without a clone or a truth clone key; an
    # ineligible member does not count towards its clone's size.
    mixed = [_truth("a0", clone="1"), _truth("a1", clone="1", clone_key=None), _truth("b0"),
             _truth("x0", clone="2", excluded="non_biological_orientation"), _truth("x1", clone="2"),
             _truth("y0", clone="3"), _truth("y1", clone="3")]
    assert _scores(mixed, [K, K, K, K, K, L, L]) == {"y0": 1.0, "y1": 1.0}


def test_clone_agreement_uses_top1_genes():
    truths = [_truth(f"c{i}", clone="7") for i in range(3)]
    preds = {
        # Same gene, other allele, other call set: the same key.
        "c0": _pred("c0", ("IGHV1-2*02", "IGHJ4*02", 48), v_set=("IGHV1-69*01",)),
        "c1": _pred("c1", ("IGHV1-2*04", "IGHJ4*03", 48), j_set=("IGHJ5*01",)),
        # Another junction length: another key.
        "c2": _pred("c2", ("IGHV1-2*02", "IGHJ4*02", 51)),
    }
    scores = clone_agreement(truths, preds)
    assert {cid: case["clone.agree_modal"] for cid, case in scores.items()} == {"c0": 1.0, "c1": 1.0, "c2": 0.0}
    assert all(set(case) == {"clone.agree_modal"} for case in scores.values())
    assert not lower_is_better("clone.agree_modal") and not is_informational("clone.agree_modal")


def test_clone_agreement_tie_rule():
    truths = [_truth(f"c{i}", clone="7") for i in range(4)]
    # A tie goes to the key with the smallest repr, whatever the order of the members.
    assert _scores(truths, [L, L, K, K]) == {"c0": 0.0, "c1": 0.0, "c2": 1.0, "c3": 1.0}
    assert _scores(truths, [K, L, K, L]) == {"c0": 1.0, "c1": 0.0, "c2": 1.0, "c3": 0.0}
    # repr order, not numeric order: "('IGHV1-2', 'IGHJ4', 51)" < "('IGHV1-2', 'IGHJ4', 9)".
    short, long_ = ("IGHV1-2*02", "IGHJ4*02", 9), ("IGHV1-2*02", "IGHJ4*02", 51)
    assert _scores(truths, [short, short, long_, long_]) == {"c0": 0.0, "c1": 0.0, "c2": 1.0, "c3": 1.0}
