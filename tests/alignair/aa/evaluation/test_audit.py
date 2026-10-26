import pytest
from alignair.aa.equivalence import build_equivalence_classes, db_hash
from alignair.aa.audit import (
    identifiability_summary, window_identifiability, prefix_identifiability_curve,
)

pytestmark = pytest.mark.component


def test_identifiability_summary_counts_classes_and_unique():
    aa = {"a*01": "MKV", "a*02": "MKV", "b*01": "MKL"}
    cmap = build_equivalence_classes("V", aa, db_hash([("a*01", "x")]))
    s = identifiability_summary(cmap)
    assert s["gene"] == "V"
    assert s["n_alleles"] == 3
    assert s["n_classes"] == 2
    assert s["n_unique_alleles"] == 1                        # only b*01 is alone
    assert abs(s["frac_alleles_uniquely_identifiable"] - 1 / 3) < 1e-9
    assert s["ambiguity_size_hist"] == {"1": 1, "2": 1}      # one class of 1, one of 2
    assert s["max_ambiguity"] == 2


def test_window_identifiability_collapses_on_short_window():
    aa = {"a": "MKVW", "b": "MKLW", "c": "MKVW"}
    # window [0,2) = "MK" for all -> 1 class, nothing unique.
    w = window_identifiability(aa, 0, 2)
    assert w["n_classes"] == 1 and w["frac_uniquely_identifiable"] == 0.0
    # full window: a==c collide, b alone -> 2 classes, only b unique.
    full = window_identifiability(aa, 0, 4)
    assert full["n_classes"] == 2
    assert abs(full["frac_uniquely_identifiable"] - 1 / 3) < 1e-9


def test_prefix_curve_is_monotonic_nondecreasing_in_classes():
    aa = {"a": "MKVW", "b": "MKLW", "c": "MQVW"}
    curve = prefix_identifiability_curve(aa, [1, 2, 3, 4])
    classes = [pt["n_classes"] for pt in curve]
    assert classes == sorted(classes)                        # more residues never merges classes
    assert curve[0]["lo"] == 0 and curve[0]["hi"] == 1
