from dataclasses import replace

import pytest

from alignair_assay.scoring.slices import d_len_bin, slice_labels
from alignair_assay.truth.bench import truth_from_bench_case


@pytest.mark.parametrize("length,label", [
    (None, "absent"), (0, "<5"), (4, "<5"), (5, "5-7"), (7, "5-7"), (8, "8-9"), (9, "8-9"), (10, "10-11"),
    (11, "10-11"), (12, "12-14"), (14, "12-14"), (15, ">=15"), (40, ">=15"),
])
def test_d_len_bin(length, label):
    assert d_len_bin(length) == label


def test_d_inverted_slice(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    d = truth.segments["d"]
    assert slice_labels(truth)["d_inverted"] == "no"
    assert slice_labels(replace(truth, segments={**truth.segments, "d": replace(d, inverted=True)}))[
        "d_inverted"] == "yes"
    no_d = replace(truth, segments={g: s for g, s in truth.segments.items() if g != "d"})
    assert slice_labels(no_d)["d_inverted"] == "absent"


def test_slices_include_label_dimensions(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    labeled = replace(truth, labels={"shm_class": "igg", "design": "fr3"})
    got = slice_labels(labeled)
    assert got["shm_class"] == "igg" and got["design"] == "fr3"
    assert set(slice_labels(truth)) == {"stratum", "orientation", "d_len_bin", "d_inverted"}


def test_label_colliding_with_fixed_dimension_raises(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    with pytest.raises(ValueError, match="stratum"):
        slice_labels(replace(truth, labels={"stratum": "x"}))
