from dataclasses import replace

import pytest

from alignair_assay.scoring.slices import d_len_bin, slice_labels, v_len_bin
from alignair_assay.truth.bench import truth_from_bench_case

pytestmark = pytest.mark.component


@pytest.mark.parametrize("length,label", [
    (None, "absent"), (0, "<5"), (4, "<5"), (5, "5-7"), (7, "5-7"), (8, "8-9"), (9, "8-9"), (10, "10-11"),
    (11, "10-11"), (12, "12-14"), (14, "12-14"), (15, ">=15"), (40, ">=15"),
])
def test_d_len_bin(length, label):
    assert d_len_bin(length) == label


@pytest.mark.parametrize("length,label", [
    (None, "absent"), (0, "<20"), (19, "<20"), (20, "20-49"), (49, "20-49"), (50, "50-99"), (99, "50-99"),
    (100, "100-149"), (149, "100-149"), (150, "150-199"), (199, "150-199"), (200, "200-249"), (249, "200-249"),
    (250, ">=250"), (320, ">=250"),
])
def test_v_len_bin(length, label):
    assert v_len_bin(length) == label


def test_v_len_bin_slice_from_truth(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    v = truth.segments["v"]
    assert slice_labels(truth)["v_len_bin"] == v_len_bin(v.observed_length)
    no_v = replace(truth, segments={g: s for g, s in truth.segments.items() if g != "v"})
    assert slice_labels(no_v)["v_len_bin"] == "absent"


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
    assert set(slice_labels(truth)) == {"stratum", "orientation", "v_len_bin", "d_len_bin", "d_inverted"}


def test_label_colliding_with_fixed_dimension_raises(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    with pytest.raises(ValueError, match="stratum"):
        slice_labels(replace(truth, labels={"stratum": "x"}))
