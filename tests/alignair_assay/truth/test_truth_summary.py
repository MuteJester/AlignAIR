from dataclasses import replace

from alignair_assay.truth.bench import truth_from_bench_case
from alignair_assay.truth.summary import simulator_disagreements, truth_summary

FIXTURE_NAMES = ("amplicon_janchor_000000.json", "orientation_000001.json", "orientation_000002.json")


def test_truth_summary(reference, load_fixture):
    truths = [truth_from_bench_case(load_fixture(name), reference) for name in FIXTURE_NAMES]
    truths[0] = replace(truths[0], warnings=("v: example warning",), unscorable=frozenset({"j"}))
    summary = truth_summary(truths)
    assert summary["cases"] == 3
    assert summary["excluded"] == {"non_biological_orientation": 1}
    assert summary["warnings"] == {truths[0].case_id: ["v: example warning"]}
    assert summary["unscorable"] == {"v": 0, "d": 0, "j": 1}
    assert summary["simulator_disagreements"] == simulator_disagreements(truths)
    assert set(summary["simulator_disagreements"]) == {"v", "d", "j"}


def test_simulator_disagreements_skip_excluded(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    v = truth.segments["v"]
    disagreeing = replace(truth, segments={**truth.segments, "v": replace(
        v, alleles=replace(v.alleles, disagrees_with_simulator=True))})
    assert simulator_disagreements([disagreeing])["v"] == 1
    assert simulator_disagreements([replace(disagreeing, excluded="x")])["v"] == 0
