from dataclasses import replace

from alignair_assay.truth.bench import truth_from_bench_case
from alignair_assay.truth.summary import simulator_calls, truth_summary

FIXTURE_NAMES = ("amplicon_janchor_000000.json", "orientation_000001.json", "orientation_000002.json")


def test_truth_summary(reference, load_fixture):
    truths = [truth_from_bench_case(load_fixture(name), reference) for name in FIXTURE_NAMES]
    truths[0] = replace(truths[0], warnings=("v: example warning",), unscorable=frozenset({"j"}))
    summary = truth_summary(truths)
    assert summary["cases"] == 3
    assert summary["excluded"] == {"non_biological_orientation": 1}
    assert summary["warnings"] == {truths[0].case_id: ["v: example warning"]}
    assert summary["unscorable"] == {"v": 0, "d": 0, "j": 1}
    assert summary["simulator_calls"] == simulator_calls(truths)
    assert set(summary["simulator_calls"]) == {"v", "d", "j"}
    assert all(set(counts) == {"call_list_differs", "recalled"} for counts in summary["simulator_calls"].values())


def test_simulator_calls_count_call_lists_and_recalls(reference, load_fixture):
    truth = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    v = truth.segments["v"]

    def with_v(**changes):
        return replace(truth, segments={**truth.segments, "v": replace(v, alleles=replace(v.alleles, **changes))})

    # The simulator's first call is another allele: the call list differs and the read was re-called.
    recalled = with_v(disagrees_with_simulator=True, primary_fallback=False, simulator_ties=frozenset({"IGHV9-99*01"}))
    assert recalled.segments["v"].alleles.recalled
    assert simulator_calls([recalled])["v"] == {"call_list_differs": 1, "recalled": 1}
    assert simulator_calls([replace(recalled, excluded="x")])["v"] == {"call_list_differs": 0, "recalled": 0}
    # A tie list holding the generative allele (listed first by GenAIRR) differs without a re-call.
    ties = with_v(disagrees_with_simulator=True, primary_fallback=False,
                  simulator_ties=frozenset({v.alleles.primary, "IGHV9-99*01"}))
    assert simulator_calls([ties])["v"] == {"call_list_differs": 1, "recalled": 0}
    # No call at all is not a re-call, and a fallback primary is the first call by definition.
    assert not with_v(primary_fallback=False, simulator_ties=frozenset()).segments["v"].alleles.recalled
    assert not with_v(primary_fallback=True, simulator_ties=frozenset({"IGHV9-99*01"})).segments["v"].alleles.recalled
