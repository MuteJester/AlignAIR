"""Coverage quotas, tracked contexts, and generation acceptance order."""
import pytest
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair_benchmark import (
    BenchmarkSpec,
    CoveragePlan,
    CoverageTracker,
    StratumSpec,
    allele_context_label,
    allele_stratification_contexts,
    case_coverage_labels,
    coverage_plan_from_spec,
    default_igh_assay_spec,
    generate_benchmark,
    generate_coverage_benchmark,
    run_online_benchmark,
)
from tests.alignair_benchmark._support import (_FakeReferenceSet, _manual_case, _small_spec)

pytestmark = pytest.mark.integration


def test_case_coverage_labels_include_core_assay_axes():
    case = generate_benchmark(_small_spec())[0]
    labels = set(case_coverage_labels(case))
    assert "stratum:clean" in labels
    assert "orientation:identity" in labels
    assert any(label.startswith("length:") for label in labels)
    assert any(label.startswith("mutation:") for label in labels)
    assert any(label.startswith("indel:") for label in labels)
    assert any(label.startswith("noise:") for label in labels)
    assert any(label.startswith("productivity:") for label in labels)
    assert any(label.startswith("junction_length:") for label in labels)
    assert any(label.startswith("segment_presence:") for label in labels)
    assert any(label.startswith("ambiguity:") for label in labels)
    assert any(label.startswith("allele:v:") for label in labels)


def test_coverage_planned_generation_accepts_until_quota_met():
    spec = BenchmarkSpec(
        name="planned",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=13,
        strata=(StratumSpec(name="orient", n=1, progress=0.2, orientation_ids=(0, 1, 2, 3)),),
    )
    plan = CoveragePlan(
        name="need_rev_comp",
        min_cases=1,
        min_counts={"orientation:reverse_complement": 1},
        max_candidates=6,
    )
    result = generate_coverage_benchmark(spec, gdata.HUMAN_IGH_OGRDB, plan=plan)
    assert result.report["satisfied"]
    assert result.report["observed_target_counts"]["orientation:reverse_complement"] == 1
    assert any(case.orientation_id == 1 for case in result.cases)
    assert len(result.cases) > 1


@pytest.mark.filterwarnings("error:.*multi-threaded.*fork.*:DeprecationWarning")
def test_parallel_coverage_planned_generation_preserves_acceptance_order():
    spec = BenchmarkSpec(
        name="planned_parallel",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=13,
        strata=(StratumSpec(name="orient", n=1, progress=0.2, orientation_ids=(0, 1, 2, 3)),),
    )
    plan = CoveragePlan(
        name="need_rev_comp",
        min_cases=1,
        min_counts={"orientation:reverse_complement": 1},
        max_candidates=6,
    )

    serial = generate_coverage_benchmark(spec, plan=plan, workers=1)
    parallel = generate_coverage_benchmark(spec, plan=plan, workers=2)
    assert [case.case_id for case in parallel.cases] == [case.case_id for case in serial.cases]
    assert parallel.report["satisfied"]
    assert parallel.report["generated_cases"] == serial.report["generated_cases"]
    assert parallel.report["generation_profile"]["mode"] == "coverage_planned"
    assert parallel.report["generation_profile"]["workers"] == 2
    assert parallel.report["generation_profile"]["generated_candidates"] == serial.report["generated_cases"]


def test_coverage_tracker_counts_targeted_allele_context_labels():
    label = allele_context_label("v", "IGHV1*01", "orientation:identity")
    plan = CoveragePlan(min_counts={label: 1})
    tracker = CoverageTracker(plan)
    case = _manual_case("c1", ("IGHV1*01",), orientation_id=0)
    assert tracker.missing_labels_for(case) == (label,)
    tracker.accept(case)
    assert tracker.to_dict()["observed_target_counts"][label] == 1
    assert tracker.satisfied


def test_coverage_plan_can_target_reference_allele_context_matrix():
    spec = BenchmarkSpec(
        name="matrix",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=19,
        strata=(StratumSpec(name="easy", n=1, progress=0.0),),
    )
    plan = coverage_plan_from_spec(
        spec,
        _FakeReferenceSet(),
        min_per_allele_context=2,
        allele_contexts=("stratum:easy",),
        min_cases=1,
    )
    assert plan.min_counts[allele_context_label("v", "IGHV1*01", "stratum:easy")] == 2
    assert plan.min_counts[allele_context_label("v", "IGHV2*01", "stratum:easy")] == 2
    assert plan.min_counts[allele_context_label("j", "IGHJ4*01", "stratum:easy")] == 2


def test_allele_stratification_contexts_include_recipe_strata_and_axes():
    spec = default_igh_assay_spec(n_per_stratum=1, n_per_focus=1)
    contexts = set(allele_stratification_contexts(spec))
    assert "stratum:clean_full" in contexts
    assert "stratum:forced_d_inversion" in contexts
    assert "orientation:reverse_complement" in contexts
    assert "tag:fragment" in contexts
    assert "mutation:>18%" in contexts


def test_coverage_planned_generation_reports_unmet_quota():
    spec = BenchmarkSpec(
        name="unmet",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=17,
        strata=(StratumSpec(name="clean", n=1, progress=0.0),),
    )
    plan = CoveragePlan(
        name="impossible",
        min_cases=1,
        min_counts={"not_a_real_label": 1},
        max_candidates=2,
    )
    result = generate_coverage_benchmark(spec, gdata.HUMAN_IGH_OGRDB, plan=plan)
    assert not result.report["satisfied"]
    assert result.report["unmet"]["not_a_real_label"] == 1
    assert result.report["generated_cases"] == 2


def test_online_benchmark_accepts_coverage_plan():
    spec = BenchmarkSpec(
        name="online_planned",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=19,
        strata=(StratumSpec(name="orient", n=1, progress=0.2, orientation_ids=(0, 1, 2, 3)),),
    )
    plan = coverage_plan_from_spec(
        spec,
        min_cases=1,
        required_labels={"orientation:reverse_complement": 1},
        max_candidates=6,
    )

    def predictor(reads):
        return [{} for _ in reads]

    report = run_online_benchmark(spec, predictor, batch_size=2, coverage_plan=plan, contract_level="minimal")
    assert report["generation_coverage"]["satisfied"]
    assert report["generation_coverage"]["observed_target_counts"]["orientation:reverse_complement"] == 1
    assert report["prediction_validation"]["valid_fraction"] == 0.0
    assert report["assay"]["summary"]["n_criteria"] == len(report["criteria"])
