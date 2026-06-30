from alignair.benchmark import (
    BenchmarkSpec,
    StratumSpec,
    criteria_catalog,
    default_igh_assay_spec,
    measurement_aligned_coverage_plan,
    measurement_required_contexts,
    measurement_scenario_catalog,
    validate_measurement_scenario_catalog,
)
from alignair.benchmark.generation.strata import adaptive_igh_strata, focused_igh_strata


def test_measurement_scenario_catalog_covers_criteria_and_default_assay_strata():
    spec = default_igh_assay_spec(n_per_stratum=1, n_per_focus=1)

    validation = validate_measurement_scenario_catalog(spec)
    scenarios = {row["name"]: row for row in measurement_scenario_catalog()}

    assert validation["valid"], validation["errors"]
    assert validation["n_represented_criteria"] == len(criteria_catalog())
    assert "adaptive_fragment_observability" in scenarios
    assert "genotype_masked_inference" in scenarios
    assert scenarios["adaptive_fragment_observability"]["status"] == "integrated"
    assert scenarios["genotype_masked_inference"]["status"] == "coverage_planned"


def test_measurement_required_contexts_feed_coverage_planning():
    spec = default_igh_assay_spec(n_per_stratum=1, n_per_focus=1)

    plan = measurement_aligned_coverage_plan(
        spec,
        min_cases=spec.n_cases,
        min_per_measurement_context=2,
        max_candidates=100,
    )

    contexts = set(measurement_required_contexts())
    assert "tag:adaptive" in contexts
    assert "measurement:allele_coverage_and_candidates" in contexts
    assert "measurement:genotype_masked_inference" in contexts
    assert "measurement:multi_locus_chain" in contexts
    assert "stratum:forced_d_inversion" in contexts
    assert "orientation:reverse_complement" in contexts
    spec_contexts = set(measurement_required_contexts(spec=spec))
    assert "measurement:allele_coverage_and_candidates" not in spec_contexts
    assert "measurement:genotype_masked_inference" not in spec_contexts
    assert "measurement:multi_locus_chain" not in spec_contexts
    assert "tag:adaptive" in spec_contexts
    assert plan.min_counts["tag:adaptive"] == 2
    assert plan.min_counts["stratum:forced_d_inversion"] == 2
    assert plan.min_counts["orientation:reverse_complement"] == 2
    assert "measurement:genotype_masked_inference" not in plan.min_counts
    assert plan.min_cases == spec.n_cases


def test_measurement_aligned_plan_keeps_explicit_dynamic_measurements():
    spec = BenchmarkSpec(
        name="explicit_genotype_measurement",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=1,
        strata=(
            StratumSpec(
                name="genotype_subject_000",
                n=1,
                progress=0.0,
                param_overrides={
                    "metadata": {
                        "benchmark_measurement": "genotype_masked_inference",
                    },
                },
            ),
        ),
    )

    plan = measurement_aligned_coverage_plan(
        spec,
        min_cases=1,
        min_per_measurement_context=2,
        max_candidates=10,
    )

    assert plan.min_counts["measurement:genotype_masked_inference"] == 2


def test_focused_strata_control_background_variables_for_isolation():
    strata = {stratum.name: stratum for stratum in focused_igh_strata(n_per_scenario=1)}

    d_inversion = strata["forced_d_inversion"].param_overrides
    assert d_inversion["invert_d_prob"] == 1.0
    assert d_inversion["mutation_rate"] == 0.005
    assert d_inversion["indel_count"] == (0, 0)
    assert d_inversion["seq_error_rate"] == 0.0
    assert d_inversion["ambiguous_count"] == (0, 0)

    high_indel = strata["high_indel_extreme"].param_overrides
    assert high_indel["indel_count"] == (6, 12)
    assert high_indel["mutation_rate"] == 0.005
    assert high_indel["seq_error_rate"] == 0.0

    ambiguous = strata["ambiguous_n_extreme"].param_overrides
    assert ambiguous["seq_error_rate"] == 0.06
    assert ambiguous["ambiguous_count"] == (12, 30)
    assert ambiguous["indel_count"] == (0, 0)


def test_adaptive_strata_are_observability_slices_not_mixed_corruption_slices():
    for stratum in adaptive_igh_strata(n_per_scenario=1):
        params = stratum.param_overrides
        assert params["mutation_rate"] == 0.005
        assert params["indel_count"] == (0, 0)
        assert params["seq_error_rate"] == 0.0
        assert params["ambiguous_count"] == (0, 0)
        assert stratum.anchor is not None
