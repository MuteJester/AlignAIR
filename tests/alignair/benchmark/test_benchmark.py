import pytest

genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata

from alignair.benchmark import (
    BenchmarkSpec,
    CoveragePlan,
    StratumSpec,
    case_coverage_labels,
    case_to_prediction,
    compact_summary,
    coverage_plan_from_spec,
    coverage_summary,
    criteria_catalog,
    default_igh_assay_spec,
    focused_igh_spec,
    generate_benchmark,
    generate_coverage_benchmark,
    load_jsonl,
    run_online_benchmark,
    save_jsonl,
    scenario_axes_catalog,
    score_cases,
    stream_benchmark,
)
from alignair.reference.reference_set import ReferenceSet


def _small_spec():
    return BenchmarkSpec(
        name="tiny",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=7,
        strata=(
            StratumSpec(name="clean", n=3, progress=0.0),
            StratumSpec(name="oriented", n=4, progress=0.3, orientation_ids=(0, 1, 2, 3)),
        ),
    )


def test_generate_benchmark_cases_and_coverage():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    cases = generate_benchmark(_small_spec(), gdata.HUMAN_IGH_OGRDB, rs)
    assert len(cases) == 7
    assert {c.stratum for c in cases} == {"clean", "oriented"}
    assert {c.orientation_id for c in cases if c.stratum == "oriented"} == {0, 1, 2, 3}
    for case in cases:
        assert case.sequence
        assert case.genes["v"].calls
        assert case.presented_genes["v"].calls == case.genes["v"].calls
        assert len(case.region_labels) == len(case.canonical_sequence)
        assert len(case.presented_region_labels) == len(case.sequence)
    cov = coverage_summary(cases)
    assert cov["n_cases"] == 7
    assert cov["by_stratum"]["clean"] == 3


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


def test_focused_igh_spec_generates_hard_to_hit_scenarios():
    cases = generate_benchmark(focused_igh_spec(n_per_scenario=1, seed=23))
    by_stratum = {case.stratum: case for case in cases}
    assert {
        "productive_only_clean",
        "forced_d_inversion",
        "receptor_revision",
        "contaminant",
        "paired_end",
        "ultra_short_fragment_40",
    }.issubset(by_stratum)
    assert by_stratum["productive_only_clean"].record["productive"] is True
    assert by_stratum["forced_d_inversion"].record["d_inverted"] is True
    assert by_stratum["receptor_revision"].record["receptor_revision_applied"] is True
    assert by_stratum["contaminant"].record["is_contaminant"] is True
    assert by_stratum["paired_end"].record["read_layout"] == "paired_end"
    assert by_stratum["paired_end"].record["r1_sequence"]
    assert by_stratum["paired_end"].record["r2_sequence"]
    assert len(by_stratum["ultra_short_fragment_40"].sequence) == 40


def test_assay_spec_combines_broad_and_focused_strata():
    spec = default_igh_assay_spec(n_per_stratum=1, n_per_focus=1, seed=29)
    names = {stratum.name for stratum in spec.strata}
    assert "clean_full" in names
    assert "forced_d_inversion" in names
    assert "contaminant" in names
    assert spec.n_cases > focused_igh_spec(n_per_scenario=1).n_cases


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


def test_reverse_orientation_maps_presented_coordinates():
    cases = generate_benchmark(
        BenchmarkSpec(
            name="rev",
            dataconfig_name="HUMAN_IGH_OGRDB",
            seed=9,
            strata=(StratumSpec(name="rev", n=1, progress=0.1, orientation_ids=(3,)),),
        )
    )
    case = cases[0]
    L = len(case.canonical_sequence)
    v = case.genes["v"]
    pv = case.presented_genes["v"]
    assert pv.sequence_start == L - v.sequence_end
    assert pv.sequence_end == L - v.sequence_start
    assert case.presented_region_labels == list(reversed(case.region_labels))


def test_jsonl_roundtrip(tmp_path):
    cases = generate_benchmark(_small_spec())
    path = tmp_path / "bench.jsonl"
    save_jsonl(cases, path)
    loaded = load_jsonl(path)
    assert [c.case_id for c in loaded] == [c.case_id for c in cases]
    assert loaded[0].genes["v"].calls == cases[0].genes["v"].calls


def test_perfect_predictions_score_high():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    summary = compact_summary(scores)
    assert summary["v"]["call"] == 1.0
    assert summary["v"]["set_f1"] == 1.0
    assert summary["v"]["seq_mae"] == [0.0, 0.0]
    assert summary["global"]["region_acc"] == 1.0
    assert summary["global"]["state_acc"] == 1.0
    assert summary["global"]["orientation_acc"] == 1.0
    assert summary["global"]["required_field_presence"] == 1.0
    assert summary["global"]["parseable_airr_rate"] == 1.0
    assert summary["global"]["junction_nt_exact"] == 1.0
    assert summary["global"]["junction_start_mae"] == 0.0
    assert summary["global"]["np1_exact"] == 1.0
    assert summary["global"]["p_region_length_mae"] == 0.0
    assert summary["global"]["vj_in_frame_acc"] == 1.0
    assert summary["global"]["stop_codon_acc"] == 1.0
    assert summary["global"]["d_inversion_acc"] == 1.0
    assert summary["global"]["region_macro_f1"] == 1.0
    assert summary["global"]["state_macro_f1"] == 1.0
    assert scores["genes"]["v"]["cigar_exact"] == 1.0
    assert scores["genes"]["v"]["cigar_edit_distance"] == 0.0
    assert scores["genes"]["v"]["trim_5_mae"] == 0.0
    assert scores["genes"]["v"]["identity_mae"] == 0.0
    assert scores["genes"]["v"]["top10_recall"] == 1.0
    assert set(scores["by_stratum"]) == {"clean", "oriented"}


def test_set_metrics_reward_set_outputs():
    cases = generate_benchmark(_small_spec())
    case = cases[0]
    pred = case_to_prediction(case)
    truth = case.genes["v"]
    if len(truth.calls) == 1:
        pred["v_calls"] = [truth.calls[0], "not-a-real-allele"]
        scores = score_cases([case], [pred])
        assert scores["genes"]["v"]["call_set_recall"] == 1.0
        assert scores["genes"]["v"]["call_set_precision"] == 0.5


def test_extended_metrics_detect_biological_annotation_errors():
    case = generate_benchmark(_small_spec())[0]
    pred = case_to_prediction(case)
    pred["junction"] = "A"
    pred["np1"] = "A"
    pred["v_cigar"] = "1M"
    pred["vj_in_frame"] = not bool(case.record["vj_in_frame"])
    pred["d_inverted"] = not bool(case.record["d_inverted"])
    scores = score_cases([case], [pred])
    assert scores["global"]["junction_nt_exact"] == 0.0
    assert scores["global"]["np1_exact"] == 0.0
    assert scores["global"]["vj_in_frame_acc"] == 0.0
    assert scores["global"]["d_inversion_acc"] == 0.0
    assert scores["genes"]["v"]["cigar_exact"] == 0.0
    assert scores["genes"]["v"]["cigar_edit_distance"] > 0.0
    empty_scores = score_cases([case], [{}])
    assert empty_scores["global"]["required_field_presence"] == 0.0
    assert empty_scores["global"]["parseable_airr_rate"] == 0.0


def test_criteria_catalog_names_core_assay_dimensions():
    catalog = criteria_catalog()
    names = {c["name"] for c in catalog}
    assert {
        "airr_schema_completeness",
        "coordinate_convention_compliance",
        "locus_and_chain_support",
        "orientation_detection",
        "prediction_completeness",
        "in_sequence_boundaries",
        "junction_sequence_and_translation",
        "np_and_p_nucleotide_recovery",
        "germline_boundaries",
        "alignment_path_and_cigar",
        "set_valued_allele_call",
        "sibling_allele_resolution",
        "genotype_mask_compliance",
        "constant_region_call",
        "d_orientation_and_inversion",
        "regional_shm_profile",
        "segment_identity_and_support",
        "allele_uncertainty_calibration",
        "fragment_observability",
        "read_layout_and_end_loss",
        "contaminant_and_out_of_scope_handling",
        "receptor_revision_cases",
        "productive_status",
    }.issubset(names)
    categories = {c["category"] for c in catalog}
    assert {
        "airr_output_contract",
        "alignment_quality",
        "allele_calling",
        "calibration",
        "chain_locus",
        "efficiency",
        "robustness",
    }.issubset(categories)
    statuses = {c["status"] for c in catalog}
    assert {"available", "partial", "planned"}.issubset(statuses)
    assert all("ground_truth_fields" in c for c in catalog)


def test_scenario_axes_catalog_names_stress_dimensions():
    axes = {a["name"]: a for a in scenario_axes_catalog()}
    assert {
        "difficulty_stratum",
        "length",
        "locus_chain",
        "mutation_burden",
        "indel_burden",
        "allele_ambiguity",
        "d_orientation",
        "genotype_size",
        "read_layout",
        "segment_presence",
        "junction_biology",
    }.issubset(axes)
    assert "fragment_50" in axes["difficulty_stratum"]["values"]


def test_stream_benchmark_yields_without_materializing():
    stream = stream_benchmark(_small_spec())
    first = next(stream)
    second = next(stream)
    assert first.case_id != second.case_id
    assert first.sequence and second.sequence


def test_online_benchmark_report_with_perfect_predictor():
    spec = _small_spec()

    def predictor(reads):
        # The online runner calls the predictor with generated reads only; for this
        # deterministic test, close over a matching stream of perfect cases.
        assert reads
        batch = [next(predictor.case_iter) for _ in reads]
        return [case_to_prediction(c) for c in batch]

    predictor.case_iter = stream_benchmark(spec)
    report = run_online_benchmark(spec, predictor, batch_size=2)
    assert report["coverage"]["n_cases"] == spec.n_cases
    assert report["coverage"]["alleles"]["v"]["n_total_reference"] > 0
    assert 0.0 < report["coverage"]["alleles"]["v"]["fraction_observed"] <= 1.0
    assert report["criteria"]
    assert report["scenario_axes"]
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0
    assert "stratum:clean" in report["results"]["by_context"]
    assert any(k.startswith("locus:") for k in report["results"]["by_context"])
    assert any(k.startswith("chain:") for k in report["results"]["by_context"])
    assert any(k.startswith("d_orientation:") for k in report["results"]["by_context"])
    assert any(k.startswith("junction_length:") for k in report["results"]["by_context"])
    assert any(k.startswith("segment_presence:") for k in report["results"]["by_context"])
    assert "ambiguity:all_single" in report["results"]["by_context"] or any(
        k.startswith("ambiguity:") for k in report["results"]["by_context"]
    )


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

    report = run_online_benchmark(spec, predictor, batch_size=2, coverage_plan=plan)
    assert report["generation_coverage"]["satisfied"]
    assert report["generation_coverage"]["observed_target_counts"]["orientation:reverse_complement"] == 1
