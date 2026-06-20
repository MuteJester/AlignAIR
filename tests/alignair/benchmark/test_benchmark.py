import csv
import json

import pytest

genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata

from alignair.benchmark import (
    BenchmarkCase,
    BenchmarkSpec,
    CoveragePlan,
    CoverageTracker,
    DEFAULT_BOOTSTRAP_METRICS,
    GeneTruth,
    PredictionValidationAccumulator,
    ReadinessThresholds,
    StratumSpec,
    allele_context_label,
    allele_stratification_contexts,
    align_predictions_to_cases,
    assess_benchmark_readiness,
    audit_criteria_report,
    build_benchmark_manifest,
    build_assay_report,
    build_benchmark_report,
    bootstrap_metric_intervals,
    build_allele_calling_diagnostics,
    build_boundary_diagnostics,
    case_coverage_labels,
    case_to_prediction,
    compact_summary,
    coverage_plan_from_spec,
    coverage_summary,
    criteria_catalog,
    default_igh_assay_spec,
    export_benchmark_inputs,
    focused_igh_spec,
    generate_benchmark,
    generate_coverage_benchmark,
    load_airr_predictions,
    load_dicts_jsonl,
    load_jsonl,
    prediction_contract,
    readiness_thresholds,
    run_benchmark_report,
    run_online_benchmark,
    save_jsonl,
    save_dicts_jsonl,
    save_airr_input,
    save_fasta,
    scenario_axes_catalog,
    score_cases,
    stream_benchmark,
    validate_prediction,
    validate_predictions,
)
from alignair.benchmark.cli import main as benchmark_cli
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


def _single_clean_case(seed=31):
    return generate_benchmark(
        BenchmarkSpec(
            name="single",
            dataconfig_name="HUMAN_IGH_OGRDB",
            seed=seed,
            strata=(StratumSpec(name="clean", n=1, progress=0.0),),
        )
    )[0]


def _manual_case(case_id, v_calls, *, stratum="manual", orientation_id=0):
    genes = {
        "v": GeneTruth(
            calls=tuple(v_calls),
            primary=v_calls[0],
            sequence_start=0,
            sequence_end=80,
            germline_start=0,
            germline_end=80,
        ),
        "d": GeneTruth(),
        "j": GeneTruth(
            calls=("IGHJ4*01",),
            primary="IGHJ4*01",
            sequence_start=80,
            sequence_end=100,
            germline_start=0,
            germline_end=20,
        ),
    }
    return BenchmarkCase(
        case_id=case_id,
        stratum=stratum,
        sequence="A" * 100,
        canonical_sequence="A" * 100,
        orientation_id=orientation_id,
        genes=genes,
        presented_genes=genes,
        record={},
    )


class _FakeReferenceGene:
    def __init__(self, names):
        self.names = tuple(names)


class _FakeReferenceSet:
    has_d = True

    def __init__(self):
        self.genes = {
            "V": _FakeReferenceGene(("IGHV1*01", "IGHV2*01")),
            "J": _FakeReferenceGene(("IGHJ4*01",)),
        }

    def gene(self, gene):
        return self.genes[gene]


def _airr_row_for_case(case):
    row = {
        "sequence_id": case.case_id,
        "sequence": case.canonical_sequence,
        "productive": "T" if bool(case.record.get("productive")) else "F",
        "vj_in_frame": "T" if bool(case.record.get("vj_in_frame")) else "F",
        "stop_codon": "T" if bool(case.record.get("stop_codon")) else "F",
        "rev_comp": "F",
    }
    for gene, truth in case.genes.items():
        row[f"{gene}_call"] = ",".join(truth.calls)
        if truth.sequence_start is not None:
            row[f"{gene}_sequence_start"] = str(truth.sequence_start + 1)
        if truth.sequence_end is not None:
            row[f"{gene}_sequence_end"] = str(truth.sequence_end)
        if truth.germline_start is not None:
            row[f"{gene}_germline_start"] = str(truth.germline_start + 1)
        if truth.germline_end is not None:
            row[f"{gene}_germline_end"] = str(truth.germline_end)
    for key in (
        "locus",
        "c_call",
        "junction",
        "junction_aa",
        "junction_length",
        "np1",
        "np2",
        "np1_length",
        "np2_length",
    ):
        if key in case.record:
            row[key] = case.record[key]
    if "junction_start" in case.record:
        row["junction_start"] = str(case.record["junction_start"] + 1)
    if "junction_end" in case.record:
        row["junction_end"] = str(case.record["junction_end"])
    return row


def _write_tsv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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


def test_readiness_profiles_distinguish_smoke_from_assay_scale():
    cases = generate_benchmark(_small_spec())
    reference_set = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    smoke = assess_benchmark_readiness(cases, reference_set=reference_set, profile="smoke")
    assay = assess_benchmark_readiness(cases, reference_set=reference_set, profile="assay")
    assert readiness_thresholds("assay").min_cases > readiness_thresholds("smoke").min_cases
    assert readiness_thresholds("allele_complete").min_per_reference_allele > 0
    assert readiness_thresholds("allele_stratified").min_per_allele_context > 0
    assert smoke["grade"] == "pass"
    assert smoke["truth_source"] == "GenAIRR benchmark cases"
    assert assay["grade"] == "fail"
    assert any(check["name"] == "case_count" and check["grade"] == "fail" for check in assay["checks"])
    assert any(check["name"] == "reference_allele_fraction" for check in assay["checks"])
    allele_check = next(check for check in assay["checks"] if check["name"] == "observed_allele_counts")
    assert allele_check["observed"]["j"]["threshold"] == len(reference_set.gene("J").names)
    assert allele_check["observed"]["j"]["threshold"] < assay["thresholds"]["min_observed_alleles_per_gene"]


def test_readiness_can_require_minimum_count_for_every_reference_allele():
    thresholds = ReadinessThresholds(
        profile="unit_allele_complete",
        min_cases=1,
        min_per_stratum=1,
        min_observed_alleles_per_gene=0,
        min_reference_allele_fraction=1.0,
        min_per_reference_allele=2,
    )
    reference_set = _FakeReferenceSet()
    low_cases = [
        _manual_case("c1", ("IGHV1*01",)),
        _manual_case("c2", ("IGHV1*01",)),
        _manual_case("c3", ("IGHV2*01",)),
    ]
    low_report = assess_benchmark_readiness(
        low_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    min_check = next(
        check for check in low_report["checks"] if check["name"] == "reference_allele_min_counts"
    )
    assert low_report["grade"] == "fail"
    assert min_check["grade"] == "fail"
    assert min_check["observed"]["v"]["n_low"] == 1
    assert min_check["details"]["low_allele_examples"]["v"] == {"IGHV2*01": 1}

    passing_cases = low_cases + [_manual_case("c4", ("IGHV2*01",))]
    passing_report = assess_benchmark_readiness(
        passing_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    passing_min_check = next(
        check for check in passing_report["checks"] if check["name"] == "reference_allele_min_counts"
    )
    assert passing_report["grade"] == "pass"
    assert passing_min_check["grade"] == "pass"
    assert passing_min_check["observed"]["v"]["min_count"] == 2


def test_readiness_can_require_reference_alleles_across_contexts():
    thresholds = ReadinessThresholds(
        profile="unit_allele_stratified",
        min_cases=1,
        min_per_stratum=1,
        min_observed_alleles_per_gene=0,
        min_reference_allele_fraction=1.0,
        min_per_reference_allele=1,
        min_per_allele_context=1,
        allele_contexts=("stratum:easy", "orientation:reverse_complement"),
    )
    reference_set = _FakeReferenceSet()
    low_cases = [
        _manual_case("c1", ("IGHV1*01",), stratum="easy", orientation_id=0),
        _manual_case("c2", ("IGHV2*01",), stratum="easy", orientation_id=0),
        _manual_case("c3", ("IGHV1*01",), stratum="hard", orientation_id=1),
    ]
    low_report = assess_benchmark_readiness(
        low_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    context_check = next(
        check for check in low_report["checks"] if check["name"] == "reference_allele_context_min_counts"
    )
    assert low_report["grade"] == "fail"
    assert context_check["grade"] == "fail"
    assert context_check["observed"]["v"]["n_low"] == 1
    assert context_check["details"]["low_cell_examples"]["v"] == [
        {"allele": "IGHV2*01", "context": "orientation:reverse_complement", "count": 0}
    ]

    passing_cases = low_cases + [_manual_case("c4", ("IGHV2*01",), stratum="hard", orientation_id=1)]
    passing_report = assess_benchmark_readiness(
        passing_cases,
        reference_set=reference_set,
        thresholds=thresholds,
    )
    passing_context_check = next(
        check for check in passing_report["checks"] if check["name"] == "reference_allele_context_min_counts"
    )
    assert passing_report["grade"] == "pass"
    assert passing_context_check["grade"] == "pass"
    assert passing_context_check["observed"]["v"]["min_count"] == 1


def test_cli_readiness_command_reports_preflight_status(tmp_path):
    cases = generate_benchmark(_small_spec())
    cases_path = tmp_path / "cases.jsonl"
    out_path = tmp_path / "readiness.json"
    save_jsonl(cases, cases_path)
    benchmark_cli(
        [
            "readiness",
            "--cases",
            str(cases_path),
            "--config",
            "HUMAN_IGH_OGRDB",
            "--profile",
            "smoke",
            "--out",
            str(out_path),
        ]
    )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["grade"] == "pass"
    assert report["profile"] == "smoke"
    assert report["n_cases"] == len(cases)


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


def test_prediction_jsonl_roundtrip(tmp_path):
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    path = tmp_path / "preds.jsonl"
    save_dicts_jsonl(preds, path)
    loaded = load_dicts_jsonl(path)
    assert loaded[0]["v_call"] == preds[0]["v_call"]
    assert loaded[0]["v_calls"] == preds[0]["v_calls"]


def test_external_input_exports_and_manifest(tmp_path):
    spec = _small_spec()
    reference_set = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    cases = generate_benchmark(spec, gdata.HUMAN_IGH_OGRDB, reference_set)
    fasta_path = tmp_path / "bench.fasta"
    airr_path = tmp_path / "bench_airr.tsv"

    save_fasta(cases, fasta_path)
    fasta_text = fasta_path.read_text(encoding="utf-8")
    assert fasta_text.startswith(f">{cases[0].case_id} ")
    first_record_lines = fasta_text.split(">")[1].splitlines()[1:]
    assert "".join(first_record_lines) == cases[0].sequence

    save_airr_input(cases, airr_path, include_metadata=True)
    with airr_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["sequence_id"] == cases[0].case_id
    assert rows[0]["sequence"] == cases[0].sequence
    assert rows[0]["benchmark_stratum"] == cases[0].stratum

    manifest = build_benchmark_manifest(
        cases,
        spec=spec,
        dataconfig_name="HUMAN_IGH_OGRDB",
        reference_set=reference_set,
        frame="presented",
    )
    assert manifest["benchmark"]["n_cases"] == len(cases)
    assert manifest["generation"]["spec"]["seed"] == spec.seed
    assert manifest["generation"]["dataconfig_name"] == "HUMAN_IGH_OGRDB"
    assert manifest["reference"]["genes"]["v"]["n_alleles"] > 0
    assert len(manifest["reference"]["sha256"]) == 64
    assert manifest["benchmark"]["sequence_id_policy"] == "sequence_id equals benchmark case_id"
    assert manifest["readiness"]["profile"] == "assay"
    assert manifest["readiness"]["truth_source"] == "GenAIRR benchmark cases"


def test_export_benchmark_inputs_writes_expected_files(tmp_path):
    spec = _small_spec()
    reference_set = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    cases = generate_benchmark(spec, gdata.HUMAN_IGH_OGRDB, reference_set)
    files = export_benchmark_inputs(
        cases,
        tmp_path,
        prefix="tiny",
        spec=spec,
        dataconfig_name="HUMAN_IGH_OGRDB",
        reference_set=reference_set,
    )
    assert set(files) == {"fasta", "airr_input_tsv", "manifest"}
    assert all(path for path in files.values())
    assert all((tmp_path / name).exists() for name in ("tiny.fasta", "tiny_airr_input.tsv", "tiny_manifest.json"))
    manifest = json.loads((tmp_path / "tiny_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["fasta"].endswith("tiny.fasta")
    assert manifest["coverage"]["n_cases"] == len(cases)
    assert manifest["readiness"]["profile"] == "assay"


def test_cli_export_writes_external_tool_inputs(tmp_path):
    cases = generate_benchmark(_small_spec())
    cases_path = tmp_path / "cases.jsonl"
    out_dir = tmp_path / "exported"
    save_jsonl(cases, cases_path)

    benchmark_cli(
        [
            "export",
            "--cases",
            str(cases_path),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "cli",
            "--config",
            "HUMAN_IGH_OGRDB",
            "--airr-metadata",
        ]
    )
    assert (out_dir / "cli.fasta").exists()
    assert (out_dir / "cli_airr_input.tsv").exists()
    manifest = json.loads((out_dir / "cli_manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"]["n_cases"] == len(cases)
    assert manifest["reference"]["has_d"] is True
    with (out_dir / "cli_airr_input.tsv").open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["sequence_id"] == cases[0].case_id
    assert row["benchmark_case_id"] == cases[0].case_id


def test_airr_tsv_predictions_normalize_and_score(tmp_path):
    case = _single_clean_case()
    path = tmp_path / "airr.tsv"
    _write_tsv(path, [_airr_row_for_case(case)])

    preds = load_airr_predictions(path)
    pred = preds[0]
    assert pred["v_call"] == case.genes["v"].calls[0]
    assert pred["v_calls"] == list(case.genes["v"].calls)
    assert pred["v_sequence_start"] == case.genes["v"].sequence_start
    assert pred["v_sequence_end"] == case.genes["v"].sequence_end
    assert pred["junction_start"] == case.record["junction_start"]
    assert pred["junction_end"] == case.record["junction_end"]
    assert pred["productive"] is bool(case.record["productive"])
    assert pred["orientation_id"] == 0

    report = build_benchmark_report([case], preds, contract_level="core")
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0
    assert report["results"]["overall"]["global"]["productive_acc"] == 1.0


def test_cli_normalizes_and_evaluates_airr_tables(tmp_path):
    case = _single_clean_case(seed=37)
    cases_path = tmp_path / "cases.jsonl"
    airr_path = tmp_path / "airr.tsv"
    preds_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    save_jsonl([case], cases_path)
    _write_tsv(airr_path, [_airr_row_for_case(case)])

    benchmark_cli(
        [
            "normalize-predictions",
            "--input",
            str(airr_path),
            "--format",
            "airr-tsv",
            "--out",
            str(preds_path),
        ]
    )
    preds = load_dicts_jsonl(preds_path)
    assert preds[0]["j_call"] == case.genes["j"].calls[0]

    benchmark_cli(
        [
            "evaluate",
            "--cases",
            str(cases_path),
            "--predictions",
            str(airr_path),
            "--prediction-format",
            "airr-tsv",
            "--contract-level",
            "core",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["results"]["overall"]["genes"]["j"]["call_top1_in_set"] == 1.0


def test_prediction_matching_recovers_reordered_outputs():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = build_benchmark_report(
        cases,
        list(reversed(preds)),
        match_by="sequence_id",
        contract_level="minimal",
    )
    assert report["prediction_matching"]["n_matched_cases"] == len(cases)
    assert report["prediction_matching"]["match_fraction"] == 1.0
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0


def test_prediction_matching_reports_missing_extra_and_duplicate_ids():
    cases = generate_benchmark(_small_spec())[:3]
    pred0 = case_to_prediction(cases[0])
    pred1 = case_to_prediction(cases[1])
    duplicate0 = dict(pred0)
    duplicate0["v_call"] = "not-a-real-allele"
    extra = dict(pred0)
    extra["sequence_id"] = "not-in-benchmark"

    result = align_predictions_to_cases(
        cases,
        [extra, duplicate0, pred1, pred0],
        id_field="sequence_id",
        duplicate_policy="first",
    )
    assert result.report["n_matched_cases"] == 2
    assert result.report["n_missing_cases"] == 1
    assert result.report["missing_case_ids"] == [cases[2].case_id]
    assert result.report["n_extra_prediction_ids"] == 1
    assert result.report["extra_prediction_ids"] == ["not-in-benchmark"]
    assert result.report["n_duplicate_prediction_ids"] == 1
    assert result.report["duplicate_prediction_ids"][0]["id"] == cases[0].case_id
    assert result.predictions[2] == {}

    report = build_benchmark_report(cases, [extra, duplicate0, pred1, pred0], match_by="sequence_id")
    assert report["prediction_matching"]["n_missing_cases"] == 1
    assert report["results"]["overall"]["genes"]["v"]["missing_call_rate"] > 0.0


def test_cli_evaluate_matches_airr_rows_by_sequence_id(tmp_path):
    cases = generate_benchmark(
        BenchmarkSpec(
            name="airr_match",
            dataconfig_name="HUMAN_IGH_OGRDB",
            seed=41,
            strata=(StratumSpec(name="clean", n=2, progress=0.0),),
        )
    )
    cases_path = tmp_path / "cases.jsonl"
    airr_path = tmp_path / "airr.tsv"
    report_path = tmp_path / "report.json"
    save_jsonl(cases, cases_path)
    _write_tsv(airr_path, [_airr_row_for_case(cases[1]), _airr_row_for_case(cases[0])])

    benchmark_cli(
        [
            "evaluate",
            "--cases",
            str(cases_path),
            "--predictions",
            str(airr_path),
            "--prediction-format",
            "airr-tsv",
            "--bootstrap",
            "5",
            "--confidence",
            "0.9",
            "--no-bootstrap-strata",
            "--out",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["prediction_matching"]["n_matched_cases"] == 2
    assert report["uncertainty"]["n_bootstrap"] == 5
    assert report["uncertainty"]["confidence"] == 0.9
    assert report["uncertainty"]["by_stratum"] == {}
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0


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


def test_bootstrap_uncertainty_uses_existing_genairr_truth():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = bootstrap_metric_intervals(cases, preds, n_bootstrap=20, confidence=0.9, seed=11)
    assert "genes.v.call_top1_in_set" in DEFAULT_BOOTSTRAP_METRICS
    assert report["method"] == "paired_case_bootstrap"
    assert report["truth_source"] == "GenAIRR benchmark cases"
    assert report["n_bootstrap"] == 20
    assert report["confidence"] == 0.9
    metric = report["overall"]["genes.v.call_top1_in_set"]
    assert metric["point"] == 1.0
    assert metric["ci_low"] == 1.0
    assert metric["ci_high"] == 1.0
    assert report["by_stratum"]["clean"]["genes.v.call_top1_in_set"]["n_cases"] == 3


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


def test_allele_calling_diagnostics_separate_allele_gene_and_family_errors():
    cases = [
        _manual_case("exact", ("IGHV1-1*01",)),
        _manual_case("same_gene_wrong_allele", ("IGHV1-1*02",)),
        _manual_case("same_family_wrong_gene", ("IGHV1-2*01",)),
        _manual_case("wrong_family", ("IGHV3-7*01",)),
        _manual_case("ambiguous", ("IGHV2-5*01", "IGHV2-5*02")),
        _manual_case("missing", ("IGHV5-10*01",)),
    ]
    preds = [
        {"v_call": "IGHV1-1*01", "v_calls": ["IGHV1-1*01"]},
        {"v_call": "IGHV1-1*01", "v_calls": ["IGHV1-1*01"]},
        {"v_call": "IGHV1-69*01", "v_calls": ["IGHV1-69*01"]},
        {"v_call": "IGHV4-4*01", "v_calls": ["IGHV4-4*01"]},
        {"v_call": "IGHV2-5*02", "v_calls": ["IGHV2-5*02"]},
        {},
    ]

    diagnostics = build_allele_calling_diagnostics(cases, preds, top_n=20)
    v_diag = diagnostics["genes"]["v"]
    summary = v_diag["summary"]

    assert summary["n_truth_cases"] == 6
    assert summary["top1_accepted_allele_rate"] == pytest.approx(2 / 6)
    assert summary["top1_same_gene_rate"] == pytest.approx(3 / 6)
    assert summary["top1_same_family_rate"] == pytest.approx(4 / 6)
    assert summary["same_gene_wrong_allele_rate"] == pytest.approx(1 / 6)
    assert summary["same_family_wrong_gene_rate"] == pytest.approx(1 / 6)
    assert summary["wrong_family_rate"] == pytest.approx(1 / 6)
    assert summary["missing_prediction_rate"] == pytest.approx(1 / 6)

    allele_rows = {row["allele"]: row for row in v_diag["per_allele"]}
    assert allele_rows["IGHV1-1*02"]["singleton_top1_exact_rate"] == 0.0
    assert allele_rows["IGHV1-1*02"]["same_gene_wrong_allele_rate"] == 1.0
    assert allele_rows["IGHV2-5*01"]["n_ambiguous_truth_cases"] == 1
    assert allele_rows["IGHV2-5*01"]["top1_accepted_allele_rate"] == 1.0
    assert allele_rows["IGHV2-5*01"]["pred_set_contains_allele_rate"] == 0.0
    assert allele_rows["IGHV5-10*01"]["missing_prediction_rate"] == 1.0

    gene_rows = {row["gene"]: row for row in v_diag["per_gene"]}
    assert gene_rows["IGHV1-2"]["same_family_wrong_gene_rate"] == 1.0
    family_rows = {row["family"]: row for row in v_diag["per_gene_family"]}
    assert family_rows["IGHV1"]["top1_same_family_rate"] == 1.0
    assert family_rows["IGHV3"]["wrong_family_rate"] == 1.0
    assert family_rows["IGHV5"]["missing_prediction_rate"] == 1.0

    allele_confusions = {
        (row["truth_allele"], row["pred_call"]): row for row in v_diag["allele_confusions"]
    }
    assert allele_confusions[("IGHV1-1*02", "IGHV1-1*01")]["error_kind"] == "same_gene_wrong_allele"
    assert allele_confusions[("IGHV1-2*01", "IGHV1-69*01")]["error_kind"] == "same_family_wrong_gene"
    assert allele_confusions[("IGHV3-7*01", "IGHV4-4*01")]["error_kind"] == "wrong_family"
    assert allele_confusions[("IGHV5-10*01", None)]["error_kind"] == "missing_prediction"
    assert "same_gene_wrong_allele" in allele_confusions[("IGHV1-1*02", "IGHV1-1*01")]["example_case_ids"]


def test_benchmark_report_includes_allele_diagnostics():
    cases = [
        _manual_case("ok", ("IGHV1-1*01",)),
        _manual_case("bad", ("IGHV3-7*01",)),
    ]
    preds = [
        {"v_call": "IGHV1-1*01", "v_calls": ["IGHV1-1*01"]},
        {"v_call": "IGHV4-4*01", "v_calls": ["IGHV4-4*01"]},
    ]

    report = build_benchmark_report(cases, preds)
    v_diag = report["diagnostics"]["allele_calling"]["genes"]["v"]
    assert v_diag["summary"]["n_truth_cases"] == 2
    assert v_diag["summary"]["wrong_family_rate"] == 0.5
    assert v_diag["family_confusions"][0]["truth_family"] == "IGHV3"
    assert report["diagnostics"]["boundaries"]["genes"]["v"]["summary"]["n_truth_segments"] == 2


def test_boundary_diagnostics_classify_coordinate_failure_modes():
    cases = [
        _manual_case("exact", ("IGHV1-1*01",)),
        _manual_case("missing", ("IGHV1-1*01",)),
        _manual_case("start_only", ("IGHV1-1*01",)),
        _manual_case("shifted", ("IGHV1-1*01",)),
        _manual_case("wrong_length", ("IGHV1-1*01",)),
        _manual_case("negative", ("IGHV1-1*01",)),
        _manual_case("wrong_trim", ("IGHV1-1*01",)),
    ]

    def pred(start, end, gs=0, ge=80):
        return {
            "v_call": "IGHV1-1*01",
            "v_sequence_start": start,
            "v_sequence_end": end,
            "v_germline_start": gs,
            "v_germline_end": ge,
        }

    preds = [
        pred(0, 80),
        {},
        pred(1, 80),
        pred(1, 81),
        pred(0, 70),
        pred(80, 0),
        pred(0, 80, gs=5, ge=85),
    ]

    diagnostics = build_boundary_diagnostics(cases, preds, top_n=20)
    v_diag = diagnostics["genes"]["v"]
    summary = v_diag["summary"]

    assert summary["n_truth_segments"] == 7
    assert summary["exact_query_span_rate"] == pytest.approx(2 / 7)
    assert summary["exact_all_coordinates_rate"] == pytest.approx(1 / 7)
    assert summary["missing_coordinates_rate"] == pytest.approx(1 / 7)
    assert summary["start_only_error_rate"] == pytest.approx(1 / 7)
    assert summary["off_by_one_rate"] == pytest.approx(2 / 7)
    assert summary["systematic_plus_one_shift_rate"] == pytest.approx(1 / 7)
    assert summary["correct_length_shifted_span_rate"] == pytest.approx(1 / 7)
    assert summary["wrong_length_rate"] == pytest.approx(3 / 7)
    assert summary["negative_span_rate"] == pytest.approx(1 / 7)
    assert summary["correct_query_span_wrong_germline_span_rate"] == pytest.approx(1 / 7)
    assert summary["correct_allele_wrong_trim_rate"] == pytest.approx(1 / 7)

    failures = {row["failure_type"]: row for row in v_diag["failure_types"]}
    assert failures["missing_coordinates"]["example_case_ids"] == ["missing"]
    assert failures["systematic_plus_one_shift"]["example_case_ids"] == ["shifted"]
    assert failures["correct_allele_wrong_trim"]["example_case_ids"] == ["wrong_trim"]


def test_boundary_diagnostics_detect_frame_confusion_and_vdj_overlap():
    canonical_genes = {
        "v": GeneTruth(("IGHV1-1*01",), "IGHV1-1*01", 0, 80, 0, 80),
        "d": GeneTruth(),
        "j": GeneTruth(("IGHJ4*01",), "IGHJ4*01", 80, 100, 0, 20),
    }
    presented_genes = {
        "v": GeneTruth(("IGHV1-1*01",), "IGHV1-1*01", 20, 100, 0, 80),
        "d": GeneTruth(),
        "j": GeneTruth(("IGHJ4*01",), "IGHJ4*01", 0, 20, 0, 20),
    }
    frame_case = BenchmarkCase(
        case_id="frame_confusion",
        stratum="manual",
        sequence="A" * 100,
        canonical_sequence="A" * 100,
        orientation_id=3,
        genes=canonical_genes,
        presented_genes=presented_genes,
        record={},
    )
    overlap_case = _manual_case("overlap", ("IGHV1-1*01",))
    preds = [
        {
            "v_sequence_start": 20,
            "v_sequence_end": 100,
            "v_germline_start": 0,
            "v_germline_end": 80,
        },
        {
            "v_sequence_start": 0,
            "v_sequence_end": 90,
            "v_germline_start": 0,
            "v_germline_end": 80,
            "j_sequence_start": 80,
            "j_sequence_end": 100,
            "j_germline_start": 0,
            "j_germline_end": 20,
        },
    ]

    diagnostics = build_boundary_diagnostics([frame_case, overlap_case], preds, top_n=20)
    v_summary = diagnostics["genes"]["v"]["summary"]
    assert v_summary["canonical_presented_frame_confusion_rate"] == pytest.approx(0.5)
    failures = {row["failure_type"]: row for row in diagnostics["global"]["failure_types"]}
    assert failures["vdj_order_missing_coordinates"]["example_case_ids"] == ["frame_confusion"]
    assert failures["vdj_order_or_overlap_error"]["example_case_ids"] == ["overlap"]


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


def test_prediction_contract_validates_core_predictions():
    case = generate_benchmark(_small_spec())[0]
    pred = case_to_prediction(case)
    fields = prediction_contract()
    assert any(field["name"] == "v_call" and field["level"] == "minimal" for field in fields)
    assert any(field["name"] == "region_labels" and field["level"] == "assay" for field in fields)
    result = validate_prediction(pred, level="core", has_d=True)
    assert result["valid"]
    assert result["coverage_fraction"] == 1.0

    malformed = dict(pred)
    malformed["orientation_id"] = 9
    malformed.pop("v_sequence_start")
    result = validate_prediction(malformed, level="core", has_d=True)
    assert not result["valid"]
    assert "v_sequence_start" in result["missing_fields"]
    assert any(item["field"] == "orientation_id" for item in result["malformed_fields"])


def test_prediction_contract_batch_and_accumulator():
    case = generate_benchmark(_small_spec())[0]
    valid = case_to_prediction(case)
    invalid = {}
    batch = validate_predictions([valid, invalid], level="minimal", has_d=True)
    assert batch["n_predictions"] == 2
    assert batch["n_valid"] == 1
    assert batch["missing_field_counts"]["v_call"] == 1

    acc = PredictionValidationAccumulator(level="minimal", has_d=True)
    acc.update([valid])
    acc.update([invalid])
    summary = acc.to_dict()
    assert summary["n_predictions"] == 2
    assert summary["n_valid"] == 1
    assert summary["missing_field_counts"]["j_call"] == 1


def test_assay_report_groups_criteria_and_missing_metrics():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    report = build_assay_report(scores, top_n_contexts=5)
    assert report["summary"]["n_cases"] == len(cases)
    assert report["summary"]["n_criteria_with_results"] > 0
    assert report["summary"]["grade"] == "pass"
    assert report["summary"]["n_failed_criteria"] == 0
    assert "allele_calling" in report["by_category"]
    assert "segmentation" in report["by_category"]
    assert report["by_category"]["allele_calling"]["grade"] == "pass"
    by_name = {entry["name"]: entry for entry in report["criteria"]}
    assert by_name["allele_top1_call"]["observed_metrics"]["call_top1_in_set"]["genes"]["v"] == 1.0
    assert by_name["allele_top1_call"]["grade"] == "pass"
    assert by_name["allele_top1_call"]["metric_assessments"][0]["grade"] == "pass"
    assert by_name["allele_uncertainty_calibration"]["n_observed_metric_keys"] == 0
    assert by_name["allele_uncertainty_calibration"]["grade"] == "planned"


def test_assay_report_surfaces_weak_contexts():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    first_stratum = next(iter(scores["by_stratum"]))
    scores["by_stratum"][first_stratum]["genes"]["v"]["call_top1_in_set"] = 0.0
    report = build_assay_report(scores, top_n_contexts=3)
    assert report["weak_contexts"]
    assert report["weak_contexts"][0]["metric"] == "call_top1_in_set"
    assert report["weak_contexts"][0]["context"] == first_stratum


def test_assay_report_grades_broken_core_criteria_as_failures():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    for pred in preds:
        pred["v_call"] = "not-a-real-allele"
        pred["v_calls"] = ["not-a-real-allele"]
    report = build_assay_report(score_cases(cases, preds), top_n_contexts=5)
    assert report["summary"]["grade"] == "fail"
    assert report["summary"]["n_failed_criteria"] > 0
    by_name = {entry["name"]: entry for entry in report["criteria"]}
    assert by_name["allele_top1_call"]["grade"] == "fail"
    assert any(row["grade"] == "fail" for row in by_name["allele_top1_call"]["metric_assessments"])
    assert any(item["name"] == "allele_top1_call" for item in report["critical_failures"])


def test_criteria_audit_tracks_metric_and_genairr_truth_coverage():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    scores = score_cases(cases, preds)
    audit = audit_criteria_report(scores, cases=cases)
    assert audit["summary"]["has_case_truth_audit"]
    assert audit["truth_source"] == "GenAIRR benchmark cases"
    assert "call_top1_in_set" in audit["observed_metric_keys"]["all"]
    assert "truth_set_size" not in audit["metric_keys_without_criteria"]
    assert "ss_within10" not in audit["metric_keys_without_criteria"]
    by_name = {entry["name"]: entry for entry in audit["criteria"]}
    assert by_name["allele_top1_call"]["metric_coverage_fraction"] == 1.0
    assert "call_top1_in_set" in by_name["allele_top1_call"]["observed_metric_keys"]
    assert audit["truth_field_availability"]["v_call"]["n_present"] == len(cases)
    assert audit["truth_field_availability"]["sequence"]["n_present"] == len(cases)


def test_cli_audit_command_writes_metric_and_truth_audit(tmp_path):
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = build_benchmark_report(cases, preds, contract_level="core")
    cases_path = tmp_path / "cases.jsonl"
    report_path = tmp_path / "report.json"
    audit_path = tmp_path / "audit.json"
    save_jsonl(cases, cases_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    benchmark_cli(
        [
            "audit",
            "--report",
            str(report_path),
            "--cases",
            str(cases_path),
            "--out",
            str(audit_path),
        ]
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["summary"]["n_criteria"] == len(criteria_catalog())
    assert audit["summary"]["has_case_truth_audit"]
    assert audit["truth_field_availability"]["j_call"]["n_present"] == len(cases)


def test_build_benchmark_report_from_saved_predictions():
    cases = generate_benchmark(_small_spec())
    preds = [case_to_prediction(c) for c in cases]
    report = build_benchmark_report(
        cases,
        preds,
        contract_level="core",
        n_bootstrap=10,
        confidence=0.9,
        bootstrap_strata=False,
    )
    assert report["coverage"]["n_cases"] == len(cases)
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["criteria_audit"]["summary"]["has_case_truth_audit"]
    assert report["criteria_audit"]["truth_field_availability"]["v_call"]["n_present"] == len(cases)
    assert report["uncertainty"]["overall"]["genes.v.call_top1_in_set"]["ci_low"] == 1.0
    assert report["uncertainty"]["confidence"] == 0.9
    assert report["uncertainty"]["by_stratum"] == {}
    assert report["assay"]["summary"]["n_criteria_with_results"] > 0
    assert report["assay"]["summary"]["grade"] == "pass"
    assert report["results"]["overall"]["genes"]["v"]["call_top1_in_set"] == 1.0
    assert "stratum:clean" in report["results"]["by_context"]


def test_run_benchmark_report_from_predictor():
    cases = generate_benchmark(_small_spec())

    def predictor(reads):
        assert len(reads) == len(cases)
        return [case_to_prediction(c) for c in cases]

    report = run_benchmark_report(cases, predictor, contract_level="core")
    assert report["prediction_validation"]["n_valid"] == len(cases)
    assert report["assay"]["by_category"]["allele_calling"]["n_with_results"] > 0


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
    report = run_online_benchmark(spec, predictor, batch_size=2, contract_level="core")
    assert report["coverage"]["n_cases"] == spec.n_cases
    assert report["coverage"]["alleles"]["v"]["n_total_reference"] > 0
    assert 0.0 < report["coverage"]["alleles"]["v"]["fraction_observed"] <= 1.0
    assert report["criteria"]
    assert report["prediction_contract"]
    assert report["prediction_validation"]["valid_fraction"] == 1.0
    assert report["scenario_axes"]
    assert report["assay"]["summary"]["n_criteria_with_results"] > 0
    assert report["diagnostics"]["allele_calling"]["genes"]["v"]["summary"]["n_truth_cases"] == spec.n_cases
    assert report["diagnostics"]["boundaries"]["genes"]["v"]["summary"]["n_truth_segments"] == spec.n_cases
    assert "allele_calling" in report["assay"]["by_category"]
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

    report = run_online_benchmark(spec, predictor, batch_size=2, coverage_plan=plan, contract_level="minimal")
    assert report["generation_coverage"]["satisfied"]
    assert report["generation_coverage"]["observed_target_counts"]["orientation:reverse_complement"] == 1
    assert report["prediction_validation"]["valid_fraction"] == 0.0
    assert report["assay"]["summary"]["n_criteria"] == len(report["criteria"])
