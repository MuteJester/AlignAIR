import json

import pytest

pytest.importorskip("GenAIRR")

from alignair.benchmark import (
    BENCHMARK_SUITE_MANIFEST,
    SuiteMeasurementReadinessThresholds,
    assess_benchmark_suite_readiness,
    default_measurement_suite_spec,
    export_benchmark_suite,
    generate_benchmark_suite,
    load_jsonl,
    validate_artifact,
)
from alignair.benchmark.cli import main as benchmark_cli


def _small_suite(**overrides):
    params = {
        "n_per_stratum": 0,
        "n_per_focus": 0,
        "n_panels": 1,
        "alleles_per_panel": 2,
        "n_per_panel": 1,
        "n_subjects": 1,
        "n_per_subject": 1,
        "n_per_locus": 1,
        "include_base_assay": False,
    }
    params.update(overrides)
    return default_measurement_suite_spec(**params)


def test_default_measurement_suite_spec_composes_expected_components():
    suite = _small_suite()

    assert [component.name for component in suite.components] == [
        "allele_panels",
        "genotype_subjects",
        "multi_locus",
    ]
    assert suite.n_cases == 5
    assert suite.components[0].measurement_focus == ("allele_coverage_and_candidates",)
    assert suite.components[1].measurement_focus == ("genotype_masked_inference",)
    assert suite.components[2].measurement_focus == ("multi_locus_chain",)


def test_generate_benchmark_suite_stamps_component_metadata():
    suite = _small_suite(seed=131)

    result = generate_benchmark_suite(suite, suite_readiness_profile="smoke")

    assert len(result.cases) == 5
    assert result.report["generation_report"]["mode"] == "benchmark_suite"
    assert result.report["suite_readiness"]["profile"] == "smoke"
    assert result.report["suite_readiness"]["n_scenarios"] == 3
    assert result.report["component_case_counts"] == {
        "allele_panels": 1,
        "genotype_subjects": 1,
        "multi_locus": 3,
    }
    components = {case.record["benchmark_component"] for case in result.cases}
    assert components == {"allele_panels", "genotype_subjects", "multi_locus"}
    assert {case.record["benchmark_suite"] for case in result.cases} == {suite.name}
    assert result.report["measurement_coverage"]["explicit_by_measurement"]["multi_locus_chain"] == 3


def test_suite_readiness_passes_targeted_measurements_with_custom_thresholds():
    suite = _small_suite(seed=133)
    result = generate_benchmark_suite(suite)
    thresholds = SuiteMeasurementReadinessThresholds(
        profile="unit",
        min_cases_per_measurement=1,
        min_explicit_cases_per_targeted_measurement=1,
        min_cases_per_required_label=0,
        required_measurements=(
            "allele_coverage_and_candidates",
            "genotype_masked_inference",
            "multi_locus_chain",
        ),
    )

    report = assess_benchmark_suite_readiness(result.cases, thresholds=thresholds)

    assert report["grade"] == "pass"
    assert report["grade_counts"] == {"pass": 3}
    assert report["measurement_coverage"]["explicit_by_measurement"]["genotype_masked_inference"] == 1


def test_suite_readiness_fails_missing_required_measurement():
    suite = _small_suite(seed=135, include_genotype_subjects=False)
    result = generate_benchmark_suite(suite)
    thresholds = SuiteMeasurementReadinessThresholds(
        profile="unit",
        min_cases_per_measurement=1,
        min_explicit_cases_per_targeted_measurement=1,
        min_cases_per_required_label=0,
        required_measurements=("genotype_masked_inference",),
    )

    report = assess_benchmark_suite_readiness(result.cases, thresholds=thresholds)

    assert report["grade"] == "fail"
    assert report["failed_scenarios"] == ["genotype_masked_inference"]
    row = report["scenarios"][0]
    assert row["n_cases"] == 0
    assert row["checks"]["explicit_targeted_case_count"] == "fail"


def test_export_benchmark_suite_writes_combined_and_component_manifests(tmp_path):
    suite = _small_suite(seed=137)
    result = generate_benchmark_suite(suite, suite_readiness_profile="smoke")

    files = export_benchmark_suite(result, tmp_path, prefix="tiny_suite", include_airr_metadata=True)

    assert (tmp_path / "tiny_suite.jsonl").exists()
    assert (tmp_path / "tiny_suite_manifest.json").exists()
    assert (tmp_path / "tiny_suite_combined_manifest.json").exists()
    assert (tmp_path / "tiny_suite_combined_airr_input.tsv").exists()
    assert (tmp_path / "components" / "allele_panels" / "allele_panels_manifest.json").exists()
    assert "combined" in files
    assert "allele_panels" in files["components"]

    manifest = json.loads((tmp_path / "tiny_suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact"]["kind"] == BENCHMARK_SUITE_MANIFEST
    assert validate_artifact(manifest, BENCHMARK_SUITE_MANIFEST, require_current_version=True)["valid"]
    assert manifest["benchmark"]["n_cases"] == 5
    assert manifest["suite_readiness"]["profile"] == "smoke"
    assert manifest["components"][0]["name"] == "allele_panels"
    assert manifest["measurement_coverage"]["explicit_by_measurement"]["genotype_masked_inference"] == 1


def test_cli_build_suite_smoke(tmp_path, capsys):
    out = tmp_path / "suite.jsonl"
    export_dir = tmp_path / "exports"

    benchmark_cli(
        [
            "build-suite",
            "--out",
            str(out),
            "--no-base-assay",
            "--n-panels",
            "1",
            "--alleles-per-panel",
            "2",
            "--n-per-panel",
            "1",
            "--n-subjects",
            "1",
            "--n-per-subject",
            "1",
            "--n-per-locus",
            "1",
            "--export-dir",
            str(export_dir),
            "--airr-metadata",
            "--suite-readiness-profile",
            "smoke",
        ]
    )

    payload = json.loads(capsys.readouterr().out.rsplit("\nwrote", 1)[0])
    cases = load_jsonl(out)
    assert len(cases) == 5
    assert payload["suite_readiness"]["profile"] == "smoke"
    assert payload["component_case_counts"]["multi_locus"] == 3
    assert payload["measurement_coverage"]["explicit_by_measurement"]["allele_coverage_and_candidates"] == 1
    assert (export_dir / "benchmark_suite_manifest.json").exists()
    assert (export_dir / "benchmark_suite_combined_manifest.json").exists()
