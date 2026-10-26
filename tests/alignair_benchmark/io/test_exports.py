"""JSONL round trips, external tool inputs, and AIRR normalization."""
import pytest
import csv
import json
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair_benchmark import (
    build_benchmark_manifest,
    build_benchmark_report,
    case_to_prediction,
    export_benchmark_inputs,
    generate_benchmark,
    load_airr_predictions,
    load_dicts_jsonl,
    load_jsonl,
    save_jsonl,
    save_dicts_jsonl,
    save_airr_input,
    save_fasta,
)
from alignair.reference.reference_set import ReferenceSet
from tests.alignair_benchmark._support import (
    _airr_row_for_case,
    _single_clean_case,
    _small_spec,
    _write_tsv,
)

pytestmark = pytest.mark.integration


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
    assert rows[0]["benchmark_measurements"]

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
    assert manifest["generation"]["genairr_feature_validation"]["valid"]
    assert manifest["generation"]["measurement_scenarios"]
    assert manifest["reference"]["genes"]["v"]["n_alleles"] > 0
    assert len(manifest["reference"]["sha256"]) == 64
    assert manifest["measurement_coverage"]["n_cases"] == len(cases)
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
    assert manifest["measurement_coverage"]["n_cases"] == len(cases)
    assert manifest["generation"]["measurement_scenario_validation"]["n_scenarios"] > 0
    assert manifest["coverage"]["n_cases"] == len(cases)
    assert manifest["readiness"]["profile"] == "assay"


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
