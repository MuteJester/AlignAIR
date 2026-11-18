"""Golden records: a model's AIRR rows on a fixed read set, and the check that another code or model file gives the
same rows (strings, integers and booleans exactly, floats within ``TOLERANCE``)."""
import json

import pytest

from alignair_assay.diagnostics.golden import TOLERANCE, check_golden, compare_rows, record_golden

pytestmark = pytest.mark.integration


def test_compare_rows_is_exact_on_strings_and_tolerant_on_floats():
    a = {"v_call": "IGHV1-2*02", "v_sequence_start": 0, "v_likelihoods": [1.0, 0.5], "mutation_rate": 0.1}
    assert TOLERANCE == 1e-4
    assert compare_rows(a, dict(a)) == []
    assert compare_rows(a, {**a, "v_likelihoods": [1.0, 0.50005]}) == []
    assert compare_rows(a, {**a, "v_likelihoods": [1.0, 0.51]}) == ["v_likelihoods"]
    assert compare_rows(a, {**a, "v_likelihoods": [1.0]}) == ["v_likelihoods"]
    assert compare_rows(a, {**a, "v_call": "IGHV1-2*04"}) == ["v_call"]
    assert compare_rows(a, {**a, "v_sequence_start": 1}) == ["v_sequence_start"]
    assert compare_rows(a, {k: v for k, v in a.items() if k != "mutation_rate"}) == ["mutation_rate"]
    assert compare_rows({"productive": True}, {"productive": 1}) == ["productive"]


def test_a_recorded_model_checks_clean_and_a_changed_row_is_named(tmp_path, toy_chip_file, toy_reads_fasta):
    golden = tmp_path / "golden.jsonl"
    assert record_golden(toy_chip_file, toy_reads_fasta, str(golden)) == 3
    header, *records = [json.loads(line) for line in golden.read_text().splitlines()]
    assert header["kind"] == "alignair-golden" and header["n"] == 3 and [r["id"] for r in records] == ["r0", "r1", "r2"]
    assert check_golden(toy_chip_file, str(golden)).differences == []
    records[0]["row"]["v_call"] = "IGHV0-0*99"
    golden.write_text("".join(json.dumps(item) + "\n" for item in (header, *records)))
    report = check_golden(toy_chip_file, str(golden))
    assert report.n == 3 and [(d[0], d[1]) for d in report.differences] == [("r0", "v_call")]
