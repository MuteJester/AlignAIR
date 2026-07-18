import csv
import logging
import pytest
from alignair.data.readers import CsvTableReader
from alignair.data.column_schema import ColumnSet


def _write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_reads_rows_and_len(tmp_path):
    p = tmp_path / "d.csv"
    header = ["sequence", "v_call", "j_call", "v_sequence_start", "v_sequence_end",
              "j_sequence_start", "j_sequence_end", "mutation_rate", "productive", "indels"]
    _write_csv(p, [{k: v for k, v in zip(header, ["ACGT", "V*01", "J*01", 0, 4, 5, 8, 0.1, "T", "{}"])}], header)
    reader = CsvTableReader(str(p), ColumnSet(has_d=False))
    assert len(reader) == 1
    row = reader[0]
    assert row["sequence"] == "ACGT"
    assert row["v_sequence_end"] == "4"  # raw string values; adapter coerces


def test_missing_required_raises(tmp_path):
    p = tmp_path / "d.csv"
    header = ["sequence", "v_call"]  # missing j_call etc.
    _write_csv(p, [{"sequence": "ACGT", "v_call": "V*01"}], header)
    with pytest.raises(ValueError):
        CsvTableReader(str(p), ColumnSet(has_d=False))


def test_defaults_missing_optional(tmp_path, caplog):
    p = tmp_path / "d.csv"
    header = ["sequence", "v_call", "j_call", "v_sequence_start", "v_sequence_end",
              "j_sequence_start", "j_sequence_end", "mutation_rate"]  # no productive/indels
    _write_csv(p, [{k: v for k, v in zip(header, ["ACGT", "V*01", "J*01", 0, 4, 5, 8, 0.1])}], header)
    with caplog.at_level(logging.WARNING):
        reader = CsvTableReader(str(p), ColumnSet(has_d=False))
    row = reader[0]
    assert row["productive"] == 1.0
    assert row["indels"] == ""
    assert any("default" in m.lower() for m in caplog.messages)
