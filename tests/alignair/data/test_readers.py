import csv
import bz2
import gzip
import lzma
import logging
import pytest
from alignair.data.readers import CsvTableReader
from alignair.data.column_schema import ColumnSet

pytestmark = pytest.mark.component


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


@pytest.mark.parametrize("sep", [",", "\t"])
@pytest.mark.parametrize("has_d", [False, True])
def test_preserves_quoted_and_empty_values(tmp_path, sep, has_d):
    schema = ColumnSet(has_d=has_d)
    header = ["unused"] + list(reversed(schema.as_list()))
    row = dict.fromkeys(header, "0")
    row.update(sequence="ACGT", v_call='V*01,V*02', j_call="NA", mutation_rate="NaN",
               productive="", indels='{"1": "I < A", "2": "D > C"}', unused="quoted\n\ntext")
    p = tmp_path / "table.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, delimiter=sep)
        writer.writeheader()
        writer.writerow(row)
    reader = CsvTableReader(p, schema, sep=sep)
    assert reader.columns == schema.as_list()
    assert reader.records == [{col: row[col] for col in schema}]


@pytest.mark.parametrize("nrows,expected", [(None, 3), (0, 0), (1, 1), (2, 2), (10, 3)])
def test_row_limit_ignores_blank_lines(tmp_path, nrows, expected):
    schema = ColumnSet(has_d=False)
    header = schema.required_columns
    p = tmp_path / "table.csv"
    p.write_text("\n  \n" + ",".join(header) + "\n\n" +
                 "\n \t\n".join(",".join([sequence] + ["0"] * (len(header) - 1))
                                   for sequence in ["ACGT", "TGCA", "AAAA"]) + "\n")
    reader = CsvTableReader(p, schema, nrows=nrows)
    assert len(reader) == expected
    assert [row["sequence"] for row in reader.records] == ["ACGT", "TGCA", "AAAA"][:expected]


@pytest.mark.parametrize("nrows", [-1, 1.5, "2"])
def test_invalid_row_limit_raises(tmp_path, nrows):
    with pytest.raises(ValueError, match="nrows"):
        CsvTableReader(tmp_path / "table.csv", ColumnSet(has_d=False), nrows=nrows)


@pytest.mark.parametrize("suffix,opener", [(".gz", gzip.open), (".bz2", bz2.open), (".xz", lzma.open)])
def test_compressed_tables(tmp_path, suffix, opener):
    schema = ColumnSet(has_d=False)
    p = tmp_path / ("table.tsv" + suffix)
    row = {col: "0" for col in schema.required_columns}
    row["sequence"] = "ACGT"
    with opener(p, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=schema.required_columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    assert CsvTableReader(p, schema, sep="\t").records == [{**row, "productive": 1.0, "indels": ""}]


def test_short_rows_keep_empty_strings(tmp_path):
    schema = ColumnSet(has_d=False)
    p = tmp_path / "table.csv"
    p.write_text(",".join(schema.as_list()) + "\nACGT,V*01,J*01\n")
    row = CsvTableReader(p, schema)[0]
    assert row == {col: {"sequence": "ACGT", "v_call": "V*01", "j_call": "J*01"}.get(col, "")
                   for col in schema}


def test_header_only_table(tmp_path):
    p = tmp_path / "table.csv"
    schema = ColumnSet(has_d=False)
    _write_csv(p, [], schema.required_columns)
    assert CsvTableReader(p, schema).records == []


def test_row_limit_does_not_parse_later_rows(tmp_path):
    schema = ColumnSet(has_d=False)
    p = tmp_path / "table.csv"
    p.write_text(",".join(schema.required_columns) + "\n" +
                 ",".join(["ACGT"] + ["0"] * (len(schema.required_columns) - 1)) + "\n" +
                 ",".join(["extra"] * (len(schema.required_columns) + 1)) + "\n")
    assert len(CsvTableReader(p, schema, nrows=1)) == 1
    with pytest.raises(ValueError, match="fields"):
        CsvTableReader(p, schema)
