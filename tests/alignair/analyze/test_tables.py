"""The analysis commands read an AIRR table one way, tab- or comma-separated, and parse its fields one way."""
import pytest

from alignair.analyze import analyze_file
from alignair.analyze.compare import read_airr
from alignair.analyze.tables import first_call, num, read_airr_table, truthy

pytestmark = pytest.mark.component


@pytest.mark.parametrize("delimiter", ["\t", ","])
def test_a_table_is_read_tab_or_comma_separated(tmp_path, delimiter):
    path = tmp_path / "rows.tsv"
    path.write_text(delimiter.join(["sequence_id", "v_call", "productive"]) + "\n"
                    + delimiter.join(["r1", "IGHV1-2*02", "T"]) + "\n" + delimiter.join(["", "IGHV3-23*01", "F"]) + "\n")
    rows, columns = read_airr_table(path)
    assert columns == ["sequence_id", "v_call", "productive"] and [row["v_call"] for row in rows] == [
        "IGHV1-2*02", "IGHV3-23*01"]
    assert list(read_airr(str(path))) == ["r1"]
    report = analyze_file(str(path))["repertoire"]
    assert report["n_reads"] == 2 and report["productive"]["n"] == 1


def test_the_shared_field_parsers():
    assert [truthy(value) for value in ("T", "true", "Y", "1", True, "F", "", None, False)] == [True] * 5 + [False] * 4
    assert (num("3.5"), num(""), num(None), num("x")) == (3.5, None, None, None)
    assert (first_call("IGHV1-2*02,IGHV1-2*04"), first_call(""), first_call(None)) == ("IGHV1-2*02", "", "")
