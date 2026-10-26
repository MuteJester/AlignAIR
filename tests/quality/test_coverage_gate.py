"""Coverage gates must reject absent evidence and uncovered branches."""
import pytest

from tests.support.coverage_gate import check_coverage

pytestmark = pytest.mark.component

POLICY = {"schema": "alignair.coverage-policy.v1",
          "files": {"critical.py": {"statements": 90, "branches": 80}}}


def _report(lines=10, branches=4):
    return {"files": {"critical.py": {"summary": {
        "num_statements": 10, "covered_lines": lines,
        "num_branches": 5, "covered_branches": branches,
    }}}}


def test_coverage_gate_requires_both_statement_and_branch_floors():
    assert check_coverage(_report(9, 4), POLICY) == []
    assert "statements" in check_coverage(_report(8, 4), POLICY)[0]
    assert "branches" in check_coverage(_report(10, 3), POLICY)[0]


def test_missing_file_cannot_disappear_from_the_gate():
    assert check_coverage({"files": {}}, POLICY) == ["critical.py: missing from coverage evidence"]


@pytest.mark.parametrize("summary", [
    {"covered_lines": 0, "num_statements": 0, "covered_branches": 0, "num_branches": 0},
    {"covered_lines": 11, "num_statements": 10, "covered_branches": 4, "num_branches": 5},
    {"covered_lines": 10, "num_statements": 10, "covered_branches": -1, "num_branches": 5},
])
def test_invalid_coverage_totals_fail_closed(summary):
    with pytest.raises(ValueError, match="invalid .* evidence"):
        check_coverage({"files": {"critical.py": {"summary": summary}}}, POLICY)


def test_file_without_branches_can_satisfy_a_branch_floor():
    report = _report()
    report["files"]["critical.py"]["summary"].update(num_branches=0, covered_branches=0)
    assert check_coverage(report, POLICY) == []
