"""The genotype pass's bookkeeping: pass 1's rows with the moved ones swapped in, and the run's assembly counts moved
from the replaced rows' outcomes to their new ones."""
import csv

import pytest

from alignair.cli.genotype_pass import merge
from alignair.cli.predict_run import RunCounts

pytestmark = pytest.mark.component


def _write(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def test_merge_swaps_in_the_moved_rows_and_keeps_the_others(tmp_path):
    header = ["sequence_id", "v_call"]
    _write(tmp_path / "first.tsv", header, [["r0", "A*01"], ["r1", "A*02,A*01"], ["r2", "B*01"],
                                            ["r3", 'C*01\t"x"'], ["r4", "A*02,B*01"]])
    _write(tmp_path / "second.tsv", header, [["r1", "A*01,A*02"], ["r4", "B*01,A*02"]])
    merge(str(tmp_path / "first.tsv"), str(tmp_path / "second.tsv"), [1, 4], str(tmp_path / "out.tsv"))
    with open(tmp_path / "out.tsv", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert rows == [header, ["r0", "A*01"], ["r1", "A*01,A*02"], ["r2", "B*01"], ["r3", 'C*01\t"x"'],
                    ["r4", "B*01,A*02"]]


def test_a_move_takes_the_replaced_rows_outcomes_to_their_new_ones():
    counts = RunCounts(complete=5, partial=2, failed=1, fail_reasons={"no_junction": 1}, partial_reasons={"no_v": 2})
    counts.move(before=[("partial", "no_v"), ("failed", "no_junction"), ("complete", None)],
                after=[("complete", None), ("partial", "no_d"), ("complete", None)])
    assert (counts.complete, counts.partial, counts.failed) == (6, 2, 0)
    assert counts.fail_reasons == {} and counts.partial_reasons == {"no_v": 1, "no_d": 1}
