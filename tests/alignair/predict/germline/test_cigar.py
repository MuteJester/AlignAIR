"""One reader of the CIGARs the germline reader writes, and the walks the prediction stages make over them."""
import pytest

from alignair.predict.germline.cigar import aligned_columns, aligned_runs, cigar_ops, matched_bases

pytestmark = pytest.mark.component


def test_a_cigar_reads_as_its_count_and_op_pairs():
    assert cigar_ops("3M1I2M2D4=1X") == ((3, "M"), (1, "I"), (2, "M"), (2, "D"), (4, "="), (1, "X"))
    assert cigar_ops("") == cigar_ops(None) == ()


def test_insertions_move_the_read_and_deletions_the_germline():
    assert list(aligned_runs("3M1I2M2D1M", 10, 4)) == [(10, 4, 3), (14, 7, 2), (16, 11, 1)]
    reads, germs = aligned_columns("2M1I1M1D1M", 5, 0)
    assert reads.tolist() == [5, 6, 8, 9] and germs.tolist() == [0, 1, 2, 4]
    assert reads.dtype.name == germs.dtype.name == "int64"
    assert aligned_columns("", 0, 0)[0].tolist() == []


def test_matched_bases_count_read_bases_on_an_equal_germline_base():
    assert matched_bases("GATTACA", "GATTACA", "7M") == 7
    assert matched_bases("GATTACA", "GATCA", "3M2I2M") == 3 + 2      # GAT|TA inserted|CA on CA
    assert matched_bases("XXGATT", "GACTT", "2M1D2M", 2, 0) == 2 + 2  # GA on GA, C deleted, TT on TT
