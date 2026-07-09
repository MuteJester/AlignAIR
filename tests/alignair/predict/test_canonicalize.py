"""Orientation canonicalization: re-applying the predicted transform recovers the forward frame
(the transforms are involutions), so coords/germline/AIRR all operate on one consistent sequence."""
from alignair.predict.pipeline import _canonicalize

_COMP = str.maketrans("ACGTN", "TGCAN")


def test_canonicalize_recovers_forward_frame():
    fwd = "ACGTACGTGGCCAN"
    assert _canonicalize(fwd, 0) == fwd                          # identity
    assert _canonicalize(fwd.translate(_COMP)[::-1], 1) == fwd   # revcomp -> forward
    assert _canonicalize(fwd.translate(_COMP), 2) == fwd         # complement -> forward
    assert _canonicalize(fwd[::-1], 3) == fwd                    # reverse -> forward


def test_canonicalize_preserves_length():
    seq = "ACGTACGT"
    for oid in (0, 1, 2, 3):
        assert len(_canonicalize(seq, oid)) == len(seq)
