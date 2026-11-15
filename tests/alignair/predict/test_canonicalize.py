"""Orientation canonicalization: re-applying the predicted transform recovers the forward frame
(the transforms are involutions), so coords/germline/AIRR all operate on one consistent sequence."""
import pytest
from alignair.core.orientations import orient as _canonicalize

pytestmark = pytest.mark.component

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


def test_a_record_stores_its_post_crop_pre_orientation_input():
    """Each record OWNS its post-crop, pre-orientation read as `input_sequence`, and its canonical
    `sequence` is exactly the re-oriented input — so the AIRR writer never needs an external list to get
    orientation right (the fix for the Python-API double-reverse defect)."""
    import numpy as np

    from alignair.predict.results import light_record, read_results
    from alignair.predict.state import GeneCall, Predictions
    fwd = "ACGTACGTAC"
    original = _canonicalize(fwd, 1)                  # the read as submitted (reverse-complemented)
    canonical = _canonicalize(original, 1)            # re-oriented back to forward == fwd
    assert canonical == fwd
    calls = {"v": [GeneCall(("IGHV1-1*01",), (0.9,))], "j": [GeneCall(("IGHJ1*01",), (0.8,))]}
    aligns = {"v": [None], "j": [None]}
    preds = Predictions(allele={}, start={}, end={}, mutation_rate=np.array([0.0]), indel_count=np.array([0.0]),
                        orientation=np.array([1]))
    (read,) = read_results([canonical], [original], preds, calls, aligns, None, None, None, None, [None])
    record = light_record(read)
    assert record["sequence"] == canonical
    assert record["input_sequence"] == original
    assert _canonicalize(record["input_sequence"], record["orientation_id"]) == record["sequence"]
