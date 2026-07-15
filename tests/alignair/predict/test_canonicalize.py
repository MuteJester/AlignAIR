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


def test_to_records_stores_post_crop_pre_orientation_input():
    """Each record OWNS its post-crop, pre-orientation read as `input_sequence`, and its canonical
    `sequence` is exactly the re-oriented input — so the AIRR writer never needs an external list to get
    orientation right (the fix for the Python-API double-reverse defect)."""
    from types import SimpleNamespace

    from alignair.predict.pipeline import _to_records
    fwd = "ACGTACGTAC"
    original = _canonicalize(fwd, 1)                  # the read as submitted (reverse-complemented)
    canonical = _canonicalize(original, 1)            # re-oriented back to forward == fwd
    assert canonical == fwd
    calls = {"v": [SimpleNamespace(names=["IGHV1-1*01"], likelihoods=[0.9])],
             "j": [SimpleNamespace(names=["IGHJ1*01"], likelihoods=[0.8])]}
    aligns = {"v": [None], "j": [None]}
    preds = SimpleNamespace(orientation=[1], mutation_rate=[0.0], indel_count=[0.0],
                            productive=[True], chain_type=None)
    recs = _to_records([canonical], calls, aligns, ("v", "j"), preds, input_sequences=[original])
    assert recs[0]["sequence"] == canonical
    assert recs[0]["input_sequence"] == original
    assert _canonicalize(recs[0]["input_sequence"], recs[0]["orientation_id"]) == recs[0]["sequence"]
