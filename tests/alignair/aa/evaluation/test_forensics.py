import pytest
from alignair.aa.forensics import germline_id_matrix, nearest_germline
from alignair.aa.reference import AAGermline
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


def _ref():
    g = GeneReference(["a*01", "a*02", "b*01"], ["", "", ""], {"a*01": 0, "a*02": 1, "b*01": 2})
    return ReferenceSet({"V": g, "J": GeneReference([], [], {})}, has_d=False)


def _germlines(seqs, anchors):
    return {n: AAGermline(n, s, 0, anchors[n], "V") for n, s in seqs.items()}


def test_germline_matrix_follows_reference_order_and_keeps_anchors():
    ref = _ref()
    gl = _germlines({"a*01": "MKVCAAAA", "a*02": "MKVCAAAA", "b*01": "MKLCAAAA"},
                    {"a*01": 3, "a*02": 3, "b*01": 3})
    names, G, glen, ganchor = germline_id_matrix(ref, gl, "V")
    assert names == ["a*01", "a*02", "b*01"]
    assert glen.tolist() == [8, 8, 8]
    assert ganchor.tolist() == [3, 3, 3]


def test_cys_anchored_identity_is_robust_to_a_frame_or_trim_offset():
    ref = _ref()
    # germline V regions before the Cys(anchor idx 4): a="MKVC?", b="MKLC?"; anchor residue 4 = the 'C'
    gl = _germlines({"a*01": "QMKVCXXXX", "a*02": "QMKVCXXXX", "b*01": "QMKLCXXXX"},
                    {"a*01": 4, "a*02": 4, "b*01": 4})
    names, G, _, ganchor = germline_id_matrix(ref, gl, "V")
    tok = AAProteinTokenizer()
    # OBSERVED read is 5'-trimmed: it lost the leading 'Q', so its V is "MKVC" with Cys at residue 3.
    # A residue-0 (prefix) anchor would misalign; Cys-anchoring (obs_cys=3 <-> ganchor=4) recovers it.
    obs = tok.encode("MKVC")
    idx, ident = nearest_germline(obs, obs_cys=3, G=G, ganchor=ganchor, min_overlap=3)
    assert names[idx] in ("a*01", "a*02")
    assert ident[0] == 1.0 and ident[2] < 1.0                     # a matches fully, b differs at V-residue
    # an obs matching the b protein selects b
    idx2, _ = nearest_germline(tok.encode("MKLC"), 3, G, ganchor, min_overlap=3)
    assert names[idx2] == "b*01"


def test_too_little_overlap_is_not_scored():
    ref = _ref()
    gl = _germlines({"a*01": "MKVCA", "a*02": "MKVCA", "b*01": "MKLCA"}, {"a*01": 3, "a*02": 3, "b*01": 3})
    _, G, _, ganchor = germline_id_matrix(ref, gl, "V")
    _, ident = nearest_germline(AAProteinTokenizer().encode("MK"), obs_cys=2, G=G, ganchor=ganchor,
                                min_overlap=15)
    assert (ident == -1.0).all()                                  # < min_overlap comparable residues
