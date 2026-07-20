import numpy as np

from alignair.aa.forensics import germline_id_matrix, nearest_germline
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.reference.reference_set import GeneReference, ReferenceSet


def _ref():
    g = GeneReference(["a*01", "a*02", "b*01"], ["", "", ""], {"a*01": 0, "a*02": 1, "b*01": 2})
    return ReferenceSet({"V": g, "J": GeneReference([], [], {})}, has_d=False)


def test_germline_matrix_follows_reference_order():
    ref = _ref()
    aa = {"a*01": "MKVC", "a*02": "MKVC", "b*01": "MKLC"}
    names, G, glen = germline_id_matrix(ref, aa, "V")
    assert names == ["a*01", "a*02", "b*01"]
    assert glen.tolist() == [4, 4, 4]


def test_nearest_germline_picks_the_aa_closest_allele():
    ref = _ref()
    aa = {"a*01": "MKVC", "a*02": "MKVC", "b*01": "MKLC"}
    names, G, _ = germline_id_matrix(ref, aa, "V")
    tok = AAProteinTokenizer()
    # exact match to the a-class protein -> nearest is a*01 or a*02 (both identical, tie -> first)
    idx, ident = nearest_germline(tok.encode("MKVC"), G)
    assert names[idx] in ("a*01", "a*02") and ident[idx] == 1.0
    assert ident[0] == ident[1] == 1.0 and ident[2] == 0.75      # b differs at position 2
    # a protein one residue from b -> nearest is b
    idx2, ident2 = nearest_germline(tok.encode("MKLC"), G)
    assert names[idx2] == "b*01" and ident2[2] == 1.0


def test_partial_observation_still_ranks_by_identity_over_the_window():
    ref = _ref()
    names, G, _ = germline_id_matrix(ref, {"a*01": "MKVC", "a*02": "MKVC", "b*01": "MKLC"}, "V")
    # only first 3 residues observed: "MKV" matches a (3/3) better than b (2/3)
    idx, ident = nearest_germline(AAProteinTokenizer().encode("MKV"), G)
    assert names[idx] in ("a*01", "a*02")
    assert ident[0] == 1.0 and abs(ident[2] - 2 / 3) < 1e-9
