import numpy as np
import pytest
from alignair.aa.tokenizer import (
    AA_TOKEN_DICT, AA_VOCAB_SIZE, InvalidResidueError,
    AAProteinTokenizer, pad_tokenize_aa,
)

pytestmark = pytest.mark.component


def test_vocab_shape_and_proline_gotcha():
    assert AA_VOCAB_SIZE == 23
    # 20 canonical residues -> 1..20, X -> 21, * -> 22 ; pad id 0 has NO character.
    assert AA_TOKEN_DICT["A"] == 1
    assert AA_TOKEN_DICT["P"] == 13          # proline is a real residue, NOT pad
    assert AA_TOKEN_DICT["Y"] == 20
    assert AA_TOKEN_DICT["X"] == 21
    assert AA_TOKEN_DICT["*"] == 22
    assert 0 not in AA_TOKEN_DICT.values()   # 0 is pad-only


def test_stop_and_unknown_are_distinct():
    assert AA_TOKEN_DICT["*"] != AA_TOKEN_DICT["X"]


def test_encode_maps_and_lowercases():
    tok = AAProteinTokenizer()
    np.testing.assert_array_equal(tok.encode("acdef"), tok.encode("ACDEF"))
    assert tok.encode("ACDEF").tolist() == [1, 2, 3, 4, 5]


def test_encode_invalid_maps_to_X_by_default():
    tok = AAProteinTokenizer()                      # on_invalid="X"
    # B, Z, J, U, O, gap and digit are non-canonical -> X (21)
    assert tok.encode("BZJUO-1").tolist() == [21] * 7


def test_encode_invalid_raises_when_configured():
    tok = AAProteinTokenizer(on_invalid="raise")
    with pytest.raises(InvalidResidueError):
        tok.encode("AB")                            # B invalid


def test_pad_tokenize_aa_right_pads_and_masks():
    tokens, mask = pad_tokenize_aa(["AC", "ACDE"])
    assert tokens.shape == (2, 4) and mask.shape == (2, 4)
    assert tokens[0].tolist() == [1, 2, 0, 0]       # right pad with 0
    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True]
