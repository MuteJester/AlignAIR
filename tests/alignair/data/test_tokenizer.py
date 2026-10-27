import pytest
import numpy as np
from alignair.data.tokenizer import (
    CenterPaddedTokenizer, AA_TOKEN_DICT, AA_VOCAB_SIZE, AAProteinTokenizer, InvalidResidueError,
)

pytestmark = pytest.mark.component


def test_token_vocab():
    tok = CenterPaddedTokenizer(max_length=10)
    enc = tok.encode("ATGCN")
    assert list(enc) == [1, 2, 3, 4, 5]


def test_unknown_char_maps_to_n():
    tok = CenterPaddedTokenizer(max_length=10)
    assert list(tok.encode("AXZ")) == [1, 5, 5]


def test_center_pad_offsets():
    tok = CenterPaddedTokenizer(max_length=10)
    padded, pad_left = tok.encode_and_pad("ATGC")  # len 4, pad total 6 -> left 3 right 3
    assert padded.shape == (10,)
    assert pad_left == 3
    assert list(padded) == [0, 0, 0, 1, 2, 3, 4, 0, 0, 0]


def test_odd_padding_left_floor():
    tok = CenterPaddedTokenizer(max_length=8)
    padded, pad_left = tok.encode_and_pad("ATGCG")  # len 5, pad total 3 -> left 1 right 2
    assert pad_left == 1
    assert list(padded) == [0, 1, 2, 3, 4, 3, 0, 0]


def test_protein_vocabulary_matches_serialized_token_ids():
    assert AA_VOCAB_SIZE == 23
    # Proline is residue 13; zero is reserved for padding, not a protein character.
    assert AA_TOKEN_DICT == {
        'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10,
        'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20,
        'X':21, '*':22,
    }


def test_protein_encoding_accepts_lowercase_with_the_same_residue_ids():
    tokenizer = AAProteinTokenizer()
    np.testing.assert_array_equal(tokenizer.encode('acdef'), tokenizer.encode('ACDEF'))
    assert tokenizer.encode('ACDEF').tolist() == [1,2,3,4,5]


def test_unknown_protein_characters_map_to_x_by_default():
    assert AAProteinTokenizer().encode('BZJUO-1').tolist() == [21]*7


def test_strict_protein_encoding_reports_the_invalid_residue():
    with pytest.raises(InvalidResidueError) as caught:
        AAProteinTokenizer(on_invalid='raise').encode('AB')
    assert caught.value.context['residue'] == 'B'
