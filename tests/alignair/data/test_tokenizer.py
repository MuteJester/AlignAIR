import numpy as np
from alignair.data.tokenizer import CenterPaddedTokenizer


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
