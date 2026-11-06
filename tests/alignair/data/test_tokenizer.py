import pytest
from alignair.data.tokenizer import NucleotideTokenizer

pytestmark = pytest.mark.component


def test_token_vocab():
    tok = NucleotideTokenizer(max_length=10)
    enc = tok.encode("ATGCN")
    assert list(enc) == [1, 2, 3, 4, 5]


def test_unknown_char_maps_to_n():
    tok = NucleotideTokenizer(max_length=10)
    assert list(tok.encode("AXZ")) == [1, 5, 5]

