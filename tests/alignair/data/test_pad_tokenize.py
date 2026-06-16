import torch
from alignair.data.tokenizer import pad_tokenize


def test_pad_tokenize_shapes_and_mask():
    tokens, mask = pad_tokenize(["ACGT", "ACG"])
    assert tokens.shape == (2, 4) and mask.shape == (2, 4)
    assert tokens.dtype == torch.long and mask.dtype == torch.bool
    assert tokens[0].tolist() == [1, 4, 3, 2]   # A C G T per TOKEN_DICT
    assert tokens[1].tolist() == [1, 4, 3, 0]   # right-padded with 0
    assert mask[0].tolist() == [True, True, True, True]
    assert mask[1].tolist() == [True, True, True, False]


def test_pad_tokenize_unknown_to_n():
    tokens, _ = pad_tokenize(["AXN"])
    assert tokens[0].tolist() == [1, 5, 5]  # X -> N(5), N -> 5
