import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.encoder.germline import GermlineEncoder


def test_forward_positions_shape_and_padding_zeroed():
    enc = GermlineEncoder(embed_dim=48)
    tokens, mask = pad_tokenize(["ACGTACGT", "ACG"])
    pos = enc.forward_positions(tokens, mask)
    assert pos.shape == (2, 8, 48)
    assert torch.allclose(pos[1, 3:], torch.zeros(5, 48), atol=1e-6)


def test_pooled_forward_still_normalized():
    enc = GermlineEncoder(embed_dim=48)
    tokens, mask = pad_tokenize(["ACGTACGT"])
    emb = enc(tokens, mask)
    assert emb.shape == (1, 48)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(1), atol=1e-5)
