import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.germline_encoder import GermlineEncoder


def test_encoder_output_is_normalized_embedding():
    enc = GermlineEncoder(embed_dim=64)
    tokens, mask = pad_tokenize(["ACGTACGTAC", "ACGT", "TTGGCCAATT"])
    emb = enc(tokens, mask)
    assert emb.shape == (3, 64)
    # L2-normalized rows (unit norm)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_encoder_masking_ignores_padding():
    enc = GermlineEncoder(embed_dim=32).eval()
    # same sequence, one padded longer in a batch with a longer neighbor
    t1, m1 = pad_tokenize(["ACGTACGT"])
    t2, m2 = pad_tokenize(["ACGTACGT", "AAAAAAAAAAAAAAAA"])
    with torch.no_grad():
        e1 = enc(t1, m1)[0]
        e2 = enc(t2, m2)[0]  # first row identical seq, but batch padded to len 16
    assert torch.allclose(e1, e2, atol=1e-5)  # padding must not change the embedding
