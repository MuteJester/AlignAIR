import torch
from alignair.nn.primitives.activations import make_activation
from alignair.nn.primitives.embedding import TokenPositionEmbedding


def test_make_activation_known():
    assert isinstance(make_activation("swish"), torch.nn.SiLU)
    assert isinstance(make_activation("gelu"), torch.nn.GELU)
    assert isinstance(make_activation("tanh"), torch.nn.Tanh)


def test_make_activation_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        make_activation("not_an_activation")


def test_embedding_output_shape():
    emb = TokenPositionEmbedding(max_len=16, vocab_size=6, embed_dim=32)
    x = torch.randint(0, 6, (4, 16))
    out = emb(x)
    assert out.shape == (4, 16, 32)


def test_embedding_adds_position():
    # Two identical token rows should produce identical embeddings (positions equal).
    emb = TokenPositionEmbedding(max_len=4, vocab_size=6, embed_dim=8)
    x = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    out = emb(x)
    assert torch.allclose(out[0], out[1])
    # But position 0 and position 1 of the same token should differ (position added).
    assert not torch.allclose(out[0, 0], out[0, 1])
