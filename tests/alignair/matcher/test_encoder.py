import torch
from alignair.matcher.config import MatcherConfig
from alignair.matcher.encoder import MatcherEncoder


def _tiny_cfg():
    return MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2))


def test_output_shapes_and_length_preserved():
    cfg = _tiny_cfg()
    enc = MatcherEncoder(cfg).eval()
    B, L = 3, 40
    ids = torch.randint(1, 6, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    emb, sal = enc(ids, mask)
    assert emb.shape == (B, L, cfg.d_match)      # length preserved (no pooling)
    assert sal.shape == (B, L)


def test_match_embeddings_are_l2_normalized_and_salience_positive():
    cfg = _tiny_cfg()
    enc = MatcherEncoder(cfg).eval()
    ids = torch.randint(1, 6, (2, 25))
    mask = torch.ones(2, 25, dtype=torch.bool)
    emb, sal = enc(ids, mask)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert (sal > 0).all()
