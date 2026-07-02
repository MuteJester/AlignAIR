import torch
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle


def test_forward_shapes_and_causal():
    cfg = AIRRConfig(vocab_size=50, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=256, max_seq=64)
    m = AIRRistotle(cfg).eval()
    ids = torch.randint(0, 50, (2, 16))
    hid, logits = m(ids)
    assert hid.shape == (2, 16, 128)
    assert logits.shape == (2, 16, 50)
    ids2 = ids.clone(); ids2[:, -1] = (ids2[:, -1] + 1) % 50
    _, l2 = m(ids2)
    assert torch.allclose(logits[:, 0], l2[:, 0], atol=1e-5)


def test_150m_config_param_count_in_range():
    cfg = AIRRConfig(vocab_size=50)
    m = AIRRistotle(cfg)
    n = m.n_params()
    assert 120_000_000 <= n <= 190_000_000, n
