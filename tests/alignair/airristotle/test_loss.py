import torch
from alignair.airristotle.config import AIRRConfig
from alignair.airristotle.model import AIRRistotle, airristotle_loss


def test_copy_logits_shape_and_loss_runs():
    cfg = AIRRConfig(vocab_size=50, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128, max_seq=64)
    m = AIRRistotle(cfg)
    ids = torch.randint(0, 50, (2, 20)); prompt_len = 12
    hid, lm = m(ids)
    cp = m.copy_logits(hid, prompt_len)
    assert cp.shape == (2, 20, prompt_len)
    batch = {"gen_target": torch.randint(0, 50, (2, 20)),
             "copy_target": torch.randint(0, prompt_len, (2, 20)),
             "is_copy": torch.randint(0, 2, (2, 20)),
             "loss_mask": torch.ones(2, 20, dtype=torch.long)}
    loss = airristotle_loss(lm, cp, batch)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_perfect_copy_gives_low_copy_loss():
    cfg = AIRRConfig(vocab_size=10, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=64, max_seq=32)
    m = AIRRistotle(cfg)
    lm = torch.zeros(1, 3, 10)
    cp = torch.full((1, 3, 4), -10.0); cp[0, 0, 2] = 10.0     # hidden0 points hard at prompt pos 2
    batch = {"gen_target": torch.zeros(1, 3, dtype=torch.long),
             "copy_target": torch.tensor([[0, 2, 0]]),        # label at position 1 = copy pos 2
             "is_copy": torch.tensor([[0, 1, 0]]),
             "loss_mask": torch.tensor([[0, 1, 0]])}
    assert airristotle_loss(lm, cp, batch).item() < 0.01
