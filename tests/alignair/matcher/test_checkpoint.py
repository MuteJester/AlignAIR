
import pytest
import torch
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.checkpoint import save_checkpoint, load_checkpoint

pytestmark = pytest.mark.component


def _tiny():
    return MatcherConfig(d_model=16, d_match=8, n_layers=2, dilations=(1,))


def test_roundtrip_preserves_step_weights_and_optimizer(tmp_path):
    cfg = _tiny()
    m = AIRRMatch(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # one update so the optimizer accumulates state (momentum/variance)
    loss = sum((p ** 2).sum() for p in m.parameters())
    loss.backward()
    opt.step()
    path = tmp_path / "ck.pt"
    save_checkpoint(path, cfg, m, opt, step=1234)

    m2 = AIRRMatch(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step = load_checkpoint(path, m2, opt2, device="cpu")

    assert step == 1234
    assert len(opt2.state) > 0                                   # optimizer state restored
    assert torch.equal(next(m.parameters()), next(m2.parameters()))  # weights match


def test_weights_only_warmstart_ignores_optimizer(tmp_path):
    cfg = _tiny()
    m = AIRRMatch(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, cfg, m, opt, step=500)

    m2 = AIRRMatch(cfg)
    step = load_checkpoint(path, m2, opt=None, device="cpu")     # warm-start, no optimizer
    assert step == 500
    assert torch.equal(next(m.parameters()), next(m2.parameters()))
