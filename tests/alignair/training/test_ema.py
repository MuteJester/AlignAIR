import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.training.ema import EMATeacher


def test_ema_tracks_student_and_is_grad_free():
    torch.manual_seed(0)
    m = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    t = EMATeacher(m, decay=0.9)
    assert not any(p.requires_grad for p in t.model.parameters())
    before = next(t.model.parameters()).clone()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    t.update(m)
    moved = (next(t.model.parameters()) - before).abs().mean().item()
    assert 0.05 < moved < 0.15            # ~ (1-decay) * delta
    assert not any(p.requires_grad for p in t.model.parameters())
