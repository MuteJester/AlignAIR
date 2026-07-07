"""Tests for the faithful hierarchical multi-task loss (proper-Kendall weighting)."""
import torch

from alignair.config.alignair_config import AlignAIRConfig
from alignair.models.losses import hierarchical_loss, make_logvars, soft_gaussian_target
from alignair.models.single_chain import SingleChainAlignAIR


def test_soft_gaussian_target_normalized():
    t = soft_gaussian_target(torch.tensor([5.0, 10.0]), length=32, sigma=1.5)
    assert t.shape == (2, 32)
    assert torch.allclose(t.sum(-1), torch.ones(2), atol=1e-5)
    assert t[0].argmax().item() == 5 and t[1].argmax().item() == 10   # peak at the target


def _targets(B, cfg):
    t = {"mutation_rate": torch.zeros(B, 1), "indel_count": torch.zeros(B, 1),
         "productive": torch.ones(B, 1)}
    for g in (["v", "j"] + (["d"] if cfg.has_d else [])):
        t[f"{g}_start"] = torch.full((B, 1), 10.0)
        t[f"{g}_end"] = torch.full((B, 1), 100.0)
        count = {"v": cfg.v_allele_count, "j": cfg.j_allele_count, "d": cfg.d_allele_count}[g]
        y = torch.zeros(B, count); y[:, 0] = 1.0
        t[f"{g}_allele"] = y
    return t


def test_hierarchical_loss_assembles_and_backprops():
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=10, d_allele_count=4,
                         j_allele_count=4, has_d=True)
    m = SingleChainAlignAIR(cfg)
    logvars = make_logvars(cfg)
    B = 2
    out = m({"tokenized_sequence": torch.randint(0, 6, (B, 256))})
    total, parts = hierarchical_loss(out, _targets(B, cfg), cfg, logvars)
    assert total.ndim == 0 and torch.isfinite(total)
    assert {"segmentation", "classification", "mutation", "indel", "productive"} <= set(parts)
    total.backward()
    assert any(p.grad is not None for p in m.parameters())
    assert any(uw.log_var.grad is not None for uw in logvars.values())   # Kendall params learn
