"""Tests for the faithful hierarchical multi-task loss (proper-Kendall weighting)."""
import torch
import pytest

from alignair.core.config import AlignAIRConfig
from alignair.core.losses import hierarchical_loss, make_logvars, soft_gaussian_target
from alignair.core import AlignAIR

pytestmark = pytest.mark.component


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
    m = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    B = 2
    out = m({"tokenized_sequence": torch.randint(0, 6, (B, 256))})
    total, parts = hierarchical_loss(out, _targets(B, cfg), cfg, logvars)
    assert total.ndim == 0 and torch.isfinite(total)
    assert {"segmentation", "classification", "mutation", "indel", "productive"} <= set(parts)
    total.backward()
    assert any(p.grad is not None for p in m.parameters())
    assert any(uw.log_var.grad is not None for uw in logvars.values())   # Kendall params learn


@pytest.mark.parametrize("gene", ["v", "d", "j"])
def test_absent_gene_predictions_do_not_contribute_to_loss(gene):
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=3, d_allele_count=3,
                         j_allele_count=3, has_d=True)
    model = AlignAIR(cfg).eval()
    out = model({"tokenized_sequence": torch.ones(2, 256, dtype=torch.long)})
    targets = _targets(2, cfg)
    targets[f"{gene}_supervise"] = torch.tensor([True, False])
    logvars = make_logvars(cfg)
    original, _ = hierarchical_loss(out, targets, cfg, logvars)
    changed = {key: value.clone() for key, value in out.items()}
    changed[f"{gene}_start"][1] = 200
    changed[f"{gene}_end"][1] = 1
    changed[f"{gene}_start_logits"][1] = torch.arange(256, dtype=torch.float32)
    changed[f"{gene}_end_logits"][1] = -torch.arange(256, dtype=torch.float32)
    changed[f"{gene}_allele"][1] = torch.tensor([0.99, 0.001, 0.001])
    other, _ = hierarchical_loss(changed, targets, cfg, logvars)
    torch.testing.assert_close(original, other)


def test_completely_absent_gene_has_zero_uncertainty_gradient():
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=3, d_allele_count=3,
                         j_allele_count=3, has_d=True)
    model = AlignAIR(cfg)
    out = model({"tokenized_sequence": torch.ones(2, 256, dtype=torch.long)})
    targets = _targets(2, cfg)
    targets["j_supervise"] = torch.zeros(2, dtype=torch.bool)
    logvars = make_logvars(cfg)
    total, _ = hierarchical_loss(out, targets, cfg, logvars)
    assert torch.isfinite(total)
    total.backward()
    for key in ("j_start", "j_end", "j_classification"):
        assert logvars[key].log_var.grad.item() == 0
