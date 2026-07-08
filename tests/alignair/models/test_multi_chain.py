"""Tests for the faithful MultiChain port: SingleChain + a chain_type head/loss term."""
import torch

from alignair.config.alignair_config import AlignAIRConfig
from alignair.models import AlignAIR
from alignair.models.losses import hierarchical_loss, make_logvars


def _cfg(has_d=True, num_chain_types=3):
    return AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                          d_allele_count=4, has_d=has_d, num_chain_types=num_chain_types)


def _batch(cfg, B=4):
    return {"tokenized_sequence": torch.randint(1, 6, (B, cfg.max_seq_length)),
            "orientation": torch.zeros(B, dtype=torch.long)}


def _targets(cfg, B=4):
    tg = {"mutation_rate": torch.rand(B, 1), "indel_count": torch.zeros(B, 1),
          "productive": torch.ones(B, 1), "orientation": torch.zeros(B, dtype=torch.long),
          "chain_type": torch.randint(0, cfg.num_chain_types, (B,))}
    genes = ["v", "j"] + (["d"] if cfg.has_d else [])
    counts = {"v": cfg.v_allele_count, "j": cfg.j_allele_count, "d": cfg.d_allele_count}
    for g in genes:
        tg[f"{g}_start"] = torch.full((B, 1), 10.0)
        tg[f"{g}_end"] = torch.full((B, 1), 100.0)
        y = torch.zeros(B, counts[g]); y[:, 0] = 1.0
        tg[f"{g}_allele"] = y
    return tg


def test_forward_emits_chain_type_logits():
    cfg = _cfg(num_chain_types=3)
    model = AlignAIR(cfg)
    out = model(_batch(cfg, B=4))
    assert out["chain_type_logits"].shape == (4, 3)          # (B, num_chain_types)
    # backbone outputs still present and unchanged in shape
    assert out["v_allele"].shape == (4, cfg.v_allele_count)
    assert "d_allele" in out and "orientation_logits" in out


def test_chain_type_gradient_flows_to_head():
    cfg = _cfg()
    model = AlignAIR(cfg)
    model(_batch(cfg))["chain_type_logits"].sum().backward()
    head = model.meta_heads["chain_type_logits"].head
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_loss_includes_chain_type_term():
    cfg = _cfg()
    model = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    assert "chain_type" in logvars                            # multi-chain -> weight exists
    out = model(_batch(cfg))
    total, parts = hierarchical_loss(out, _targets(cfg), cfg, logvars)
    assert "chain_type" in parts
    assert torch.isfinite(total)


def test_light_chain_multichain_has_no_d_but_has_chain_type():
    cfg = _cfg(has_d=False, num_chain_types=2)
    model = AlignAIR(cfg)
    out = model(_batch(cfg))
    assert "d_allele" not in out
    assert out["chain_type_logits"].shape == (4, 2)


def test_single_chain_emits_no_chain_type():
    """Isolation: the base SingleChain (num_chain_types=1) must not grow a chain_type head/output."""
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                         d_allele_count=4, has_d=True)
    model = AlignAIR(cfg)
    out = model(_batch(cfg))
    assert "chain_type_logits" not in out
    assert "chain_type" not in make_logvars(cfg)
