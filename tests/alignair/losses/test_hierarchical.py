import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.core.multi_chain import MultiChainAlignAIR
from alignair.losses.hierarchical import AlignAIRLoss


def _targets_for(cfg, B):
    L = cfg.max_seq_length
    y = {
        "v_start": torch.full((B, 1), 1.0), "v_end": torch.full((B, 1), float(L // 2)),
        "j_start": torch.full((B, 1), float(L // 2 + 1)), "j_end": torch.full((B, 1), float(L - 1)),
        "v_allele": torch.zeros(B, cfg.v_allele_count),
        "j_allele": torch.zeros(B, cfg.j_allele_count),
        "mutation_rate": torch.full((B, 1), 0.1),
        "indel_count": torch.full((B, 1), 1.0),
        "productive": torch.ones(B, 1),
    }
    y["v_allele"][:, 0] = 1.0
    y["j_allele"][:, 0] = 1.0
    if cfg.has_d_gene:
        y["d_start"] = torch.full((B, 1), float(L // 2 - 2))
        y["d_end"] = torch.full((B, 1), float(L // 2))
        y["d_allele"] = torch.zeros(B, cfg.d_allele_count)
        y["d_allele"][:, 0] = 1.0
    return y


def test_loss_is_finite_and_backprops(tiny_config_d, dummy_tokens):
    model = SingleChainAlignAIR(tiny_config_d)
    loss_fn = AlignAIRLoss(tiny_config_d)
    out = model(dummy_tokens)
    y = _targets_for(tiny_config_d, dummy_tokens.shape[0])
    total, components = loss_fn(y, out.as_dict())
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert "segmentation_loss" in components and "classification_loss" in components


def test_loss_multichain_has_chain_type_component():
    cfg = ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                      d_allele_count=4, has_d_gene=True, number_of_chains=2,
                      chain_types=["IGH", "IGK"])
    model = MultiChainAlignAIR(cfg)
    loss_fn = AlignAIRLoss(cfg)
    x = torch.randint(0, 6, (2, 256))
    out = model(x)
    y = _targets_for(cfg, 2)
    y["chain_type"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    total, components = loss_fn(y, out.as_dict())
    assert torch.isfinite(total)
    assert "chain_type_loss" in components
