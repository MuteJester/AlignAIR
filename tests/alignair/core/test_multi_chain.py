import torch
from alignair.config.model_config import ModelConfig
from alignair.core.multi_chain import MultiChainAlignAIR


def _multi_config():
    return ModelConfig(
        max_seq_length=256, v_allele_count=5, j_allele_count=3, d_allele_count=4,
        has_d_gene=True, number_of_chains=2, chain_types=["IGH", "IGK"],
    )


def test_multi_chain_has_chain_type():
    model = MultiChainAlignAIR(_multi_config())
    out = model(torch.randint(0, 6, (2, 256)))
    assert out.chain_type is not None
    assert out.chain_type.shape == (2, 2)
    assert torch.allclose(out.chain_type.sum(dim=-1), torch.ones(2), atol=1e-5)
