import pytest
import torch


@pytest.fixture
def tiny_config_d():
    """A tiny D-gene (heavy-chain-like) ModelConfig for fast tests."""
    from alignair.config.model_config import ModelConfig
    return ModelConfig(
        max_seq_length=16,
        v_allele_count=5,
        j_allele_count=3,
        d_allele_count=4,
        has_d_gene=True,
    )


@pytest.fixture
def tiny_config_no_d():
    """A tiny non-D (light-chain-like) ModelConfig."""
    from alignair.config.model_config import ModelConfig
    return ModelConfig(
        max_seq_length=16,
        v_allele_count=5,
        j_allele_count=3,
        d_allele_count=None,
        has_d_gene=False,
    )


@pytest.fixture
def dummy_tokens():
    """A batch of (B=2, L=16) integer tokens in [0, 5]."""
    torch.manual_seed(0)
    return torch.randint(0, 6, (2, 16))
