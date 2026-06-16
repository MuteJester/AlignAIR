import pytest
import torch


# The feature extractors downsample aggressively (the 6-conv-layer
# classification extractor is ~/128), so test sequences must be long enough to
# survive it. 256 leaves comfortable margin while keeping CPU tests fast.
TEST_SEQ_LEN = 256


@pytest.fixture
def tiny_config_d():
    """A small D-gene (heavy-chain-like) ModelConfig for fast tests."""
    from alignair.config.model_config import ModelConfig
    return ModelConfig(
        max_seq_length=TEST_SEQ_LEN,
        v_allele_count=5,
        j_allele_count=3,
        d_allele_count=4,
        has_d_gene=True,
    )


@pytest.fixture
def tiny_config_no_d():
    """A small non-D (light-chain-like) ModelConfig."""
    from alignair.config.model_config import ModelConfig
    return ModelConfig(
        max_seq_length=TEST_SEQ_LEN,
        v_allele_count=5,
        j_allele_count=3,
        d_allele_count=None,
        has_d_gene=False,
    )


@pytest.fixture
def dummy_tokens():
    """A batch of (B=2, L=TEST_SEQ_LEN) integer tokens in [0, 5]."""
    torch.manual_seed(0)
    return torch.randint(0, 6, (2, TEST_SEQ_LEN))
