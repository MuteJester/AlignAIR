import pytest
import torch


TEST_SEQ_LEN = 256


@pytest.fixture
def dummy_tokens():
    """A batch of (B=2, L=TEST_SEQ_LEN) integer tokens in [0, 5]."""
    torch.manual_seed(0)
    return torch.randint(0, 6, (2, TEST_SEQ_LEN))
