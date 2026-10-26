import pytest
import torch

from alignair.aa.guard.encoder import ApplicabilityEncoder

pytestmark = pytest.mark.component


def _inp(real_ids, total_len):
    ids = real_ids + [0] * (total_len - len(real_ids))
    mask = [True] * len(real_ids) + [False] * (total_len - len(real_ids))
    return torch.tensor([ids]), torch.tensor([mask], dtype=torch.bool)


def test_output_shapes_and_determinism_in_eval():
    torch.manual_seed(0)
    enc = ApplicabilityEncoder(width=128, depth=4, heads=4).eval()
    ids, mask = _inp([0, 3, 5, 7, 9], 16)              # CLS + 4 residues, padded
    with torch.no_grad():
        c1, m1 = enc(ids, mask)
        c2, m2 = enc(ids, mask)
    assert c1.shape == (1, 128) and m1.shape == (1, 128)
    assert torch.allclose(c1, c2) and torch.allclose(m1, m2)


def test_padding_invariance():
    torch.manual_seed(0)
    enc = ApplicabilityEncoder(width=64, depth=2, heads=4).eval()
    real = [0, 3, 5, 7, 9, 11, 2]                      # CLS + 6 residues
    with torch.no_grad():
        c1, m1 = enc(*_inp(real, 10))
        c2, m2 = enc(*_inp(real, 28))                  # same content, more padding
    assert torch.allclose(c1, c2, atol=1e-5) and torch.allclose(m1, m2, atol=1e-5)
