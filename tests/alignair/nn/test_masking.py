import torch
from alignair.nn.primitives.masking import SoftCutout


def test_softcutout_shape():
    m = SoftCutout(max_size=16, k=3.0)
    start = torch.tensor([[4.0], [0.0]])
    end = torch.tensor([[10.0], [16.0]])
    mask = m(start, end)
    assert mask.shape == (2, 16)


def test_softcutout_high_inside_low_outside():
    m = SoftCutout(max_size=20, k=1.0)
    start = torch.tensor([[5.0]])
    end = torch.tensor([[15.0]])
    mask = m(start, end)[0]
    # Middle of the interval should be near 1, far outside near 0.
    assert mask[10] > 0.9
    assert mask[0] < 0.1
    assert mask[19] < 0.1


def test_softcutout_enforces_min_width():
    # end <= start should be bumped to start + 1, producing a non-degenerate mask.
    m = SoftCutout(max_size=10, k=1.0)
    start = torch.tensor([[5.0]])
    end = torch.tensor([[5.0]])
    mask = m(start, end)
    assert torch.isfinite(mask).all()
    assert mask.max() > 0.0


def test_softcutout_differentiable():
    m = SoftCutout(max_size=10, k=1.0)
    start = torch.tensor([[3.0]], requires_grad=True)
    end = torch.tensor([[7.0]], requires_grad=True)
    m(start, end).sum().backward()
    assert start.grad is not None and torch.isfinite(start.grad).all()
