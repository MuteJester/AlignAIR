"""Backbone transfer: shape-matched partial weight load."""
import torch
import torch.nn as nn

from alignair.train.transfer import summarize, transfer_compatible_weights


class _Net(nn.Module):
    def __init__(self, head_dim: int, shared_dim: int = 4):
        super().__init__()
        self.shared = nn.Linear(shared_dim, shared_dim)     # reference-agnostic (fixed shape)
        self.head = nn.Linear(shared_dim, head_dim)         # reference-specific (varies)


def test_transfer_copies_matching_shapes_only():
    src, tgt = _Net(head_dim=3), _Net(head_dim=5)
    with torch.no_grad():
        src.shared.weight.fill_(7.0)
        tgt.shared.weight.zero_()
    transferred, skipped = transfer_compatible_weights(tgt, src.state_dict())

    assert set(transferred) == {"shared.weight", "shared.bias"}
    assert set(skipped) == {"head.weight", "head.bias"}
    # the shared backbone was copied in place...
    assert torch.allclose(tgt.shared.weight, torch.full((4, 4), 7.0))
    # ...and the shape-mismatched head kept its own init (dim 5, not overwritten by src's dim 3)
    assert tgt.head.weight.shape[0] == 5


def test_transfer_ignores_extra_source_keys():
    tgt = _Net(head_dim=5)
    src_sd = dict(_Net(head_dim=5).state_dict())
    src_sd["nonexistent.weight"] = torch.zeros(2, 2)        # a key not in the target
    transferred, skipped = transfer_compatible_weights(tgt, src_sd)
    assert "nonexistent.weight" not in transferred and not skipped  # all target keys matched
    assert len(transferred) == len(tgt.state_dict())


def test_summarize_reports_percentage():
    s = summarize(["a.w", "a.b"], ["c.w"])
    assert "2/3" in s and "67%" in s
