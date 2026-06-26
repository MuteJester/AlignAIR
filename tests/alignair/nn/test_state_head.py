import torch
from alignair.nn.heads.state import PerPositionStateHead, state_counts, STATES, STATE_INDEX


def test_state_head_shape_and_backprop():
    head = PerPositionStateHead(d_model=16)
    h = torch.randn(2, 7, 16)
    logits = head(h)
    assert logits.shape == (2, 7, len(STATES))
    assert len(STATES) == 4
    logits.sum().backward()
    assert head.fc.weight.grad is not None


def test_state_counts_from_labels():
    L = 6
    names = ["germline", "substitution", "substitution", "insertion", "deletion", "germline"]
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, STATE_INDEX[nm]] = 10.0
    counts = state_counts(logits, torch.ones(1, L, dtype=torch.bool))
    assert counts["substitution_count"].tolist() == [2]
    assert counts["indel_count"].tolist() == [2]


def test_state_counts_respects_mask():
    L = 4
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i in range(L):
        logits[0, i, STATE_INDEX["substitution"]] = 10.0
    counts = state_counts(logits, torch.tensor([[True, True, False, False]]))
    assert counts["substitution_count"].tolist() == [2]
