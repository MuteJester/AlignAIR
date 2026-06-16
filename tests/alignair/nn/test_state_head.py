import torch
from alignair.nn.state_head import PerPositionStateHead, state_counts, STATES, STATE_INDEX


def test_state_head_shape_and_backprop():
    head = PerPositionStateHead(d_model=16)
    h = torch.randn(2, 7, 16)
    logits = head(h)
    assert logits.shape == (2, 7, len(STATES))
    logits.sum().backward()
    assert head.fc.weight.grad is not None


def test_state_counts_from_labels():
    L = 8
    names = ["germline", "germline", "mutation", "noise", "noise", "insertion", "deletion", "germline"]
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, STATE_INDEX[nm]] = 10.0
    mask = torch.ones(1, L, dtype=torch.bool)
    counts = state_counts(logits, mask)
    assert counts["noise_count"].tolist() == [2]
    assert counts["mutation_count"].tolist() == [1]
    assert counts["indel_count"].tolist() == [2]   # 1 insertion + 1 deletion


def test_state_counts_respects_mask():
    L = 4
    names = ["noise", "noise", "noise", "noise"]
    logits = torch.full((1, L, len(STATES)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, STATE_INDEX[nm]] = 10.0
    mask = torch.tensor([[True, True, False, False]])
    counts = state_counts(logits, mask)
    assert counts["noise_count"].tolist() == [2]   # padded positions ignored
