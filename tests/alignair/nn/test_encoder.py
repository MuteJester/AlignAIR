import torch
from alignair.nn.encoder.shared import SharedNucleotideEncoder


def _enc():
    return SharedNucleotideEncoder(d_model=64, n_layers=2, nhead=4, max_len=128)


def test_shapes_and_pooled_normalized():
    enc = _enc()
    B, L = 3, 20
    tokens = torch.randint(0, 6, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    reps = enc.forward_positions(tokens, mask)
    assert reps.shape == (B, L, 64)
    emb = enc(tokens, mask)
    assert emb.shape == (B, 64)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(B), atol=1e-5)  # L2-normalized


def test_padding_does_not_leak_into_valid_positions():
    # valid-position reps must not change when padding tokens change (causal-of-padding)
    enc = _enc().eval()
    B, L, n = 2, 16, 10
    tokens = torch.randint(1, 6, (B, L))
    mask = torch.zeros(B, L, dtype=torch.bool); mask[:, :n] = True
    with torch.no_grad():
        r1 = enc.forward_positions(tokens, mask)
        t2 = tokens.clone(); t2[:, n:] = torch.randint(1, 6, (B, L - n))  # change pad region
        r2 = enc.forward_positions(t2, mask)
    assert torch.allclose(r1[:, :n], r2[:, :n], atol=1e-5)


def test_token_type_changes_representation():
    enc = _enc().eval()
    tokens = torch.randint(1, 6, (2, 12)); mask = torch.ones(2, 12, dtype=torch.bool)
    with torch.no_grad():
        as_read = enc.forward_positions(tokens, mask, SharedNucleotideEncoder.READ)
        as_germ = enc.forward_positions(tokens, mask, SharedNucleotideEncoder.GERMLINE)
    assert not torch.allclose(as_read, as_germ)


def test_gradient_flows():
    enc = _enc()
    tokens = torch.randint(0, 6, (2, 18)); mask = torch.ones(2, 18, dtype=torch.bool)
    out = enc(tokens, mask).sum()
    out.backward()
    assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)
