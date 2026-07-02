"""Bidirectional read<->reference fusion (GLIP BiAttentionBlock, 1-D)."""
import torch

from alignair.nn.sota.fusion import BiAttentionBlock, ReferenceFusion


def test_fusion_preserves_shapes():
    B, Lr, Lc, d = 3, 20, 7, 32
    read = torch.randn(B, Lr, d); cand = torch.randn(B, Lc, d)
    r2, c2 = ReferenceFusion(d, nhead=4, n_layers=2)(read, cand)
    assert r2.shape == read.shape and c2.shape == cand.shape


def test_zero_init_gate_is_identity_at_init():
    """Flamingo-style: at init the gate is 0, so fusion returns its inputs unchanged — a fusion
    block can be added to a converged encoder without disrupting it."""
    torch.manual_seed(0)
    B, Lr, Lc, d = 2, 12, 5, 16
    read = torch.randn(B, Lr, d); cand = torch.randn(B, Lc, d)
    blk = BiAttentionBlock(d, nhead=4).eval()
    r2, c2 = blk(read, cand)
    assert torch.allclose(r2, read, atol=1e-6) and torch.allclose(c2, cand, atol=1e-6)


def test_fusion_changes_output_once_gated_and_is_masked():
    torch.manual_seed(0)
    B, Lr, Lc, d = 2, 12, 6, 16
    read = torch.randn(B, Lr, d); cand = torch.randn(B, Lc, d)
    blk = BiAttentionBlock(d, nhead=4)
    with torch.no_grad():
        blk.gate_r.fill_(1.0); blk.gate_c.fill_(1.0)
    cand_mask = torch.ones(B, Lc, dtype=torch.bool); cand_mask[:, 3:] = False
    r2, _ = blk(read, cand, cand_mask=cand_mask)
    assert not torch.allclose(r2, read)                 # fusion now moves the read
    assert torch.isfinite(r2).all()                     # masked candidates don't break it


def test_fusion_gradients_flow():
    B, Lr, Lc, d = 2, 10, 5, 16
    read = torch.randn(B, Lr, d, requires_grad=True); cand = torch.randn(B, Lc, d, requires_grad=True)
    blk = BiAttentionBlock(d, nhead=4)
    blk.gate_r.data.fill_(0.5)
    (blk(read, cand)[0].sum()).backward()
    assert read.grad is not None and blk.gate_r.grad is not None
