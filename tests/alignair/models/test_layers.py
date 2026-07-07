"""Unit tests for the faithful PyTorch AlignAIR layers (port of TF Models/Layers/Layers.py)."""
import torch

from alignair.models.layers import (Conv1DBatchNorm, ConvResidualFeatureExtractionBlock,
                                     SoftCutoutLayer, TokenAndPositionEmbedding, same_pad1d)
from alignair.nn.weighting import UncertaintyWeight


def test_token_pos_embedding_shape_and_additivity():
    emb = TokenAndPositionEmbedding(vocab_size=6, embed_dim=32, maxlen=16)
    x = torch.randint(0, 6, (4, 16))
    out = emb(x)
    assert out.shape == (4, 16, 32)
    # position component is input-independent -> difference of two token seqs isolates token emb
    x2 = torch.randint(0, 6, (4, 16))
    assert torch.allclose((emb(x) - emb(x2)), emb.token(x) - emb.token(x2), atol=1e-6)


def test_same_pad1d_matches_tf():
    assert same_pad1d(2) == (0, 1)   # even kernel -> asymmetric (TF 'same')
    assert same_pad1d(3) == (1, 1)
    assert same_pad1d(5) == (2, 2)


def test_conv_bn_shapes_and_pooling():
    blk = Conv1DBatchNorm(in_channels=8, filters=8, kernel=2, pool=2).eval()
    x = torch.randn(2, 8, 17)                       # channels-first (B, C, L)
    y = blk(x)
    assert y.shape == (2, 8, 8)                      # same-pad keeps L=17 -> pool2 -> 8


def test_resblock_output_and_halving():
    # N=4 -> length halved N+1=5 times: 512 -> 16; flatten(128*16) -> Linear(576)
    blk = ConvResidualFeatureExtractionBlock(in_channels=32, filters=128, N=4,
                                             kernels=[3, 3, 3, 2, 5], max_len=512, out=576).eval()
    x = torch.randn(2, 512, 32)                     # (B, L, C) channels-last at block boundary
    y = blk(x)
    assert y.shape == (2, 576)


def test_soft_cutout_mask_peaks_inside_interval():
    m = SoftCutoutLayer(max_size=10, k=3.0)
    mask = m(torch.tensor([[2.0]]), torch.tensor([[6.0]]))
    assert mask.shape == (1, 10)
    assert mask[0, 4] > mask[0, 0] and mask[0, 4] > mask[0, 9]   # soft indicator peaks in [2,6)


def test_soft_cutout_enforces_min_span():
    m = SoftCutoutLayer(max_size=10, k=3.0)
    mask = m(torch.tensor([[5.0]]), torch.tensor([[5.0]]))       # end==start -> forced end>=start+1
    assert mask.sum() > 0                                        # non-degenerate


def test_kendall_uncertainty_weight_reuse():
    uw = UncertaintyWeight()
    loss = torch.tensor(2.0)
    contrib = loss * uw.forward() + uw.penalty()                 # proper Kendall: loss*e^-s + 0.5*s
    assert torch.isclose(contrib, loss * torch.exp(-uw.log_var) + 0.5 * uw.log_var)
    uw.log_var.data.fill_(5.0); uw.apply_constraints()
    assert uw.log_var.item() == 3.0                             # clamp to max_log_var
