import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment
from alignair.nn.heads.region import REGION_INDEX


def test_extract_segment_left_aligns_gene_positions():
    B, L, d = 1, 8, 4
    reps = torch.arange(B * L * d, dtype=torch.float32).reshape(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    # region labels: positions 2,3,4 are V
    labels = torch.zeros(B, L, dtype=torch.long)
    labels[0, 2:5] = REGION_INDEX["V"]
    seg, seg_mask = extract_segment(reps, mask, labels, "V")
    assert seg.shape[0] == 1 and seg_mask[0].sum().item() == 3
    # first extracted row equals reps at original position 2
    assert torch.allclose(seg[0, 0], reps[0, 2])
    assert torch.allclose(seg[0, 2], reps[0, 4])


def test_germline_coords_shapes():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=2, nhead=4, dim_feedforward=64)
    model = DNAlignAIR(cfg)
    B, Ls, Lg = 2, 5, 12
    seg = torch.randn(B, Ls, cfg.d_model)
    seg_mask = torch.ones(B, Ls, dtype=torch.bool)
    germ = torch.randn(B, Lg, cfg.d_model)
    germ_mask = torch.ones(B, Lg, dtype=torch.bool)
    start_logits, end_logits = model.germline_coords(seg, seg_mask, germ, germ_mask)
    assert start_logits.shape == (B, Lg) and end_logits.shape == (B, Lg)
