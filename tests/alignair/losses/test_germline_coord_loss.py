import torch
from alignair.losses.dnalignair_loss import germline_coord_loss


def test_loss_minimized_at_truth():
    B, Lg = 4, 30
    gt_s = torch.tensor([5, 10, 3, 12])
    gt_e = torch.tensor([20, 25, 18, 28])
    pos = torch.arange(Lg).float()
    # sharp logits centered on truth vs centered 5nt off
    good_s = -(pos[None] - gt_s[:, None].float()) ** 2
    good_e = -(pos[None] - (gt_e[:, None].float() - 1)) ** 2
    bad_s = -(pos[None] - (gt_s[:, None].float() + 5)) ** 2
    bad_e = -(pos[None] - (gt_e[:, None].float() - 1 + 5)) ** 2
    good = germline_coord_loss(good_s, good_e, gt_s, gt_e).mean()
    bad = germline_coord_loss(bad_s, bad_e, gt_s, gt_e).mean()
    assert good < bad


def test_consistency_term_penalizes_span_mismatch_on_indel_free_rows():
    # end far from start+span should cost more when indel_free=True
    B, Lg = 1, 40
    gt_s = torch.tensor([5]); gt_e = torch.tensor([25])      # span 20
    pos = torch.arange(Lg).float()
    s = -(pos[None] - 5.0) ** 2
    e_ok = -(pos[None] - 24.0) ** 2                          # end 24 -> span 20 (matches)
    e_bad = -(pos[None] - 34.0) ** 2                         # end 34 -> span 30 (mismatch)
    free = torch.tensor([True])
    l_ok = germline_coord_loss(s, e_ok, gt_s, gt_e, indel_free=free).mean()
    l_bad = germline_coord_loss(s, e_bad, gt_s, gt_e, indel_free=free).mean()
    assert l_bad > l_ok


def test_returns_per_row_vector():
    B, Lg = 3, 20
    s = torch.randn(B, Lg); e = torch.randn(B, Lg)
    out = germline_coord_loss(s, e, torch.tensor([1, 2, 3]), torch.tensor([5, 6, 7]))
    assert out.shape == (B,)
