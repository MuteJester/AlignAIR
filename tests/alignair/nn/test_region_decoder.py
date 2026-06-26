import torch
from alignair.nn.heads.region_decoder import RegionMaskSpanDecoder, GENES
from alignair.nn.heads.region import REGIONS


def test_shapes_and_masking():
    torch.manual_seed(0)
    dec = RegionMaskSpanDecoder(d_model=32, nhead=4)
    B, L, d = 2, 20, 32
    reps = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 15:] = False                                   # sample 0 padded after 15
    out = dec(reps, mask)
    assert out["region_logits"].shape == (B, L, len(REGIONS))
    for g in GENES:
        assert out["start_logits"][g].shape == (B, L)
        assert out["end_logits"][g].shape == (B, L)
    # padded positions are masked to a large negative (never selected by argmax/softmax)
    assert (out["region_logits"][0, 15:] < -1e30).all()
    assert (out["start_logits"]["V"][0, 15:] < -1e30).all()


def test_can_learn_a_fixed_segmentation():
    # the decoder should be able to fit a simple region layout + V boundaries
    torch.manual_seed(0)
    B, L, d = 4, 24, 32
    dec = RegionMaskSpanDecoder(d_model=d, nhead=4)
    reps = torch.randn(B, L, d, requires_grad=False)
    mask = torch.ones(B, L, dtype=torch.bool)
    # target: positions 0-3 pre, 4-11 V, rest J (indices into REGIONS)
    from alignair.nn.heads.region import REGION_INDEX
    region_tgt = torch.full((B, L), REGION_INDEX["J"], dtype=torch.long)
    region_tgt[:, :4] = REGION_INDEX["pre"]
    region_tgt[:, 4:12] = REGION_INDEX["V"]
    v_start = torch.full((B,), 4); v_end = torch.full((B,), 11)
    opt = torch.optim.Adam(dec.parameters(), lr=5e-3)
    first = last = None
    for step in range(150):
        out = dec(reps, mask)
        loss = torch.nn.functional.cross_entropy(
            out["region_logits"].reshape(-1, out["region_logits"].shape[-1]), region_tgt.reshape(-1))
        loss = loss + torch.nn.functional.cross_entropy(out["start_logits"]["V"], v_start)
        loss = loss + torch.nn.functional.cross_entropy(out["end_logits"]["V"], v_end)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0: first = loss.item()
        last = loss.item()
    assert last < first * 0.3, (first, last)
    out = dec(reps, mask)
    assert out["start_logits"]["V"].argmax(-1).float().mean().item() == 4.0
    assert out["end_logits"]["V"].argmax(-1).float().mean().item() == 11.0
