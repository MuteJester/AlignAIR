import torch
from alignair.nn.heads.cross_attn_matcher import CrossAttnMatcher, xattn_match


def _setup(B=2, C=3, S=5, K=6, Lg=7, d=16):
    torch.manual_seed(0)
    matcher = CrossAttnMatcher(d_model=d, nhead=4)
    seg = torch.randn(B, S, d)
    sm = torch.ones(B, S, dtype=torch.bool)
    pos_reps = torch.randn(K, Lg, d)
    pos_mask = torch.ones(K, Lg, dtype=torch.bool)
    cand_idx = torch.tensor([[0, 3, 5], [1, 2, 4]])[:B, :C]
    return matcher, seg, sm, pos_reps, pos_mask, cand_idx


def test_output_keys_and_shapes():
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    assert out["allele_logits"].shape == (2, 3)
    assert out["best_idx"].shape == (2,) and out["best_global_idx"].shape == (2,)
    assert out["germ_start"].shape == (2,) and out["germ_end"].shape == (2,)
    assert out["gstart_logits"].shape == (2, 3, 7)


def test_best_global_idx_maps_through_cand_idx():
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    bi = out["best_idx"]
    expected_global = ci[torch.arange(2), bi]
    assert torch.equal(out["best_global_idx"], expected_global)


def test_germ_coords_are_argmax_of_chosen_candidate_pointers():
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    bi = out["best_idx"]
    chosen_gs = out["gstart_logits"][torch.arange(2), bi]      # (B,Lg)
    assert torch.equal(out["germ_start"], chosen_gs.argmax(-1))


def test_gathered_candidate_reps_match_reference():
    # the gather must pull pos_reps[cand_idx]; verify via a matcher-independent identity:
    # building cand_reps directly and calling the matcher gives the same logits.
    matcher, seg, sm, pr, pm, ci = _setup()
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    cand_reps = pr[ci]                                         # (B,C,Lg,d)
    cand_mask = pm[ci]
    match, _, _ = matcher(seg, sm, cand_reps, cand_mask)
    assert torch.allclose(out["allele_logits"], match)


def test_differentiable_through_seg_reps():
    matcher, seg, sm, pr, pm, ci = _setup()
    seg.requires_grad_(True)
    out = xattn_match(matcher, seg, sm, pr, pm, ci)
    out["allele_logits"].sum().backward()
    assert seg.grad is not None and torch.isfinite(seg.grad).all()
