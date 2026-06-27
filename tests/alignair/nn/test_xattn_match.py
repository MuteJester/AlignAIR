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


def test_real_encoder_end_to_end_shapes():
    # encode a few germlines (GERMLINE type) + a read (READ type) with the real encoder, then run
    # xattn_match on a constructed candidate pool. Locks that the real reps flow through cleanly.
    from alignair.nn.encoder.shared import SharedNucleotideEncoder
    from alignair.data.tokenizer import pad_tokenize
    torch.manual_seed(0)
    d = 32
    enc = SharedNucleotideEncoder(d_model=d, n_layers=1, nhead=4).eval()
    germlines = ["ACGTACGTACGTACGT", "ACGTACGTACGTACGA", "TTTTGGGGCCCCAAAA"]
    gtok, gmsk = pad_tokenize(germlines)
    with torch.no_grad():
        pos_reps = enc.forward_positions(gtok, gmsk, SharedNucleotideEncoder.GERMLINE)  # (3,Lg,d)
        rtok, rmsk = pad_tokenize([germlines[0]])                                        # read == allele 0
        seg = enc.forward_positions(rtok, rmsk, SharedNucleotideEncoder.READ)           # (1,S,d)
    matcher = CrossAttnMatcher(d_model=d, nhead=4)
    cand_idx = torch.tensor([[1, 0, 2]])                                                 # pool incl. true (0)
    out = xattn_match(matcher, seg, rmsk, pos_reps, gmsk, cand_idx)
    assert out["allele_logits"].shape == (1, 3)
    assert int(out["best_global_idx"][0]) in (0, 1, 2)
    assert 0 <= int(out["germ_start"][0]) < pos_reps.shape[1]
