"""Dynamic-reference matcher: mechanism (shapes, dynamic-ness, gradient) + a matching overfit."""
import torch

from alignair.nn.singlechain import SingleChainAlignAIR, hierarchical_loss
from alignair.nn.singlechain.dynamic import DynamicAlleleMatcher, contrastive_match_loss

L, V, D, J = 256, 8, 5, 4


def _model():
    return SingleChainAlignAIR(L, V, J, D, embed_dim=16, filter_size=16, feature_dim=128)


def test_matcher_shapes_and_is_reference_agnostic():
    torch.manual_seed(0)
    model = _model()
    matcher = DynamicAlleleMatcher(feature_dim=128)
    read = torch.randint(1, 6, (6, L))
    out = model(read)
    alleles = torch.randint(1, 6, (V, L))                 # a reference of V candidate germlines
    allele_feat = model.encode_alleles("V", alleles)
    scores = matcher.match("v", out["v_feature"], allele_feat)
    assert scores.shape == (6, V)
    # dynamic: hand it a DIFFERENT-sized reference (a novel/extended genotype) -> still scores, no new params
    alleles2 = torch.randint(1, 6, (V + 3, L))
    scores2 = matcher.match("v", out["v_feature"], model.encode_alleles("V", alleles2))
    assert scores2.shape == (6, V + 3) and torch.isfinite(scores2).all()


def test_matcher_gradients_flow_to_encoder_and_matcher():
    model = _model()
    matcher = DynamicAlleleMatcher(feature_dim=128)
    read = torch.randint(1, 6, (4, L))
    out = model(read)
    allele_feat = model.encode_alleles("V", torch.randint(1, 6, (V, L)))
    scores = matcher.match("v", out["v_feature"], allele_feat)
    scores.sum().backward()
    assert matcher.proj["v"].weight.grad is not None
    assert model.v_cls_fblock.proj.weight.grad is not None      # gradient reaches the shared encoder


def test_dynamic_mode_model_forward_loss_and_heldout_mask():
    torch.manual_seed(0)
    model = SingleChainAlignAIR(L, V, J, D, embed_dim=16, filter_size=16, feature_dim=128,
                                allele_mode="dynamic")
    B = 4
    read = torch.randint(1, 6, (B, L))
    reference = {"V": torch.randint(1, 6, (V, L)), "D": torch.randint(1, 6, (D, L)),
                 "J": torch.randint(1, 6, (J, L))}
    out = model(read, reference)
    assert out["v_allele_scores"].shape == (B, V) and out["d_allele_scores"].shape == (B, D)
    assert "v_allele" not in out                            # dynamic: scores, not fixed probs

    tgt = {k: torch.full((B,), float(v)) for k, v in
           dict(v_start=10, v_end=120, d_start=125, d_end=140, j_start=145, j_end=170).items()}
    for g, n in (("v", V), ("d", D), ("j", J)):
        oh = torch.zeros(B, n); oh[torch.arange(B), torch.arange(B) % n] = 1.0
        tgt[f"{g}_allele"] = oh
    tgt["mutation_rate"] = torch.full((B,), 0.05); tgt["indel_count"] = torch.full((B,), 1.0)
    tgt["productive"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss, logs = hierarchical_loss(model, out, tgt)
    assert torch.isfinite(loss)
    loss.backward()

    # held-out / reference-subset masking: a disabled allele can never be scored
    cm = {"V": torch.ones(V, dtype=torch.bool)}; cm["V"][2] = False
    out2 = model(read, reference, candidate_mask=cm)
    assert torch.isinf(out2["v_allele_scores"][:, 2]).all()
    assert torch.isfinite(out2["v_allele_scores"][:, [0, 1, 3]]).all()


def test_matcher_overfits_identity():
    """With read segments that ARE the reference alleles (position-aligned), the matcher must learn
    to rank each allele against itself — validates the matching objective end-to-end."""
    torch.manual_seed(1)
    K = 6
    model = _model()
    matcher = DynamicAlleleMatcher(feature_dim=128)
    alleles = torch.randint(1, 6, (K, L))                 # K distinct germlines
    target = torch.eye(K)                                 # read i's true allele is i
    params = list(model.parameters()) + list(matcher.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        read_feat = model.encode_alleles("V", alleles)    # "read segment" = the allele itself
        allele_feat = model.encode_alleles("V", alleles)
        scores = matcher.match("v", read_feat, allele_feat)
        loss = contrastive_match_loss(scores, target)
        loss.backward(); opt.step()
    with torch.no_grad():
        scores = matcher.match("v", model.encode_alleles("V", alleles),
                               model.encode_alleles("V", alleles))
    assert (scores.argmax(-1) == torch.arange(K)).all(), scores.argmax(-1)
