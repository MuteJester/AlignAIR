"""SingleChainAlignAIR PyTorch port: shapes + end-to-end overfit sanity."""
import torch

from alignair.nn.singlechain import SingleChainAlignAIR, hierarchical_loss

L, V, D, J = 256, 8, 5, 4


def _model():
    return SingleChainAlignAIR(max_seq_length=L, v_allele_count=V, j_allele_count=J,
                               d_allele_count=D, embed_dim=16, filter_size=16, feature_dim=128)


def _fixed_batch(B=6):
    torch.manual_seed(0)
    tokens = torch.randint(1, 6, (B, L))
    bounds = {"v_start": 10, "v_end": 120, "d_start": 125, "d_end": 140, "j_start": 145, "j_end": 170}
    tgt = {k: torch.full((B,), float(v)) for k, v in bounds.items()}
    for g, n in (("v", V), ("d", D), ("j", J)):
        true = torch.arange(B) % n
        oh = torch.zeros(B, n); oh[torch.arange(B), true] = 1.0
        tgt[f"{g}_allele"] = oh
        tgt[f"_{g}_true"] = true
    tgt["mutation_rate"] = torch.full((B,), 0.05)
    tgt["indel_count"] = torch.full((B,), 1.0)
    tgt["productive"] = torch.tensor([1.0, 0.0] * (B // 2))
    return tokens, tgt


def test_forward_shapes():
    tokens, _ = _fixed_batch()
    out = _model()(tokens)
    B = tokens.shape[0]
    for s in ("v_start", "v_end", "j_start", "j_end", "d_start", "d_end"):
        assert out[f"{s}_logits"].shape == (B, L)
        assert out[s].shape == (B, 1)
    assert out["v_allele"].shape == (B, V) and out["j_allele"].shape == (B, J)
    assert out["d_allele"].shape == (B, D)
    for k in ("mutation_rate", "indel_count", "productive"):
        assert out[k].shape == (B, 1)


def test_overfits_a_fixed_batch():
    tokens, tgt = _fixed_batch()
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    first = None
    for step in range(200):
        opt.zero_grad()
        out = model(tokens)
        loss, logs = hierarchical_loss(model, out, tgt)
        loss.backward(); opt.step()
        if step == 0:
            first = logs["total"]
    assert logs["total"] < 0.6 * first, f"loss did not drop: {first:.2f} -> {logs['total']:.2f}"

    model.eval()
    with torch.no_grad():
        out = model(tokens)
    # boundaries land near target (segmentation-first localization works)
    v_start_pred = out["v_start_logits"].argmax(-1).float()
    assert (v_start_pred - tgt["v_start"]).abs().mean() < 5.0
    # V allele called correctly
    assert (out["v_allele"].argmax(-1) == tgt["_v_true"]).all()
