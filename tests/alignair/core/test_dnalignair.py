import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.region_head import REGIONS
from alignair.nn.state_head import STATES


def test_dense_and_scalar_outputs():
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    model = DNAlignAIR(cfg)
    tokens, mask = pad_tokenize(["ACGTACGTACGT", "ACGTAC"])
    out = model.forward_dense(tokens, mask)
    B, L = tokens.shape
    assert out["orientation_logits"].shape == (B, 4)
    assert out["region_logits"].shape == (B, L, len(REGIONS))
    assert out["state_logits"].shape == (B, L, len(STATES))
    for k in ("noise_count", "mutation_rate", "indel_count", "productive"):
        assert out[k].shape == (B, 1)
    # bounded scalars
    assert (out["mutation_rate"] >= 0).all() and (out["mutation_rate"] <= 1).all()
    assert (out["productive"] >= 0).all() and (out["productive"] <= 1).all()
    assert (out["noise_count"] >= 0).all()
    # backbone reps exposed for downstream heads
    assert out["reps"].shape == (B, L, cfg.d_model)
