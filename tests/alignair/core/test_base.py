import torch
from alignair.core.base import BaseAlignAIR


def test_base_forward_keys_with_d(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    d = out.as_dict()
    for k in ["v_start_logits", "v_end_logits", "j_start_logits", "j_end_logits",
              "d_start_logits", "d_end_logits",
              "v_start", "v_end", "j_start", "j_end", "d_start", "d_end",
              "v_allele", "j_allele", "d_allele",
              "mutation_rate", "indel_count", "productive"]:
        assert k in d, f"missing {k}"


def test_base_forward_shapes(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    B, L = dummy_tokens.shape
    assert out.v_start_logits.shape == (B, L)
    assert out.v_start.shape == (B, 1)
    assert out.v_allele.shape == (B, tiny_config_d.v_allele_count)
    assert out.d_allele.shape == (B, tiny_config_d.d_allele_count)
    assert out.mutation_rate.shape == (B, 1)


def test_base_no_d_omits_d_keys(tiny_config_no_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_no_d)
    out = model(dummy_tokens)
    d = out.as_dict()
    assert "d_allele" not in d and "d_start_logits" not in d


def test_base_backprop(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    out = model(dummy_tokens)
    out.v_start_logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_regularization_loss_is_finite_scalar(tiny_config_d, dummy_tokens):
    model = BaseAlignAIR(tiny_config_d)
    _ = model(dummy_tokens)
    reg = model.regularization_loss()
    assert reg.ndim == 0 and torch.isfinite(reg)
