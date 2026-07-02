"""Decoupled span + objectness head (YOLOX-style, 1-D)."""
import torch

from alignair.nn.sota.span_head import SpanHead


def test_span_and_objectness_shapes_and_range():
    B, G, d = 4, 3, 32
    q = torch.randn(B, G, d)
    out = SpanHead(d)(q)
    assert out["span"].shape == (B, G, 2)
    assert out["objectness"].shape == (B, G)
    assert (out["span"] >= 0).all() and (out["span"] <= 1).all()   # normalized fractions


def test_reg_and_obj_branches_are_decoupled():
    """YOLOX's finding: cls/reg get separate branches. The two stems must be independent params."""
    head = SpanHead(16)
    reg_params = {id(p) for p in head.reg_stem.parameters()} | {id(p) for p in head.reg.parameters()}
    obj_params = {id(p) for p in head.obj_stem.parameters()} | {id(p) for p in head.obj.parameters()}
    assert reg_params.isdisjoint(obj_params)


def test_span_head_gradients_flow():
    q = torch.randn(2, 3, 16, requires_grad=True)
    out = SpanHead(16)(q)
    (out["span"].sum() + out["objectness"].sum()).backward()
    assert q.grad is not None
