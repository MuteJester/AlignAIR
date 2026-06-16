import math
import torch
from alignair.nn.weighting import UncertaintyWeight


def test_initial_precision_is_one():
    w = UncertaintyWeight(initial_value=1.0)
    # log_var initialized to log(1)=0 -> precision exp(-0)=1
    assert abs(w().item() - 1.0) < 1e-6


def test_regularization_term_nonnegative():
    w = UncertaintyWeight()
    assert w.regularization().item() >= 0.0


def test_clamp_constraint_bounds_log_var():
    w = UncertaintyWeight(min_log_var=-3.0, max_log_var=1.0)
    with torch.no_grad():
        w.log_var.fill_(5.0)
    w.apply_constraints()
    assert w.log_var.item() <= 1.0 + 1e-6


def test_precision_is_differentiable():
    w = UncertaintyWeight()
    loss = torch.tensor(2.0)
    weighted = loss * w()
    weighted.backward()
    assert w.log_var.grad is not None
