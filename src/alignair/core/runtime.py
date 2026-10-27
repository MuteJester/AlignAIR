"""Shared model execution contexts used by training, evaluation, and prediction."""
from contextlib import contextmanager

import torch


@contextmanager
def evaluation_mode(model):
    """Disable gradients and restore every module's prior mode, including on failure."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
