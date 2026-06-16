"""String-keyed activation factory (mirrors the legacy string-based API)."""
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.3),  # Keras LeakyReLU default alpha=0.3
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "mish": nn.Mish,
}


def make_activation(name: str) -> "nn.Module":
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Known: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()
