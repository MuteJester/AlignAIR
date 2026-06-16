"""Convolutional building blocks (port of legacy Conv1D_and_BatchNorm and
ConvResidualFeatureExtractionBlock). All ops operate channel-first: (B, C, L)."""
from typing import List, Union

import torch
import torch.nn as nn

from .activations import make_activation


def _same_padding(kernel: int) -> int:
    # 'same' padding for odd kernels; for even kernels PyTorch can't do exact
    # 'same' with a single int, so we use floor(kernel/2) (length preserved for
    # odd kernels; even kernels shift by at most 1, matching the legacy warning).
    return kernel // 2


class Conv1dBatchNorm(nn.Module):
    """Three stacked same-convolutions -> BatchNorm -> activation -> MaxPool.

    Mirrors legacy ``Conv1D_and_BatchNorm`` (3 convs, BN(momentum=0.1, eps=0.8),
    LeakyReLU, MaxPool1d). BN momentum is converted 0.1(Keras) -> 0.9(PyTorch).
    """

    def __init__(self, in_channels: int, filters: int = 16, kernel: int = 3,
                 max_pool: int = 2, activation: str = "leaky_relu"):
        super().__init__()
        pad = _same_padding(kernel)
        self.conv1 = nn.Conv1d(in_channels, filters, kernel, padding=pad)
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding=pad)
        self.conv3 = nn.Conv1d(filters, filters, kernel, padding=pad)
        self.batch_norm = nn.BatchNorm1d(filters, eps=0.8, momentum=0.9)
        self.activation = make_activation(activation)
        self.max_pool = nn.MaxPool1d(max_pool)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for conv in (self.conv1, self.conv2, self.conv3):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x
