"""Convolutional building blocks (port of legacy Conv1D_and_BatchNorm and
ConvResidualFeatureExtractionBlock). All ops operate channel-first: (B, C, L).

Convolutions use ``padding='same'`` (stride 1) so the sequence length is
preserved exactly for any kernel parity, matching Keras ``padding='same'``.
Length reduction happens only through the explicit MaxPool layers. This is
required for the residual add to align between the feature and residual streams
when even-sized kernels are used.
"""
from typing import List, Union

import torch
import torch.nn as nn

from .activations import make_activation


class Conv1dBatchNorm(nn.Module):
    """Three stacked same-convolutions -> BatchNorm -> activation -> MaxPool.

    Mirrors legacy ``Conv1D_and_BatchNorm`` (3 convs, BN(momentum=0.1, eps=0.8),
    LeakyReLU, MaxPool1d). BN momentum is converted 0.1(Keras) -> 0.9(PyTorch).
    """

    def __init__(self, in_channels: int, filters: int = 16, kernel: int = 3,
                 max_pool: int = 2, activation: str = "leaky_relu"):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filters, kernel, padding="same")
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding="same")
        self.conv3 = nn.Conv1d(filters, filters, kernel, padding="same")
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


class ConvResidualFeatureExtractor(nn.Module):
    """Residual conv feature extractor (port of ConvResidualFeatureExtractionBlock).

    Input:  (B, L, E) embeddings (channel-last, as produced by the embedding).
    Output: (B, out_features).

    With ``kernel_sizes`` as a list of length N, the first N-1 kernels become
    Conv1dBatchNorm layers and the last kernel sizes the residual projection conv.

    Note: each Conv1dBatchNorm and each residual MaxPool halves the length, so the
    total downsampling factor is roughly ``2 ** (num_layers + 1)``. The input
    length ``L`` must be large enough to survive this (e.g. the 6-conv-layer
    classification extractor needs ``L >= 128``).
    """

    def __init__(self, in_channels: int, filter_size: int = 128,
                 kernel_sizes: Union[int, List[int]] = 5, max_pool_size: int = 2,
                 out_features: int = 576, activation: str = "tanh"):
        super().__init__()
        if isinstance(kernel_sizes, int):
            conv_kernels = [kernel_sizes] * 5
            residual_kernel = kernel_sizes
        else:
            conv_kernels = list(kernel_sizes[:-1])
            residual_kernel = kernel_sizes[-1]

        self.num_layers = len(conv_kernels)

        # First conv-batch layer maps in_channels -> filter_size; the rest map
        # filter_size -> filter_size (the residual stream is filter_size-wide).
        self.conv_layers = nn.ModuleList()
        for i, ks in enumerate(conv_kernels):
            cin = in_channels if i == 0 else filter_size
            self.conv_layers.append(
                Conv1dBatchNorm(cin, filters=filter_size, kernel=ks,
                                max_pool=max_pool_size, activation=activation)
            )

        self.residual_channel = nn.Conv1d(
            in_channels, filter_size, residual_kernel, padding="same")
        nn.init.xavier_uniform_(self.residual_channel.weight)
        if self.residual_channel.bias is not None:
            nn.init.zeros_(self.residual_channel.bias)

        self.pools = nn.ModuleList([nn.MaxPool1d(2) for _ in range(self.num_layers)])
        self.acts = nn.ModuleList([make_activation("leaky_relu") for _ in range(self.num_layers)])
        self.proj = nn.LazyLinear(out_features)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # (B, L, E) -> (B, E, L) for channel-first conv.
        x = embeddings.transpose(1, 2)

        residual = self.residual_channel(x)
        residual = self.pools[0](residual)

        feat = self.conv_layers[0](x)
        residual = feat + residual
        residual = self.acts[0](residual)
        residual = self.pools[0](residual)

        for i in range(1, self.num_layers):
            feat = self.conv_layers[i](residual)
            residual = self.pools[i](residual)
            residual = feat + residual
            residual = self.acts[i](residual)

        residual = torch.flatten(residual, start_dim=1)
        return self.proj(residual)
