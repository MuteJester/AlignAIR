"""PyTorch port of the AlignAIR custom layers (originally TensorFlow/Keras `Models/Layers/Layers.py`).

Faithful reimplementations of the proven conv-based AlignAIR building blocks. Interfaces keep the
Keras channel-last convention (B, L, C) at the boundary and transpose to torch's (B, C, L) inside
conv blocks. See the legacy source for provenance.

Notes on faithful-but-unusual choices (kept for parity; flagged as patch candidates):
  - BatchNorm uses eps=0.8 (very high — mild normalization) exactly as the original.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAndPositionEmbedding(nn.Module):
    """Token embedding + learned absolute positional embedding (added). vocab 6, dim 32."""

    def __init__(self, maxlen: int, vocab_size: int = 6, embed_dim: int = 32):
        super().__init__()
        self.maxlen = maxlen
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):                                   # x (B, L) long -> (B, L, d)
        pos = torch.arange(self.maxlen, device=x.device)
        return self.token_emb(x) + self.pos_emb(pos)[None]


class Conv1DBatchNorm(nn.Module):
    """Three stacked same-conv → BatchNorm → activation → MaxPool(2). Halves length.

    (Legacy `Conv1D_and_BatchNorm`.) Works in (B, C, L)."""

    def __init__(self, filters: int, kernel: int, max_pool: int = 2, activation: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.LazyConv1d(filters, kernel, padding="same")
        self.conv2 = nn.Conv1d(filters, filters, kernel, padding="same")
        self.conv3 = nn.Conv1d(filters, filters, kernel, padding="same")
        self.bn = nn.LazyBatchNorm1d(eps=0.8)              # faithful: eps=0.8
        self.act = activation if activation is not None else nn.LeakyReLU()
        self.pool = nn.MaxPool1d(max_pool)

    def forward(self, x):                                  # (B, C, L) -> (B, filters, L/2)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.pool(self.act(self.bn(x)))


class ConvResidualFeatureExtractionBlock(nn.Module):
    """Residual conv feature extractor → flatten → Dense(out_shape). (Legacy block.)

    Input (B, L, C_in); output a flat feature vector (B, out_shape). `kernel_size` is a list: the
    first N-1 kernels are the conv-batchnorm layers, the last is the residual conv's kernel."""

    def __init__(self, filter_size: int, num_conv_batch_layers: int, kernel_size, max_pool_size: int = 2,
                 conv_activation: nn.Module | None = None, out_shape: int = 576):
        super().__init__()
        self.filter_size = filter_size
        act = conv_activation
        if isinstance(kernel_size, int):
            kernels = [kernel_size] * num_conv_batch_layers
            res_kernel = kernel_size
        else:
            kernels = list(kernel_size[:-1])
            res_kernel = kernel_size[-1]
        self.conv_layers = nn.ModuleList(
            Conv1DBatchNorm(filter_size, k, max_pool_size, activation=act) for k in kernels)
        self.residual_channel = nn.LazyConv1d(filter_size, res_kernel, padding="same")
        self.pools = nn.ModuleList(nn.MaxPool1d(2) for _ in range(num_conv_batch_layers))
        self.acts = nn.ModuleList(
            (act if act is not None else nn.LeakyReLU()) for _ in range(num_conv_batch_layers))
        self.proj = nn.LazyLinear(out_shape)

    def forward(self, embeddings):                          # (B, L, C) -> (B, out_shape)
        x = embeddings.transpose(1, 2)                      # (B, C, L)
        residual = self.pools[0](self.residual_channel(x))  # residual stream, pooled once
        feat = self.conv_layers[0](x)                       # conv-bn (pools once) -> matches residual
        residual = self.acts[0](feat + residual)
        residual = self.pools[0](residual)                  # block-0 pools again (faithful)
        for i in range(1, len(self.pools)):
            feat = self.conv_layers[i](residual)
            residual = self.pools[i](residual)
            residual = self.acts[i](feat + residual)
        return self.proj(residual.flatten(1))


class SoftCutout(nn.Module):
    """Differentiable soft mask over L positions from (start, end): sigmoid ramps of width k.
    (Legacy `SoftCutoutLayer`.) start/end are (B,1) floats -> mask (B, L)."""

    def __init__(self, max_size: int, k: float = 3.0):
        super().__init__()
        self.max_size = max_size
        self.k = k

    def forward(self, start, end):
        start = start.reshape(-1, 1).clamp(0.0, float(self.max_size))
        end = end.reshape(-1, 1).clamp(0.0, float(self.max_size))
        end = torch.maximum(end, start + 1.0)
        idx = torch.arange(self.max_size, device=start.device, dtype=start.dtype)[None]
        return torch.sigmoid((idx - start) / self.k) * torch.sigmoid((end - idx) / self.k)


class RegularizedConstrainedLogVar(nn.Module):
    """Kendall-style dynamic loss weighting: a scalar log-variance clamped to [min,max]; returns
    precision exp(-log_var) and exposes a regularizer discouraging log_var < -2.
    (Legacy `RegularizedConstrainedLogVar`.)"""

    def __init__(self, initial_value: float = 1.0, min_log_var: float = -3.0, max_log_var: float = 1.0,
                 regularizer_weight: float = 0.01):
        super().__init__()
        import math
        self.log_var = nn.Parameter(torch.tensor(float(math.log(initial_value))))
        self.min_log_var, self.max_log_var = min_log_var, max_log_var
        self.regularizer_weight = regularizer_weight

    def forward(self):
        lv = self.log_var.clamp(self.min_log_var, self.max_log_var)
        reg = self.regularizer_weight * F.relu(-lv - 2.0)
        return torch.exp(-lv), reg                          # (precision, reg_loss)
