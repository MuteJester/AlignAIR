"""Output heads for AlignAIR models."""
import torch
import torch.nn as nn

from ..primitives.activations import make_activation


class SegmentationHead(nn.Module):
    """Per-position boundary logits: (B, in_features) -> (B, max_seq_length)."""

    def __init__(self, in_features: int, max_seq_length: int):
        super().__init__()
        self.linear = nn.Linear(in_features, max_seq_length)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class AlleleClassificationHead(nn.Module):
    """Latent dense (swish) -> sigmoid allele probabilities."""

    def __init__(self, in_features: int, latent_dim: int, num_alleles: int,
                 mid_activation: str = "swish"):
        super().__init__()
        self.mid = nn.Linear(in_features, latent_dim)
        self.act = make_activation(mid_activation)
        self.out = nn.Linear(latent_dim, num_alleles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mid(x))
        return torch.sigmoid(self.out(x))


class _ScalarRegressionHead(nn.Module):
    """Shared body for mutation-rate / indel-count heads: Dense(L, gelu) ->
    dropout -> Dense(1, relu), with output clamped to [min_val, max_val]."""

    def __init__(self, in_features: int, max_seq_length: int,
                 min_val: float, max_val: float, dropout: float = 0.05):
        super().__init__()
        self.mid = nn.Linear(in_features, max_seq_length)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(max_seq_length, 1)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mid(x))
        x = self.dropout(x)
        x = torch.relu(self.out(x))
        return torch.clamp(x, self.min_val, self.max_val)

    @torch.no_grad()
    def apply_constraints(self) -> None:
        # MinMaxValueConstraint on the output kernel.
        self.out.weight.clamp_(self.min_val, self.max_val)


class MutationRateHead(_ScalarRegressionHead):
    def __init__(self, in_features: int, max_seq_length: int):
        super().__init__(in_features, max_seq_length, min_val=0.0, max_val=1.0)


class IndelCountHead(_ScalarRegressionHead):
    def __init__(self, in_features: int, max_seq_length: int):
        super().__init__(in_features, max_seq_length, min_val=0.0, max_val=50.0)


class ProductivityHead(nn.Module):
    """Flatten meta features -> dropout -> sigmoid scalar."""

    def __init__(self, in_features: int, dropout: float = 0.05):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.dropout(x)))


class ChainTypeHead(nn.Module):
    """Dense(L, gelu) -> dropout -> softmax over chain types."""

    def __init__(self, in_features: int, max_seq_length: int, num_types: int,
                 dropout: float = 0.05):
        super().__init__()
        self.mid = nn.Linear(in_features, max_seq_length)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(max_seq_length, num_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.mid(x))
        x = self.dropout(x)
        return torch.softmax(self.out(x), dim=-1)
