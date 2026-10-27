"""In-model orientation: token transforms (identity / reverse-complement /
complement / reverse) and a 4-class orientation head. All transforms are
involutions, so canonicalizing = re-applying the predicted transform."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Complement lookup by token id: A(1)<->T(2), G(3)<->C(4), N(5)->N, pad(0)->pad.
_COMPLEMENT = torch.tensor([0, 2, 1, 4, 3, 5], dtype=torch.long)

# Transform ids
IDENTITY, REVERSE_COMPLEMENT, COMPLEMENT, REVERSE = 0, 1, 2, 3
NUM_ORIENTATIONS = 4


def complement(tokens: torch.Tensor) -> torch.Tensor:
    return _COMPLEMENT.to(tokens.device)[tokens]


def reverse_valid(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse only the valid (unpadded) prefix of each row; pad stays at the end."""
    B, L = tokens.shape
    lengths = mask.sum(dim=1, keepdim=True)                   # (B,1)
    ar = torch.arange(L, device=tokens.device).unsqueeze(0)   # (1,L)
    valid = ar < lengths
    rev_idx = (lengths - 1 - ar).clamp(min=0)                 # (B,L)
    reversed_tokens = torch.gather(tokens, 1, rev_idx)
    return torch.where(valid, reversed_tokens, torch.zeros_like(tokens))


def apply_orientation(tokens: torch.Tensor, mask: torch.Tensor,
                      transform_ids: torch.Tensor) -> torch.Tensor:
    """Apply a per-row transform id in {0,1,2,3} to the token batch."""
    comp = complement(tokens)
    rev = reverse_valid(tokens, mask)
    revcomp = reverse_valid(comp, mask)
    variants = torch.stack([tokens, revcomp, comp, rev], dim=0)  # (4,B,L)
    B = tokens.shape[0]
    return variants[transform_ids, torch.arange(B, device=tokens.device)]


class OrientationHead(nn.Module):
    """Light encoder -> 4-class orientation logits."""

    def __init__(self, d: int = 64, vocab_size: int = 6):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.conv = nn.Conv1d(d, d, 7, padding="same")
        self.fc = nn.Linear(d, NUM_ORIENTATIONS)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        h = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2) * m
        pooled = h.sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)
