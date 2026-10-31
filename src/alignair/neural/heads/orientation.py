"""In-model orientation: token transforms (identity / reverse-complement /
complement / reverse), a 4-class orientation head, and the bank orientation (the
transform under which a read's k-mers are the bound reference's). All transforms
are involutions, so canonicalizing = re-applying the predicted transform."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignair.errors import ConfigurationError

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


# Bases in a bank-orientation k-mer: a random 12-mer is one of 4**12 ≈ 16.8M, so against the ~70k k-mers of a human
# IGH reference a k-mer hits by chance about 0.4% of the time, while an unmutated germline stretch hits all along.
BANK_KMER = 12
_BASE_TOKENS = {"A": 1, "T": 2, "G": 3, "C": 4}


class BankOrientation(nn.Module):
    """The orientation a bound reference reads: for each of a read's 4 transforms, how many of its k-mers (A/C/G/T
    only) are k-mers of the reference's germlines. ``bind`` indexes the reference's k-mers (sorted codes in a buffer
    that is not saved with the weights: the reference is an input); ``vote`` gives each read the transform with most
    hits and whether that transform won outright (some hit and no tie)."""

    def __init__(self, k: int = BANK_KMER):
        super().__init__()
        self.k = k
        self.bound = False
        self.register_buffer("kmers", torch.zeros(0, dtype=torch.long), persistent=False)

    def _codes(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) tokens -> (B, L - k + 1) base-4 codes of their k-mers; -1 for a k-mer holding a non-A/C/G/T token."""
        if tokens.shape[1] < self.k:
            return tokens.new_full((tokens.shape[0], 0), -1)
        acgt = (tokens >= 1) & (tokens <= 4)
        windows = (tokens - 1).clamp(0, 3).unfold(1, self.k, 1)
        powers = 4 ** torch.arange(self.k - 1, -1, -1, device=tokens.device)
        return torch.where(acgt.unfold(1, self.k, 1).all(-1), (windows * powers).sum(-1), -1)

    def bind(self, reference) -> None:
        codes = [self._codes(torch.tensor([[_BASE_TOKENS.get(base, 5) for base in sequence.upper()]]))[0]
                 for gene in reference.genes.values() for sequence in gene.sequences]
        codes = torch.cat(codes) if codes else torch.zeros(0, dtype=torch.long)
        self.kmers = torch.unique(codes[codes >= 0]).to(self.kmers.device)
        self.bound = True

    def hits(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, 4): per transform id, the read's k-mers found in the bound reference."""
        if not self.bound:
            raise ConfigurationError("bank_orientation reads a reference: bind one with model.bind_reference")
        counts = []
        for transform in range(NUM_ORIENTATIONS):
            ids = torch.full((tokens.shape[0],), transform, dtype=torch.long, device=tokens.device)
            codes = self._codes(apply_orientation(tokens, mask, ids))
            counts.append((torch.isin(codes, self.kmers) & (codes >= 0)).sum(-1))
        return torch.stack(counts, -1)

    def vote(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(transform ids (B,), won (B,) bool)``: each read's transform with most hits, and whether it won outright."""
        hits = self.hits(tokens, mask)
        best = hits.max(-1)
        won = (best.values > 0) & ((hits == best.values[:, None]).sum(-1) == 1)
        return best.indices, won
