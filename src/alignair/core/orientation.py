"""The four orientations a read can come in, in the order of the orientation head's classes, and orienting a read:
as a string (``orient``), as a token batch (``apply_orientation``), by the 4-class orientation head on the shared
embedding (``EmbeddingOrientationHead``) and by the bound reference (``BankOrientation``: the orientation under which a
read's k-mers are the reference's). Each orientation is an involution: applying a read's orientation to it again gives
the forward read."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import ConfigurationError
from .tokens import COMPLEMENT_TOKENS, encode, is_base

ORIENTATIONS = ("forward", "reverse_complement", "complement", "reverse")
FORWARD, REVERSE_COMPLEMENT, COMPLEMENT, REVERSE = range(len(ORIENTATIONS))
NUM_ORIENTATIONS = len(ORIENTATIONS)

_COMPLEMENT_BASES = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    """The reverse complement of ``sequence`` (A/C/G/T/N, either case); any other character stays."""
    return sequence.translate(_COMPLEMENT_BASES)[::-1]


def orient(sequence: str, orientation: int) -> str:
    """``sequence`` under ``orientation`` (an index of ``ORIENTATIONS``); a character that is not a base or N stays."""
    if orientation == FORWARD:
        return sequence
    if orientation == REVERSE_COMPLEMENT:
        return reverse_complement(sequence)
    if orientation == COMPLEMENT:
        return sequence.translate(_COMPLEMENT_BASES)
    if orientation == REVERSE:
        return sequence[::-1]
    raise ValueError(f"orientation {orientation!r} is not one of 0..{NUM_ORIENTATIONS - 1}")


_COMPLEMENT_TOKENS = torch.tensor(COMPLEMENT_TOKENS, dtype=torch.long)


def complement(tokens: torch.Tensor) -> torch.Tensor:
    """The complement of base tokens; padding and N stay."""
    return _COMPLEMENT_TOKENS.to(tokens.device)[tokens]


def reverse_valid(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse only the valid (unpadded) prefix of each row; pad stays at the end."""
    L = tokens.shape[1]
    lengths = mask.sum(dim=1, keepdim=True)                   # (B,1)
    ar = torch.arange(L, device=tokens.device).unsqueeze(0)   # (1,L)
    valid = ar < lengths
    rev_idx = (lengths - 1 - ar).clamp(min=0)                 # (B,L)
    reversed_tokens = torch.gather(tokens, 1, rev_idx)
    return torch.where(valid, reversed_tokens, torch.zeros_like(tokens))


def apply_orientation(tokens: torch.Tensor, mask: torch.Tensor,
                      transform_ids: torch.Tensor) -> torch.Tensor:
    """Apply a per-row orientation (an index of ``core.orientations.ORIENTATIONS``) to the token batch."""
    comp = complement(tokens)
    oriented = {FORWARD: tokens, REVERSE_COMPLEMENT: reverse_valid(comp, mask), COMPLEMENT: comp,
                REVERSE: reverse_valid(tokens, mask)}
    variants = torch.stack([oriented[k] for k in range(NUM_ORIENTATIONS)], dim=0)  # (4,B,L)
    B = tokens.shape[0]
    return variants[transform_ids, torch.arange(B, device=tokens.device)]


class EmbeddingOrientationHead(nn.Module):
    """4-class orientation logits from the model's shared initial embeddings.

    Order-sensitive (a width-7 convolution over the masked embeddings) so forward vs reverse is distinguishable — a
    plain mean-pool could not tell a read from its reverse. Feeds the in-model correct-and-re-embed step.
    """

    def __init__(self, embed_dim: int, num_orientations: int = NUM_ORIENTATIONS):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, 7, padding="same")
        self.fc = nn.Linear(embed_dim, num_orientations)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # (B,L,C), (B,L)
        m = mask.unsqueeze(-1).to(emb.dtype)
        h = F.gelu(self.conv((emb * m).transpose(1, 2))).transpose(1, 2) * m
        pooled = h.sum(1) / m.sum(1).clamp(min=1.0)
        return self.fc(pooled)


# Bases in a bank-orientation k-mer: a random 12-mer is one of 4**12 ≈ 16.8M, so against the ~70k k-mers of a human
# IGH reference a k-mer hits by chance about 0.4% of the time, while an unmutated germline stretch hits all along.
BANK_KMER = 12


class BankOrientation(nn.Module):
    """The orientation a bound reference reads: for each of a read's 4 transforms, how many of its k-mers (A/C/G/T
    only) are k-mers of the reference's germlines. ``bind`` indexes the reference's k-mers (a bitmap over every
    possible k-mer code, 4**k booleans, in a buffer that is not saved with the weights: the reference is an input);
    ``vote`` gives each read the transform with most hits and whether that transform won outright (some hit and no
    tie)."""

    def __init__(self, k: int = BANK_KMER):
        super().__init__()
        self.k = k
        self.bound = False
        self.register_buffer("present", torch.zeros(0, dtype=torch.bool), persistent=False)

    def _codes(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) tokens -> (B, L - k + 1) base-4 codes of their k-mers; -1 for a k-mer holding a non-A/C/G/T token."""
        if tokens.shape[1] < self.k:
            return tokens.new_full((tokens.shape[0], 0), -1)
        acgt = is_base(tokens)
        windows = (tokens - 1).clamp(0, 3).unfold(1, self.k, 1)
        powers = 4 ** torch.arange(self.k - 1, -1, -1, device=tokens.device)
        return torch.where(acgt.unfold(1, self.k, 1).all(-1), (windows * powers).sum(-1), -1)

    def bind(self, reference) -> None:
        """Hold the k-mers of every allele of ``reference``."""
        codes = [self._codes(torch.from_numpy(encode(sequence))[None])[0]
                 for gene in reference.genes.values() for sequence in gene.sequences]
        codes = torch.cat(codes) if codes else torch.zeros(0, dtype=torch.long)
        present = torch.zeros(4 ** self.k, dtype=torch.bool)
        present[codes[codes >= 0]] = True
        self.present = present.to(self.present.device)
        self.bound = True

    def hits(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, 4): per transform id, the read's k-mers found in the bound reference."""
        if not self.bound:
            raise ConfigurationError("bank_orientation reads a reference: bind one with model.bind_reference")
        counts = []
        for transform in range(NUM_ORIENTATIONS):
            ids = torch.full((tokens.shape[0],), transform, dtype=torch.long, device=tokens.device)
            codes = self._codes(apply_orientation(tokens, mask, ids))
            counts.append((self.present[codes.clamp_min(0)] & (codes >= 0)).sum(-1))
        return torch.stack(counts, -1)

    def vote(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(transform ids (B,), won (B,) bool)``: each read's transform with most hits, and whether it won
        outright."""
        hits = self.hits(tokens, mask)
        best = hits.max(-1)
        won = (best.values > 0) & ((hits == best.values[:, None]).sum(-1) == 1)
        return best.indices, won
