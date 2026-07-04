import torch

PAD, A, C, G, T, N = 0, 1, 2, 3, 4, 5
VOCAB_SIZE = 6
_CHAR2ID = {"A": A, "C": C, "G": G, "T": T, "N": N}
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


def encode(seq: str) -> torch.Tensor:
    return torch.tensor([_CHAR2ID.get(ch, N) for ch in seq.upper()], dtype=torch.long)


def revcomp(seq: str) -> str:
    return "".join(_COMP.get(ch, "N") for ch in reversed(seq.upper()))
