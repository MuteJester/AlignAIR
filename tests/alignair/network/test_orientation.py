"""Orientation: the token transforms (involutions, padding kept at the end) and the embedding orientation head."""
import pytest
import torch

from alignair.data.batching import pad_vectors
from alignair.data.tokenizer import TOKEN_DICT
from alignair.network.orientation import (NUM_ORIENTATIONS, EmbeddingOrientationHead, apply_orientation, complement,
                                          reverse_valid)

pytestmark = pytest.mark.component

# token ids back to letters for readability
INV = {0: "-", 1: "A", 2: "T", 3: "G", 4: "C", 5: "N"}


def pad_tokenize(sequences):
    """Right-padded token ids and the valid-position mask of a batch of reads."""
    tokens = pad_vectors([[TOKEN_DICT[c] for c in s] for s in sequences])
    return tokens, tokens != 0


def to_str(tok, msk):
    return "".join(INV[int(t)] for t, m in zip(tok, msk) if m)


def test_complement():
    tok, msk = pad_tokenize(["ACGT"])
    assert to_str(complement(tok)[0], msk[0]) == "TGCA"


def test_reverse_valid_respects_padding():
    tok, msk = pad_tokenize(["ACGT", "AC"])
    r = reverse_valid(tok, msk)
    assert to_str(r[0], msk[0]) == "TGCA"
    assert to_str(r[1], msk[1]) == "CA"
    assert r[1].tolist()[2:] == [0, 0]


def test_reverse_complement():
    tok, msk = pad_tokenize(["AAAC"])
    assert to_str(apply_orientation(tok, msk, torch.tensor([1]))[0], msk[0]) == "GTTT"


def test_transforms_are_involutions():
    tok, msk = pad_tokenize(["ACGTACG", "ACG"])
    for tid in range(NUM_ORIENTATIONS):
        ids = torch.full((2,), tid)
        assert torch.equal(apply_orientation(apply_orientation(tok, msk, ids), msk, ids), tok), tid


def test_the_embedding_head_gives_one_logit_per_orientation_and_trains():
    head = EmbeddingOrientationHead(embed_dim=16)
    emb = torch.randn(3, 20, 16)
    mask = torch.ones(3, 20, dtype=torch.bool)
    mask[2, 10:] = False
    logits = head(emb, mask)
    assert logits.shape == (3, NUM_ORIENTATIONS)
    logits.sum().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_the_embedding_head_tells_a_read_from_its_reverse():
    torch.manual_seed(0)
    head = EmbeddingOrientationHead(embed_dim=8).eval()
    emb = torch.randn(1, 12, 8)
    mask = torch.ones(1, 12, dtype=torch.bool)
    assert not torch.allclose(head(emb, mask), head(emb.flip(1), mask))
