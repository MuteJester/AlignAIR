import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.orientation import apply_orientation, complement, reverse_valid, OrientationHead

# helper to decode tokens back to letters for readability
INV = {0: "-", 1: "A", 2: "T", 3: "G", 4: "C", 5: "N"}


def to_str(tok, msk):
    return "".join(INV[int(t)] for t, m in zip(tok, msk) if m)


def test_complement():
    tok, msk = pad_tokenize(["ACGT"])
    c = complement(tok)
    assert to_str(c[0], msk[0]) == "TGCA"  # A->T C->G G->C T->A


def test_reverse_valid_respects_padding():
    tok, msk = pad_tokenize(["ACGT", "AC"])  # second is right-padded to len 4
    r = reverse_valid(tok, msk)
    assert to_str(r[0], msk[0]) == "TGCA"
    assert to_str(r[1], msk[1]) == "CA"     # only the 2 valid bases reversed
    # padding stays at the end
    assert r[1].tolist()[2:] == [0, 0]


def test_reverse_complement():
    tok, msk = pad_tokenize(["AAAC"])
    out = apply_orientation(tok, msk, torch.tensor([1]))  # revcomp
    assert to_str(out[0], msk[0]) == "GTTT"  # comp(AAAC)=TTTG, reverse=GTTT


def test_transforms_are_involutions():
    tok, msk = pad_tokenize(["ACGTACG", "ACG"])
    for tid in (0, 1, 2, 3):
        ids = torch.full((2,), tid)
        once = apply_orientation(tok, msk, ids)
        twice = apply_orientation(once, msk, ids)
        assert torch.equal(twice, tok), f"transform {tid} is not an involution"


def test_orientation_head_shape_and_backprop():
    head = OrientationHead(d=32)
    tok, msk = pad_tokenize(["ACGTACGT", "ACG"])
    logits = head(tok, msk)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert all(p.grad is not None for p in head.parameters())
