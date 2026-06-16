import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.backbone import SequenceBackbone


def test_backbone_output_shape_and_padding_zeroed():
    bb = SequenceBackbone(d_model=64, n_layers=2, nhead=4)
    tok, msk = pad_tokenize(["ACGTACGT", "ACG"])
    h = bb(tok, msk)
    assert h.shape == (2, 8, 64)
    # padded positions are zeroed in the output
    assert torch.allclose(h[1, 3:], torch.zeros(5, 64), atol=1e-6)


def test_backbone_padding_invariance():
    bb = SequenceBackbone(d_model=64, n_layers=2, nhead=4).eval()
    t1, m1 = pad_tokenize(["ACGTACGT"])
    t2, m2 = pad_tokenize(["ACGTACGT", "AAAAAAAAAAAAAAAA"])  # forces longer padding on row 0
    with torch.no_grad():
        h1 = bb(t1, m1)[0]
        h2 = bb(t2, m2)[0, :8]
    assert torch.allclose(h1, h2, atol=1e-5)  # valid positions independent of padding
