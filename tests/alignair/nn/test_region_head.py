import torch
from alignair.nn.region_head import RegionTagger, decode_boundaries, REGIONS, REGION_INDEX


def test_region_tagger_shape():
    tagger = RegionTagger(d_model=32)
    h = torch.randn(2, 10, 32)
    logits = tagger(h)
    assert logits.shape == (2, 10, len(REGIONS))


def test_decode_boundaries_from_labels():
    # build one sample whose argmax labels are: pre pre V V V N1 D D J J
    L = 10
    names = ["pre", "pre", "V", "V", "V", "N1", "D", "D", "J", "J"]
    logits = torch.full((1, L, len(REGIONS)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, REGION_INDEX[nm]] = 10.0
    mask = torch.ones(1, L, dtype=torch.bool)
    rec = decode_boundaries(logits, mask, has_d=True)[0]
    assert rec["v_start"] == 2 and rec["v_end"] == 5   # V at [2,5)
    assert rec["d_start"] == 6 and rec["d_end"] == 8   # D at [6,8)
    assert rec["j_start"] == 8 and rec["j_end"] == 10  # J at [8,10)


def test_decode_absent_gene_is_minus_one():
    L = 6
    names = ["V", "V", "V", "N1", "J", "J"]  # no D
    logits = torch.full((1, L, len(REGIONS)), -10.0)
    for i, nm in enumerate(names):
        logits[0, i, REGION_INDEX[nm]] = 10.0
    rec = decode_boundaries(logits, torch.ones(1, L, dtype=torch.bool), has_d=True)[0]
    assert rec["d_start"] == -1 and rec["d_end"] == -1
    assert rec["v_start"] == 0 and rec["j_end"] == 6
