import torch
from alignair.reference.reference_set import ReferenceSet, GeneReference
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch


def _toy_ref():
    def gr(names, seqs):
        return GeneReference(names, seqs, {n: i for i, n in enumerate(names)})
    return ReferenceSet({
        "V": gr(["V1*01", "V2*01"], ["ACGTACGTACGTACGTACGT", "TTTTGGGGCCCCAAAATTTT"]),
        "D": gr(["D1*01", "D2*01"], ["GGGGTTTTGGGG", "CCCCAAAACCCC"]),
        "J": gr(["J1*01", "J2*01"], ["TTTTAAAACCCCGGGG", "GGGGCCCCAAAATTTT"]),
    }, has_d=True)


def test_overfits_tiny_set():
    """Learning sanity: the model must fit a handful of exact-concat reads."""
    torch.manual_seed(0)
    from alignair.matcher.data import build_sequence_groups, encode_batch, target_mask
    ref = _toy_ref()
    groups = {g: build_sequence_groups(ref.gene(g)) for g in "VDJ"}
    picks = [(0, 0, 0), (1, 1, 1)]
    reads = ["".join([ref.gene(g).sequences[i] for g, i in zip("VDJ", p)]) for p in picks]
    rid, rmask = encode_batch(reads, 128)
    genes = {}
    for gi, g in enumerate("VDJ"):
        cid, cmask = encode_batch(groups[g].unique_seqs, 128)
        tgt = torch.stack([target_mask([ref.gene(g).names[p[gi]]], groups[g]) for p in picks])
        genes[g] = {"cand_ids": cid, "cand_mask": cmask, "target": tgt, "groups": groups[g]}
    batch = {"read_ids": rid, "read_mask": rmask, "read_rc_ids": rid, "read_rc_mask": rmask,
             "genes": genes}
    m = AIRRMatch(MatcherConfig(d_model=48, d_match=24, n_layers=4, dilations=(1, 2)))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        opt.zero_grad()
        loss = m.training_loss(batch)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    out = m.predict(ref, reads[0])
    assert "V1*01" in out["V"]["names"]
