import torch
from alignair.reference.reference_set import ReferenceSet, GeneReference
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch


def _toy_ref():
    def gr(names, seqs):
        return GeneReference(names, seqs, {n: i for i, n in enumerate(names)})
    genes = {
        "V": gr(["V1*01", "V2*01"], ["ACGTACGTACGTACGT", "TTTTGGGGCCCCAAAA"]),
        "D": gr(["D1*01", "D2*01"], ["GGGGTTTT", "CCCCAAAA"]),
        "J": gr(["J1*01", "J2*01"], ["TTTTAAAACCCC", "GGGGCCCCAAAA"]),
    }
    return ReferenceSet(genes, has_d=True)


def test_predict_returns_names_and_sequences_for_each_gene():
    ref = _toy_ref()
    m = AIRRMatch(MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2))).eval()
    read = ref.gene("V").sequences[0] + ref.gene("D").sequences[0] + ref.gene("J").sequences[0]
    out = m.predict(ref, read)
    assert set(out) == {"V", "D", "J"}
    assert out["V"]["seq"] in ref.gene("V").sequences
    assert isinstance(out["V"]["names"], list) and out["V"]["names"]


def test_training_loss_is_scalar_and_differentiable():
    from alignair.matcher.data import build_sequence_groups, encode_batch, target_mask
    ref = _toy_ref()
    m = AIRRMatch(MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2)))
    groups = {g: build_sequence_groups(ref.gene(g)) for g in "VDJ"}
    reads = [ref.gene("V").sequences[0] + ref.gene("D").sequences[0] + ref.gene("J").sequences[0]]
    rid, rmask = encode_batch(reads, 64)
    genes = {}
    for g in "VDJ":
        cid, cmask = encode_batch(groups[g].unique_seqs, 64)
        genes[g] = {"cand_ids": cid, "cand_mask": cmask,
                    "target": torch.stack([target_mask([ref.gene(g).names[0]], groups[g])]),
                    "groups": groups[g]}
    batch = {"read_ids": rid, "read_mask": rmask, "read_rc_ids": rid, "read_rc_mask": rmask,
             "genes": genes}
    loss = m.training_loss(batch)
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())
