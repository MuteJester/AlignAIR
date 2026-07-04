import torch
from alignair.reference.reference_set import GeneReference
from alignair.matcher.data import build_sequence_groups, target_mask, encode_batch


def _gene():
    # two names share an identical sequence -> must collapse to one group.
    names = ["V1*01", "V1*02", "V2*01"]
    seqs = ["ACGTACGT", "ACGTACGT", "TTTTGGGG"]
    index = {n: i for i, n in enumerate(names)}
    return GeneReference(names, seqs, index)


def test_sequence_identical_alleles_collapse():
    g = build_sequence_groups(_gene())
    assert g.n == 2                                  # 3 alleles -> 2 unique sequences
    assert g.name_to_group["V1*01"] == g.name_to_group["V1*02"]
    assert set(g.group_names[g.name_to_group["V1*01"]]) == {"V1*01", "V1*02"}


def test_target_mask_is_set_aware_over_groups():
    g = build_sequence_groups(_gene())
    m = target_mask(["V1*02", "V2*01"], g)           # comma-list truth
    assert m.dtype == torch.bool and m.sum().item() == 2
    assert m[g.name_to_group["V1*01"]]               # V1*02 lights up the shared group


def test_encode_batch_pads_and_masks():
    ids, mask = encode_batch(["ACG", "ACGTA"], max_len=8)
    assert ids.shape == (2, 5) and mask.shape == (2, 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert ids[0, 3:].sum().item() == 0              # PAD id = 0


def test_stream_batches_smoke_igh():
    import GenAIRR.data as gd
    from alignair.gym.curriculum import Curriculum
    from alignair.matcher.config import MatcherConfig
    from alignair.matcher.data import stream_batches
    cfg = MatcherConfig(max_len=576)
    it = stream_batches(gd.HUMAN_IGH_OGRDB, cfg, dict(Curriculum().params(0.2)),
                        batch_size=4, seed=1)
    batch = next(iter(it))
    assert batch["read_ids"].shape[0] == 4
    assert set(batch["genes"]) == {"V", "D", "J"}
    assert batch["genes"]["V"]["target"].shape[0] == 4
    assert batch["genes"]["V"]["target"].any()          # some read has a true V group
