"""AA↔gym bridge: turn REAL gym records into batches for the REAL AlignAIR model + loss, with
only the input (AA tokens) and coordinate space (AA residues) changed. Allele targets stay the
full nucleotide allele multi-hot, so ambiguity must EMERGE, never be hand-encoded."""
import numpy as np
import pytest
import torch

from alignair.aa.gym_bridge import aa_usable, build_batch_aa
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.integration


def _ref():
    genes = {
        "V": GeneReference(["IGHV1*01", "IGHV1*02", "IGHV2*01"], ["", "", ""],
                           {"IGHV1*01": 0, "IGHV1*02": 1, "IGHV2*01": 2}),
        "D": GeneReference(["IGHD1*01", "IGHD2*01"], ["", ""], {"IGHD1*01": 0, "IGHD2*01": 1}),
        "J": GeneReference(["IGHJ1*01", "IGHJ2*01"], ["", ""], {"IGHJ1*01": 0, "IGHJ2*01": 1}),
    }
    return ReferenceSet(genes, has_d=True)


class _Cfg:
    has_d = True
    max_seq_length = 192


def _rec(**over):
    # 24 nt, frame 0 -> 8 residues. V=[0,12) D=[12,15) J=[15,24).
    base = {"sequence": "A" * 24, "sequence_aa": "QVQLCWGQ", "junction_start": 0, "productive": True,
            "mutation_rate": 0.05, "n_indels": 0, "d_inverted": False,
            "v_call": "IGHV1*01,IGHV1*02", "d_call": "IGHD1*01", "j_call": "IGHJ1*01",
            "v_sequence_start": 0, "v_sequence_end": 12,
            "d_sequence_start": 12, "d_sequence_end": 15,
            "j_sequence_start": 15, "j_sequence_end": 24}
    base.update(over)
    return base


def test_aa_usable_requires_junction_and_aa():
    assert aa_usable(_rec())
    assert not aa_usable(_rec(junction_start=None))
    assert not aa_usable(_rec(sequence_aa=""))


def test_build_batch_aa_shapes_projected_coords_and_full_allele_multihot():
    ref = _ref()
    bi, tg = build_batch_aa([_rec()], ref, _Cfg())
    assert bi["tokenized_sequence"].shape == (1, 192)
    # AA-projected coords (nt //3): V owns residues 0..3, D 4..4, J 5..7
    assert tg["v_start"][0, 0] == 0 and tg["v_end"][0, 0] == 4
    assert tg["j_start"][0, 0] == 5 and tg["j_end"][0, 0] == 8
    # allele targets are the FULL nucleotide allele multi-hot; a co-listed pair -> two hots
    assert tg["v_allele"].shape == (1, 3)
    assert tg["v_allele"][0].tolist() == [1.0, 1.0, 0.0]        # IGHV1*01 + IGHV1*02
    assert tg["j_allele"][0].tolist() == [1.0, 0.0]
    assert tg["mutation_rate"][0, 0] == pytest.approx(0.05)
    assert tg["productive"][0, 0] == 1.0


def test_inverted_d_is_excluded_from_d_multihot():
    ref = _ref()
    _, tg = build_batch_aa([_rec(d_inverted=True)], ref, _Cfg())
    assert tg["d_allele"][0].tolist() == [0.0, 0.0]            # inverted-D not supervised (as DNA gym)


@pytest.mark.slow
def test_full_path_real_gym_real_model_real_loss():
    """gym record -> AA batch -> REAL AlignAIR(use_orientation=False) -> hierarchical_loss -> backward."""
    import GenAIRR as ga
    from alignair.core.config import AlignAIRConfig
    from alignair.core.losses import hierarchical_loss, make_logvars
    from alignair.core.model import AlignAIR
    from alignair.train.gym import Curriculum, build_experiment

    dc = ga.HUMAN_IGH_OGRDB
    ref = ReferenceSet.from_dataconfigs(dc)
    cfg = AlignAIRConfig.from_dataconfigs(dc, max_seq_length=192, vocab_size=23, use_orientation=False)
    assert cfg.v_allele_count == 198 and cfg.d_allele_count == 33 and cfg.j_allele_count == 7

    params = dict(Curriculum().params(0.6)); params.setdefault("invert_d_prob", 0.05)
    exp = build_experiment(dc, params, allow_curatable=True)
    recs = []
    for r in exp.stream_records(n=None, seed=0):
        if aa_usable(r):
            recs.append(r)
        if len(recs) >= 8:
            break

    bi, tg = build_batch_aa(recs, ref, cfg)
    model = AlignAIR(cfg)                          # the REAL model, orientation off
    logvars = make_logvars(cfg)
    out = model(bi)
    assert "orientation_logits" not in out         # RC path off
    assert out["v_allele"].shape == (8, 198)       # full nucleotide allele head
    total, parts = hierarchical_loss(out, tg, cfg, logvars)
    assert torch.isfinite(total)
    total.backward()                               # gradients flow through the whole thing
    assert any(p.grad is not None for p in model.parameters())
