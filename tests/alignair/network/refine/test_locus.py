"""One locus per read (``refine.locus``): a read takes the locus of its most probable same-locus V-J pair, and its V,
D and J are restricted to that locus; a read of a locus without D is called D-absent."""
from types import SimpleNamespace

import pytest
import torch

from alignair.core.d_presence import ABSENT, D_PRESENCE_LOGITS
from alignair.network.refine.locus import read_loci, restrict_to_locus

pytestmark = pytest.mark.component


def _bank(v_of_b=(3, 4), j_of_b=(2, 3)):
    """Two loci: A holds V 0-2, J 0-1 and D 0-1; B holds V ``v_of_b``, J ``j_of_b`` and no D."""
    def alleles(n, a, b):
        mask = torch.zeros(2, n, dtype=torch.bool)
        mask[0, list(a)], mask[1, list(b)] = True, True
        return mask
    return SimpleNamespace(locus_count=2, locus_alleles_v=alleles(5, (0, 1, 2), v_of_b),
                           locus_alleles_j=alleles(4, (0, 1), j_of_b), locus_alleles_d=alleles(2, (0, 1), ()),
                           locus_has_d=torch.tensor([True, False]))


BANK = _bank()


def _out(v, j, d_presence=None):
    v, j = torch.log_softmax(torch.tensor(v), -1), torch.log_softmax(torch.tensor(j), -1)
    d = torch.log_softmax(torch.tensor([[0.5, -0.5]] * len(v)), -1)
    return {"v_log_posterior": v, "j_log_posterior": j, "d_log_posterior": d,
            "v_allele": torch.exp(v - v.amax(-1, keepdim=True)), "j_allele": torch.exp(j - j.amax(-1, keepdim=True)),
            "d_allele": torch.exp(d - d.amax(-1, keepdim=True)),
            D_PRESENCE_LOGITS: torch.tensor(d_presence if d_presence is not None else [[0., 0., 0.]] * len(v))}


def test_the_read_takes_the_locus_of_its_best_same_locus_v_j_pair():
    # V slightly prefers locus A (0 against -1), J strongly prefers B (0 against -9): A scores 0 - 9, B scores -1 + 0
    out = _out(v=[[0., -5, -5, -1, -5]], j=[[-9., -9, 0, -5]])
    assert read_loci(out, BANK).tolist() == [1]


def test_a_locus_without_v_or_j_alleles_is_never_a_reads_locus():
    out = _out(v=[[-3., -3, -3, 0, 0]], j=[[-9., -9, 0, 0]])
    assert read_loci(out, _bank(v_of_b=())).tolist() == [0]           # B keeps its J alleles, no V
    assert read_loci(out, _bank(j_of_b=())).tolist() == [0]           # B keeps its V alleles, no J


def test_the_restricted_alleles_score_one_at_the_best_in_locus_allele():
    out = _out(v=[[-200., -300, -250, 0, -1]], j=[[-200., -201, 0, -1]])
    restrict_to_locus(out, torch.tensor([0]), BANK)
    assert out["v_allele"][0, 0] == 1 and (out["v_allele"][0, 3:] == 0).all()
    assert out["j_allele"][0, 0] == 1 and (out["j_allele"][0, 2:] == 0).all()
    assert torch.allclose(out["v_log_posterior"].exp().sum(-1), torch.ones(1))
    assert out["read_locus"].tolist() == [0]


def test_a_read_of_a_locus_without_d_is_called_d_absent():
    out = _out(v=[[-9., -9, -9, 0, -1]], j=[[-9., -9, 0, -1]], d_presence=[[-3., 0., 2.]])
    kept = out["d_log_posterior"].clone()
    restrict_to_locus(out, torch.tensor([1]), BANK)
    assert out[D_PRESENCE_LOGITS].argmax(-1).tolist() == [ABSENT]
    assert torch.equal(out["d_log_posterior"], kept)                   # its D posterior is left as it was


def test_a_read_of_a_locus_with_d_keeps_its_d_presence_and_its_locus_d_alleles():
    out = _out(v=[[0., -9, -9, -9, -9], [-9., -9, -9, 0, -9]], j=[[0., -9, -9, -9], [-9., -9, 0, -9]],
               d_presence=[[-3., 0., 2.], [-3., 0., 2.]])
    restrict_to_locus(out, read_loci(out, BANK), BANK)
    assert out["read_locus"].tolist() == [0, 1]
    assert out[D_PRESENCE_LOGITS].argmax(-1).tolist() == [2, ABSENT]


def test_a_shared_v_allele_lets_a_trd_read_take_trd():
    """TRA + TRD: a read whose V is TRAV29/DV5*01 (listed by both loci) and whose J is TRDJ1*01 is a TRD read."""
    import GenAIRR.data as gd
    import numpy as np
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRA_IMGT, gd.HUMAN_TCRD_IMGT)
    loci = ref.locus_names()
    bank = SimpleNamespace(**{f"locus_alleles_{g}": torch.from_numpy(np.stack([ref.locus_mask(l, g.upper())
                                                                               for l in loci])) for g in "vdj"})
    v, j = ref.gene("V"), ref.gene("J")
    v_lp = torch.full((1, len(v)), -20.)
    v_lp[0, v.index["TRAV29/DV5*01"]] = 0.
    j_lp = torch.full((1, len(j)), -20.)
    j_lp[0, j.index["TRDJ1*01"]] = 0.
    out = {"v_log_posterior": torch.log_softmax(v_lp, -1), "j_log_posterior": torch.log_softmax(j_lp, -1)}
    assert loci[int(read_loci(out, bank))] == "TRD"
