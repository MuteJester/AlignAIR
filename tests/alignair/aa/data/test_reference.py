import pytest
from alignair.aa.reference import (
    ANCHOR_AA, AAGermline, translate_germline, build_aa_germlines,
)

pytestmark = pytest.mark.integration


def test_translate_germline_passes_anchor_qc():
    # frame 0, Cys (TGT) codon at nt index 6 -> anchor 6, residue index 2.
    seq = "ATGAAATGTGGG"          # M K C G
    germ, reason = translate_germline("IGHV-test*01", seq, anchor=6, gene="V")
    assert reason is None
    assert isinstance(germ, AAGermline)
    assert germ.aa_seq == "MKCG"
    assert germ.frame == 0
    assert germ.anchor_aa == 2
    assert germ.aa_seq[germ.anchor_aa] == ANCHOR_AA["V"]


def test_translate_germline_quarantines_wrong_anchor_residue():
    # anchor points at GGG (Gly), not Cys -> quarantined.
    seq = "ATGAAAGGGTGG"          # M K G W
    germ, reason = translate_germline("IGHV-bad*01", seq, anchor=6, gene="V")
    assert germ is None
    assert reason is not None and "C" in reason


def test_translate_germline_respects_nonzero_frame():
    # frame 2 (anchor 2 -> 2 % 3 == 2). Translate from base 2: TGT GGG AAA -> C G K ; anchor residue 0.
    seq2 = "AATGTGGGAAA"          # bases[2:] = TGTGGGAAA -> C G K ; anchor 2 -> residue 0
    germ2, reason2 = translate_germline("IGHV-f2b*01", seq2, anchor=2, gene="V")
    assert reason2 is None and germ2.frame == 2 and germ2.anchor_aa == 0
    assert germ2.aa_seq[0] == "C"


@pytest.mark.slow
def test_build_aa_germlines_real_igh_all_v_and_j_pass_anchor_qc():
    import GenAIRR as ga
    dc = ga.HUMAN_IGH_OGRDB
    gset = build_aa_germlines(dc, genes=("V", "J"))
    # every V allele translation carries Cys at its anchor residue; every J carries Trp.
    assert len(gset.genes["V"]) >= 190          # 198 V alleles, near-all pass
    assert len(gset.genes["J"]) == 7            # all 7 J pass
    for germ in gset.genes["V"].values():
        assert germ.aa_seq[germ.anchor_aa] == "C"
    for germ in gset.genes["J"].values():
        assert germ.aa_seq[germ.anchor_aa] == "W"
    # quarantine, if any, is small and reported with a reason string.
    assert all(isinstance(r, str) for r in gset.quarantined["V"].values())
