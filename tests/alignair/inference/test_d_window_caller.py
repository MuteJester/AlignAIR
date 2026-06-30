import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.inference.wfa_caller import call_d_in_window

_RC = str.maketrans("ACGTN", "TGCAN")
def _rc(s): return s.translate(_RC)[::-1]


def _d_gene():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    g = rs.gene("D")
    return g.names, g.sequences


def test_finds_forward_d_in_window_with_coords():
    names, seqs = _d_gene()
    d = seqs[5]; pad = "TTTTTTTTTT"
    res = call_d_in_window(pad + d + pad, names, seqs)
    assert res is not None
    assert names[res.idx] == names[5]
    assert res.inverted is False
    # window-relative coords locate the D after the pad
    assert res.t_start == len(pad)
    assert res.t_end == len(pad) + len(d)


def test_finds_inverted_d_in_window():
    names, seqs = _d_gene()
    d = seqs[5]; pad = "TTTTTTTTTT"
    res = call_d_in_window(pad + _rc(d) + pad, names, seqs)
    assert res is not None
    assert names[res.idx] == names[5]
    assert res.inverted is True


def test_no_d_below_min_score_returns_none():
    names, seqs = _d_gene()
    res = call_d_in_window("ACGT", names, seqs)        # too short to contain any D
    assert res is None
