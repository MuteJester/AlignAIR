import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.align import SeedPrefilter, get_aligner
from alignair.inference.wfa_caller import call_segment


def test_classical_rescore_picks_true_allele_from_neural_pool():
    # given a neural top-k pool that CONTAINS the true allele, classical raw-base rescore
    # (call_segment over the pool, no seed additions) must pick it — even when the true allele
    # is NOT first in the pool (the sibling-rescue Layer 1 is meant to provide).
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    sp, al = SeedPrefilter(rs, k=11), get_aligner()
    vg = rs.gene("V")
    true_idx = 11
    seg = vg.sequences[true_idx]                              # read segment == true germline
    neural_pool = [3, 7, true_idx, 9, 2]                      # true allele present but not first
    call = call_segment(seg, "V", neural_pool, rs, sp, al, m_seed=0)
    assert call.best_idx == true_idx                          # best_idx is the GLOBAL allele index
