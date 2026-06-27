import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.align import SeedPrefilter, get_aligner
from alignair.inference.wfa_caller import call_segment, SegmentCall

RS = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
SP = SeedPrefilter(RS, k=11)
AL = get_aligner()
VG = RS.gene("V")


def test_identity_segment_picks_its_allele_with_full_germline_span():
    gseq = VG.sequences[0]
    call = call_segment(gseq, "V", topk_idx=[1, 2, 0], reference_set=RS,
                        seed_prefilter=SP, aligner=AL)
    assert call.best_idx == 0
    assert call.set_idx[0] == 0
    assert call.germ_start == 0 and call.germ_end == len(gseq)


def test_true_allele_reachable_only_via_seed_pool_is_still_called():
    # retrieval top-k OMITS the true allele (idx 0); the k-mer seed prefilter must admit it so
    # WFA can still pick it — the swap-robustness guarantee at the caller level.
    gseq = VG.sequences[0]
    call = call_segment(gseq, "V", topk_idx=[1, 2, 3], reference_set=RS,
                        seed_prefilter=SP, aligner=AL)
    assert 0 in call.pool_idx          # admitted by the seed prefilter, not retrieval
    assert call.best_idx == 0


def test_short_segment_returns_none():
    assert call_segment("ACG", "V", topk_idx=[0], reference_set=RS,
                        seed_prefilter=SP, aligner=AL) is None


def test_set_is_ordered_and_genotype_restricted():
    gseq = VG.sequences[0]
    call = call_segment(gseq, "V", topk_idx=[0, 1, 2, 3, 4], reference_set=RS,
                        seed_prefilter=SP, aligner=AL, allowed={0, 1})
    assert set(call.pool_idx) <= {0, 1}            # allowed restricts the pool
    assert set(call.set_idx) <= {0, 1}
    assert call.set_idx == sorted(call.set_idx, key=lambda i: -call.scores[call.pool_idx.index(i)])
