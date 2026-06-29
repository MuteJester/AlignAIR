import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.gym.gym import build_experiment
from alignair.gym.crop import crop_one_sided, anchor_c0


def _clean_record(seed=1):
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, dict(
        mutation_rate=0.0, productive_only=False, end_loss_5=(0, 0), end_loss_3=(0, 0),
        indel_count=(0, 0), seq_error_rate=0.0, ambiguous_count=(0, 0)))
    return list(exp.stream_records(n=1, seed=seed))[0]


def test_v_germline_anchor_drops_5prime_v_keeps_cdr3_and_j():
    r = _clean_record()
    g_start = 220
    c0 = anchor_c0(r, ("v_germline", g_start))
    cr = crop_one_sided(r, c0)
    assert cr["v_sequence_start"] == 0
    assert cr["v_germline_start"] >= g_start
    assert cr["j_sequence_start"] is not None
    assert (cr["v_sequence_end"] - cr["v_sequence_start"]) < (r["v_sequence_end"] - r["v_sequence_start"])
    assert len(cr["sequence"]) == len(str(r["sequence"])) - c0


def test_j_anchored_keeps_3prime_len():
    r = _clean_record(2)
    c0 = anchor_c0(r, ("j", 100))
    cr = crop_one_sided(r, c0)
    assert len(cr["sequence"]) == min(100, len(str(r["sequence"])))
    assert cr["j_sequence_end"] == len(cr["sequence"])


def test_fully_cut_gene_becomes_absent():
    r = _clean_record(3)
    c0 = int(r["j_sequence_start"])
    cr = crop_one_sided(r, c0)
    assert cr.get("v_sequence_start") is None


def test_build_targets_tolerates_absent_v():
    """A deep J-anchored crop can drop V entirely; the benchmark truth builder must not crash."""
    from alignair.gym.crop import crop_one_sided
    from alignair.gym.targets import build_targets
    from alignair.reference.reference_set import ReferenceSet
    import GenAIRR.data as gdata
    r = _clean_record(3)
    cr = crop_one_sided(r, int(r["j_sequence_start"]))      # cut everything 5' of J -> V (and D) gone
    assert cr.get("v_sequence_start") is None
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(cr, rs, has_d=rs.has_d)               # must not raise
    assert len(t["region_labels"]) == len(cr["sequence"])
    assert "J" in t["calls"]                                # J retained and labeled
