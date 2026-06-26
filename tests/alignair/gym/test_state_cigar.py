import numpy as np
import pytest

from alignair.gym.targets import _parse_cigar, _label_segment_states
from alignair.nn.heads.state import STATE_INDEX


def test_parse_cigar():
    assert _parse_cigar("41M13D") == [(41, "M"), (13, "D")]
    assert _parse_cigar("10M2I5M") == [(10, "M"), (2, "I"), (5, "M")]
    assert _parse_cigar("") == [] and _parse_cigar(None) == []


def test_match_and_substitution():
    seq = "ACGTACGT"
    gref = "ACCTACGT"  # differs at index 2 (G vs C)
    state = np.zeros(len(seq), dtype=np.int64)
    _label_segment_states(state, seq, 0, gref, 0, "8M")
    assert state[2] == STATE_INDEX["substitution"]
    assert (np.delete(state, 2) == STATE_INDEX["germline"]).all()


def test_insertion_consumes_observed_only():
    # seq has 2 inserted bases not in germline: 4M 2I 4M
    seq = "ACGTXXACGT"
    gref = "ACGTACGT"  # 8 germline bases
    state = np.zeros(len(seq), dtype=np.int64)
    ndel = _label_segment_states(state, seq, 0, gref, 0, "4M2I4M")
    assert state[4] == STATE_INDEX["insertion"] and state[5] == STATE_INDEX["insertion"]
    assert ndel == 0
    # the trailing 4M realign to germline[4:8] -> all germline
    assert (state[6:] == STATE_INDEX["germline"]).all()


def test_deletion_tags_next_observed_and_counts():
    # 4M 2D 4M: germline has 2 extra bases deleted from the read
    seq = "ACGTACGT"       # 8 observed
    gref = "ACGTGGACGT"    # 10 germline (GG deleted)
    state = np.zeros(len(seq), dtype=np.int64)
    ndel = _label_segment_states(state, seq, 0, gref, 0, "4M2D4M")
    assert ndel == 2
    # observed index 4 (first base after the deletion) is tagged deletion
    assert state[4] == STATE_INDEX["deletion"]


def test_segment_offset_respected():
    seq = "NNNNACGT"
    gref = "ACCT"
    state = np.zeros(len(seq), dtype=np.int64)
    _label_segment_states(state, seq, 4, gref, 0, "4M")  # segment starts at obs index 4
    assert state[6] == STATE_INDEX["substitution"]  # obs[6]=G vs gref[2]=C
    assert (state[:4] == 0).all()


def test_real_genairr_record_has_indel_states():
    pytest.importorskip("GenAIRR")
    import GenAIRR.data as gdata
    from alignair.reference.reference_set import ReferenceSet
    from alignair.gym.gym import build_experiment
    from alignair.gym.curriculum import Curriculum
    from alignair.gym.targets import build_targets

    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    params = Curriculum().params(0.6)
    params["indel_count"] = (3, 6)  # force indels
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, params)
    saw_ins = saw_del = False
    for rec in exp.stream_records(n=300, seed=2):
        if int(rec.get("n_indels", 0) or 0) == 0:
            continue
        b = build_targets(rec, rs, has_d=True)
        st = b["state_labels"]
        saw_ins |= bool((st == STATE_INDEX["insertion"]).any())
        saw_del |= bool((st == STATE_INDEX["deletion"]).any())
        if saw_ins and saw_del:
            break
    assert saw_ins, "no insertion states produced from indel records"
    assert saw_del, "no deletion states produced from indel records"
