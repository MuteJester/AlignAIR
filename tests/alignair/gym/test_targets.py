import pytest
import numpy as np
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from GenAIRR import Experiment
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.targets import build_targets
from alignair.nn.region_head import REGION_INDEX
from alignair.nn.state_head import STATE_INDEX


def _record(seed=5):
    exp = (Experiment.on(gdata.HUMAN_IGH_OGRDB).recombine().mutate(model="s5f", rate=0.05)
           .end_loss_5prime(length=(0, 8)).compile())
    return next(exp.stream_records(n=1, seed=seed))


def test_targets_region_spans_match_coords():
    rec = _record()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(rec, rs, has_d=True)
    seq = rec["sequence"]
    assert t["tokens"].shape == (len(seq),)
    region = t["region_labels"]
    vs, ve = int(rec["v_sequence_start"]), int(rec["v_sequence_end"])
    # all positions inside the V span are labeled V
    assert (region[vs:ve] == REGION_INDEX["V"]).all()
    js, je = int(rec["j_sequence_start"]), int(rec["j_sequence_end"])
    assert (region[js:je] == REGION_INDEX["J"]).all()


def test_targets_coords_and_calls_and_scalars():
    rec = _record()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(rec, rs, has_d=True)
    assert t["germline"]["v"] == (int(rec["v_germline_start"]), int(rec["v_germline_end"]))
    assert t["inseq"]["v"] == (int(rec["v_sequence_start"]), int(rec["v_sequence_end"]))
    assert rec["v_call"].split(",")[0] in t["calls"]["V"]
    assert t["orientation_id"] == 0
    assert t["noise_count"] == rec["n_quality_errors"] + rec.get("n_pcr_errors", 0)
    assert abs(t["mutation_rate"] - rec["mutation_rate"]) < 1e-6
    assert t["indel_count"] == rec["n_indels"]
    assert t["productive"] in (0.0, 1.0)


def test_targets_state_marks_some_substitutions():
    rec = _record()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    t = build_targets(rec, rs, has_d=True)
    state = t["state_labels"]
    # with ~5% SHM there should be at least one substitution in the V span (no-indel records)
    if rec["n_indels"] == 0:
        vs, ve = int(rec["v_sequence_start"]), int(rec["v_sequence_end"])
        assert (state[vs:ve] == STATE_INDEX["substitution"]).sum() >= 1
