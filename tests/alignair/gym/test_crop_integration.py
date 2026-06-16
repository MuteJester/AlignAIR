"""Cropped records must produce valid target bundles and flow through the gym."""
import numpy as np
import pytest

from alignair.gym.crop import crop_record
from alignair.gym.targets import build_targets
from alignair.gym.curriculum import Curriculum

genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import AlignAIRGym, build_experiment


def _valid_bundle(b, has_d):
    n = len(b["tokens"])
    assert (b["region_labels"] >= 0).all() and len(b["region_labels"]) == n
    genes = ["v", "j"] + (["d"] if has_d else [])
    for g in genes:
        gs, ge = b["germline"][g]
        ss, ee = b["inseq"][g]
        assert 0 <= gs <= ge, (g, gs, ge)
        assert 0 <= ss <= ee <= n, (g, ss, ee, n)
        assert len(b["calls"][g.upper()]) >= 1


def test_cropped_genairr_record_builds_valid_targets():
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    params = Curriculum().params(1.0)
    exp = build_experiment(dc, params)
    n_cropped = 0
    for rec in exp.stream_records(n=40, seed=1):
        c = crop_record(rec, 60)
        if len(c["sequence"]) < len(rec["sequence"]):
            n_cropped += 1
            assert len(c["sequence"]) <= 120  # junction-centered window stays small
        b = build_targets(c, rs, has_d=True)
        _valid_bundle(b, has_d=True)
    assert n_cropped > 30  # most 60bp crops actually shrink a ~300bp read


def test_gym_yields_fragments_at_high_p():
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    gym = AlignAIRGym([dc], rs, n=60, seed=0)
    gym.set_progress(1.0)
    lengths = [len(b["tokens"]) for b in gym]
    lengths = np.array(lengths)
    # at p=1 a mix: some full reads (~300bp) and a meaningful fraction of short fragments
    assert (lengths < 120).mean() > 0.2, lengths
    assert (lengths > 200).mean() > 0.1, "full reads should still be present"
