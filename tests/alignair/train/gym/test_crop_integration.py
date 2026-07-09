"""Cropped records must produce valid target bundles and flow through the gym."""
import numpy as np
import pytest

from alignair.train.gym.crop import crop_record
from alignair.train.gym.targets import build_targets
from alignair.train.gym.curriculum import Curriculum

genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.train.gym.gym import AlignAIRGym, build_experiment


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


def test_training_mix_yields_genairr_short_reads():
    """The training stream shortens reads via GenAIRR end-loss (no post-hoc crop): every batch spans
    full-length rehearsal + a meaningful fraction of short amplicon/fragment reads, all with
    engine-correct coordinates."""
    import itertools
    from alignair.train.trainer import _mixed_stream
    dc = gdata.HUMAN_IGH_OGRDB
    recs = list(itertools.islice(_mixed_stream(dc, (0.3, 0.6, 0.9), 0.25, 0), 1200))
    lengths = np.array([len(str(r["sequence"])) for r in recs])
    assert (lengths < 160).mean() > 0.15, lengths          # meaningful short-read exposure
    assert (lengths > 300).mean() > 0.3, "full-length rehearsal must remain"
    # GenAIRR keeps every coordinate valid, even on the shortest reads
    for r in recs:
        L = len(str(r["sequence"]))
        for g in ("v", "d", "j"):
            ss = r.get(f"{g}_sequence_start")
            if ss is not None:
                assert 0 <= int(ss) <= int(r[f"{g}_sequence_end"]) <= L
