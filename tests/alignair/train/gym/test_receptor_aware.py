"""build_experiment must generate valid samples for BOTH BCR (SHM) and TCR (no SHM)."""

import pytest
import GenAIRR.data as gd

from alignair.train.gym.experiment import build_experiment
from alignair.train.gym.curriculum import Curriculum

pytestmark = pytest.mark.integration


def _params():
    return {**Curriculum().params(0.3), "invert_d_prob": 0.05}


def test_tcr_generates_without_shm_error():
    # TCRs do not undergo somatic hypermutation; the builder must skip mutate() for TCR
    # and still produce a valid recombined record.
    exp = build_experiment(gd.HUMAN_TCRB_IMGT, _params(), allow_curatable=True)
    rec = next(iter(exp.stream_records(n=1, seed=0)))
    assert rec["sequence"] and rec["v_call"] and rec["j_call"]


def test_bcr_still_generates_with_shm():
    exp = build_experiment(gd.HUMAN_IGH_OGRDB, _params(), allow_curatable=True)
    rec = next(iter(exp.stream_records(n=1, seed=0)))
    assert rec["sequence"] and rec["v_call"] and rec["j_call"]
