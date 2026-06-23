import pytest
import torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.evaluator import LatticeEvaluator


def test_eval_cell_returns_bounded_competence_with_ci():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    lat = FrozenLattice.standard(seed=0)
    ev = LatticeEvaluator(model, rs, lat, CompetenceMetric(), [gdata.HUMAN_IGK_OGRDB],
                          device="cpu")
    cell = next(c for c in lat.cells if c.name == "clean")
    out = ev.eval_cell(cell, n=16)
    assert out["n"] > 0
    assert 0.0 <= out["lo"] <= out["S"] <= out["hi"] <= 1.0
