import pytest
import torch
pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.competence import CompetenceMetric
from alignair.gym.instrument.evaluator import LatticeEvaluator


def test_competence_field_covers_all_cells():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=64, n_layers=2, nhead=4, dim_feedforward=128)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)
    model = DNAlignAIR(cfg)
    lat = FrozenLattice.standard(seed=0)
    field = LatticeEvaluator(model, rs, lat, CompetenceMetric(), [gdata.HUMAN_IGK_OGRDB],
                             device="cpu").eval_all(n_per_cell=8)
    assert {c.name for c in lat.cells} <= set(field)
    assert all(0.0 <= v["S"] <= 1.0 for v in field.values())
