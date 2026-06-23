import os
import sys

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("GenAIRR")

# repo root on path so the experiment script under scripts/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from scripts.exp_ramp_vs_factored import run_arm


def test_run_arm_returns_competence_field():
    field = run_arm("factored", steps=2, n_per_cell=4, batch_size=8, seed=0)
    assert "heavy_shm_fulllen" in field
    assert 0.0 <= field["heavy_shm_fulllen"]["S"] <= 1.0
