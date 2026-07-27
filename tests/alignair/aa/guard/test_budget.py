import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts/aa/applicability").resolve()))
import budget
from train_candidate import choose_production_seed

from alignair.aa.guard.model import GuardModel


def test_plan_counts_every_seed_replicate_and_model():
    e = budget.plan(["C1", "C2", "C3", "C4"], seed_replicates=3, loco_folds=4)
    assert e.main_seed_runs == 12 and e.loco_models == 16
    assert e.total_models == 12 + 4 + 16 + 1 + 4          # main-seeds + chosen + loco + C0 + C0-withheld


def test_choose_production_seed_deterministic_and_dev_only():
    res = {0: {"dev_loss": 0.5}, 1: {"dev_loss": 0.3}, 2: {"dev_loss": 0.3}}
    assert choose_production_seed(res) == 1               # min loss, tie -> lowest seed
    with pytest.raises(ValueError):
        choose_production_seed({0: {"dev_loss": 0.1, "fresh_selection_acc": 0.9}})


def test_record_and_total_roundtrip(tmp_path):
    p = tmp_path / "b.json"
    budget.record("C1.s0", 100, 0.5, 1000, path=p)
    budget.record("C1.s1", 200, 1.0, 1000, path=p)
    t = budget.total(path=p)
    assert t["wall_s"] == 300 and t["gpu_h"] == 1.5 and t["n_runs"] == 2


def test_guardmodel_forward_shapes_and_c1_needs_no_fusion():
    import torch
    m = GuardModel("C4", hmm_dim=6, width=64, depth=2, heads=4, n_sources=3)
    ids = torch.zeros(2, 10, dtype=torch.long); mask = torch.ones(2, 10, dtype=torch.bool)
    out = m(ids, mask, hmm_norm=torch.zeros(2, 6))
    assert out["applicability"].shape == (2, 1) and out["domain_type"].shape == (2, 8)
    assert out["source_logits"].shape == (2, 3) and out["energy"].shape == (2,)
    m1 = GuardModel("C1", hmm_dim=6, width=64, depth=2)
    o1 = m1(ids, mask)                                    # C1: no fusion required
    assert o1["source_logits"].shape == (2, 1)
