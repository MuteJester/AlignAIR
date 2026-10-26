import json

import pytest

from alignair.aa.guard.training import budget

pytestmark = pytest.mark.component


def test_plan_counts_every_seed_replicate_and_model():
    e = budget.plan(["C1", "C2", "C3", "C4"], seed_replicates=3, loco_folds=4)
    assert e.main_seed_runs == 12 and e.loco_models == 16
    assert e.total_models == 12 + 4 + 16 + 1 + 4          # main-seeds + chosen + loco + C0 + C0-withheld


def test_record_and_total_roundtrip(tmp_path):
    p = tmp_path / "b.json"
    budget.record("C1.s0", 100, 0.5, 1000, path=p)
    budget.record("C1.s1", 200, 1.0, 1000, path=str(p))
    t = budget.total(path=p)
    assert t["wall_s"] == 300 and t["gpu_h"] == 1.5 and t["n_runs"] == 2


def test_resumed_run_replaces_its_totals_and_preserves_other_records(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({
        "study": "guard", "runs": {"C1.s0": {"wall_s": 10, "gpu_h": 0.1, "params": 100}},
    }))
    budget.record("C2.s0", 20, 0.2, 200, path=path)
    budget.record("C1.s0", 30, 0.3, 100, path=path)
    assert budget.total(path=path) == {"wall_s": 50, "gpu_h": 0.5, "n_runs": 2}
    assert json.loads(path.read_text())["study"] == "guard"


def test_missing_ledger_has_zero_totals_without_creating_a_file(tmp_path):
    path = tmp_path / "missing.json"
    assert budget.total(path=path) == {"wall_s": 0, "gpu_h": 0, "n_runs": 0}
    assert not path.exists()


def test_malformed_ledger_is_not_overwritten(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text("incomplete ledger")
    with pytest.raises(json.JSONDecodeError):
        budget.record("C1.s0", 1, 0.1, 100, path=path)
    assert path.read_text() == "incomplete ledger"
