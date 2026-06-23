import json
from alignair.gym.control.state import GateStatus, AxisStat, GymState
from alignair.gym.control.reporter import StruggleReporter


def _state():
    return GymState(level=2, level_name="SHM Caverns", n_levels=10, step=5000,
                    gates=(GateStatus("v_call", 0.7, 0.9, "higher"),),
                    axes=(AxisStat("shm", (("0-0.05", 0.99, 50), ("0.15-1", 0.61, 40))),),
                    rooms_cleared=2, patience_used=1, patience_max=8,
                    best_level=2, headline="v_call")


def test_to_dict_is_json_serializable_and_complete():
    d = StruggleReporter("/tmp/gymtest").to_dict(_state())
    json.dumps(d)                                  # must not raise
    assert d["level"] == 2 and d["step"] == 5000
    assert d["blocking"] == ["v_call"]
    assert any(g["name"] == "v_call" for g in d["gates"])
    assert any(a["axis"] == "shm" for a in d["axes"])


def test_write_creates_files_and_appends_curve(tmp_path):
    rep = StruggleReporter(str(tmp_path))
    path = rep.write(_state())
    assert path.endswith(".json")
    assert (tmp_path / "climb_curve.jsonl").exists()
    md = rep.markdown(_state())
    assert "SHM Caverns" in md and "v_call" in md
