from alignair.train.gym.control.state import AxisStat
from alignair.train.gym.control.exam import bucket_axis, axis_breakdown


def _recs():
    # two easy (low SHM, correct) + two hard (high SHM, wrong)
    return [
        {"mutation_rate": 0.01, "indel_count": 0, "noise_count": 0, "length": 300, "correct": 1.0},
        {"mutation_rate": 0.02, "indel_count": 0, "noise_count": 0, "length": 300, "correct": 1.0},
        {"mutation_rate": 0.20, "indel_count": 3, "noise_count": 1, "length": 80, "correct": 0.0},
        {"mutation_rate": 0.25, "indel_count": 4, "noise_count": 2, "length": 60, "correct": 0.0},
    ]


def test_bucket_axis_splits_and_means():
    st = bucket_axis(_recs(), axis="mutation_rate", edges=[0.0, 0.05, 1.0], metric_key="correct")
    assert isinstance(st, AxisStat) and st.axis == "mutation_rate"
    # bin 0 (<=0.05): both correct -> 1.0, n=2 ; bin 1 (>0.05): both wrong -> 0.0, n=2
    labels = {b[0]: (b[1], b[2]) for b in st.bins}
    assert any(v == (1.0, 2) for v in labels.values())
    assert any(v == (0.0, 2) for v in labels.values())


def test_axis_breakdown_returns_standard_axes():
    axes = axis_breakdown(_recs())
    names = {a.axis for a in axes}
    assert {"shm", "indel", "noise", "length"} <= names
