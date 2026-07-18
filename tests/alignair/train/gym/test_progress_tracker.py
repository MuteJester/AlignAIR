from alignair.train.gym.targeting import ProgressTracker


def test_regret_ranks_low_competence_cells_first():
    t = ProgressTracker(target=0.95, alp_weight=0.0, regret_weight=1.0)
    t.update({"clean": {"S": 0.97}, "heavy_shm": {"S": 0.50}})
    pr = t.priorities()
    assert pr["heavy_shm"] > pr["clean"]              # bigger regret -> higher priority
    assert t.top_cell() == "heavy_shm"
    assert abs(sum(pr.values()) - 1.0) < 1e-9         # normalized


def test_alp_rewards_moving_cells():
    t = ProgressTracker(target=0.95, alp_weight=1.0, regret_weight=0.0)
    t.update({"a": {"S": 0.40}, "b": {"S": 0.40}})
    t.update({"a": {"S": 0.60}, "b": {"S": 0.41}})    # a moved +0.20, b +0.01
    assert t.top_cell() == "a"


def test_empty_tracker():
    t = ProgressTracker()
    assert t.priorities() == {} and t.top_cell() is None
