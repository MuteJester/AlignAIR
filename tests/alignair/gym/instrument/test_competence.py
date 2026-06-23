from alignair.gym.instrument.competence import CompetenceMetric


def _perfect():
    return {"v_call_correct": 1, "d_call_correct": 1, "j_call_correct": 1,
            "coord_errs": [0.0, 0.0, 1.0], "region_acc": 1.0, "junction_exact": 1}


def test_perfect_read_scores_one():
    assert abs(CompetenceMetric().score(_perfect()) - 1.0) < 1e-9


def test_zero_read_scores_zero():
    rec = {"v_call_correct": 0, "d_call_correct": 0, "j_call_correct": 0,
           "coord_errs": [50.0, 50.0], "region_acc": 0.0, "junction_exact": 0}
    assert CompetenceMetric(coord_tol=2.0).score(rec) == 0.0


def test_coord_tolerance_counts_within_band():
    m = CompetenceMetric(weights={"coords": 1.0}, coord_tol=2.0)
    # 2 of 4 boundaries within 2nt -> coord sub-score 0.5
    rec = {"coord_errs": [0.0, 1.0, 5.0, 9.0]}
    assert abs(m.score(rec) - 0.5) < 1e-9


def test_missing_submetric_is_excluded_not_zero():
    m = CompetenceMetric()
    # junction omitted entirely: a perfect-everything-else read must score 1.0,
    # NOT be capped at (1 - junction_weight). Renormalize over present sub-metrics.
    rec = {"v_call_correct": 1, "d_call_correct": 1, "j_call_correct": 1,
           "coord_errs": [0.0, 0.0], "region_acc": 1.0}     # no junction_exact key
    assert abs(m.score(rec) - 1.0) < 1e-9
    # excluding D (unsupervised row) must not penalize: drop d_call_correct -> still 1.0
    rec.pop("d_call_correct")
    assert abs(m.score(rec) - 1.0) < 1e-9


def test_present_zero_submetric_still_counts():
    m = CompetenceMetric(weights={"v_call": 1.0, "junction": 1.0})
    # junction present and 0 -> counts (0.5), unlike when it is absent
    assert abs(m.score({"v_call_correct": 1, "junction_exact": 0}) - 0.5) < 1e-9
    assert m.score({"v_call_correct": 1}) == 1.0          # junction absent -> excluded


def test_aggregate_returns_ci():
    m = CompetenceMetric()
    out = m.aggregate([_perfect(), _perfect()], seed=1)
    assert out["n"] == 2 and out["S"] == 1.0 and out["lo"] == 1.0
