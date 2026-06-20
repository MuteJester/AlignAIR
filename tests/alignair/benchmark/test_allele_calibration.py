import math

from alignair.benchmark.evaluation.allele_calibration import (
    multipos_nll, fit_temperature, sweep_epsilon, fit_calibration, _set_stats)


def test_lr_band_set_membership():
    # keep iff (s_top - s_c)/T <= eps. s=[0,-0.5,-3], T=1, eps=1 -> keep {0,1}, drop 2.
    rows = [([0.0, -0.5, -3.0], [0])]
    size, rec, f1 = _set_stats(rows, T=1.0, eps=1.0)
    assert size == 2.0 and rec == 1.0
    size0, _, _ = _set_stats(rows, T=1.0, eps=0.0)
    assert size0 == 1.0                                   # only the top survives eps=0


def test_fit_temperature_prefers_sharp_when_truth_is_top():
    # truth is consistently the clear top -> low T (sharper) minimizes NLL
    rows = [([3.0, 0.0, -1.0], [0]) for _ in range(20)]
    T = fit_temperature(rows)
    assert T <= 0.5
    assert multipos_nll(rows, T) < multipos_nll(rows, 5.0)


def test_fit_temperature_prefers_soft_when_truth_not_top():
    # truth is the 2nd-highest candidate (model top-1 is wrong) -> a sharp T over-penalizes;
    # a softer T spreads posterior mass and fits better.
    rows = [([0.5, 0.0], [1]) for _ in range(20)]            # truth idx 1, but idx 0 scores higher
    assert multipos_nll(rows, 2.0) < multipos_nll(rows, 0.1)
    assert fit_temperature(rows) >= 1.0


def test_sweep_epsilon_smallest_set_meeting_recall():
    # two-allele truth that needs a wider band to fully recover (legacy recall objective)
    rows = [([0.0, -0.5, -5.0], [0, 1]) for _ in range(10)]
    sw = sweep_epsilon(rows, T=1.0, objective="recall", target_recall=0.95)
    assert sw["hit_target"] is True
    assert sw["set_recall"] >= 0.95
    assert sw["epsilon"] >= 0.5                           # must include the -0.5 positive
    # a stricter band can't reach full recall on the 2-allele truth
    strict, _, _ = _set_stats(rows, T=1.0, eps=0.2)
    assert strict == 1.0                                  # only top kept -> recall 0.5


def test_sweep_epsilon_fallback_when_target_unreachable():
    # truth allele is absent from candidate scores' top region (here truth idx 2 is far)
    rows = [([0.0, -0.1, -9.0], [2]) for _ in range(5)]
    sw = sweep_epsilon(rows, T=1.0, objective="recall", target_recall=0.99,
                       grid=[0.0, 0.5, 1.0])
    assert sw["hit_target"] is False                      # never reaches target on this grid


def test_f1_objective_avoids_set_blowup_on_easy_data():
    # single-positive, clearly-top truth: the F1 objective should pick a TIGHT set
    # (tiny epsilon), whereas a recall-only objective would widen needlessly.
    rows = [([2.0, -0.3, -0.5, -0.8], [0]) for _ in range(20)]
    f1 = sweep_epsilon(rows, T=1.0, objective="f1", min_recall=0.8)
    rec = sweep_epsilon(rows, T=1.0, objective="recall", target_recall=0.95)
    assert f1["set_recall"] >= 0.8
    assert f1["mean_set_size"] <= rec["mean_set_size"]    # F1 keeps it tight
    assert f1["set_f1"] >= 0.9                            # near-perfect F1 on clean tops


def test_fit_calibration_shape():
    per_gene = {"V": [([2.0, 0.0], [0])] * 8, "J": [([1.0, 0.5, 0.0], [0, 1])] * 8}
    cal = fit_calibration(per_gene, target_recall=0.95)
    assert set(cal) == {"V", "J"}
    for G in ("V", "J"):
        assert {"temperature", "epsilon", "mean_set_size", "set_recall", "n"} <= set(cal[G])
        assert cal[G]["temperature"] > 0
