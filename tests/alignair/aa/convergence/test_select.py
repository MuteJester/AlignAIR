"""Tests for the FROZEN blinded selection rule (§5 of
Notes and Plans/docs/superpowers/specs/2026-07-22-aa-convergence-study-design.md).

These synthetic-metric-dict tests pin down the exact gate thresholds, the weakness-objective
count, both tie-break stages, and the null (incumbent-wins) outcome -- this rule is scientifically
load-bearing and MUST NOT be re-tuned after the fact, so every clause of §5 gets a dedicated
assertion here.
"""
import pytest
from alignair.aa.convergence.select import select_checkpoint

pytestmark = pytest.mark.component


def make_metrics(name, step, **overrides):
    """A fully-valid, gate-passing, no-improvement metrics dict (mirrors incumbent scale) with
    any field overridden per test."""
    base = {
        "name": name, "step": step,
        "clean_v_acc": 0.986, "clean_j_acc": 0.992,
        "v_macro": 0.95, "j_macro": 0.95,
        "conformal": {"coverage": 0.95},
        "d_allele_macro": 0.70, "d_gene_acc": 0.85,
        "trunc_v_acc": 0.60, "shortread_acc": 0.55,
        "macro_minus_micro": -0.05,
        "conf_set_mean": 3.0, "conf_set_p95": 6.0,
    }
    base.update(overrides)
    return base


def pc_by_name(result, name):
    return next(pc for pc in result["per_candidate"] if pc["name"] == name)


INCUMBENT = make_metrics("incumbent", 0)


# ---------------------------------------------------------------------------
# (a) a candidate failing a gate is rejected
# ---------------------------------------------------------------------------

def test_gate_failure_rejects_candidate_even_if_it_would_improve_weaknesses():
    cand = make_metrics("cand_fails_clean_v", 25_000, clean_v_acc=0.97, d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_fails_clean_v")
    assert pc["gates_passed"] is False
    assert any("clean_v_acc" in f for f in pc["gate_failures"])
    assert pc["objective_count"] == 0
    assert result["winner"] == "incumbent"


def test_gate_rejects_v_or_j_macro_regression_over_half_point():
    cand = make_metrics("cand_macro_regress", 25_000, v_macro=0.95 - 0.006, d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_macro_regress")
    assert pc["gates_passed"] is False
    assert any("v_macro" in f for f in pc["gate_failures"])
    assert result["winner"] == "incumbent"


def test_gate_rejects_low_conformal_coverage():
    cand = make_metrics("cand_low_coverage", 25_000, conformal={"coverage": 0.93},
                        d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_low_coverage")
    assert pc["gates_passed"] is False
    assert any("coverage" in f for f in pc["gate_failures"])
    assert result["winner"] == "incumbent"


def test_gate_boundary_values_pass_inclusive():
    """Gates are '>=' / 'no regression > 0.5pt', so exact-boundary values PASS."""
    cand = make_metrics(
        "cand_at_boundary", 25_000,
        clean_v_acc=0.980, clean_j_acc=0.988,
        v_macro=0.95 - 0.005, j_macro=0.95 - 0.005,
        conformal={"coverage": 0.94},
        d_allele_macro=0.90,
    )
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_at_boundary")
    assert pc["gates_passed"] is True
    assert pc["gate_failures"] == []


# ---------------------------------------------------------------------------
# (b) a gate-passer improving 2 weaknesses beats one improving 1
# ---------------------------------------------------------------------------

def test_more_weaknesses_improved_wins():
    cand_one = make_metrics("cand_one_improve", 25_000, d_allele_macro=0.75)          # +0.05, 1 metric
    cand_two = make_metrics("cand_two_improve", 75_000, d_allele_macro=0.75,
                            trunc_v_acc=0.65)                                        # +0.05, +0.05: 2 metrics
    result = select_checkpoint(INCUMBENT, [cand_one, cand_two])

    pc_one = pc_by_name(result, "cand_one_improve")
    pc_two = pc_by_name(result, "cand_two_improve")
    assert pc_one["objective_count"] == 1
    assert set(pc_one["weaknesses_improved"]) == {"d_allele_macro"}
    assert pc_two["objective_count"] == 2
    assert set(pc_two["weaknesses_improved"]) == {"d_allele_macro", "trunc_v_acc"}
    assert result["winner"] == "cand_two_improve"


def test_accuracy_improvement_threshold_is_exclusive_below_one_point():
    # +0.009 must NOT count; +0.01 exactly must count (>= threshold).
    cand_under = make_metrics("cand_under_threshold", 25_000, d_gene_acc=0.85 + 0.009)
    cand_at = make_metrics("cand_at_threshold", 25_000, d_gene_acc=0.85 + 0.01)

    result_under = select_checkpoint(INCUMBENT, [cand_under])
    result_at = select_checkpoint(INCUMBENT, [cand_at])

    assert pc_by_name(result_under, "cand_under_threshold")["objective_count"] == 0
    assert result_under["winner"] == "incumbent"

    assert pc_by_name(result_at, "cand_at_threshold")["objective_count"] == 1
    assert result_at["winner"] == "cand_at_threshold"


# ---------------------------------------------------------------------------
# (c) tie in objective_count broken by earliest step (and gain-tiebreak precedes step)
# ---------------------------------------------------------------------------

def test_tie_in_objective_count_broken_by_earliest_step():
    cand_a = make_metrics("cand_a_later", 150_000, d_allele_macro=0.75)   # +0.05, objective_count 1
    cand_b = make_metrics("cand_b_earlier", 75_000, d_allele_macro=0.75)  # same gain, earlier step
    result = select_checkpoint(INCUMBENT, [cand_a, cand_b])

    assert pc_by_name(result, "cand_a_later")["objective_count"] == 1
    assert pc_by_name(result, "cand_b_earlier")["objective_count"] == 1
    assert result["winner"] == "cand_b_earlier"


def test_tiebreak_prefers_larger_combined_gain_over_earlier_step():
    """§5 tie-break order is (1) combined D-allele-macro + trunc-V gain, THEN (2) earliest step --
    a later checkpoint with the bigger gain must still win over an earlier one with a smaller gain."""
    cand_high_gain_late = make_metrics("cand_high_gain_late", 300_000, d_allele_macro=0.85)  # +0.15
    cand_low_gain_early = make_metrics("cand_low_gain_early", 25_000, d_allele_macro=0.72)   # +0.02
    result = select_checkpoint(INCUMBENT, [cand_high_gain_late, cand_low_gain_early])

    assert pc_by_name(result, "cand_high_gain_late")["objective_count"] == 1
    assert pc_by_name(result, "cand_low_gain_early")["objective_count"] == 1
    assert result["winner"] == "cand_high_gain_late"


# ---------------------------------------------------------------------------
# (d) no candidate improves any weakness -> winner is "incumbent" (honest null result)
# ---------------------------------------------------------------------------

def test_null_result_when_no_gate_passer_improves_any_weakness():
    cand_no_improve = make_metrics("cand_no_improve", 25_000)  # identical to incumbent
    result = select_checkpoint(INCUMBENT, [cand_no_improve])

    pc = pc_by_name(result, "cand_no_improve")
    assert pc["gates_passed"] is True
    assert pc["objective_count"] == 0
    assert pc["weaknesses_improved"] == []
    assert result["winner"] == "incumbent"
    assert "did not help" in result["rationale"].lower()


def test_null_result_when_all_candidates_fail_gates():
    cand = make_metrics("cand_fails_everything", 25_000, clean_v_acc=0.5, clean_j_acc=0.5)
    result = select_checkpoint(INCUMBENT, [cand])
    assert result["winner"] == "incumbent"


# ---------------------------------------------------------------------------
# (e) set-size improvement counts only at >=5% relative reduction
# ---------------------------------------------------------------------------

def test_set_size_improvement_requires_five_percent_relative_reduction():
    cand_exact = make_metrics("cand_exact_5pct", 25_000, conf_set_mean=3.0 * 0.95)   # exactly 5%: counts
    cand_under = make_metrics("cand_under_5pct", 25_000, conf_set_mean=2.90)          # ~3.3%: does not count

    result_exact = select_checkpoint(INCUMBENT, [cand_exact])
    result_under = select_checkpoint(INCUMBENT, [cand_under])

    pc_exact = pc_by_name(result_exact, "cand_exact_5pct")
    assert pc_exact["objective_count"] == 1
    assert "conf_set_mean" in pc_exact["weaknesses_improved"]
    assert result_exact["winner"] == "cand_exact_5pct"

    pc_under = pc_by_name(result_under, "cand_under_5pct")
    assert pc_under["objective_count"] == 0
    assert result_under["winner"] == "incumbent"


def test_set_size_p95_improvement_also_recognized():
    cand = make_metrics("cand_p95_improves", 25_000, conf_set_p95=6.0 * 0.90)  # 10% reduction
    result = select_checkpoint(INCUMBENT, [cand])
    pc = pc_by_name(result, "cand_p95_improves")
    assert "conf_set_p95" in pc["weaknesses_improved"]
    assert pc["objective_count"] == 1
    assert result["winner"] == "cand_p95_improves"


# ---------------------------------------------------------------------------
# (f) isolated gate failures on non-macro-regression gates (clean_j, coverage)
# ---------------------------------------------------------------------------

def test_gate_failure_clean_j_acc_below_threshold():
    """A candidate failing clean_j_acc gate alone (0.987 < 0.988) while all other gates pass."""
    cand = make_metrics("cand_clean_j_fail", 25_000, clean_j_acc=0.987, d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_clean_j_fail")
    assert pc["gates_passed"] is False
    assert any("clean_j_acc" in f for f in pc["gate_failures"])
    assert pc["objective_count"] == 0
    assert result["winner"] == "incumbent"


def test_gate_failure_j_macro_regression_over_threshold():
    """A candidate failing j_macro gate alone (regression > 0.5pt) while all other gates pass."""
    cand = make_metrics("cand_j_macro_regress", 25_000, j_macro=0.95 - 0.006, d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_j_macro_regress")
    assert pc["gates_passed"] is False
    assert any("j_macro" in f for f in pc["gate_failures"])
    assert pc["objective_count"] == 0
    assert result["winner"] == "incumbent"


def test_gate_failure_conformal_coverage_missing():
    """A candidate with conformal.coverage missing (None) is treated as gate failure (fail-closed)."""
    cand = make_metrics("cand_coverage_missing", 25_000, conformal={}, d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_coverage_missing")
    assert pc["gates_passed"] is False
    assert any("coverage" in f for f in pc["gate_failures"])
    assert pc["objective_count"] == 0
    assert result["winner"] == "incumbent"


def test_gate_failure_conformal_dict_missing_entirely():
    """A candidate with conformal field missing entirely (None) is treated as gate failure."""
    cand = make_metrics("cand_no_conformal", 25_000, conformal=None, d_allele_macro=0.90)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_no_conformal")
    assert pc["gates_passed"] is False
    assert any("coverage" in f for f in pc["gate_failures"])
    assert pc["objective_count"] == 0
    assert result["winner"] == "incumbent"


# ---------------------------------------------------------------------------
# (g) single-weakness improvements: shortread_acc and macro_minus_micro
# ---------------------------------------------------------------------------

def test_shortread_acc_improvement_only_wins():
    """A candidate improving only shortread_acc (>= incumbent + 0.01) should have objective_count==1
    and beat incumbent."""
    cand = make_metrics("cand_shortread_only", 25_000, shortread_acc=0.55 + 0.01)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_shortread_only")
    assert pc["gates_passed"] is True
    assert pc["objective_count"] == 1
    assert set(pc["weaknesses_improved"]) == {"shortread_acc"}
    assert result["winner"] == "cand_shortread_only"


def test_macro_minus_micro_improvement_only_wins():
    """A candidate improving only macro_minus_micro (>= incumbent + 0.01) should have
    objective_count==1 and beat incumbent."""
    cand = make_metrics("cand_macro_minus_micro_only", 25_000, macro_minus_micro=-0.05 + 0.01)
    result = select_checkpoint(INCUMBENT, [cand])

    pc = pc_by_name(result, "cand_macro_minus_micro_only")
    assert pc["gates_passed"] is True
    assert pc["objective_count"] == 1
    assert set(pc["weaknesses_improved"]) == {"macro_minus_micro"}
    assert result["winner"] == "cand_macro_minus_micro_only"


# ---------------------------------------------------------------------------
# Rationale / per_candidate shape sanity
# ---------------------------------------------------------------------------

def test_result_shape_has_required_top_level_keys():
    cand = make_metrics("cand", 25_000, d_allele_macro=0.80)
    result = select_checkpoint(INCUMBENT, [cand])
    assert set(result.keys()) >= {"winner", "rationale", "per_candidate"}
    assert isinstance(result["rationale"], str) and result["rationale"]
    pc = result["per_candidate"][0]
    for key in ("name", "gates_passed", "gate_failures", "weaknesses_improved", "objective_count"):
        assert key in pc
