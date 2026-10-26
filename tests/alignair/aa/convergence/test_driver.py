"""Tests for the pure selection-input derivation in the convergence driver
(``flatten_row``): a checkpoint metrics dict shaped exactly like the ``select.py`` §5 rule
expects (see that module's docstring), derived from an ``evaluator.evaluate_checkpoint``-shaped
output. This is the only piece of ``driver.py`` that is pure-function unit-testable here; the
end-to-end training->per-checkpoint-eval orchestration is exercised later by the smoke test
(the driver imports real GenAIRR/torch machinery that isn't worth mocking for this task).
"""
import pytest
from alignair.aa.convergence.driver import flatten_row

pytestmark = pytest.mark.component


def make_evaluator_output(**overrides) -> dict:
    """A fully-populated synthetic `evaluate_checkpoint`-shaped dict (mirrors the real function's
    documented return shape), with any field overridden per test."""
    base = {
        "clean_v_acc": 0.986, "clean_j_acc": 0.992,
        "v_macro": 0.95, "j_macro": 0.96,
        "d_allele_acc": 0.72, "d_allele_macro": 0.70, "d_gene_acc": 0.85,
        "by_stratum": {
            "clean": {"n": 100, "v_acc": 0.986},
            "aa_truncate_50_80": {"n": 40, "v_acc": 0.60},
            "short_janchor": {"n": 40, "v_acc": 0.55},
        },
        "freq_bins": {"v": {}, "j": {}},
        "boundary_mae": {"v": {"start": 0.5, "end": 0.6}, "j": {"start": 0.4, "end": 0.3}},
        "nll": {"v": 0.1, "j": 0.1, "d": 0.2},
        "brier": {"v": 0.05, "j": 0.05, "d": 0.1},
        "conformal": {"coverage": 0.95, "set_mean": 3.0, "set_p95": 6.0},
    }
    base.update(overrides)
    return base


def test_flatten_row_pulls_trunc_and_shortread_v_acc_from_by_stratum():
    out = make_evaluator_output()
    row = flatten_row("cont_step25000", 83000, out)
    assert row["trunc_v_acc"] == 0.60
    assert row["shortread_acc"] == 0.55


def test_flatten_row_computes_macro_minus_micro_as_v_macro_minus_clean_v_acc():
    out = make_evaluator_output(v_macro=0.95, clean_v_acc=0.986)
    row = flatten_row("incumbent", 58000, out)
    assert row["macro_minus_micro"] == out["v_macro"] - out["clean_v_acc"]
    assert abs(row["macro_minus_micro"] - (-0.036)) < 1e-9


def test_flatten_row_pulls_conformal_set_sizes():
    out = make_evaluator_output()
    row = flatten_row("incumbent", 58000, out)
    assert row["conf_set_mean"] == 3.0
    assert row["conf_set_p95"] == 6.0
    assert row["conformal"] == out["conformal"]


def test_flatten_row_passes_through_required_selection_keys():
    out = make_evaluator_output()
    row = flatten_row("incumbent", 58000, out)
    for key in ("clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
               "d_allele_macro", "d_gene_acc"):
        assert row[key] == out[key]
    assert row["name"] == "incumbent"
    assert row["step"] == 58000


def test_flatten_row_missing_stratum_degrades_to_none_not_a_crash():
    out = make_evaluator_output(by_stratum={"clean": {"n": 100, "v_acc": 0.986}})
    row = flatten_row("cont_step25000", 83000, out)
    assert row["trunc_v_acc"] is None
    assert row["shortread_acc"] is None


def test_flatten_row_missing_conformal_degrades_to_none():
    out = make_evaluator_output(conformal={"coverage": None, "set_mean": None, "set_p95": None,
                                           "note": "insufficient rows for conformal fit"})
    row = flatten_row("cont_step25000", 83000, out)
    assert row["conf_set_mean"] is None
    assert row["conf_set_p95"] is None


def test_flatten_row_missing_macro_or_micro_degrades_macro_minus_micro_to_none():
    out = make_evaluator_output(v_macro=None)
    row = flatten_row("cont_step25000", 83000, out)
    assert row["macro_minus_micro"] is None
