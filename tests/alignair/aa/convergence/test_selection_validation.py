"""Frozen selection rejects malformed evidence before evaluating any candidate."""
from copy import deepcopy
from types import MappingProxyType

import pytest

from alignair.aa.convergence import select as module
from alignair.errors import InputError, NumericalError, current_error_context

pytestmark = pytest.mark.component


def row(name="candidate", step=25000):
    return {"name": name, "step": step, "clean_v_acc": .986, "clean_j_acc": .992,
            "v_macro": .95, "j_macro": .95, "conformal": {"coverage": .95},
            "d_allele_macro": .8, "d_gene_acc": .85, "trunc_v_acc": .6,
            "shortread_acc": .55, "macro_minus_micro": -.05,
            "conf_set_mean": 3., "conf_set_p95": 6.}


def incumbent():
    result = row("incumbent", 0)
    result["d_allele_macro"] = .7
    return result


@pytest.mark.parametrize("field", ["clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
    "d_allele_macro", "d_gene_acc", "trunc_v_acc", "shortread_acc", "macro_minus_micro",
    "conf_set_mean", "conf_set_p95", "conformal.coverage"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_candidate_evidence_cannot_pass_gates_or_change_ranking(field, value):
    candidate = row()
    if field == "conformal.coverage": candidate["conformal"]["coverage"] = value
    else: candidate[field] = value
    with pytest.raises(NumericalError) as caught:
        module.select_checkpoint(incumbent(), [candidate])
    assert caught.value.context == {"stage": "convergence_selection", "role": "candidate",
                                   "candidate_index": 0, "field": field}


@pytest.mark.parametrize("field,value", [("v_macro", None), ("j_macro", True), ("clean_v_acc", 1.1),
    ("d_gene_acc", -.1), ("macro_minus_micro", -1.1), ("macro_minus_micro", 1.1),
    ("conf_set_mean", -1), ("conf_set_p95", ".5"), ("d_allele_macro", 2 ** 10000)])
def test_malformed_metrics_are_rejected_with_field_context(field, value):
    candidate = row(); candidate[field] = value
    with pytest.raises((InputError, NumericalError)) as caught:
        module.select_checkpoint(incumbent(), [candidate])
    assert caught.value.context["field"] == field


@pytest.mark.parametrize("source", [None, [], {"v_macro": float("nan"), "j_macro": .95},
                                   {"v_macro": .95}, {"v_macro": .95, "j_macro": .95, "conf_set_mean": float("inf")}])
def test_incumbent_evidence_is_checked_even_when_there_are_no_candidates(source):
    with pytest.raises((InputError, NumericalError)) as caught:
        module.select_checkpoint(source, [])
    assert caught.value.context["role"] == "incumbent"


@pytest.mark.parametrize("source", [None, {}, "candidate", [None], [{"name": "x"}]])
def test_malformed_candidate_collections_and_rows_have_input_errors(source):
    with pytest.raises(InputError):
        module.select_checkpoint(incumbent(), source)


@pytest.mark.parametrize("field,value", [("name", None), ("name", " "), ("name", "incumbent"),
                                        ("step", None), ("step", -1), ("step", True), ("step", 1.5)])
def test_candidates_require_unambiguous_names_and_orderable_checkpoint_steps(field, value):
    candidate = row(); candidate[field] = value
    with pytest.raises(InputError) as caught:
        module.select_checkpoint(incumbent(), [candidate])
    assert caught.value.context["field"] == field


def test_duplicate_names_are_rejected_before_any_gate_computation(monkeypatch):
    monkeypatch.setattr(module, "_check_gates", lambda *_: pytest.fail("selection began before preflight completed"))
    with pytest.raises(InputError) as caught:
        module.select_checkpoint(incumbent(), [row("same", 1), row("same", 2)])
    assert caught.value.context["candidate_index"] == 1
    assert caught.value.context["field"] == "name"


def test_bad_late_candidate_cannot_leave_a_partial_decision(monkeypatch):
    later = row("later", 2); later["v_macro"] = float("nan")
    monkeypatch.setattr(module, "_check_gates", lambda *_: pytest.fail("selection began before preflight completed"))
    with pytest.raises(NumericalError):
        module.select_checkpoint(incumbent(), iter([row("first", 1), later]))
    assert current_error_context() == {}


@pytest.mark.parametrize("conformal", [[], "0.95", True, {"coverage": True}, {"coverage": 1.1}])
def test_invalid_conformal_containers_or_values_are_not_treated_as_missing(conformal):
    candidate = row(); candidate["conformal"] = conformal
    with pytest.raises(InputError):
        module.select_checkpoint(incumbent(), [candidate])


@pytest.mark.parametrize("conformal", [None, {}, {"coverage": None}])
def test_missing_coverage_retains_the_frozen_gate_failure(conformal):
    candidate = row(); candidate["conformal"] = conformal
    report = module.select_checkpoint(incumbent(), [candidate])
    assert report["winner"] == "incumbent" and not report["per_candidate"][0]["gates_passed"]


def test_optional_missing_weaknesses_remain_uncompared_and_inputs_are_unchanged():
    baseline, candidate = incumbent(), row()
    baseline.pop("d_gene_acc"); candidate["d_gene_acc"] = None
    baseline["conf_set_mean"] = None; candidate.pop("conf_set_p95")
    original = deepcopy((baseline, candidate))
    report = module.select_checkpoint(MappingProxyType(baseline), (MappingProxyType(candidate) for _ in range(1)))
    assert report["winner"] == "candidate"
    assert "d_gene_acc" not in report["per_candidate"][0]["gain"]
    assert "conf_set_mean" not in report["per_candidate"][0]["gain"]
    assert "conf_set_p95" not in report["per_candidate"][0]["gain"]
    assert (baseline, candidate) == original


def test_empty_candidate_set_preserves_null_result_and_complete_rank_ties_preserve_order():
    report = module.select_checkpoint(incumbent(), [])
    assert report["winner"] == "incumbent" and report["per_candidate"] == [] and report["rationale"]
    assert module.select_checkpoint(incumbent(), [row("first"), row("second")])["winner"] == "first"
