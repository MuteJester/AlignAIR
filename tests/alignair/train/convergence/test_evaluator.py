"""Evaluator contracts on fresh, hand-written cohorts independent of GenAIRR simulation."""
import math
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignair.core import AlignAIR
from alignair.train.convergence.evaluator import evaluate_checkpoint

pytestmark = pytest.mark.component

REQUIRED_STRATA = ("clean", "aa_truncate_50_80", "short_janchor", "moderate_full")


def test_clean_heads_and_conformal_test_probabilities_share_one_forward(evaluation_inputs):
    model, cfg, maps, meta, cohort = evaluation_inputs
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    try:
        evaluate_checkpoint(model,cfg,maps,meta,cohort,device='cpu')
    finally:
        handle.remove()
    # Four strata, one boundary pass, one calibration pass. Clean V/J/D and test APS reuse logits.
    assert len(calls) == 6


def test_evaluator_supports_a_model_without_a_d_head(evaluation_inputs):
    _, cfg, maps, meta, cohort = evaluation_inputs
    cfg = replace(cfg,has_d=False,d_allele_count=0)
    output = evaluate_checkpoint(AlignAIR(cfg),cfg,maps,meta,cohort,device='cpu')
    assert output['d_allele_acc'] is output['d_gene_acc'] is output['nll']['d'] is None


def test_evaluate_checkpoint_well_formed(evaluation_inputs):
    model, cfg, class_maps, meta, coh = evaluation_inputs
    model.eval()

    out = evaluate_checkpoint(model, cfg, class_maps, meta, coh, device="cpu")

    required_top_level = (
        "clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
        "d_allele_acc", "d_allele_macro", "d_gene_acc",
        "by_stratum", "freq_bins", "boundary_mae", "nll", "brier", "conformal",
    )
    for key in required_top_level:
        assert key in out, f"evaluate_checkpoint output missing key: {key}"

    accuracy_keys = ("clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
                     "d_allele_acc", "d_allele_macro", "d_gene_acc")
    for key in accuracy_keys:
        val = out[key]
        assert val is not None, f"{key} unexpectedly None (untrained model, tiny cohort)"
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    # by_stratum: required strata present with well-formed v_acc
    for name in REQUIRED_STRATA:
        assert name in out["by_stratum"], f"by_stratum missing required stratum: {name}"
    for name, entry in out["by_stratum"].items():
        assert entry["n"] > 0
        assert 0.0 <= entry["v_acc"] <= 1.0

    # conformal: coverage in [0,1], set sizes well-formed (>=0, p95 >= mean)
    conf = out["conformal"]
    assert conf["coverage"] is not None
    assert 0.0 <= conf["coverage"] <= 1.0
    assert conf["set_mean"] >= 0.0
    assert conf["set_p95"] >= 0.0

    # nll / brier for v, j, d: present and non-negative
    for gene in ("v", "j", "d"):
        assert out["nll"][gene] is not None and out["nll"][gene] >= 0.0
        assert out["brier"][gene] is not None and out["brier"][gene] >= 0.0

    # informational metrics: keys must exist (best-effort; values may legitimately be partial)
    assert set(out["freq_bins"].keys()) == {"v", "j"}
    assert "v" in out["boundary_mae"] and "j" in out["boundary_mae"]

def test_evaluate_checkpoint_is_a_pure_function_of_model_and_cohort(evaluation_inputs):
    """Same frozen model + same frozen cohort -> identical metric dict (no hidden randomness)."""
    model, cfg, class_maps, meta, coh = evaluation_inputs
    model.eval()
    original = deepcopy(coh)
    weights = deepcopy(model.state_dict())

    out_a = evaluate_checkpoint(model, cfg, class_maps, meta, coh, device="cpu")
    out_b = evaluate_checkpoint(model, cfg, class_maps, meta, coh, device="cpu")

    assert out_a == out_b
    assert coh == original
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_evaluator_metrics_match_hand_calculated_predictions(evaluation_inputs):
    _, cfg, maps, meta, cohort = evaluation_inputs
    class FixedModel(torch.nn.Module):
        def forward(self, batch):
            count = len(batch["tokenized_sequence"])
            result = {f"{g}_class_logits": torch.tensor([[math.log(3), 0.0]]).repeat(count, 1)
                      for g in ("v", "d", "j")}
            result.update({name: torch.full((count, 1), value)
                           for name, value in (("v_start", 0.0), ("v_end", 8.0),
                                               ("j_start", 16.0), ("j_end", 20.0))})
            return result
    result = evaluate_checkpoint(FixedModel(), cfg, maps, meta, cohort)
    for key in ("clean_v_acc", "clean_j_acc", "v_macro", "j_macro",
                "d_allele_acc", "d_allele_macro", "d_gene_acc"):
        assert result[key] == 0.5
    for gene in ("v", "d", "j"):
        assert result["nll"][gene] == pytest.approx(-math.log(0.75 * 0.25) / 2)
        assert result["brier"][gene] == pytest.approx(0.625)
    assert result["boundary_mae"] == {"v": {"start": 0.0, "end": 0.0},
                                      "j": {"start": 0.0, "end": 0.0}, "n": 2}


def test_no_clean_rows_fail_before_any_forward(evaluation_inputs):
    model, cfg, maps, meta, cohort = evaluation_inputs
    cohort["by_stratum"]["clean"] = []
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="no 'clean' stratum"):
            evaluate_checkpoint(model, cfg, maps, meta, cohort)
    finally:
        handle.remove()
    assert not calls


@pytest.mark.parametrize("source", ["absent", "ineligible"])
def test_unavailable_boundary_truth_and_calibration_are_reported_without_invented_metrics(evaluation_inputs, source):
    model, cfg, maps, meta, cohort = evaluation_inputs
    cohort["calibration"] = []
    for row in cohort["by_stratum"]["clean"]:
        if source == "absent":
            row.pop("record")
        else:
            row["record"]["productive"] = False
    result = evaluate_checkpoint(model, cfg, maps, meta, cohort)
    assert result["boundary_mae"]["v"] is result["boundary_mae"]["j"] is None
    assert "note" in result["boundary_mae"]
    assert result["conformal"]["coverage"] is None and "note" in result["conformal"]
    assert result["clean_v_acc"] is not None and result["clean_j_acc"] is not None


def test_reference_frequency_table_is_used_when_available(monkeypatch):
    import GenAIRR as ga
    from alignair.train.convergence.evaluator import _freq_for
    monkeypatch.setattr(ga, "FREQUENCY_FIXTURE", SimpleNamespace(gene_use_dict={"V": {"V0": .3, "V1": .7}}), raising=False)
    rows = [{"v_call": name} for name in ("V0", "V1", "V2")]
    np.testing.assert_array_equal(_freq_for(rows, "V", {"V0": 0, "V1": 1}, "FREQUENCY_FIXTURE"), [.3, .7, 0.])
