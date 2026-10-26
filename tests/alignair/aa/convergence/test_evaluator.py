"""Evaluator contracts on fresh, hand-written cohorts independent of GenAIRR simulation."""
import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from alignair.aa.config import aa_igh_config
from alignair.aa.equivalence import EquivalenceClassMap
from alignair.aa.model import AAAlignAIR
from alignair.aa.translation import CODON_TABLE
from alignair.aa.convergence.evaluator import evaluate_checkpoint

pytestmark = pytest.mark.component

REQUIRED_STRATA = ("clean", "aa_truncate_50_80", "short_janchor", "moderate_full")


@pytest.fixture
def evaluation_inputs():
    class_maps = {g: EquivalenceClassMap(g, "fixture", {g+"0*01": 0, g+"1*01": 1},
                                       {0: [g+"0*01"], 1: [g+"1*01"]}) for g in ("V", "J")}
    cfg = aa_igh_config(2, 2, 2, max_seq_length=128, has_d=True)
    meta = {"d_index": {"D0*01": 0, "D1*01": 1}}
    codons = {aa: codon for codon, aa in CODON_TABLE.items()}
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "YWVTSRQPNMLKIHGFEDCA"]
    def row(sequence, index):
        calls = {g.lower()+"_call": g+str(index)+"*01" for g in ("V", "D", "J")}
        record = {"sequence": "".join(codons[aa] for aa in sequence), "sequence_aa": sequence,
                  "productive": True, "junction_start": 24, **calls,
                  "v_sequence_start": 0, "v_sequence_end": 24,
                  "d_sequence_start": 24, "d_sequence_end": 48,
                  "j_sequence_start": 48, "j_sequence_end": 60}
        return {"aa": sequence, **calls, "d_inverted": False, "record": record}
    clean = [row(s, i) for i, s in enumerate(sequences)]
    strata = {name: deepcopy(clean) for name in REQUIRED_STRATA}
    for item in strata["aa_truncate_50_80"]:
        item["aa"] = item["aa"][:12]
        item.pop("record")
    for item in strata["short_janchor"]:
        item["aa"] = item["aa"][-8:]
        item.pop("record")
    calibration = [row(s[1:]+s[:1], i) for i, s in enumerate(sequences)]
    cohort = {"by_stratum": strata, "calibration": calibration, "dc_name": "fixture"}
    return AAAlignAIR(cfg).eval(), cfg, class_maps, meta, cohort


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
