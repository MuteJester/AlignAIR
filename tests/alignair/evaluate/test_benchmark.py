"""Phase 3: in-package `alignair benchmark` — score + generate + end-to-end."""
import math

import pytest

from alignair.evaluate import default_strata, format_text, score


def _truth():
    return [
        {"v_call": "IGHV1-2*01", "d_call": "IGHD3-3*01", "j_call": "IGHJ4*02", "productive": True,
         "junction": "TGTGCGAAA", "v_sequence_start": 0, "v_sequence_end": 30,
         "j_sequence_start": 40, "j_sequence_end": 60},
        {"v_call": "IGHV3-23*01", "d_call": "", "j_call": "IGHJ6*03", "productive": True,
         "junction": "TGTGCGAGAGGG", "v_sequence_start": 0, "v_sequence_end": 28,
         "j_sequence_start": 38, "j_sequence_end": 58},
    ]


def test_score_perfect_predictions():
    truth = _truth()
    m = score(truth, [dict(t) for t in truth])
    assert m["v_call_acc"] == 1.0 and m["j_call_acc"] == 1.0
    assert m["junction_nt_exact"] == 1.0 and m["junction_len_mae"] == 0.0
    assert m["v_sequence_end_mae"] == 0.0
    assert m["productive_acc"] == 1.0


def test_score_penalizes_wrong_calls_and_junction():
    truth = _truth()
    preds = [dict(t) for t in truth]
    preds[0]["v_call"] = "IGHV9-9*01"          # wrong V
    preds[0]["junction"] = "TGTGCGAAT"          # 1nt off
    preds[1]["v_sequence_end"] = 25             # coord off by 3
    m = score(truth, preds)
    assert m["v_call_acc"] == 0.5
    assert m["junction_nt_exact"] == 0.5
    assert m["v_sequence_end_mae"] == 1.5       # (0 + 3) / 2


def test_score_in_set_call_counts_as_correct():
    truth = [{"v_call": "IGHV1-2*01,IGHV1-2*02", "j_call": "IGHJ4*02", "productive": True}]
    m = score(truth, [{"v_call": "IGHV1-2*02", "j_call": "IGHJ4*02", "productive": True}])
    assert m["v_call_acc"] == 1.0               # predicted a sibling in the truth set


def test_default_strata_are_named_param_dicts():
    strata = default_strata()
    assert set(strata) >= {"clean", "moderate", "high_shm", "short_janchor"}
    assert isinstance(strata["clean"], dict)


def test_format_text_renders_per_stratum():
    results = {"clean": {"n": 100, "v_call_acc": 0.99, "junction_nt_exact": 0.85},
               "high_shm": {"n": 100, "v_call_acc": 0.80, "junction_nt_exact": 0.60}}
    text = format_text(results)
    assert "clean" in text and "high_shm" in text and "junc_nt" in text


@pytest.mark.slow
def test_run_benchmark_end_to_end_small():
    import GenAIRR.data as gd
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.evaluate import run_benchmark
    from alignair.reference.reference_set import ReferenceSet
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model, ref = AlignAIR(cfg).eval(), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    results = run_benchmark(model, ref, gd.HUMAN_IGH_OGRDB, n=6, seed=0,
                            strata_names=["clean", "high_shm"], device="cpu", batch_size=6)
    assert set(results) == {"clean", "high_shm"}
    for s, m in results.items():
        assert m["n"] == 6 and "v_call_acc" in m and "junction_nt_exact" in m


@pytest.mark.slow
def test_cli_benchmark_uses_card_dataconfig(tmp_path):
    import json

    import GenAIRR.data as gd
    from alignair import model_file as mf
    from alignair.cli.main import build_parser
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    p = str(tmp_path / "m.alignair")
    mf.save_model(p, AlignAIR(cfg).eval(), dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 1, "batch_size": 1}, include_trusted_pickle=False,
                  model_id="m", model_version="1.0.0")           # no --dataconfig: read it from the card
    out = str(tmp_path / "bench.json")
    args = build_parser().parse_args(["benchmark", "--model", p, "--n", "4", "--strata", "clean",
                                      "--device", "cpu", "--format", "json", "--out", out])
    assert args.func(args) == 0
    assert json.load(open(out))["clean"]["n"] == 4
