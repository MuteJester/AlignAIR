"""Improved `train --plan` and reference discovery (#96)."""
import json

import pytest

from alignair import cli


def _run(capsys, argv):
    code, msg = 0, ""
    try:
        cli.main(argv)
    except SystemExit as e:
        if isinstance(e.code, str):              # SystemExit("error: ...") carries the message here
            msg, code = e.code, 1
        else:
            code = e.code or 0
    cap = capsys.readouterr()
    return cap.out, cap.err + msg, code


def test_reference_list(capsys):
    pytest.importorskip("GenAIRR")
    out, _, code = _run(capsys, ["reference", "list"])
    assert code == 0 and "built-in GenAIRR references" in out
    assert "HUMAN" in out and "BCR_HEAVY" in out      # dataconfigs + chain types both shown


def test_reference_list_json(capsys):
    pytest.importorskip("GenAIRR")
    out, _, _ = _run(capsys, ["reference", "list", "--json"])
    payload = json.loads(out)
    assert "HUMAN_IGH_OGRDB" in payload["dataconfigs"]
    assert "TCR_BETA" in payload["chain_types"]


def test_unknown_reference_suggests_and_points_to_list(capsys):
    pytest.importorskip("GenAIRR")
    out, err, code = _run(capsys, ["train", "--reference", "HUMAN_IGH_OGRD", "-o", "/tmp/x", "--plan"])
    msg = out + err
    assert code != 0
    assert "HUMAN_IGH_OGRDB" in msg                   # near-match suggestion
    assert "reference list" in msg                    # pointer to the catalog


def test_bad_chain_type_lists_valid(capsys):
    pytest.importorskip("GenAIRR")
    out, err, code = _run(capsys, ["train", "--v-fasta", "a.fa", "--j-fasta", "b.fa",
                                   "--chain-type", "NOPE", "-o", "/tmp/x", "--plan"])
    assert code != 0 and "BCR_HEAVY" in (out + err)


def test_train_plan_reports_time_memory_and_expectations(capsys):
    pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    out, _, code = _run(capsys, ["train", "--reference", "HUMAN_IGH_OGRDB", "-o", "/tmp/plan_unit",
                                 "--preset", "smoke", "--plan", "--device", "cpu"])
    assert code == 0
    for label in ("reference :", "anchors   :", "model     :", "time est. :",
                  "memory est:", "expected  :", "outputs   :"):
        assert label in out
    assert "sanity only" in out                       # smoke preset expectation text
    assert "no training performed" in out


def test_genairr_helpers():
    pytest.importorskip("GenAIRR")
    from alignair.cli import _genairr_dataconfigs, _genairr_chain_types
    dcs, cts = _genairr_dataconfigs(), _genairr_chain_types()
    assert "HUMAN_IGH_OGRDB" in dcs and len(dcs) > 50
    assert set(cts) >= {"BCR_HEAVY", "TCR_BETA"}
