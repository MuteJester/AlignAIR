"""A checkpoint against IgBLAST on the lab's dev suites (``igblast``): IgBLAST's per-read scores from an assay run on
the dev build's reads, paired read by read with the checkpoint's diagnostics (the product pipeline and the
truth-substitution chain), with the assay's own paired statistics."""
import json

import pytest

from alignair_lab import igblast

pytestmark = pytest.mark.component

METRIC = "v.top1_in_set"


def _case(baseline, **variants):
    return {"head": {}, "baseline": {METRIC: baseline},
            "variants": {name: {METRIC: value} for name, value in variants.items()}}


def _suite(n, igblast_right, chip_right, prefix="S1"):
    """n reads: IgBLAST right on the first ``igblast_right``, the chip on the first ``chip_right`` (true orientation
    always right)."""
    ids = [f"{prefix}:full:{i:04d}" for i in range(n)]
    theirs = {case: {METRIC: float(i < igblast_right)} for i, case in enumerate(ids)}
    ours = {case: _case(float(i < chip_right), true_orientation=1.0) for i, case in enumerate(ids)}
    return theirs, ours


def test_pairing_counts_the_reads_each_gets_and_tests_them():
    theirs, ours = _suite(40, igblast_right=40, chip_right=25)
    result = igblast.pair({"S1": theirs}, {"S1": ours}, {}, metrics=(METRIC,))
    suite = result["suites"]["S1"]
    compared = suite["comparison"][METRIC]
    assert compared["mean_a_paired"] == 1.0 and compared["mean_b_paired"] == 25 / 40
    assert compared["verdict"] == "a_better"                                   # IgBLAST better: 15 reads, none back
    assert suite["changes"]["metrics"][METRIC]["broken"] == 15                 # IgBLAST right, the chip wrong
    assert suite["changes"]["metrics"][METRIC]["fixed"] == 0
    assert suite["chain"]["true_orientation"][METRIC] == 1.0


def test_the_pooled_comparison_holds_every_suite():
    s1 = _suite(30, igblast_right=30, chip_right=30, prefix="S1")
    s2 = _suite(30, igblast_right=10, chip_right=30, prefix="S2")
    result = igblast.pair({"S1": s1[0], "S2": s2[0]}, {"S1": s1[1], "S2": s2[1]}, {}, metrics=(METRIC,))
    pooled = result["pooled"]["comparison"][METRIC]
    assert pooled["n_paired"] == 60 and pooled["verdict"] == "b_better"         # the chip better: 20 reads
    assert result["pooled"]["changes"]["metrics"][METRIC]["fixed"] == 20


def test_a_comparison_never_pairs_a_subset():
    theirs, ours = _suite(20, igblast_right=20, chip_right=20)
    del theirs[next(iter(theirs))]
    with pytest.raises(ValueError, match="IgBLAST has no scores"):
        igblast.pair({"S1": theirs}, {"S1": ours}, {}, metrics=(METRIC,))


def _build(root, suite, **settings):
    directory = root / "suites" / suite
    directory.mkdir(parents=True)
    record = {"suite": suite, "split": "dev", "seed": 20261019, "scale": 0.1, "reference": "HUMAN_IGH_OGRDB",
              **settings}
    (directory / "build.json").write_text(json.dumps(record))


def test_igblast_must_have_run_on_the_dev_builds_reads(tmp_path):
    dev, run = tmp_path / "dev", tmp_path / "run"
    for suite in ("S1", "S2"):
        _build(dev, suite)
    _build(run, "S1")
    _build(run, "S2", seed=7)
    igblast.check_same_builds(run, dev, ["S1"])
    with pytest.raises(ValueError, match="S2"):
        igblast.check_same_builds(run, dev, ["S1", "S2"])
    with pytest.raises(ValueError, match="S3"):
        igblast.check_same_builds(run, dev, ["S3"])


def test_the_igblast_run_takes_the_dev_builds_settings(tmp_path):
    for suite in ("S1", "S2"):
        _build(tmp_path, suite)
    argv = igblast.igblast_dev_argv(tmp_path / "run", ["S1", "S2"], igblast.dev_settings(tmp_path, ["S1", "S2"]),
                                    tmp_path / "igblast")
    assert argv[:1] == ["all"] and "igblast,oracle" in argv
    for flag, value in (("--suites", "S1,S2"), ("--split", "dev"), ("--seed", "20261019"), ("--scale", "0.1"),
                        ("--reference", "HUMAN_IGH_OGRDB")):
        assert argv[argv.index(flag) + 1] == value
    _build(tmp_path, "S3", scale=0.2)
    with pytest.raises(ValueError, match="scale"):
        igblast.dev_settings(tmp_path, ["S1", "S3"])


def test_the_report_leads_with_the_pooled_table():
    theirs, ours = _suite(40, igblast_right=40, chip_right=25)
    text = igblast.report_markdown(igblast.pair({"S1": theirs}, {"S1": ours}, {}, metrics=(METRIC,)), name="chip-s0")
    pooled, s1 = text.index("## All suites"), text.index("## S1")
    assert pooled < s1
    assert "| v.top1_in_set | 100.0 | 62.5 | -37.5 pp |" in text
    assert "IgBLAST better" in text


def test_the_commands_hand_their_arguments_to_the_module(monkeypatch, tmp_path, capsys):
    from alignair_lab import cli
    seen = {}
    monkeypatch.setattr(igblast, "run_against_igblast",
                        lambda name, checkpoint, **kwargs: seen.update(name=name, checkpoint=checkpoint, **kwargs)
                        or "# report\n")
    monkeypatch.setattr(igblast, "run_igblast_dev", lambda **kwargs: seen.update(dev=kwargs) or tmp_path)
    assert cli.main(["against-igblast", "--name", "chip-s0", "--checkpoint", "m.alignair", "--model-option",
                     "bank_orientation=true", "--suites", "S1,S6", "--igblast-run", str(tmp_path)]) == 0
    assert seen["name"] == "chip-s0" and str(seen["checkpoint"]) == "m.alignair"
    assert seen["options"] == {"bank_orientation": True} and seen["suites"] == ["S1", "S6"]
    assert str(seen["igblast_run"]) == str(tmp_path)
    assert "# report" in capsys.readouterr().out
    assert cli.main(["igblast-dev", "--run-dir", str(tmp_path), "--suites", "S1"]) == 0
    assert str(seen["dev"]["run_dir"]) == str(tmp_path) and seen["dev"]["suites"] == ["S1"]
