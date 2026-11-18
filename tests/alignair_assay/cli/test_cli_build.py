import json

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.reference import DEFAULT_REFERENCE  # noqa: E402
from alignair_assay.truth.io import read_truth_header  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites


def test_cli_build(tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert main(["build", "--suites", "S0", "--run-dir", str(run_dir), "--seed", "3", "--scale", "0.2",
                 "--workers", "2"]) == 0
    printed = capsys.readouterr().out.splitlines()
    layout = RunLayout(run_dir)
    build = json.loads(layout.build_json("S0").read_text())
    assert (build["split"], build["seed"], build["scale"]) == ("dev", 3, 0.2)
    assert build["reference"] == DEFAULT_REFERENCE == "HUMAN_IGH_OGRDB"
    assert build["n_cases"] == 10  # plain 6 + with_d 4
    header = read_truth_header(layout.truth("S0"))
    assert (header["suite"], header["split"], header["seed"], header["scale"]) == ("S0", "dev", 3, 0.2)
    assert len(printed) == 1 and printed[0].startswith("S0: 10 cases (plain 6, with_d 4)")
    assert str(layout.suite_dir("S0")) in printed[0]


def test_cli_build_split_and_suites(tmp_path, capsys):
    run_dir = tmp_path / "run"
    assert main(["build", "--suites", "S0, S0", "--run-dir", str(run_dir), "--split", "test", "--scale", "0.1",
                 "--workers", "1"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1  # a repeated suite is built once
    assert json.loads(RunLayout(run_dir).build_json("S0").read_text())["split"] == "test"

    # An unknown suite is an input error, reported before anything is built.
    bad = tmp_path / "bad"
    assert main(["build", "--suites", "S0,S99", "--run-dir", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "unknown suite 'S99'" in err and "S0_EXHAUST" in err
    assert not bad.exists()
    with pytest.raises(SystemExit):
        main(["build", "--suites", "S0", "--run-dir", str(bad), "--split", "train"])
    assert main(["build", "--suites", "S0", "--run-dir", str(bad), "--scale", "-1"]) == 2
    assert "scale" in capsys.readouterr().err


def test_cli_build_exhausted_stratum_is_an_error_not_a_traceback(tmp_path, capsys):
    # A stratum that rejection sampling cannot fill raises RuntimeError in the engine.
    assert main(["build", "--suites", "S0_EXHAUST", "--run-dir", str(tmp_path / "run"), "--workers", "1"]) == 2
    err = capsys.readouterr().err
    assert err == "error: S0_EXHAUST/never: accepted 0 of 5 after 50 rounds\n"
