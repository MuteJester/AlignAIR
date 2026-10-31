"""``python -m alignair_lab replicates``: a candidate's screens of several init seeds, named after their seeds, reused
when finished and of the same request, trained when missing, and pooled into one report that sets the candidate's
seeds against the calibration's baseline seeds. Records by hand, then on the rig (``rig``)."""
import json
import shutil

import pytest

from alignair.errors import ConfigurationError
from alignair_lab.cli import main
from alignair_lab.headline import POOLED_HEADING
from alignair_lab.noise import pooled
from alignair_lab.replicates import check_replicates, replicate_name, report_name, request_of
from tests.alignair_lab.rig import candidate_arguments, common, options, write_null_band


def record(seed, **changes):
    """A finished candidate screen's record, replicate ``seed`` of ``cand``; ``changes`` by dotted field."""
    found = {"name": replicate_name("cand", seed), "status": "done", "role": "candidate", "init_seed": seed,
             "architecture": "lab-wide-v1", "model_options": {"width": 2}, "steps": 2, "schedule": "cosine",
             "protocol": {"recipe": "v3", "cache": "c" * 64, "steps": 2,
                          "schedule": {"policy": "cosine-v1", "total_steps": 2, "min_lr": 3e-6}},
             "suites": ["S0"], "suite_builds": {"S0": "b" * 64}, "max_cases": None, "variants": ["baseline"],
             "baseline": {"kind": "screen", "name": "base0", "init_seed": 0, "options": {},
                          "checkpoint": "/screens/base0/train/model.alignair"}}
    for key, value in changes.items():
        owner, _, field = key.rpartition(".")
        target = found
        for part in owner.split(".") if owner else []:
            target = target[part]
        target[field] = value
    return found


@pytest.mark.component
def test_replicates_are_named_after_their_init_seeds():
    assert [replicate_name("cand", seed) for seed in (0, 1, 12)] == ["cand", "cand-init1", "cand-init12"]
    assert report_name("cand", [0, 1, 2]) == "report-cand-seeds012"
    assert report_name("cand", [0, 10]) == "report-cand-seeds0_10"


@pytest.mark.component
def test_replicates_of_one_request_but_their_init_seeds_pass():
    records = {seed: record(seed) for seed in (0, 1, 2)}
    assert check_replicates(records) == request_of(records[0])
    assert "init_seed" not in request_of(records[0]) and "name" not in request_of(records[0])
    assert check_replicates(records, expected=request_of(record(0))) == request_of(records[0])


@pytest.mark.component
@pytest.mark.parametrize("seed,changes,problem", [
    (1, {"protocol.steps": 3}, "protocol"),
    (2, {"model_options": {"width": 3}}, "model_options"),
    (1, {"baseline.checkpoint": "/other/model.alignair"}, "baseline.checkpoint"),
    (2, {"suite_builds": {"S0": "a" * 64}}, "suite_builds"),
    (1, {"max_cases": 50}, "max_cases"),
    (1, {"init_seed": 2}, "init seed 2"),
    (2, {"status": "training"}, "has not finished"),
    (1, {"role": "baseline"}, "not a candidate"),
])
def test_replicates_of_different_requests_are_refused_naming_the_screen_and_what_differs(seed, changes, problem):
    records = {found: record(found) for found in (0, 1, 2)}
    records[seed] = record(seed, **changes)
    with pytest.raises(ConfigurationError, match=problem) as caught:
        check_replicates(records)
    assert replicate_name("cand", seed) in str(caught.value)


@pytest.mark.component
def test_replicates_of_another_request_than_the_one_asked_for_are_refused():
    records = {seed: record(seed) for seed in (0, 1)}
    with pytest.raises(ConfigurationError, match="architecture") as caught:
        check_replicates(records, expected=request_of(record(0, architecture="lab-other-v1")))
    assert "the replicate cand (init seed 0)" in str(caught.value)


# --- on the rig ---------------------------------------------------------------------------------------------------

@pytest.fixture
def replicate_root(rig, baseline, tmp_path):
    """A screens directory holding the registered baseline screen, a null band of its protocol (baseline screens of
    init seeds 0, 1 and 2) and the candidate screen ``rep`` (the small conv-v1 of init seed 0) trained on the rig."""
    root = tmp_path / "screens"
    shutil.copytree(baseline, root / "base0")
    write_null_band(rig, root, json.loads((baseline / "screen.json").read_text())["protocol"])
    assert main(candidate_arguments(rig, "rep", out_root=root)) == 0
    return root


def copy_replicate(root, seed, shift, **changes):
    """``rep`` copied as its replicate of init seed ``seed``, every headline Δ raised by ``shift``, its record's
    ``changes`` applied."""
    target = root / replicate_name("rep", seed)
    shutil.copytree(root / "rep", target)
    found = json.loads((target / "screen.json").read_text())
    found.update(name=target.name, init_seed=seed, screen_dir=str(target), **changes)
    for suite in found["comparison"]["suites"].values():
        for entry in suite["headline"].values():
            if entry.get("delta") is not None:
                entry["delta"] += shift
    (target / "screen.json").write_text(json.dumps(found))


def replicate_arguments(rig, root, *extra, seeds="0,1,2"):
    return ["replicates", "--name", "rep", "--init-seeds", seeds, "--steps", "2", *options("--model-option"),
            *options("--baseline-model-option"), *common(rig, root), *extra]


@pytest.mark.integration
def test_replicates_pool_finished_screens_into_a_report_without_training(rig, replicate_root, capsys):
    copy_replicate(replicate_root, 1, .01)
    copy_replicate(replicate_root, 2, -.02)
    capsys.readouterr()
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 0
    report = replicate_root / "report-rep-seeds012"
    found = json.loads((report / "report.json").read_text())
    assert found["status"] == "done" and found["init_seeds"] == [0, 1, 2]
    assert [(found["screens"][seed]["name"], found["screens"][seed]["reused"]) for seed in ("0", "1", "2")] == \
        [("rep", True), ("rep-init1", True), ("rep-init2", True)]
    screens = [json.loads((replicate_root / name / "screen.json").read_text())
               for name in ("rep", "rep-init1", "rep-init2")]
    assert found["baseline"]["name"] == "base0" and found["calibration"]["init_seeds"] == [0, 1, 2]
    decided = [(metric, entry) for metric, entry in found["decisions"].items() if entry["pooled"] is not None]
    assert decided
    for metric, entry in decided:              # each seed's own pooled Δ, against the baseline seeds of the null band
        assert entry["candidate"] == pytest.approx([pooled(screen["comparison"], metric) for screen in screens])
        assert (entry["k"], entry["n"]) == (3, 3) and entry["decision"] in ("beyond noise", "within noise")
    headline = (report / "headline.txt").read_text()
    assert headline.strip() in capsys.readouterr().out
    assert POOLED_HEADING in headline and "init seed 0 alone (rep)" in headline
    assert sorted(path.name for path in replicate_root.iterdir()) == \
        ["base0", "calibration", "rep", "rep-init1", "rep-init2", "report-rep-seeds012"]          # nothing trained
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 2                      # a report is kept
    assert "exists" in capsys.readouterr().err


@pytest.mark.integration
def test_a_missing_replicate_or_one_of_another_request_is_refused_before_anything_is_written(rig, replicate_root,
                                                                                           capsys):
    copy_replicate(replicate_root, 1, 0.)
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 2
    assert "rep-init2" in capsys.readouterr().err
    other = json.loads((replicate_root / "rep" / "screen.json").read_text())["protocol"]
    copy_replicate(replicate_root, 2, 0., protocol={**other, "steps": 3})
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 2
    error = capsys.readouterr().err
    assert "rep-init2" in error and "protocol" in error
    assert main(replicate_arguments(rig, replicate_root, "--reuse", "--model-option", "embed_dim=8", seeds="0,1")) == 2
    assert "model_options" in capsys.readouterr().err
    assert main(replicate_arguments(rig, replicate_root, "--parallel", "3")) == 2              # two at once at most
    assert "parallel" in capsys.readouterr().err
    assert not any(path.name.startswith("report-") for path in replicate_root.iterdir())


@pytest.mark.integration
def test_replicates_train_the_missing_seeds_and_reuse_the_finished(rig, replicate_root, capsys):
    assert main(replicate_arguments(rig, replicate_root, "--parallel", "2", seeds="0,1")) == 0
    report = replicate_root / "report-rep-seeds01"
    found = json.loads((report / "report.json").read_text())
    assert (found["screens"]["0"]["reused"], found["screens"]["1"]["reused"]) == (True, False)
    trained = json.loads((replicate_root / "rep-init1" / "screen.json").read_text())
    assert (trained["status"], trained["init_seed"], trained["baseline"]["name"]) == ("done", 1, "base0")
    assert (report / "rep-init1.log").is_file() and "[replicates] trained rep-init1" in capsys.readouterr().out
    assert all(entry["k"] == 2 for entry in found["decisions"].values() if entry["pooled"] is not None)
