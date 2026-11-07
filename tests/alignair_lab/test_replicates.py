"""``python -m alignair_lab replicates``: a candidate's screens of several init seeds, named after their seeds, reused
when finished and of the same request, trained when missing, and pooled into one report that sets the candidate's
seeds against the calibration's baseline seeds. Records by hand, then on the rig (``rig``)."""
import json
import shutil

import pytest

from alignair.errors import ConfigurationError
from alignair_lab.cli import main
from alignair_lab.headline import POOLED_HEADING
from alignair_lab.noise import BETTER, MIXED, WITHIN, WORSE, pooled
from alignair_lab.screen import code_digests
from alignair_lab.secondary import METRIC, merged
from alignair_lab.baselines import against_calibration, calibration_checkpoint, calibration_code_warning
from alignair_lab.replicates import (
    check_calibration,
    check_pairing,
    check_replicates,
    code_warning,
    current_code_warning,
    replicate_name,
    report_name,
    request_of,
    run_replicates,
)
from tests.alignair_lab.rig import candidate_arguments, common, flat, options, write_null_band


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
def test_two_init_seeds_are_refused_saying_why(tmp_path):
    """Two seeds' spread is an SD of two values: the pooled test calls a candidate without an effect better too often.
    One seed, or three or more."""
    with pytest.raises(ValueError, match="one init seed or three or more, not two") as caught:
        run_replicates("cand", init_seeds=(0, 1), reuse=True, out_root=tmp_path / "screens", cache=tmp_path / "none",
                       dev_build=tmp_path / "none")
    assert "3.2–4.4%" in str(caught.value) and "3.5%" in str(caught.value)
    assert not (tmp_path / "screens").exists()


@pytest.mark.component
def test_a_calibration_without_the_baselines_init_seed_leaves_the_replicates_uncalibrated():
    """The replicates compared with a baseline screen of init seed 3, and the calibration has seeds 0, 1 and 2: its
    seeds' Δs cannot be set against that screen's, so the report is uncalibrated, not refused. A seed-0 screen that is
    another checkpoint than the replicates' baseline is refused, as before."""
    key = "a" * 12 + "-" + "b" * 12
    records = {seed: record(seed, comparison={"baseline": key}) for seed in (0, 1, 2)}
    band = {"path": "/screens/calibration/null_band.json", "init_seeds": [0, 1, 2], "band": {},
            "screens": {str(seed): {"model_key": key} for seed in (0, 1, 2)},
            "seeds": {str(seed): {"pooled": {}, "suites": {}} for seed in (0, 1, 2)}}
    check_calibration(records, band, 3)
    assert against_calibration(band, 3)["seeds"] is None
    assert calibration_checkpoint(band, 3, key) == ("the calibration has no baseline screen of init seed 3, the "
                                                   "baseline screen's, so its seeds' Δs cannot be set against it")
    other = {**band, "screens": {"0": {"model_key": "c" * 12 + "-" + "b" * 12}}}
    with pytest.raises(ConfigurationError, match="another checkpoint"):
        check_calibration(records, other, 0)


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


CODE = {"alignair": "a" * 64, "alignair_lab": "b" * 64, "alignair_assay": "c" * 64}


@pytest.mark.component
def test_replicates_of_other_lab_or_assay_code_are_refused():
    records = {seed: record(seed, code=dict(CODE)) for seed in (0, 1, 2)}
    assert check_replicates(records) == request_of(records[0]) and code_warning(records) is None
    for field in ("alignair_lab", "alignair_assay"):
        records[2] = record(2, code={**CODE, field: "d" * 64})
        with pytest.raises(ConfigurationError, match=f"code.{field}") as caught:
            check_replicates(records)
        assert "cand-init2" in str(caught.value)


@pytest.mark.component
def test_a_calibration_measured_under_other_lab_or_assay_code_is_warned_of_naming_the_digests():
    """The calibration records the lab's and the assay's code it was measured under (``calibrate``), and a report
    warns when the candidates' differ, as it does for AlignAIR's; an older calibration records none."""
    band = {"code": dict(CODE)}
    assert calibration_code_warning(band, CODE, "the replicates") is None
    lab = {**CODE, "alignair_lab": "e" * 64}
    assert calibration_code_warning(band, lab, "this screen") == (
        f"Warning: the calibration was measured under other lab or assay code than this screen: alignair_lab "
        f"{'b' * 12}, this screen's {'e' * 12}.")
    both = {**lab, "alignair_assay": "f" * 64}
    assert calibration_code_warning(band, both, "the replicates") == (
        f"Warning: the calibration was measured under other lab or assay code than the replicates: alignair_lab "
        f"{'b' * 12}, the replicates' {'e' * 12}; alignair_assay {'c' * 12}, the replicates' {'f' * 12}.")
    assert calibration_code_warning({"code": None}, CODE, "the replicates") == (
        "Warning: the calibration records no alignair_lab or alignair_assay code digest, so the lab's and the assay's "
        "code it was measured under cannot be checked to be the replicates'.")
    assert calibration_code_warning(band, None, "the replicates") is None          # their own warning says so


@pytest.mark.component
def test_replicates_screened_under_other_code_than_this_process_are_pooled_with_a_warning_naming_the_digests():
    """Pooling finished replicates under newer code is reuse's normal case: never refused, but each digest that differs
    from this process's code is named, with the replicates it differs for."""
    records = {seed: record(seed, code=dict(CODE)) for seed in (0, 1, 2)}
    assert current_code_warning(records, CODE) is None
    records[2]["code"]["alignair"] = "d" * 64
    now = {**CODE, "alignair_lab": "e" * 64}
    assert check_replicates(records) == request_of(records[0])              # the replicates' own code is checked alike
    assert current_code_warning(records, now) == (
        f"Warning: this report runs under other code than its replicates were screened under: alignair differs for "
        f"cand-init2 (now {'a' * 12}); alignair_lab differs for cand, cand-init1, cand-init2 (now {'e' * 12}). "
        "Their comparisons are the ones their records hold, not diagnosed again under this code.")
    # a replicate that records no code digests is checked on the AlignAIR code its comparison was diagnosed under
    old = {0: record(0, comparison={"code": {"this": "f" * 64}}), 1: record(1)}
    assert current_code_warning(old, CODE) == (
        f"Warning: this report runs under other code than its replicates were screened under: alignair differs for "
        f"cand (now {'a' * 12}). Their comparisons are the ones their records hold, not diagnosed again under "
        "this code.")


@pytest.mark.component
def test_replicates_without_code_digests_are_pooled_with_a_warning():
    """Screens recorded before the lab's and the assay's code digests were cannot be checked: they are pooled with a
    warning naming them."""
    records = {0: record(0, code=dict(CODE)), 1: record(1), 2: record(2)}
    assert check_replicates(records) == request_of(records[0])
    assert code_warning(records) == ("Warning: code digest unknown for cand-init1, cand-init2: they record no "
                                     "alignair_lab or alignair_assay code digest, so the lab's and the assay's code "
                                     "cannot be checked to be the same.")


def compared(suites, cases=500):
    """A comparison summary of ``suites``: suite -> metric -> paired reads."""
    return {"suites": {name: {"cases": cases, "headline": {metric: {"delta": 0., "n_paired": paired}
                                                            for metric, paired in metrics.items()}}
                       for name, metrics in suites.items()}}


@pytest.mark.component
def test_replicates_must_share_their_suites_and_each_suites_pairing():
    """Their Δs are pooled suite by suite: each replicate compares the same suites, on the same reads of each."""
    metrics = ["clone_key.exact", "junction.nt_exact"]
    same = {"S1": {"clone_key.exact": 100, "junction.nt_exact": 98}, "S2": {"clone_key.exact": 80}}
    records = {seed: record(seed, comparison=compared(same)) for seed in (0, 1, 2)}
    check_pairing(records, metrics)
    records[2] = record(2, comparison=compared({"S1": same["S1"]}))
    with pytest.raises(ConfigurationError, match="suites S2") as caught:
        check_pairing(records, metrics)
    assert "cand-init2" in str(caught.value)
    records[2] = record(2, comparison=compared({**same, "S1": {"clone_key.exact": 100, "junction.nt_exact": 97}}))
    with pytest.raises(ConfigurationError, match="S1 junction.nt_exact: 97 paired reads, not 98"):
        check_pairing(records, metrics)
    records[2] = record(2, comparison=compared(same, cases=499))
    with pytest.raises(ConfigurationError, match="S1 cases: 499, not 500"):
        check_pairing(records, metrics)


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
    init seeds 0, 1 and 2) and the candidate screen ``rep`` (the small chip-v1 of init seed 0) trained on the rig."""
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
    copy_replicate(replicate_root, 2, -.02, code=None)                  # screened before the code digests were kept
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
    assert decided and METRIC in found["decisions"]             # the secondary metric too, from the screens' records
    for metric, entry in decided:              # each seed's own pooled Δ, against the baseline seeds of the null band
        assert entry["candidate"] == pytest.approx([pooled(merged(screen["comparison"], screen["secondary"]), metric)
                                                    for screen in screens])
        if metric == METRIC:               # the rig's calibration predates it, and has no cached scores to give it
            assert (entry["k"], entry["n"], entry["decision"]) == (3, 0, "uncalibrated")
        else:
            assert (entry["k"], entry["n"]) == (3, 3) and entry["decision"] in (BETTER, WORSE, MIXED, WITHIN)
    headline = (report / "headline.txt").read_text()
    if found["decisions"][METRIC]["pooled"] is not None:
        assert "D short abstain is uncalibrated: the calibration holds no d_short.abstain_ok values" in flat(headline)
    assert headline.strip() in capsys.readouterr().out
    assert POOLED_HEADING in flat(headline) and "init seed 0 alone (rep)" in flat(headline)
    assert "Warning: code digest unknown for rep-init2:" in flat(headline)
    assert "Warning: the calibration records no alignair_lab or alignair_assay code digest" in flat(headline)
    assert "other code than its replicates" not in flat(headline)         # screened by the code that pools them
    assert found["code"] == code_digests()                                 # the report's own code, by content
    assert sorted(path.name for path in replicate_root.iterdir()) == \
        ["base0", "calibration", "rep", "rep-init1", "rep-init2", "report-rep-seeds012"]          # nothing trained
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 2                      # a report is kept
    assert "exists" in capsys.readouterr().err


@pytest.mark.integration
def test_finished_replicates_are_pooled_from_their_records_alone(rig, replicate_root, tmp_path, capsys):
    """``--from-records`` reads the replicates' screen records and the calibration only: no record cache, dev build or
    baseline checkpoint, and none of the request's arguments. It pools as the full ``--reuse`` path does."""
    copy_replicate(replicate_root, 1, .01)
    copy_replicate(replicate_root, 2, -.02)
    assert main(replicate_arguments(rig, replicate_root, "--reuse", "--report", "full")) == 0
    missing = tmp_path / "nowhere"
    assert main(["replicates", "--name", "rep", "--init-seeds", "0,1,2", "--reuse", "--from-records", "--report",
                 "records", "--out-root", str(replicate_root), "--cache", str(missing),
                 "--dev-build", str(missing)]) == 0
    full, records = (json.loads((replicate_root / name / "report.json").read_text()) for name in ("full", "records"))
    assert records["decisions"] == full["decisions"] and records["request"] == full["request"]
    assert records["baseline"] == full["baseline"] and records["from_records"] and not full["from_records"]
    # replicates against a baseline screen of init seed 3, which the calibration (seeds 0, 1, 2) has not: uncalibrated
    for name in ("rep", "rep-init1", "rep-init2"):
        path = replicate_root / name / "screen.json"
        found = json.loads(path.read_text())
        found["baseline"]["init_seed"] = 3
        path.write_text(json.dumps(found))
    assert main(["replicates", "--name", "rep", "--reuse", "--from-records", "--report", "seed3", "--out-root",
                 str(replicate_root)]) == 0
    seed3 = json.loads((replicate_root / "seed3" / "report.json").read_text())
    assert {entry["decision"] for entry in seed3["decisions"].values() if entry["pooled"] is not None} == \
        {"uncalibrated"}
    headline = flat((replicate_root / "seed3" / "headline.txt").read_text())
    assert "Uncalibrated: the calibration has no baseline screen of init seed 3, the replicates' baseline screen's" \
        in headline and "their Δs against init seed 3's screen" not in headline
    capsys.readouterr()
    assert main(["replicates", "--name", "rep", "--from-records", "--out-root", str(replicate_root)]) == 2
    assert "--reuse" in capsys.readouterr().err


@pytest.mark.integration
def test_replicates_screened_under_older_lab_code_are_pooled_with_a_warning_naming_it(rig, replicate_root, capsys):
    """The lab changed since the replicates were screened: they are pooled, and the headline says which digest
    differs."""
    path = replicate_root / "rep" / "screen.json"
    found = json.loads(path.read_text())
    found["code"]["alignair_lab"] = "d" * 64
    path.write_text(json.dumps(found))
    copy_replicate(replicate_root, 1, .01)
    copy_replicate(replicate_root, 2, -.02)
    band = replicate_root / "calibration" / "null_band.json"     # a calibration measured under this process's code
    band.write_text(json.dumps({**json.loads(band.read_text()), "code": code_digests()}))
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 0
    report = replicate_root / "report-rep-seeds012"
    assert json.loads((report / "report.json").read_text())["status"] == "done"
    headline = flat((report / "headline.txt").read_text())
    now = code_digests()["alignair_lab"][:12]
    assert (f"Warning: this report runs under other code than its replicates were screened under: alignair_lab differs "
            f"for rep, rep-init1, rep-init2 (now {now}).") in headline
    assert "alignair differs" not in headline and "alignair_assay differs" not in headline
    assert (f"Warning: the calibration was measured under other lab or assay code than the replicates: alignair_lab "
            f"{now}, the replicates' {'d' * 12}.") in headline


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
    assert main(replicate_arguments(rig, replicate_root, "--reuse", "--model-option", "embed_dim=8")) == 2
    assert "model_options" in capsys.readouterr().err
    assert main(replicate_arguments(rig, replicate_root, "--parallel", "3")) == 2              # two at once at most
    assert "parallel" in capsys.readouterr().err
    assert not any(path.name.startswith("report-") for path in replicate_root.iterdir())


@pytest.mark.integration
def test_replicates_against_another_seed_0_checkpoint_than_the_calibrations_are_refused(rig, replicate_root, capsys):
    copy_replicate(replicate_root, 1, 0.)
    copy_replicate(replicate_root, 2, 0.)
    protocol = json.loads((replicate_root / "rep" / "screen.json").read_text())["protocol"]
    shutil.rmtree(replicate_root / "calibration")
    other = "0" * 12 + "-" + "c" * 12
    write_null_band(rig, replicate_root, protocol, screens={"0": {"name": "base0", "model_key": other}})
    assert main(replicate_arguments(rig, replicate_root, "--reuse")) == 2
    error = capsys.readouterr().err
    assert "another checkpoint" in error and other in error
    assert not any(path.name.startswith("report-") for path in replicate_root.iterdir())


@pytest.mark.integration
def test_replicates_train_the_missing_seeds_and_reuse_the_finished(rig, replicate_root, capsys):
    copy_replicate(replicate_root, 2, .01)
    assert main(replicate_arguments(rig, replicate_root, "--parallel", "2")) == 0
    report = replicate_root / "report-rep-seeds012"
    found = json.loads((report / "report.json").read_text())
    assert [found["screens"][seed]["reused"] for seed in ("0", "1", "2")] == [True, False, True]
    trained = json.loads((replicate_root / "rep-init1" / "screen.json").read_text())
    assert (trained["status"], trained["init_seed"], trained["baseline"]["name"]) == ("done", 1, "base0")
    assert (report / "rep-init1.log").is_file() and "[replicates] trained rep-init1" in capsys.readouterr().out
    assert all(entry["k"] == 3 for entry in found["decisions"].values() if entry["pooled"] is not None)


@pytest.mark.component
@pytest.mark.parametrize("novel_seed", [0, 1])
def test_replicates_of_a_novel_allele_screen_are_refused(novel_seed):
    """A novel-allele screen trains from another record cache: pooled with plain screens it would mix two treatments,
    so replicates of it (or with it) are refused; a novel-allele screen is run per init seed by novel-screen."""
    spec = {"generator": 1, "gene_fraction": 0.4, "snps": [1, 6], "epoch_records": 4096}
    records = {seed: record(seed, **({"novel_alleles": spec, "training_cache": {"path": "novel"}}
                                     if seed == novel_seed else {})) for seed in (0, 1, 2)}
    with pytest.raises(ConfigurationError, match="novel"):
        check_replicates(records)
