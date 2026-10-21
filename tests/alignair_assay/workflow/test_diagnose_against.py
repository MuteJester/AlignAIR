"""``diagnose``'s per-case scores and ``--against``: two checkpoints compared on the same reads.

Each suite record of ``workflow.diagnose.evaluate`` also stores its per-case scores (``cases.jsonl.gz``: the head
metrics, the end-to-end baseline and each variant), which its cache checks by digest. ``against`` pairs the model with
another on those per-case scores: the other is diagnosed first through the same cache and settings (a checkpoint or
an in-memory pair) or read by its model key, and anything but the same reads under the same settings is refused.

Everything runs on the CPU with small random-init models (``conftest.make_small_model``) on S0 and S1 built at scale
0.02 (2 and 100 reads). The released checkpoint is compared with itself only when ``ALIGNAIR_ASSAY_MODEL`` names a
``.alignair`` model file, on the GPU when ``ALIGNAIR_ASSAY_GPU=1``.
"""
import json
import math
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.model_file import save_model  # noqa: E402

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.jsonl import read_header, read_jsonl  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.core.records import write_json, write_jsonl  # noqa: E402
from alignair_assay.diagnostics.identity import alignair_code  # noqa: E402
from alignair_assay.diagnostics.replay import load_model_checked  # noqa: E402
from alignair_assay.metrics import is_informational, lower_is_better  # noqa: E402
from alignair_assay.report.diagnostics import AGAINST_SECTIONS, SUMMARY_SECTIONS  # noqa: E402
from alignair_assay.scoring.pipeline import truth_reference  # noqa: E402
from alignair_assay.stats import summarize, to_dict  # noqa: E402
from alignair_assay.workflow import diagnose as diagnose_module  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.diagnose import evaluate, read_case_scores, select_cases  # noqa: E402

MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
REFERENCE = "HUMAN_IGH_OGRDB"
SCALE = 0.02
FAST = dict(device="cpu", batch_size=16, variants=("baseline",), profile=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A run directory with S0 (2 cases) and S1 (100 cases, with donors) built on the dev split at scale 0.02."""
    layout = RunLayout(tmp_path_factory.mktemp("against_run"))
    for name in ("S0", "S1"):
        build_suite(name, layout, reference_name=REFERENCE, split="dev", seed=5, scale=SCALE, workers=4)
    return layout


def _copy(layout: RunLayout, tmp_path) -> RunLayout:
    """A fresh run directory with ``layout``'s builds and nothing else."""
    shutil.copytree(layout.root / "suites", tmp_path / "run" / "suites")
    return RunLayout(tmp_path / "run")


@pytest.fixture
def save_checkpoint(make_small_model):
    """``save(path, seed=0) -> path``: a small random-init model written as an ``.alignair`` model file."""
    def save(path, seed=0):
        model, _ = make_small_model(REFERENCE, seed=seed)
        save_model(str(path), model, dataconfigs=[REFERENCE], training={"steps": 0, "batch_size": 0},
                   include_trusted_pickle=False)
        return path

    return save


@pytest.fixture
def replays(monkeypatch):
    """The read ids of every baseline replay ``evaluate`` runs (one per suite it computes), in call order."""
    seen = []
    original = diagnose_module.replay

    def spy(model, reference_set, sequences, ids, **kwargs):
        seen.append(list(ids))
        return original(model, reference_set, sequences, ids, **kwargs)

    monkeypatch.setattr(diagnose_module, "replay", spy)
    return seen


def _suites(replays) -> list[str]:
    return [ids[0].split(":")[0] for ids in replays]


def _key(checkpoint) -> str:
    return f"{sha256_file(checkpoint)[:12]}-{alignair_code()['code_sha256'][:12]}"


# --- per-case scores --------------------------------------------------------------------------------------------

def test_each_suite_record_stores_its_per_case_scores(built, tmp_path, small_model):
    layout = _copy(built, tmp_path)
    result = evaluate(small_model, layout, ["S1"], **{**FAST, "variants": ("baseline", "argmax_decoder")},
                      max_cases=40)
    suite = result.suites["S1"]
    path = layout.diagnostics_suite_cases(result.model_key, "S1")
    # one line per diagnosed case, in diagnosed order: its head metrics, its end-to-end baseline, each variant's
    lines = read_jsonl(path)
    assert [line["case_id"] for line in lines] == [
        case["case_id"] for case in select_cases(read_cases(layout.cases("S1")), 40)]
    assert all(set(line) == {"case_id", "head", "baseline", "variants"} for line in lines)
    assert all(set(line["variants"]) == {"argmax_decoder"} for line in lines)
    header = read_header(path)
    assert (header["model_key"], header["suite"]) == (result.model_key, "S1")
    # the record names the file by its digest
    assert suite["case_scores"] == {"file": "cases.jsonl.gz", "cases": 40, "sha256": sha256_file(path)}
    # the stored scores are those the record summarizes; an applicable metric not covered is null, read back as NaN
    scores = read_case_scores(layout, result.model_key, "S1")
    assert list(scores) == [line["case_id"] for line in lines]

    def summary(field, variant=None):
        found = {case: (line["variants"][variant] if variant else line[field]) for case, line in scores.items()}
        found = {case: values for case, values in found.items() if values is not None}
        return to_dict(summarize(found, list(found)))

    assert summary("head") == suite["heads"]
    assert summary("baseline") == suite["variants"]["baseline"]["scores"]
    assert summary(None, "argmax_decoder") == suite["variants"]["argmax_decoder"]["scores"]
    for line in lines:
        read = scores[line["case_id"]]["baseline"] or {}
        for name, value in (line["baseline"] or {}).items():
            assert (value is None) == math.isnan(read[name])
    assert any(value is None for line in lines for value in (line["baseline"] or {}).values())


def test_a_record_without_its_per_case_scores_is_stale(built, tmp_path, small_model, replays):
    layout = _copy(built, tmp_path)
    first = evaluate(small_model, layout, ["S0"], **FAST)
    path = layout.diagnostics_suite_cases(first.model_key, "S0")
    record_path = layout.diagnostics_suite_json(first.model_key, "S0")

    def computed() -> bool:
        before = len(replays)
        evaluate(small_model, layout, ["S0"], **FAST)
        return len(replays) > before

    assert not computed()
    path.unlink()
    assert computed() and not computed()
    # other bytes than the record's digest: another model's file over it, or a truncated one
    write_jsonl(path, read_jsonl(path)[:1], header=read_header(path))
    assert computed() and not computed()
    # a record that names no per-case file, as one written before they were stored
    record = json.loads(record_path.read_text())
    del record["case_scores"]
    write_json(record_path, record)
    assert computed() and not computed()
    # read directly, a missing per-case file is refused, never read as no cases
    path.unlink()
    with pytest.raises(ValueError, match="per-case scores of model .* missing"):
        read_case_scores(layout, first.model_key, "S0")


# --- the comparison ---------------------------------------------------------------------------------------------

def _families(suite: dict) -> dict:
    return {family: suite[family] for family in ("heads", "baseline") if suite.get(family)}


def test_the_same_model_under_two_inputs_compares_to_zero(built, tmp_path, save_checkpoint, make_small_model):
    # The checkpoint file and the same weights in memory are two model keys, diagnosed independently: every delta
    # must be exactly zero, every verdict a tie, no read fixed or broken.
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair", seed=3)
    result = evaluate(checkpoint, layout, ["S0", "S1"], **FAST, against=make_small_model(REFERENCE, seed=3))
    against = result.against
    assert against["model_key"] != result.model_key and result.model_key == _key(checkpoint)
    assert list(against["suites"]) == ["S0", "S1"]
    for name, suite in against["suites"].items():
        assert set(_families(suite)) == {"heads", "baseline"}, name
        for family, found in _families(suite).items():
            assert found["metrics"] and not any(is_informational(metric) for metric in found["metrics"])
            assert list(found["comparison"]) == found["metrics"]
            for metric, entry in found["comparison"].items():
                assert entry["delta"] in (0.0, None) and entry["verdict"] == "tie_or_inconclusive", metric
                assert entry["wins"] == entry["losses"] == 0
            changes = found["changes"]
            assert changes["reads"] == {"fixed": 0, "broken": 0, "fixed_examples": [], "broken_examples": []}
            assert (changes["a_only"], changes["b_only"]) == (0, 0)
            assert all(m["fixed"] == m["broken"] == 0 for m in changes["metrics"].values())
    s1 = against["suites"]["S1"]
    assert s1["cases"] == 100 and s1["baseline"]["comparison"]["junction.nt_exact"]["n_paired"] > 0
    assert s1["heads"]["comparison"]["head.orientation.correct"]["n_paired"] > 0
    assert "head.v.start_err.soft" not in s1["heads"]["metrics"]  # informational: not compared
    assert against["summary"]["suites"]["S1"]["reads"]["baseline"] == {"fixed": 0, "broken": 0}

    scalars = result.scalars()
    compared = {key: value for key, value in scalars.items() if "/against/" in key}
    assert compared and set(compared.values()) == {0.0}
    for key in ("S1/against/junction.nt_exact.delta", "S1/against/junction.nt_exact.fixed",
                "S1/against/junction.nt_exact.broken", "S1/against/head.orientation.correct.delta"):
        assert key in compared, key
    assert scalars["S1/baseline/junction.nt_exact"] == result.suites["S1"]["variants"]["baseline"]["scores"][
        "junction.nt_exact"]["mean"]

    # The comparison leads the summary page; each suite's detail is a file of its own.
    md = result.report_md
    assert md.index("\n## Checkpoint comparison\n") < md.index(f"\n## {SUMMARY_SECTIONS[0]}\n")
    section = md[md.index("\n## Checkpoint comparison\n"):md.index(f"\n## {SUMMARY_SECTIONS[0]}\n")]
    assert against["model_key"] in section and "| S1 | junction.nt_exact |" in section
    assert f"[S1](S1/against_{against['model_key']}.md)" in section
    assert layout.diagnostics_md(result.model_key).read_text(encoding="utf-8") == md
    report = json.loads(layout.diagnostics_json(result.model_key).read_text())
    assert report["against"]["model_key"] == against["model_key"] and report["summary"]["against"]
    for name in ("S0", "S1"):
        detail = layout.diagnostics_against_md(result.model_key, name, against["model_key"]).read_text("utf-8")
        positions = [detail.index(f"\n## {title}\n") for title in AGAINST_SECTIONS]
        assert positions == sorted(positions) and "(../diagnostics.md)" in detail
    assert not list(layout.diagnostics_root.rglob("*.tmp"))


def test_a_perturbed_model_changes_reads_and_they_are_counted(built, tmp_path, make_small_model):
    layout = _copy(built, tmp_path)
    model, reference_set = make_small_model(REFERENCE, seed=4)
    perturbed, _ = make_small_model(REFERENCE, seed=4)
    with torch.no_grad():  # its orientation head now calls every read forward
        perturbed.orientation_head.fc.weight.zero_()
        perturbed.orientation_head.fc.bias.zero_()
    result = evaluate((model, reference_set), layout, ["S1"], **FAST, against=(perturbed, reference_set))
    suite = result.against["suites"]["S1"]
    other = result.against["model_key"]
    assert other != result.model_key
    # A direct recount from the two per-case files, as written (null: not covered).
    a = {line["case_id"]: line for line in read_jsonl(layout.diagnostics_suite_cases(result.model_key, "S1"))}
    b = {line["case_id"]: line for line in read_jsonl(layout.diagnostics_suite_cases(other, "S1"))}
    assert set(a) == set(b)
    changed = 0
    for family, field in (("heads", "head"), ("baseline", "baseline")):
        metrics = suite[family]["changes"]["metrics"]
        assert metrics, family
        for metric, found in metrics.items():
            sign = -1 if lower_is_better(metric) else 1
            pairs = [(case, a[case][field][metric], b[case][field][metric]) for case in a
                     if a[case][field] and b[case][field] and a[case][field].get(metric) is not None
                     and b[case][field].get(metric) is not None]
            fixed = sorted(case for case, x, y in pairs if sign * (y - x) > 0)
            broken = sorted(case for case, x, y in pairs if sign * (y - x) < 0)
            assert (found["n_paired"], found["fixed"], found["broken"]) == (len(pairs), len(fixed), len(broken))
            assert (found["fixed_examples"], found["broken_examples"]) == (fixed[:10], broken[:10])
            changed += len(fixed) + len(broken)
            assert result.scalars()[f"S1/against/{metric}.fixed"] == len(fixed)
            assert result.scalars()[f"S1/against/{metric}.broken"] == len(broken)
    orientation = suite["heads"]["changes"]["metrics"]["head.orientation.correct"]
    assert orientation["fixed"] > 0 and orientation["broken"] > 0 and changed > 0
    assert any(entry["delta"] for entry in suite["heads"]["comparison"].values())
    totals = result.against["summary"]["suites"]["S1"]["reads"]
    assert totals["baseline"]["fixed"] == suite["baseline"]["changes"]["reads"]["fixed"]
    assert totals["heads"]["broken"] == suite["heads"]["changes"]["reads"]["broken"] > 0
    section = result.report_md[result.report_md.index("\n## Checkpoint comparison\n"):]
    assert f"{totals['heads']['fixed']} / {totals['heads']['broken']}" in section


# --- the other model ------------------------------------------------------------------------------------------

def test_the_other_model_is_diagnosed_first_with_the_same_settings(built, tmp_path, save_checkpoint, small_model,
                                                                   replays):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "other.alignair", seed=2)
    options = dict(device="cpu", batch_size=8, variants=("argmax_decoder",), profile=False, max_cases=7)
    result = evaluate(small_model, layout, ["S0", "S1"], **options, against=str(checkpoint))
    other = result.against["model_key"]
    assert other == _key(checkpoint) and result.against["summary"]["source"] == str(checkpoint)
    # the other model first, then this one; each record under its own key, with the same configuration
    assert _suites(replays) == ["S0", "S1", "S0", "S1"]
    for name in ("S0", "S1"):
        mine = json.loads(layout.diagnostics_suite_json(result.model_key, name).read_text())["key"]
        theirs = json.loads(layout.diagnostics_suite_json(other, name).read_text())["key"]
        assert mine["model"] != theirs["model"]
        assert {k: v for k, v in mine.items() if k != "model"} == {k: v for k, v in theirs.items() if k != "model"}
        assert (theirs["max_cases"], theirs["batch_size"], theirs["device"]) == (7, 8, "cpu")
    # Again, by path or by model key: nothing is computed, the same comparison.
    for again in (checkpoint, other):
        repeated = evaluate(small_model, layout, ["S0", "S1"], **options, against=again)
        assert len(replays) == 4 and repeated.against["suites"] == result.against["suites"]
    # A missing checkpoint is refused before any work.
    with pytest.raises(FileNotFoundError, match="no AlignAIR model file"):
        evaluate(small_model, layout, ["S0"], **options, against=tmp_path / "gone.alignair")
    assert len(replays) == 4


def test_a_comparison_of_other_reads_or_settings_is_refused(built, tmp_path, small_model, make_small_model,
                                                            replays):
    layout = _copy(built, tmp_path)
    other = evaluate(make_small_model(REFERENCE, seed=9), layout, ["S0", "S1"], **FAST, max_cases=10).model_key
    before = len(replays)
    # Other settings: refused before any work, naming the setting.
    for change, field in ((dict(max_cases=5), "max_cases"), (dict(batch_size=8), "batch_size"),
                          (dict(variants=("argmax_decoder",)), "variants"), (dict(device="cpu:0"), "device")):
        with pytest.raises(ValueError, match=rf"differ.*\b{field}\b"):
            evaluate(small_model, layout, ["S0"], **{**FAST, "max_cases": 10, **change}, against=other)
    # A model key with no diagnostics of a suite.
    with pytest.raises(ValueError, match=r"no diagnostics of suite S0"):
        evaluate(small_model, layout, ["S0"], **FAST, max_cases=10, against="0" * 12 + "-" + "1" * 12)
    assert len(replays) == before

    # Another build of the suite: refused naming the build field.
    build_suite("S0", layout, reference_name=REFERENCE, split="dev", seed=6, scale=SCALE, workers=1)
    with pytest.raises(ValueError, match=r"differ.*\bsuite\.(cases_sha256|identity\.seed)\b"):
        evaluate(small_model, layout, ["S0"], **FAST, max_cases=10, against=other)
    assert len(replays) == before

    # The same build and settings, but per-case scores of other case ids (a doctored file whose digest the record
    # names): refused naming the case ids, never paired on the subset.
    path = layout.diagnostics_suite_cases(other, "S1")
    write_jsonl(path, read_jsonl(path)[:-1], header=read_header(path))
    record_path = layout.diagnostics_suite_json(other, "S1")
    record = json.loads(record_path.read_text())
    record["case_scores"]["sha256"] = sha256_file(path)
    write_json(record_path, record)
    with pytest.raises(ValueError, match=r"case ids.*1 only here"):
        evaluate(small_model, layout, ["S1"], **FAST, max_cases=10, against=other)


# --- the diagnose command ---------------------------------------------------------------------------------------

def test_diagnose_command_compares_against_a_checkpoint_or_a_model_key(built, tmp_path, save_checkpoint, capsys):
    layout = _copy(built, tmp_path)
    mine, theirs = save_checkpoint(tmp_path / "a.alignair", seed=0), save_checkpoint(tmp_path / "b.alignair", seed=1)
    base = ["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(mine), "--suites", "S0,S1", "--device",
            "cpu", "--variants", "baseline", "--no-profile", "--max-cases", "6"]
    assert main([*base, "--against", str(theirs)]) == 0
    out = capsys.readouterr().out
    other = _key(theirs)
    assert f"against {other}" in out and "S1 against" in out and "fixed" in out and "broken" in out
    assert str(layout.diagnostics_against_md(_key(mine), "S1", other)) in out
    # by its model key: read from the run directory
    assert main([*base, "--against", other]) == 0
    assert "S1: reused" in capsys.readouterr().out
    # a missing checkpoint, a model key without diagnostics
    assert main([*base, "--against", str(tmp_path / "gone.alignair")]) == 2
    assert "no AlignAIR model file" in capsys.readouterr().err
    assert main([*base, "--against", "0" * 12 + "-" + "1" * 12]) == 2
    assert "no diagnostics of suite S0" in capsys.readouterr().err


# --- the released checkpoint ------------------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("ALIGNAIR_ASSAY_GPU") != "1", reason="set ALIGNAIR_ASSAY_GPU=1"))])
def test_the_released_checkpoint_against_itself(built, tmp_path, device):
    # The file against itself (one model key), and against itself loaded in memory (another key, diagnosed again):
    # both must compare to zero.
    layout = _copy(built, tmp_path)
    checkpoint = Path(MODEL).expanduser()
    in_memory = load_model_checked(checkpoint, truth_reference(layout, ["S1"]), device=device)
    for against in (checkpoint, in_memory):
        result = evaluate(checkpoint, layout, ["S1"], device=device, batch_size=64, max_cases=60, profile=False,
                          against=against)
        summary = result.against["summary"]["suites"]["S1"]
        assert set(summary["deltas"].values()) <= {0.0, None} and summary["significant"] == []
        assert summary["reads"] == {"heads": {"fixed": 0, "broken": 0}, "baseline": {"fixed": 0, "broken": 0}}
