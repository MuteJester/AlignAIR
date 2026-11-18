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
from alignair_assay.report.diagnostics import SUMMARY_SECTIONS  # noqa: E402
from alignair_assay.report.diagnostics_against import AGAINST_SECTIONS  # noqa: E402
from alignair_assay.scoring.pipeline import truth_reference  # noqa: E402
from alignair_assay.stats import summarize, to_dict  # noqa: E402
from alignair_assay.stats.compare import sample  # noqa: E402
from alignair_assay.workflow import diagnose as diagnose_module  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.diagnose import evaluate, read_case_scores, select_cases  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
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
        model, reference_set = make_small_model(REFERENCE, seed=seed)
        save_model(str(path), model, reference_set, dataconfigs=[REFERENCE], training={"steps": 0, "batch_size": 0},
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
    result = evaluate(small_model, layout, ["S1"], **{**FAST, "variants": ("baseline", "soft_decoder")},
                      max_cases=40)
    suite = result.suites["S1"]
    path = layout.diagnostics_suite_cases(result.model_key, "S1")
    # one line per diagnosed case, in diagnosed order: its head metrics, its end-to-end baseline, each variant's
    lines = read_jsonl(path)
    assert [line["case_id"] for line in lines] == [
        case["case_id"] for case in select_cases(read_cases(layout.cases("S1")), 40)]
    assert all(set(line) == {"case_id", "head", "baseline", "variants"} for line in lines)
    assert all(set(line["variants"]) == {"soft_decoder"} for line in lines)
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
    assert summary(None, "soft_decoder") == suite["variants"]["soft_decoder"]["scores"]
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
    assert against["baseline"] != result.model_key and result.model_key == _key(checkpoint)
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
            assert changes["reads"] == {"fixed": 0, "broken": 0, "fixed_examples": [], "broken_examples": [],
                                        "groups": {}}
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
    assert against["baseline"] in section and "| S1 | junction.nt_exact |" in section
    assert f"[S1](S1/against_{against['baseline']}.md)" in section
    assert "Against the baseline" in section and "Δ (this − baseline)" in section
    assert layout.diagnostics_md(result.model_key).read_text(encoding="utf-8") == md
    report = json.loads(layout.diagnostics_json(result.model_key).read_text())
    assert report["against"]["baseline"] == against["baseline"] and report["summary"]["against"]["baseline"]
    for name in ("S0", "S1"):
        detail = layout.diagnostics_against_md(result.model_key, name, against["baseline"]).read_text("utf-8")
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
    # The perturbed model is the baseline: Δ = this model − the baseline, fixed = the baseline wrong, this one right.
    result = evaluate((model, reference_set), layout, ["S1"], **FAST, against=(perturbed, reference_set))
    suite = result.against["suites"]["S1"]
    other = result.against["baseline"]
    assert other != result.model_key
    # A direct recount from the two per-case files, as written (null: not covered).
    mine = {line["case_id"]: line for line in read_jsonl(layout.diagnostics_suite_cases(result.model_key, "S1"))}
    base = {line["case_id"]: line for line in read_jsonl(layout.diagnostics_suite_cases(other, "S1"))}
    assert set(mine) == set(base)
    changed = 0
    for family, field in (("heads", "head"), ("baseline", "baseline")):
        metrics = suite[family]["changes"]["metrics"]
        assert metrics, family
        for metric, found in metrics.items():
            right, wrong = (0.0, 1.0) if lower_is_better(metric) else (1.0, 0.0)
            pairs = [(case, base[case][field][metric], mine[case][field][metric]) for case in mine
                     if mine[case][field] and base[case][field] and mine[case][field].get(metric) is not None
                     and base[case][field].get(metric) is not None]
            fixed = sorted(case for case, was, now in pairs if (was, now) == (wrong, right))
            broken = sorted(case for case, was, now in pairs if (was, now) == (right, wrong))
            assert (found["n_paired"], found["fixed"], found["broken"]) == (len(pairs), len(fixed), len(broken))
            # up to 10 examples, evenly spaced over the sorted ids (not the first strata's)
            assert (found["fixed_examples"], found["broken_examples"]) == (sample(fixed, 10), sample(broken, 10))
            changed += len(fixed) + len(broken)
            assert result.scalars()[f"S1/against/{metric}.fixed"] == len(fixed)
            assert result.scalars()[f"S1/against/{metric}.broken"] == len(broken)
    orientation = suite["heads"]["changes"]["metrics"]["head.orientation.correct"]
    assert orientation["fixed"] > 0 and orientation["broken"] > 0 and changed > 0
    # the reads changed, counted per stratum
    for family in ("heads", "baseline"):
        reads = suite[family]["changes"]["reads"]
        assert set(reads["groups"]) <= {case.split(":")[1] for case in mine}
        assert sum(counts["fixed"] for counts in reads["groups"].values()) == reads["fixed"]
        assert sum(counts["broken"] for counts in reads["groups"].values()) == reads["broken"]
    entry = suite["heads"]["comparison"]["head.orientation.correct"]
    values = [(base[case]["head"]["head.orientation.correct"], mine[case]["head"]["head.orientation.correct"])
              for case in mine if mine[case]["head"]]
    assert entry["mean_b_paired"] == pytest.approx(sum(now for _, now in values) / len(values))  # this model
    assert entry["delta"] == pytest.approx(sum(now - was for was, now in values) / len(values))  # this − baseline
    assert result.scalars()["S1/against/head.orientation.correct.delta"] == pytest.approx(entry["delta"])
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
    options = dict(device="cpu", batch_size=8, variants=("soft_decoder",), profile=False, max_cases=7)
    result = evaluate(small_model, layout, ["S0", "S1"], **options, against=str(checkpoint))
    other = result.against["baseline"]
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


def test_each_models_alignair_code_is_named_and_another_code_flagged(built, tmp_path, small_model,
                                                                     make_small_model):
    layout = _copy(built, tmp_path)
    code = alignair_code()["code_sha256"]
    base = evaluate(make_small_model(REFERENCE, seed=9), layout, ["S0"], **FAST).model_key
    result = evaluate(small_model, layout, ["S0"], **FAST, against=base)
    assert result.against["code"] == {"this": code, "baseline": code}
    assert result.against["summary"]["code"] == {"this": code, "baseline": code, "same": True}
    assert f"Both ran under AlignAIR code `{code[:12]}`." in result.report_md
    # The baseline diagnosed before an AlignAIR edit, kept by its model key: its records name the other code.
    before = f"{base.split('-')[0]}-{'0' * 12}"
    shutil.copytree(layout.diagnostics_dir(base), layout.diagnostics_dir(before))
    record = json.loads(layout.diagnostics_suite_json(before, "S0").read_text())
    record["key"]["model"]["code_sha256"] = "0" * 64
    write_json(layout.diagnostics_suite_json(before, "S0"), record)
    result = evaluate(small_model, layout, ["S0"], **FAST, against=before)
    assert result.against["summary"]["code"] == {"this": code, "baseline": "0" * 64, "same": False}
    assert (f"**Different AlignAIR code**: this model ran under `{code[:12]}`, the baseline under `{'0' * 12}`"
            in result.report_md)
    assert "Different AlignAIR code" in layout.diagnostics_against_md(result.model_key, "S0", before).read_text()


def test_a_report_keeps_only_the_comparison_it_holds(built, tmp_path, small_model, make_small_model):
    # each suite's against_<key>.md is the detail of the last report's comparison: an older one is removed
    layout = _copy(built, tmp_path)
    first = evaluate(make_small_model(REFERENCE, seed=9), layout, ["S0"], **FAST).model_key
    second = evaluate(make_small_model(REFERENCE, seed=8), layout, ["S0"], **FAST).model_key
    mine = evaluate(small_model, layout, ["S0"], **FAST, against=first).model_key

    def details():
        return sorted(path.name for path in layout.diagnostics_suite_dir(mine, "S0").glob("against_*.md"))

    assert details() == [f"against_{first}.md"]
    evaluate(small_model, layout, ["S0"], **FAST, against=second)
    assert details() == [f"against_{second}.md"]
    evaluate(small_model, layout, ["S0"], **FAST)
    assert details() == []


def test_a_comparison_of_other_reads_or_settings_is_refused(built, tmp_path, small_model, make_small_model,
                                                            replays):
    layout = _copy(built, tmp_path)
    other = evaluate(make_small_model(REFERENCE, seed=9), layout, ["S0", "S1"], **FAST, max_cases=10).model_key
    before = len(replays)
    # Other settings: refused before any work, naming the setting.
    for change, field in ((dict(max_cases=5), "max_cases"), (dict(batch_size=8), "batch_size"),
                          (dict(variants=("soft_decoder",)), "variants"), (dict(device="cpu:0"), "device")):
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
    assert f"against the baseline {other}" in out and "Δ = this model − the baseline" in out
    assert f"Both ran under AlignAIR code `{alignair_code()['code_sha256'][:12]}`." in out
    assert f"S1 against {other}" in out and "fixed" in out and "broken" in out
    assert str(layout.diagnostics_against_md(_key(mine), "S1", other)) in out
    # by its model key: read from the run directory
    assert main([*base, "--against", other]) == 0
    assert "S1: reused" in capsys.readouterr().out
    # a missing checkpoint, a model key without diagnostics
    assert main([*base, "--against", str(tmp_path / "gone.alignair")]) == 2
    assert "no AlignAIR model file" in capsys.readouterr().err
    assert main([*base, "--against", "0" * 12 + "-" + "1" * 12]) == 2
    assert "no diagnostics of suite S0" in capsys.readouterr().err


def test_diagnose_takes_a_labelled_checkpoint(built, tmp_path, save_checkpoint, capsys):
    layout = _copy(built, tmp_path)
    mine, theirs = save_checkpoint(tmp_path / "a.alignair", seed=0), save_checkpoint(tmp_path / "b.alignair", seed=1)
    base = ["diagnose", "--run-dir", str(layout.root), "--suites", "S0", "--device", "cpu", "--variants", "baseline",
            "--no-profile"]
    assert main([*base, "--checkpoint", f"cand={mine}", "--against", str(theirs)]) == 0
    capsys.readouterr()
    # the label names the model in the report; the model key is the file's (a label is not part of its identity)
    summary = layout.diagnostics_md(_key(mine)).read_text()
    assert f"- Model `{_key(mine)}`, labelled `cand` ({mine})" in summary
    assert f"This model: `{_key(mine)}`, labelled `cand` ({mine})" in layout.diagnostics_against_md(
        _key(mine), "S0", _key(theirs)).read_text()
    # a path that holds = and names a file is a path; an unlabelled path has no label
    odd = save_checkpoint(tmp_path / "x=y.alignair", seed=0)
    assert main([*base, "--checkpoint", str(odd)]) == 0
    assert ", labelled `" not in layout.diagnostics_md(_key(odd)).read_text()


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


@pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
def test_the_released_checkpoint_with_its_j_end_head_zeroed_is_seen_getting_worse(built, tmp_path):
    # The audit's positive control through the real files: the perturbed model is diagnosed against the released one
    # on the same reads, under its own model key and per-case file, and the comparison sees its J end get worse. On
    # the soft-argmax J end the released model is almost never exact (it stops short of a J that runs to the read
    # end), so a broken head cannot make that metric worse: the control reads the logits' argmax, where it is right.
    from alignair_assay.workflow import audit

    layout = _copy(built, tmp_path)
    device = "cuda" if os.environ.get("ALIGNAIR_ASSAY_GPU") == "1" else "cpu"
    cfg = audit.AuditConfig(sim_run=str(tmp_path / "sim"), real_run=str(tmp_path / "real"),
                            data_root=str(tmp_path / "data"), checkpoint=str(Path(MODEL).expanduser()), device=device,
                            seed=5, null_suites=("S1",), null_scale=SCALE, diagnose_suites=("S1",), batch_size=64,
                            diagnose_variants=("baseline",))
    found = audit.positive_diagnose(cfg, layout, RunLayout(tmp_path / "positive"))
    evidence = found["evidence"]
    assert found["status"] == "pass", found["reason"]
    assert evidence["model_key"] != evidence["baseline"]
    assert evidence["rates"]["baseline"] > 0.5 and evidence["rates"]["this"] < evidence["rates"]["baseline"]
    # the zeroed head reached the replay: the J end's signed error differs on every read with a scored J end, in the
    # two per-case files, and the control rests on all of those reads
    mine = read_case_scores(layout, evidence["model_key"], "S1")
    base = read_case_scores(layout, evidence["baseline"], "S1")
    metric = "head.j.end_err.soft"
    errors = [(base[case]["head"][metric], mine[case]["head"][metric]) for case in mine
              if base[case]["head"] and metric in base[case]["head"]]
    assert found["n"] == len(errors) > 90 and all(was != now for was, now in errors)
