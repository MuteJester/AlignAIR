"""The in-process evaluator (``workflow.diagnose``) and the ``diagnose`` command: diagnostics of a model on built
dev suites, cached per suite and never served stale.

Everything runs on the CPU with small random-init models (``conftest.make_small_model``) on S0 and S1 built at
scale 0.02 (2 and 100 reads). The test of the released checkpoint runs only when ``ALIGNAIR_ASSAY_MODEL`` names a
``.alignair`` model file, on the GPU when ``ALIGNAIR_ASSAY_GPU=1``.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.model_file import save_model  # noqa: E402

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS, SINGLE_SWAPS, VARIANTS  # noqa: E402
from alignair_assay.diagnostics import identity as identity_module  # noqa: E402
from alignair_assay.diagnostics import pin as pin_module  # noqa: E402
from alignair_assay.diagnostics.identity import alignair_code, weights_sha256  # noqa: E402
from alignair_assay.report.diagnostics import SUITE_SECTIONS, SUMMARY_SECTIONS  # noqa: E402
from alignair_assay.runners.alignair import local_code_identity  # noqa: E402
from alignair_assay.workflow import diagnose as diagnose_module  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.diagnose import (  # noqa: E402
    DiagnosticsResult,
    UnverifiedReplayWarning,
    cluster_counts,
    evaluate,
    select_cases,
)

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
REFERENCE = "HUMAN_IGH_OGRDB"
SCALE = 0.02
BATCH = 16
FAST = dict(device="cpu", batch_size=BATCH, variants=("baseline",), profile=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A run directory with S0 (2 cases) and S1 (100 cases, with donors) built on the dev split at scale 0.02."""
    layout = RunLayout(tmp_path_factory.mktemp("evaluate_run"))
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


def _cached(result: DiagnosticsResult, suite: str) -> dict:
    return {key: value for key, value in result.suites[suite].items() if key != "reused"}


# --- end to end -------------------------------------------------------------------------------------------------

class Forward(torch.nn.Module):
    """An orientation head that calls every read forward, so the forward reads (one of S0's two, about half of S1's)
    are in the truth's frame and their boundaries are scored."""

    def forward(self, emb, mask):
        return torch.nn.functional.one_hot(torch.zeros_like(mask[:, 0], dtype=torch.long), 4).float()


def _calls_every_d_callable(model) -> None:
    """Make the chip's D presence head call every read's D callable (a forward hook on its core's outputs)."""
    def hook(module, args, out):
        logits = torch.zeros_like(out["d_presence_logits"])
        logits[:, 2] = 5.0
        return {**out, "d_presence_logits": logits}
    model.core.register_forward_hook(hook)


def test_a_d_presence_head_is_diagnosed_with_its_confusion(built, make_small_model):
    """A model with a D presence head gets its head metrics and its truth × head confusion in each suite's result;
    a head that calls every D callable gates none away, and its accuracy is the share of callable Ds."""
    model, reference_set = make_small_model(REFERENCE, seed=0)
    _calls_every_d_callable(model)
    result = evaluate((model, reference_set), built, ["S0"], device="cpu", batch_size=32, variants=())
    suite = result.suites["S0"]
    confusion = suite["d_presence"]
    assert confusion["classes"] == ["absent", "short", "callable"] and confusion["callable_gated"] == 0
    assert all(row[:2] == [0, 0] for row in confusion["counts"])           # nothing called absent or short
    assert confusion["n"] == suite["heads"]["head.d_presence.correct"]["n"]
    assert suite["heads"]["head.d_presence.correct"]["mean"] == pytest.approx(confusion["counts"][2][2] / confusion["n"])
    assert suite["heads"]["head.d_presence.callable_gated"]["mean"] == 0


def test_end_to_end_report_renders_every_section(built, make_small_model):
    # the head's orientation alone (no bank vote), so the misoriented reads are the head's wrong calls
    model, reference_set = make_small_model(REFERENCE, seed=0, bank_orientation=False)
    model.orientation_head = Forward()
    result = evaluate((model, reference_set), built, ["S0", "S1"], device="cpu", batch_size=32, profile=True)
    assert isinstance(result, DiagnosticsResult)
    assert list(result.suites) == ["S0", "S1"]
    for name, suite in result.suites.items():
        assert suite["reused"] is False and suite["suite"] == name
        assert list(suite["variants"]) == list(DEFAULT_VARIANTS)
        assert [step["to"] for step in suite["contributions"]] == list(VARIANTS[1:])
        assert [swap["to"] for swap in suite["swaps"]] == list(SINGLE_SWAPS)
        assert set(suite["calibration"]) == {"v", "d", "j"}
        assert {"head.v.start_exact.soft", "head.v.start_exact.argmax", "head.orientation.correct"} <= set(
            suite["heads"])
        assert suite["d_presence"] is not None and "head.d_presence.correct" in suite["heads"]   # the chip's head
        # The suite's primary metrics lead the compared family.
        primary = {"S0": ["v.top1_in_set", "j.top1_in_set", "junction.nt_exact"],
                   "S1": ["clone_key.exact", "junction.nt_exact"]}[name]
        assert suite["metrics"][:len(primary)] == primary
        for step in suite["contributions"] + suite["swaps"]:
            for entry in step["metrics"].values():
                assert {"mean_from_paired", "mean_to_paired", "delta", "verdict"} <= set(entry)
    s1 = result.suites["S1"]
    cropped = sum(len(case["sequence"]) > 576 for case in read_cases(built.cases("S1")))
    wrong = round(s1["reads"]["scored"] * (1 - s1["heads"]["head.orientation.correct"]["mean"]))
    assert s1["reads"] == {"built": 100, "diagnosed": 100, "scored": s1["reads"]["scored"],
                           "excluded": 100 - s1["reads"]["scored"], "cropped": cropped, "misoriented": wrong}
    assert s1["heads"]["head.orientation.correct"]["n"] == s1["reads"]["scored"]
    assert s1["heads"]["head.misoriented"]["n"] == s1["reads"]["scored"]
    # S1 has donors: its variants are scored against each read's donor genotype, as all scores the suite.
    assert "v.outside_genotype" in s1["variants"]["baseline"]["scores"]

    assert result.profile is not None and result.profile["device"] == "cpu" and result.profile["reads"] == 102

    # diagnostics.md is the one-page summary: its sections in order, with real rows per suite.
    md = result.report_md
    positions = [md.index(f"\n## {title}\n") for title in SUMMARY_SECTIONS]
    assert positions == sorted(positions)
    assert len(md.splitlines()) < 120
    boundaries = md[md.index("\n## Boundaries: decoder vs network\n"):md.index("\n## Metadata heads\n")]
    for name in ("S0", "S1"):
        # The head table's rows exist under the names heads.py gives them: a soft→argmax cell for an edge.
        assert re.search(rf"^\| {name} \| \d+→\d+ \|", boundaries, re.MULTILINE), name
        assert f"[{name}]({name}/report.md)" in md
    errors = md[md.index("\n### junction.nt_exact\n"):md.index("\n## Calls\n")]
    assert re.search(r"^\| S1 \| \d+\.\d% \(n=\d+, \d+%\) \|", errors, re.MULTILINE)
    assert "[profile.md](profile.md)" in md and "reads/s on cpu" in md
    assert result.summary["flags"][-1].startswith("Holm is per comparison")
    scalars = result.scalars()
    assert isinstance(scalars["S1/baseline/junction.nt_exact"], float)
    assert scalars["S1/baseline/junction.nt_exact"] == s1["variants"]["baseline"]["scores"]["junction.nt_exact"][
        "mean"]
    assert scalars["S0/heads/head.v.start_exact.soft"] == result.suites["S0"]["heads"]["head.v.start_exact.soft"][
        "mean"]
    assert "speed/reads_per_second" in scalars and all(isinstance(value, float) for value in scalars.values())

    # Written beside the per-suite caches, under the model key: the summary, each suite's detail and the profile.
    assert built.diagnostics_md(result.model_key).read_text(encoding="utf-8") == md
    report = json.loads(built.diagnostics_json(result.model_key).read_text(encoding="utf-8"))
    assert report["model"]["model_key"] == result.model_key and report["model"]["source"] == "in-memory model"
    assert report["suites"]["S1"]["variants"] == s1["variants"] and report["summary"] == result.summary
    for name in ("S0", "S1"):
        assert json.loads(built.diagnostics_suite_json(result.model_key, name).read_text())["suite"] == name
        detail = built.diagnostics_suite_md(result.model_key, name).read_text(encoding="utf-8")
        positions = [detail.index(f"\n## {title}\n") for title in SUITE_SECTIONS]
        assert positions == sorted(positions) and "(../diagnostics.md)" in detail
    assert built.diagnostics_profile_json(result.model_key).is_file()
    assert "| Module |" in built.diagnostics_profile_md(result.model_key).read_text(encoding="utf-8")
    assert not list(built.diagnostics_dir(result.model_key).rglob("*.tmp"))


def test_the_model_key_is_its_weights_and_the_alignair_code(built, tmp_path, save_checkpoint):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    result = evaluate(checkpoint, layout, ["S0"], **FAST)
    code = alignair_code()
    assert result.model_key == f"{sha256_file(checkpoint)[:12]}-{code['code_sha256'][:12]}"
    # The code this process imported: in this checkout, the code the alignair-local runner fingerprints. The modules
    # the stages import lazily are loaded with the evaluator, so the digest is of the code that runs.
    assert code == local_code_identity()
    assert {"alignair.predict.heuristic_matcher", "alignair.model_file"} <= set(sys.modules)
    report = json.loads(layout.diagnostics_json(result.model_key).read_text())
    assert report["model"]["weights_sha256"] == weights_sha256(checkpoint) == sha256_file(checkpoint)
    assert report["model"]["code_sha256"] == code["code_sha256"]
    assert report["model"]["source"] == str(checkpoint)


# --- an in-memory model from a training loop --------------------------------------------------------------------

def _training_model(make_small_model):
    """A small model in training mode with an Adam optimizer that has taken one step."""
    model, reference_set = make_small_model(REFERENCE, seed=7)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    _step(model, optimizer)
    return model, reference_set, optimizer


def _step(model, optimizer):
    device = next(model.parameters()).device
    tokens = torch.randint(1, 5, (2, model.cfg.max_seq_length), device=device)
    optimizer.zero_grad()
    model({"tokenized_sequence": tokens})["v_allele"].sum().backward()
    optimizer.step()


def _spy_moves(model):
    """Record the device of every ``model.to`` call (the move still happens); undo with ``del model.to``."""
    moves = []
    real = type(model).to

    def to(*args, **kwargs):
        moves.append(torch.device(args[0]) if args else kwargs.get("device"))
        return real(model, *args, **kwargs)

    model.to = to
    return moves


def test_an_in_memory_model_keeps_its_device_mode_and_optimizer(built, tmp_path, make_small_model):
    layout = _copy(built, tmp_path)
    model, reference_set, optimizer = _training_model(make_small_model)
    # No device: the model's own. No profile for an in-memory model unless asked.
    result = evaluate((model, reference_set), layout, ["S0"], batch_size=BATCH, variants=("baseline",))
    report = json.loads(layout.diagnostics_json(result.model_key).read_text())
    assert report["config"]["device"] == "cpu" and result.profile is None and report["config"]["profile"] is False
    evaluate((model, reference_set), layout, ["S0"], device="cpu", batch_size=BATCH, variants=("baseline",),
             profile=True)
    assert model.training and {p.device.type for p in model.parameters()} == {"cpu"}
    _step(model, optimizer)  # the training loop carries on


def test_the_model_is_put_back_even_when_a_suite_fails(built, tmp_path, make_small_model, monkeypatch):
    layout = _copy(built, tmp_path)
    model, reference_set, _ = _training_model(make_small_model)

    def broken(*args, **kwargs):
        raise RuntimeError("replay failed")

    monkeypatch.setattr(diagnose_module, "replay", broken)
    moves = _spy_moves(model)
    try:
        with pytest.raises(RuntimeError, match="replay failed"):
            evaluate((model, reference_set), layout, ["S0"], **FAST)
        # Moved to the device asked for, then back to its own after the failure.
        assert moves == [torch.device("cpu"), torch.device("cpu")]
    finally:
        del model.to
    assert model.training


def test_a_wrapped_model_is_diagnosed_as_the_model_it_wraps(built, tmp_path, make_small_model):
    layout = _copy(built, tmp_path)
    model, reference_set = make_small_model(REFERENCE, seed=8)
    plain = evaluate((model, reference_set), layout, ["S0"], **FAST)
    # torch.compile wraps lazily (nothing is compiled until the wrapper is called, which the replay never does).
    wrapped = evaluate((torch.compile(model), reference_set), layout, ["S0"], **FAST)
    assert wrapped.model_key == plain.model_key and wrapped.suites["S0"]["reused"] is True


@pytest.mark.skipif(os.environ.get("ALIGNAIR_ASSAY_GPU") != "1", reason="set ALIGNAIR_ASSAY_GPU=1")
def test_a_cpu_model_diagnosed_on_cuda_is_put_back_on_the_cpu(built, tmp_path, make_small_model):
    layout = _copy(built, tmp_path)
    model, reference_set, optimizer = _training_model(make_small_model)
    result = evaluate((model, reference_set), layout, ["S0"], device="cuda", batch_size=BATCH, profile=True)
    assert result.profile["device"] == "cuda"
    assert model.training and {p.device.type for p in model.parameters()} == {"cpu"}
    _step(model, optimizer)


def test_the_device_and_profile_defaults(built, tmp_path, save_checkpoint, monkeypatch, capsys):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    # A model file is profiled unless told otherwise; an in-memory model is not.
    profiled = evaluate(checkpoint, layout, ["S0"], device="cpu", batch_size=BATCH, variants=(), max_cases=2)
    assert profiled.profile is not None and profiled.profile["reads"] == 2
    # A model file runs on CUDA unless told otherwise; an API caller is told the keyword to change it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError) as error:
        evaluate(checkpoint, layout, ["S0"], variants=())
    assert 'device="cpu"' in str(error.value) and "--device" not in str(error.value)
    # The command names its own option.
    assert main(["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(checkpoint), "--suites", "S0",
                 "--no-profile"]) == 2
    err = capsys.readouterr().err
    assert "--device cpu" in err and 'device="cpu"' not in err


def test_in_memory_model_key_follows_its_weights(small_model):
    model, reference_set = small_model
    first = weights_sha256((model, reference_set))
    assert weights_sha256((model, reference_set)) == first
    parameter = next(model.parameters())
    with torch.no_grad():
        parameter.view(-1)[0] += 1.0
    try:
        assert weights_sha256((model, reference_set)) != first
    finally:
        with torch.no_grad():
            parameter.view(-1)[0] -= 1.0
    assert weights_sha256((model, reference_set)) == first


# --- resume -----------------------------------------------------------------------------------------------------

def test_resume_reuses_a_cached_suite_until_the_checkpoint_bytes_change(built, tmp_path, save_checkpoint, replays):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    first = evaluate(checkpoint, layout, ["S0"], **FAST)
    assert _suites(replays) == ["S0"] and first.suites["S0"]["reused"] is False

    second = evaluate(checkpoint, layout, ["S0"], **FAST)
    assert _suites(replays) == ["S0"]  # nothing replayed
    assert second.model_key == first.model_key and second.suites["S0"]["reused"] is True
    assert _cached(second, "S0") == _cached(first, "S0")

    # New checkpoint bytes: another model key, and the suite is computed again.
    save_checkpoint(checkpoint, seed=1)
    third = evaluate(checkpoint, layout, ["S0"], **FAST)
    assert third.model_key != first.model_key and third.model_key.split("-")[1] == first.model_key.split("-")[1]
    assert _suites(replays) == ["S0", "S0"] and third.suites["S0"]["reused"] is False
    assert layout.diagnostics_suite_json(first.model_key, "S0").is_file()  # the old model's diagnostics stay


def test_cached_diagnostics_are_never_served_stale(built, tmp_path, save_checkpoint, replays, monkeypatch):
    # One record per suite and model key: the latest configuration's. Each change below must recompute it; each
    # unchanged call must reuse it.
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    base = evaluate(checkpoint, layout, ["S0", "S1"], **FAST, max_cases=5)
    assert _suites(replays) == ["S0", "S1"]

    def computed(**kwargs) -> bool:
        before = len(replays)
        result = evaluate(checkpoint, layout, ["S0"], **{**FAST, "max_cases": 5, **kwargs})
        assert result.suites["S0"]["reused"] is (len(replays) == before)
        return len(replays) > before

    assert not computed()
    # Another variant set, case count, batch size or device: computed again.
    for change in (dict(variants=("baseline", "soft_decoder")), dict(max_cases=1), dict(batch_size=8),
                   dict(device="cpu:0")):
        assert computed(**change) and not computed(**change), change
    assert computed() and not computed()

    # A changed AlignAIR code digest: another model key.
    code = alignair_code()
    monkeypatch.setattr(diagnose_module, "alignair_code", lambda: {**code, "code_sha256": "0" * 64})
    assert computed()
    assert evaluate(checkpoint, layout, ["S0"], **FAST, max_cases=5).model_key == (
        base.model_key.split("-")[0] + "-" + "0" * 12)
    # Another torch, or another assay scoring code: the same model key, computed again.
    monkeypatch.setattr(diagnose_module, "alignair_code", lambda: {**code, "torch_version": "0.0"})
    assert computed() and not computed()
    monkeypatch.setattr(diagnose_module, "alignair_code", lambda: dict(code))
    assert computed() and not computed()
    monkeypatch.setattr(diagnose_module, "assay_code_sha256", lambda: "1" * 64)
    assert computed() and not computed()
    monkeypatch.undo()
    monkeypatch.setattr(diagnose_module, "replay", _counting(replays))
    assert computed() and not computed()

    # A changed suite build (S0 rebuilt with another seed): computed again under the same model key; S1 is kept.
    build_suite("S0", layout, reference_name=REFERENCE, split="dev", seed=6, scale=SCALE, workers=1)
    before = len(replays)
    result = evaluate(checkpoint, layout, ["S0", "S1"], **FAST, max_cases=5)
    assert _suites(replays[before:]) == ["S0"] and result.model_key == base.model_key
    assert result.suites["S1"]["reused"] is True

    # A cached record whose key does not match in full is never served, whatever it holds.
    path = layout.diagnostics_suite_json(base.model_key, "S1")
    record = json.loads(path.read_text())
    record["key"]["suite"]["truth_sha256"] = "0" * 64
    record["heads"] = {}
    path.write_text(json.dumps(record))
    before = len(replays)
    result = evaluate(checkpoint, layout, ["S1"], **FAST, max_cases=5)
    assert _suites(replays[before:]) == ["S1"] and result.suites["S1"]["heads"]
    # An unreadable record likewise.
    path.write_text("{not json")
    before = len(replays)
    evaluate(checkpoint, layout, ["S1"], **FAST, max_cases=5)
    assert _suites(replays[before:]) == ["S1"]


def _counting(replays):
    """A replay that records its read ids in ``replays`` (``monkeypatch.undo`` removed the fixture's)."""
    original = diagnose_module.replay

    def spy(model, reference_set, sequences, ids, **kwargs):
        replays.append(list(ids))
        return original(model, reference_set, sequences, ids, **kwargs)

    return spy


def test_a_checkpoint_rewritten_while_it_loads_is_refused(built, tmp_path, save_checkpoint, monkeypatch):
    # A training job saving over the file between the hash and the load: the loaded weights may not be the hashed
    # ones, so nothing may be computed or cached under that key.
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    original = diagnose_module.load_model_checked

    def load_while_training_saves(path, reference, *, device):
        loaded = original(path, reference, device=device)
        save_checkpoint(checkpoint, seed=1)
        return loaded

    monkeypatch.setattr(diagnose_module, "load_model_checked", load_while_training_saves)
    with pytest.raises(ValueError, match="changed while it was being loaded"):
        evaluate(checkpoint, layout, ["S0"], **FAST)
    assert not layout.diagnostics_root.exists()


def test_the_profile_is_cached_with_its_own_key(built, tmp_path, save_checkpoint, monkeypatch):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    profiles = []
    original = diagnose_module.profile_model

    def spy(*args, **kwargs):
        profiles.append(kwargs["batch_size"])
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnose_module, "profile_model", spy)
    options = {**FAST, "profile": True, "max_cases": 2}
    first = evaluate(checkpoint, layout, ["S0"], **options)
    assert profiles == [BATCH] and first.profile["reads"] == 2 and first.profile["reused"] is False
    second = evaluate(checkpoint, layout, ["S0"], **options)
    assert profiles == [BATCH] and second.profile["reused"] is True
    assert {k: v for k, v in second.profile.items() if k != "reused"} == {
        k: v for k, v in first.profile.items() if k != "reused"}
    evaluate(checkpoint, layout, ["S0"], **{**options, "batch_size": 4})
    assert profiles == [BATCH, 4]
    # The replay it times is assay code: another assay code digest measures again.
    monkeypatch.setattr(diagnose_module, "assay_code_sha256", lambda: "1" * 64)
    evaluate(checkpoint, layout, ["S0"], **{**options, "batch_size": 4})
    assert profiles == [BATCH, 4, 4]


def test_the_profile_times_exactly_the_diagnosed_reads(built, tmp_path, small_model, monkeypatch, replays):
    layout = _copy(built, tmp_path)
    # reads.fasta is a copy of the cases' reads for the tools; the profile must not depend on it.
    layout.reads("S1").write_text(">other\nACGTACGTACGT\n", encoding="utf-8")
    timed = []
    monkeypatch.setattr(diagnose_module, "profile_model",
                        lambda model, reference_set, sequences, **kwargs: timed.append(list(sequences)) or {
                            "device": "cpu", "batch_size": BATCH, "reads": len(sequences), "repeats": 1,
                            "reads_per_second": 1.0, "seconds_per_run": 1.0, "modules": {}, "stages": {}})
    evaluate(small_model, layout, ["S0", "S1"], **{**FAST, "profile": True}, max_cases=7)
    diagnosed = [case["sequence"] for name in ("S0", "S1")
                 for case in select_cases(read_cases(layout.cases(name)), 7)]
    assert timed == [diagnosed]
    assert [case_id for ids in replays for case_id in ids] == [
        case["case_id"] for name in ("S0", "S1") for case in select_cases(read_cases(layout.cases(name)), 7)]


def test_a_bare_suite_name_is_one_suite(built, tmp_path, small_model):
    result = evaluate(small_model, _copy(built, tmp_path), "S0", **FAST)
    assert list(result.suites) == ["S0"]


# --- the replay pin ---------------------------------------------------------------------------------------------

CHANGED = {"alignair.predict.pipeline.predict": "f" * 64}


def test_an_unverified_replay_is_refused_unless_allowed(built, tmp_path, small_model, monkeypatch, replays):
    layout = _copy(built, tmp_path)
    verified = evaluate(small_model, layout, ["S0"], **FAST)
    assert verified.replay_verified is True and verified.replay_changes == {}
    assert verified.suites["S0"]["replay_verified"] is True and verified.suites["S0"]["replay_changes"] == {}

    monkeypatch.setattr(diagnose_module, "replay_pin", lambda: ("1" * 64, dict(CHANGED)))
    before = len(replays)
    with pytest.raises(ValueError) as error:
        evaluate(small_model, layout, ["S0"], **FAST)
    for part in ("alignair.predict.pipeline.predict", "f" * 64, "update diagnostics/replay.py", "test_replay.py",
                 "PINNED", "allow_unverified_replay=True"):
        assert part in str(error.value), part
    assert len(replays) == before

    with pytest.warns(UnverifiedReplayWarning, match="alignair.predict.pipeline.predict"):
        unverified = evaluate(small_model, layout, ["S0"], **FAST, allow_unverified_replay=True)
    assert unverified.replay_verified is False and unverified.replay_changes == CHANGED
    # Never served from the verified record: the verification and the product code digest are in the key.
    suite = unverified.suites["S0"]
    assert suite["reused"] is False and len(replays) == before + 1
    assert suite["replay_verified"] is False and suite["replay_changes"] == CHANGED
    assert suite["key"]["replay"] == {"verified": False, "sha256": "1" * 64}
    record = json.loads(layout.diagnostics_suite_json(unverified.model_key, "S0").read_text())
    assert record["replay_verified"] is False and record["replay_changes"] == CHANGED
    flags = unverified.summary["flags"]
    assert flags[0].startswith("Unverified replay") and "alignair.predict.pipeline.predict" in flags[0]
    assert f"- {flags[0]}" in unverified.report_md
    detail = layout.diagnostics_suite_md(unverified.model_key, "S0").read_text(encoding="utf-8")
    assert "**Unverified replay.**" in detail


def test_the_replay_pin_is_taken_at_import_beside_the_code_digest(built, tmp_path, small_model, monkeypatch):
    # The pin and the model key's code digest both describe the code this process imported: the pin is never taken
    # again, so a product edit after the import cannot make the two describe different code.
    layout = _copy(built, tmp_path)
    pinned = identity_module.replay_pin()
    assert pinned == (pin_module.replay_digest(), pin_module.pin_changes())
    assert diagnose_module.replay_pin is identity_module.replay_pin

    def taken_again(function):
        raise AssertionError("the replay pin was taken after the import")

    monkeypatch.setattr(pin_module, "code_digest", taken_again)
    for _ in range(2):
        result = evaluate(small_model, layout, ["S0"], **FAST)
        assert result.suites["S0"]["key"]["replay"] == {"verified": not pinned[1], "sha256": pinned[0]}
    assert identity_module.replay_pin() == pinned


def test_diagnose_command_refuses_an_unverified_replay_unless_allowed(built, tmp_path, save_checkpoint, monkeypatch,
                                                                       capsys):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    monkeypatch.setattr(diagnose_module, "replay_pin", lambda: ("1" * 64, dict(CHANGED)))
    argv = ["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(checkpoint), "--suites", "S0",
            "--device", "cpu", "--variants", "baseline", "--no-profile", "--max-cases", "2"]
    assert main(argv) == 2
    assert "--allow-unverified-replay" in capsys.readouterr().err
    with pytest.warns(UnverifiedReplayWarning):
        assert main([*argv, "--allow-unverified-replay"]) == 0
    captured = capsys.readouterr()
    assert "unverified" in captured.err and "alignair.predict.pipeline.predict" in captured.err


# --- refusals and subsets ---------------------------------------------------------------------------------------

def test_the_test_split_is_refused(tmp_path, small_model):
    layout = RunLayout(tmp_path / "run")
    build_suite("S0", layout, reference_name=REFERENCE, split="test", seed=5, scale=SCALE, workers=1)
    with pytest.raises(ValueError, match="S0 was built on the test split.*locked"):
        evaluate(small_model, layout, ["S0"], **FAST)
    with pytest.raises(ValueError, match="suite S1 is not built"):
        evaluate(small_model, layout, ["S1"], **FAST)
    assert not layout.diagnostics_root.exists()


def test_a_model_on_another_reference_is_refused(built, tmp_path, make_small_model):
    layout = _copy(built, tmp_path)
    with pytest.raises(ValueError, match=r"not the truth reference.*first difference: IGHD"):
        evaluate(make_small_model("HUMAN_IGK_OGRDB"), layout, ["S0"], **FAST)
    assert not layout.diagnostics_root.exists()


def test_bad_arguments_are_refused_before_anything_runs(built, tmp_path, small_model):
    layout = _copy(built, tmp_path)
    for kwargs, message in ((dict(variants=("oracle",)), "unknown variant 'oracle'"),
                            (dict(max_cases=0), "max_cases"), (dict(batch_size=0), "batch_size")):
        with pytest.raises(ValueError, match=message):
            evaluate(small_model, layout, ["S0"], **{**FAST, **kwargs})
    with pytest.raises(ValueError, match="unknown suite"):
        evaluate(small_model, layout, ["S99"], **FAST)
    with pytest.raises(ValueError, match="no suite"):
        evaluate(small_model, layout, [], **FAST)
    with pytest.raises(FileNotFoundError, match="no AlignAIR model file"):
        evaluate(tmp_path / "missing.alignair", layout, ["S0"], **FAST)
    assert not layout.diagnostics_root.exists()


def test_max_cases_takes_an_evenly_spaced_subset(built, tmp_path, small_model, replays):
    layout = _copy(built, tmp_path)
    cases = read_cases(layout.cases("S1"))
    assert len(cases) == 100
    result = evaluate(small_model, layout, ["S1"], **FAST, max_cases=7)
    # The speed protocol's rule: positions floor(i * N / n).
    assert replays[-1] == [cases[i * 100 // 7]["case_id"] for i in range(7)] == [
        case["case_id"] for case in select_cases(cases, 7)]
    reads = result.suites["S1"]["reads"]
    assert (reads["built"], reads["diagnosed"]) == (100, 7)
    assert result.suites["S1"]["variants"]["baseline"]["scores"]["v.top1_in_set"]["n"] <= 7
    # Asking for more than the suite has diagnoses every case.
    assert select_cases(cases, 1000) == cases and select_cases(cases, None) == cases
    evaluate(small_model, layout, ["S1"], **FAST, max_cases=1000)
    assert replays[-1] == [case["case_id"] for case in cases]


def test_the_baseline_is_always_scored_with_another_variant(built, tmp_path, small_model):
    layout = _copy(built, tmp_path)
    result = evaluate(small_model, layout, ["S0"], **{**FAST, "variants": ("soft_decoder",)})
    assert list(result.suites["S0"]["variants"]) == ["baseline", "soft_decoder"]
    assert [swap["to"] for swap in result.suites["S0"]["swaps"]] == ["soft_decoder"]
    # No variant at all: the heads, the calibration and the profile only.
    result = evaluate(small_model, layout, ["S0"], **{**FAST, "variants": ()})
    suite = result.suites["S0"]
    assert suite["variants"] == {} and suite["contributions"] == [] and suite["heads"]


def test_comparisons_are_clustered_by_the_suites_groups(built, tmp_path, small_model, monkeypatch):
    # No built suite here has clones or pairs, so the S1 reads are clustered by donor: the evaluator must build the
    # clusters from the suite's truth once and pass them to every comparison.
    layout = _copy(built, tmp_path)
    plain = evaluate(small_model, layout, ["S1"], **{**FAST, "variants": ("soft_decoder",)})
    assert plain.suites["S1"]["clusters"] == {"keys": [], "clusters": 0, "cases": 0}
    entry = plain.suites["S1"]["swaps"][0]["metrics"]["v.top1_in_set"]
    assert entry["n_clusters"] == entry["n_paired"]

    seen = []

    def by_donor(truths):
        # ids in ``cluster_ids``' form, across strata: each S1 stratum has a donor once, so per stratum none pairs up
        seen.append(len(truths))
        return {t.case_id: f"all/subject:{t.groups['subject']}" for t in truths if "subject" in t.groups}

    monkeypatch.setattr(diagnose_module, "cluster_ids", by_donor)
    monkeypatch.setattr(diagnose_module, "CLUSTER_KEYS", ("subject",))
    result = evaluate(small_model, _copy(built, tmp_path / "again"), ["S1"],
                      **{**FAST, "variants": ("true_orientation", "soft_decoder")})
    assert seen == [100]
    suite = result.suites["S1"]
    donors = len({case["groups"]["subject"] for case in read_cases(built.cases("S1"))})
    assert suite["clusters"] == {"keys": ["subject"], "clusters": donors, "cases": 100}
    for comparison in suite["contributions"] + suite["swaps"]:
        for entry in comparison["metrics"].values():
            assert entry["n_clusters"] <= donors < entry["n_paired"] or entry["n_paired"] == 0
            assert entry["test"] in ("too_few_clusters", "too_few_pairs", "clustered_mcnemar", "clustered_z")


def test_a_suite_of_singleton_clusters_is_not_cluster_aware(built, tmp_path, small_model, monkeypatch):
    # S11 with ``max_cases`` keeping only its ``:fwd`` reads: every read has a pair id, but no pair has two reads.
    monkeypatch.setattr(diagnose_module, "cluster_ids", lambda truths: {t.case_id: f"pair:{t.case_id}" for t in truths})
    monkeypatch.setattr(diagnose_module, "CLUSTER_KEYS", ("subject",))
    layout = _copy(built, tmp_path)
    result = evaluate(small_model, layout, ["S1"], **{**FAST, "variants": ("soft_decoder",)})
    assert result.suites["S1"]["clusters"] == {"keys": [], "clusters": 0, "cases": 0}
    assert result.summary["clustered"] == {}
    assert "- Cluster-aware comparisons: none" in result.report_md
    detail = layout.diagnostics_suite_md(result.model_key, "S1").read_text(encoding="utf-8")
    assert re.search(r"^\| 100 \| 100 \|.*\| none \|$", detail, re.MULTILINE), detail


def test_only_clusters_of_two_or_more_reads_count():
    def truth(case_id, **groups):
        return SimpleNamespace(case_id=case_id, groups=groups)

    truths = [truth("a:fwd", pair="a"), truth("a:rev", pair="a"), truth("b:fwd", pair="b"), truth("c", clone="7"),
              truth("d", clone="7", pair="d"), truth("e")]
    clusters = {"a:fwd": "S/pair:a", "a:rev": "S/pair:a", "b:fwd": "S/pair:b", "c": "S/clone:7", "d": "S/clone:7"}
    assert cluster_counts(truths, clusters) == {"keys": ["clone", "pair"], "clusters": 2, "cases": 4}
    # ``d`` has a pair id but is clustered by its clone (``cluster_ids`` takes the first key a read has): the keys
    # are those of the cluster ids, not every key a member read has.
    assert cluster_counts(truths[3:5], {"c": "S/clone:7", "d": "S/clone:7"}) == {
        "keys": ["clone"], "clusters": 1, "cases": 2}
    assert cluster_counts(truths[2:3], {"b:fwd": "S/pair:b"}) == {"keys": [], "clusters": 0, "cases": 0}
    assert cluster_counts([], {}) == {"keys": [], "clusters": 0, "cases": 0}


# --- the diagnose command ---------------------------------------------------------------------------------------

def test_diagnose_command_writes_the_report_and_reuses_it(built, tmp_path, save_checkpoint, capsys):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    argv = ["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(checkpoint), "--suites", "S0",
            "--device", "cpu", "--batch-size", "16", "--variants", "soft_decoder", "--no-profile",
            "--max-cases", "2"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    (key,) = [path.name for path in layout.diagnostics_root.iterdir()]
    report = json.loads(layout.diagnostics_json(key).read_text())
    assert report["config"]["variants"] == ["baseline", "soft_decoder"]
    assert (report["config"]["max_cases"], report["config"]["batch_size"], report["profile"]) == (2, 16, None)
    assert layout.diagnostics_md(key).read_text().startswith(f"# AlignAIR diagnostics: model {key}")
    assert "S0: computed 2 reads (of 2)" in out and f"report: {layout.diagnostics_md(key)}" in out

    assert main(argv) == 0
    assert "S0: reused" in capsys.readouterr().out


def test_diagnose_command_input_errors_exit_2(built, tmp_path, save_checkpoint, capsys):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    base = ["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(checkpoint), "--device", "cpu",
            "--no-profile"]
    assert main([*base, "--suites", "S0", "--variants", "oracle"]) == 2
    assert "unknown variant 'oracle'" in capsys.readouterr().err
    assert main([*base, "--suites", "S5"]) == 2
    assert "suite S5 is not built" in capsys.readouterr().err
    test_layout = RunLayout(tmp_path / "test_run")
    build_suite("S0", test_layout, reference_name=REFERENCE, split="test", seed=5, scale=SCALE, workers=1)
    assert main([*base[:2], str(test_layout.root), *base[3:], "--suites", "S0"]) == 2
    assert "test split" in capsys.readouterr().err
    assert not layout.diagnostics_root.exists() and not test_layout.diagnostics_root.exists()


def test_diagnose_command_warns_about_a_stale_build(built, tmp_path, save_checkpoint, capsys):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    record = json.loads(layout.build_json("S0").read_text())
    record["identity"]["truth_source"] = "edited"
    layout.build_json("S0").write_text(json.dumps(record))
    assert main(["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(checkpoint), "--suites", "S0",
                 "--device", "cpu", "--variants", "baseline", "--no-profile"]) == 0
    err = capsys.readouterr().err
    assert "warning: S0's build is not current (changed: truth_source)" in err
    (key,) = [path.name for path in layout.diagnostics_root.iterdir()]
    summary = layout.diagnostics_md(key).read_text(encoding="utf-8")
    assert "Stale builds" in summary and "S0 (truth_source)" in summary


# --- the released checkpoint ------------------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("ALIGNAIR_ASSAY_GPU") != "1", reason="set ALIGNAIR_ASSAY_GPU=1"))])
def test_the_released_checkpoint(built, tmp_path, device):
    layout = _copy(built, tmp_path)
    checkpoint = Path(MODEL).expanduser()
    result = evaluate(checkpoint, layout, ["S1"], device=device, batch_size=64, max_cases=60)
    s1 = result.suites["S1"]
    assert result.model_key.startswith(sha256_file(checkpoint)[:12]) and s1["reads"]["diagnosed"] == 60
    # A trained model, so the numbers mean something: P2b's 300-read figures were orientation 99.7%, J top-1 97.3%.
    assert s1["heads"]["head.orientation.correct"]["mean"] > 0.9
    assert s1["heads"]["head.j.allele_top1"]["mean"] > 0.8
    # Feeding every stage's truth leaves a better junction than the product's.
    chain = s1["variants"]
    assert chain["true_germline"]["scores"]["junction.nt_exact"]["mean"] > chain["baseline"]["scores"][
        "junction.nt_exact"]["mean"]
    assert result.profile["device"] == device and result.profile["reads_per_second"] > 0
    again = evaluate(checkpoint, layout, ["S1"], device=device, batch_size=64, max_cases=60)
    assert again.suites["S1"]["reused"] and again.profile["reused"]
