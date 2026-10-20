"""The in-process evaluator (``workflow.diagnose``) and the ``diagnose`` command: diagnostics of a model on built
dev suites, cached per suite and never served stale.

Everything runs on the CPU with small random-init models (``conftest.make_small_model``) on S0 and S1 built at
scale 0.02 (2 and 100 reads). The test of the released checkpoint runs only when ``ALIGNAIR_ASSAY_MODEL`` names a
``.alignair`` model file, on the GPU when ``ALIGNAIR_ASSAY_GPU=1``.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("GenAIRR")

import torch  # noqa: E402
from alignair.model_file import save_model  # noqa: E402

from alignair_assay.cli import main  # noqa: E402
from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.provenance import sha256_file  # noqa: E402
from alignair_assay.diagnostics.ablation import DEFAULT_VARIANTS, SINGLE_SWAPS, VARIANTS  # noqa: E402
from alignair_assay.diagnostics.identity import alignair_code, weights_sha256  # noqa: E402
from alignair_assay.report.diagnostics import SECTIONS  # noqa: E402
from alignair_assay.runners.alignair import local_code_identity  # noqa: E402
from alignair_assay.workflow import diagnose as diagnose_module  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402
from alignair_assay.workflow.diagnose import DiagnosticsResult, evaluate, select_cases  # noqa: E402

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

def test_end_to_end_report_renders_every_section(built, small_model):
    result = evaluate(small_model, built, ["S0", "S1"], device="cpu", batch_size=32, profile=True)
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
        # The suite's primary metrics lead the compared family.
        primary = {"S0": ["v.top1_in_set", "j.top1_in_set", "junction.nt_exact"],
                   "S1": ["clone_key.exact", "junction.nt_exact"]}[name]
        assert suite["metrics"][:len(primary)] == primary
        for step in suite["contributions"] + suite["swaps"]:
            for entry in step["metrics"].values():
                assert {"mean_from_paired", "mean_to_paired", "delta", "verdict"} <= set(entry)
    s1 = result.suites["S1"]
    cropped = sum(len(case["sequence"]) > 576 for case in read_cases(built.cases("S1")))
    assert s1["reads"] == {"built": 100, "diagnosed": 100, "scored": s1["reads"]["scored"],
                           "excluded": 100 - s1["reads"]["scored"], "cropped": cropped}
    assert s1["heads"]["head.orientation.correct"]["n"] == s1["reads"]["scored"]
    # S1 has donors: its variants are scored against each read's donor genotype, as all scores the suite.
    assert "v.outside_genotype" in s1["variants"]["baseline"]["scores"]

    assert result.profile is not None and result.profile["device"] == "cpu" and result.profile["reads"] == 102
    md = result.report_md
    positions = [md.index(f"\n## {title}\n") for title in SECTIONS]
    assert positions == sorted(positions)
    assert md.count("\n### S0\n") == 4 and md.count("\n### S1\n") == 4

    # Written beside the per-suite caches, under the model key.
    assert built.diagnostics_md(result.model_key).read_text(encoding="utf-8") == md
    report = json.loads(built.diagnostics_json(result.model_key).read_text(encoding="utf-8"))
    assert report["model"]["model_key"] == result.model_key and report["model"]["source"] == "in-memory model"
    assert report["suites"]["S1"]["variants"] == s1["variants"]
    for name in ("S0", "S1"):
        assert json.loads(built.diagnostics_suite_json(result.model_key, name).read_text())["suite"] == name
    assert built.diagnostics_profile_json(result.model_key).is_file()


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
    for change in (dict(variants=("baseline", "argmax_decoder")), dict(max_cases=1), dict(batch_size=8),
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
    result = evaluate(small_model, layout, ["S0"], **{**FAST, "variants": ("argmax_decoder",)})
    assert list(result.suites["S0"]["variants"]) == ["baseline", "argmax_decoder"]
    assert [swap["to"] for swap in result.suites["S0"]["swaps"]] == ["argmax_decoder"]
    # No variant at all: the heads, the calibration and the profile only.
    result = evaluate(small_model, layout, ["S0"], **{**FAST, "variants": ()})
    suite = result.suites["S0"]
    assert suite["variants"] == {} and suite["contributions"] == [] and suite["heads"]


def test_comparisons_are_clustered_by_the_suites_groups(built, tmp_path, small_model, monkeypatch):
    # No built suite here has clones or pairs, so the S1 reads are clustered by donor: the evaluator must build the
    # clusters from the suite's truth once and pass them to every comparison.
    layout = _copy(built, tmp_path)
    plain = evaluate(small_model, layout, ["S1"], **{**FAST, "variants": ("argmax_decoder",)})
    assert plain.suites["S1"]["clusters"] == {"keys": ["clone", "pair"], "clusters": 0, "cases": 0}
    entry = plain.suites["S1"]["swaps"][0]["metrics"]["v.top1_in_set"]
    assert entry["n_clusters"] == entry["n_paired"]

    seen = []

    def by_donor(truths):
        seen.append(len(truths))
        return {t.case_id: t.groups["subject"] for t in truths if "subject" in t.groups}

    monkeypatch.setattr(diagnose_module, "cluster_ids", by_donor)
    monkeypatch.setattr(diagnose_module, "CLUSTER_KEYS", ("subject",))
    result = evaluate(small_model, _copy(built, tmp_path / "again"), ["S1"],
                      **{**FAST, "variants": ("true_orientation", "argmax_decoder")})
    assert seen == [100]
    suite = result.suites["S1"]
    donors = len({case["groups"]["subject"] for case in read_cases(built.cases("S1"))})
    assert suite["clusters"] == {"keys": ["subject"], "clusters": donors, "cases": 100}
    for comparison in suite["contributions"] + suite["swaps"]:
        for entry in comparison["metrics"].values():
            assert entry["n_clusters"] <= donors < entry["n_paired"] or entry["n_paired"] == 0
            assert entry["test"] in ("too_few_clusters", "too_few_pairs", "clustered_mcnemar", "clustered_z")


# --- the diagnose command ---------------------------------------------------------------------------------------

def test_diagnose_command_writes_the_report_and_reuses_it(built, tmp_path, save_checkpoint, capsys):
    layout = _copy(built, tmp_path)
    checkpoint = save_checkpoint(tmp_path / "model.alignair")
    argv = ["diagnose", "--run-dir", str(layout.root), "--checkpoint", str(checkpoint), "--suites", "S0",
            "--device", "cpu", "--batch-size", "16", "--variants", "argmax_decoder", "--no-profile",
            "--max-cases", "2"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    (key,) = [path.name for path in layout.diagnostics_root.iterdir()]
    report = json.loads(layout.diagnostics_json(key).read_text())
    assert report["config"]["variants"] == ["baseline", "argmax_decoder"]
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
