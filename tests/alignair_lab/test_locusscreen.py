"""The locus screen (``locusscreen.run_locus_screen``): a chip shared by several loci against one chip per locus, each
locus scored on its own dev build, its noise from its per-locus seeds. Every run is the rig's small model on the CPU
with one data worker for a few steps; the IGK dev build is an S0 build made here."""
import json

import pytest

from alignair_lab.locusscreen import run_locus_screen
from tests.alignair_lab.rig import SMALL

pytestmark = pytest.mark.integration

IGH, IGK = "HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB"
ARMS = {"both": ((IGH, IGK), 4, ("IGH", "IGK")), "IGH": ((IGH,), 2, ("IGH",)), "IGK": ((IGK,), 2, ("IGK",))}


@pytest.fixture(scope="module")
def igk_build(tmp_path_factory):
    from alignair_assay.core.layout import RunLayout
    from alignair_assay.workflow.build import build_suite
    root = tmp_path_factory.mktemp("igk")
    build_suite("S0", RunLayout(root), reference_name=IGK, split="dev", seed=5, scale=0.02, workers=1)
    return root


def _options(rig, igk_build, tmp_path, **extra):
    return {"arms": ARMS, "loci": {"IGH": rig.dev, "IGK": igk_build}, "seeds": (0, 1), "schedule": "constant",
            "parallel": 3, "model_options": SMALL, "device": "cpu", "data_workers": 1, "eval_batch_size": 16,
            "out_root": tmp_path, "name": "screen", **extra}


@pytest.mark.timeout(1800)
def test_the_locus_screen_trains_each_arm_and_scores_each_locus(rig, igk_build, tmp_path, monkeypatch):
    """A screen whose scoring fails is recorded failed; run again, it resumes with every finished run reused, then
    scores each locus."""
    import alignair_assay.workflow.diagnose as diagnose
    from alignair_lab import locusscreen

    def unavailable(*args, **kwargs):
        raise RuntimeError("the assay is unavailable")

    evaluate = diagnose.evaluate
    monkeypatch.setattr(diagnose, "evaluate", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        run_locus_screen(**_options(rig, igk_build, tmp_path))
    assert json.loads((tmp_path / "screen" / "locus_screen.json").read_text())["status"] == "failed"
    monkeypatch.setattr(diagnose, "evaluate", evaluate)
    plans, train_screens = [], locusscreen.train_screens

    def watched(plan, *args, **kwargs):
        plans.append(plan)
        return train_screens(plan, *args, **kwargs)

    monkeypatch.setattr(locusscreen, "train_screens", watched)
    record = run_locus_screen(**_options(rig, igk_build, tmp_path))
    assert plans == [[]]                                                         # every run reused
    assert record["kind"] == "alignair_lab.locus_screen" and record["status"] == "done"
    assert set(record["runs"]) == {"both", "IGH", "IGK"} and set(record["runs"]["both"]) == {"0", "1"}
    for locus in ("IGH", "IGK"):
        assert set(record["pairs"][locus]) == {"0", "1"}                         # shared seed s against locus seed s
        assert record["bands"][locus]                                            # from the per-locus seed pair
        for metric, call in record["verdict"][locus].items():
            assert call["call"] in ("worse", "better", "within")              # every compared metric has a band
            assert call["band"] == max(record["bands"][locus][metric], 0.001)
        assert all(record["speed"][arm][locus][where] > 0 for arm in ("both", locus) for where in ("device", "cpu8"))
    assert isinstance(record["passes"], bool)
    assert json.loads((tmp_path / "screen" / "locus_screen.json").read_text())["verdict"] == record["verdict"]
    assert "IGK" in (tmp_path / "screen" / "locus_screen.md").read_text()


def test_a_locus_build_of_another_reference_is_refused_before_training(rig, igk_build, tmp_path):
    with pytest.raises(ValueError, match="HUMAN_IGK_OGRDB"):
        run_locus_screen(**_options(rig, igk_build, tmp_path, loci={"IGH": rig.dev, "IGK": rig.dev}))
    assert not any(tmp_path.iterdir())


def test_an_arm_scored_on_a_locus_without_a_build_is_refused(rig, igk_build, tmp_path):
    with pytest.raises(ValueError, match="IGK"):
        run_locus_screen(**_options(rig, igk_build, tmp_path, loci={"IGH": rig.dev}))
    assert not any(tmp_path.iterdir())


def test_reused_runs_are_checked_and_a_mismatch_stops_the_screen(rig, igk_build, tmp_path):
    from alignair_lab.runs import train_run
    reused = tmp_path / "reused"
    for seed in (0, 1):                                  # seed 1 of another length than the arm's 2 steps
        train_run(reused / f"run-seed{seed}", seed=seed, steps=2 if seed == 0 else 3, schedule="constant",
                  dataconfigs=(IGH,), device="cpu", data_workers=1, model_options=SMALL)
    with pytest.raises(FileExistsError, match="run-seed1"):
        run_locus_screen(**_options(rig, igk_build, tmp_path, reuse={"IGH": reused}))


def _pairs(*deltas_by_seed):
    return {str(seed): {"pooled": deltas, "suites": {}} for seed, deltas in enumerate(deltas_by_seed)}


def test_a_headline_metric_compared_without_a_band_fails_the_screen():
    from alignair_lab.locusscreen import _passes, _verdict
    verdict = _verdict(_pairs({"j.top1_in_set": -0.3, "v.top1_in_set": 0.0}, {"j.top1_in_set": -0.3,
                                                                              "v.top1_in_set": 0.0}),
                       {"v.top1_in_set": 0.01})
    assert verdict["j.top1_in_set"]["call"] == "unscored" and _passes({"IGH": verdict}) is False


def test_a_headline_metric_some_seeds_lack_is_unscored():
    from alignair_lab.locusscreen import _verdict
    verdict = _verdict(_pairs({"v.top1_in_set": -0.001}, {}), {"v.top1_in_set": 0.01})
    assert verdict["v.top1_in_set"]["call"] == "unscored"


def test_a_metric_no_seed_compares_is_not_applicable_and_does_not_gate():
    from alignair_lab.locusscreen import _passes, _verdict
    verdict = _verdict(_pairs({"v.top1_in_set": 0.0}, {"v.top1_in_set": 0.0}), {"v.top1_in_set": 0.01})
    assert "d.top1_in_set" not in verdict and _passes({"IGK": verdict}) is True


def test_a_band_of_zero_is_floored_at_a_tenth_of_a_point():
    from alignair_lab.locusscreen import BAND_FLOOR, _verdict
    assert BAND_FLOOR == 0.001
    within = _verdict(_pairs({"j.top1_in_set": -0.0003}, {"j.top1_in_set": -0.0003}), {"j.top1_in_set": 0.0})
    worse = _verdict(_pairs({"j.top1_in_set": -0.002}, {"j.top1_in_set": -0.002}), {"j.top1_in_set": 0.0})
    assert within["j.top1_in_set"]["call"] == "within" and within["j.top1_in_set"]["band"] == 0.001
    assert worse["j.top1_in_set"]["call"] == "worse"


def test_the_per_locus_runs_train_before_the_shared_arms():
    from alignair_lab.locusscreen import training_order
    assert training_order({"all": ((IGH, IGK), 8, ("IGH", "IGK")), "IGH": ((IGH,), 4, ("IGH",)),
                           "IGK": ((IGK,), 4, ("IGK",))}) == ["IGH", "IGK", "all"]
