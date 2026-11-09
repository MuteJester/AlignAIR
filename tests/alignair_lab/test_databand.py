"""The data-seed band: runs of the retrain recipe trained live from several data seeds (``runs.train_run``, the lab's
``train`` command), compared pairwise with the assay (``databand.run_data_band``). Every run is the rig's small model
on the CPU, with one data worker, for 2 steps."""
import json

import pytest

from alignair_lab.databand import run_data_band
from alignair_lab.runs import train_run
from tests.alignair_lab.rig import SMALL

pytestmark = pytest.mark.integration


def _small(rig, out_root):
    return {"model_options": SMALL, "device": "cpu", "data_workers": 1, "eval_batch_size": 16, "dev_build": rig.dev,
            "out_root": out_root}


@pytest.mark.parametrize("seeds", [(0,), (0, 0), (-1, 2)])
def test_a_data_band_that_cannot_run_is_refused_before_anything_runs(rig, tmp_path, seeds):
    with pytest.raises(ValueError):
        run_data_band(seeds=seeds, steps=2, **_small(rig, tmp_path))
    assert not any(tmp_path.iterdir())


def test_a_data_band_in_use_is_refused(rig, tmp_path):
    (tmp_path / "band").mkdir()
    with pytest.raises(FileExistsError):
        run_data_band(seeds=(0, 1), steps=2, name="band", **_small(rig, tmp_path))


def test_a_run_left_without_done_json_is_refused_not_reused(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError, match="remove"):
        train_run(tmp_path / "run", seed=0, steps=2, schedule="constant", device="cpu", data_workers=1,
                  model_options=SMALL)


@pytest.mark.timeout(900)
def test_the_data_band_trains_each_seed_and_pairs_every_two(rig, tmp_path):
    record = run_data_band(seeds=(0, 1, 2), steps=2, schedule="constant", name="band", parallel=3,
                           **_small(rig, tmp_path))
    assert record["kind"] == "alignair_lab.data_band" and record["status"] == "done"
    assert record["data_seeds"] == [0, 1, 2] and record["relative_to"] == 0
    assert [(p["a"], p["b"]) for p in record["pairs"]] == [(0, 1), (0, 2), (1, 2)]
    for metric, value in record["band"].items():
        assert value == max(abs(p["pooled"][metric]) for p in record["pairs"] if metric in p["pooled"])
    assert json.loads((tmp_path / "band" / "data_band.json").read_text())["band"] == record["band"]
    done = json.loads((tmp_path / "band" / "run-seed1" / "done.json").read_text())
    assert done["seed"] == 1 and done["steps"] == 2
    again = train_run(tmp_path / "band" / "run-seed1", seed=1, steps=2, schedule="constant", device="cpu",
                      data_workers=1, model_options=SMALL)
    assert again == {**done, "reused": True}                                   # a finished run is reused


def test_a_finished_run_of_another_request_is_refused_not_reused(tmp_path):
    first = train_run(tmp_path / "run", seed=0, steps=2, schedule="constant", device="cpu", data_workers=1,
                      model_options=SMALL)
    assert first["reused"] is False
    with pytest.raises(FileExistsError, match="another run"):
        train_run(tmp_path / "run", seed=1, steps=2, schedule="constant", device="cpu", data_workers=1,
                  model_options=SMALL)
    with pytest.raises(FileExistsError, match="another run"):
        train_run(tmp_path / "run", seed=0, steps=2, schedule="constant", dataconfigs=("HUMAN_IGK_OGRDB",),
                  device="cpu", data_workers=1, model_options=SMALL)
    assert train_run(tmp_path / "run", seed=0, steps=2, schedule="constant", device="cpu", data_workers=1,
                     model_options=SMALL) == {**first, "reused": True}


@pytest.mark.timeout(900)
def test_a_failed_data_band_resumes_with_its_finished_runs(rig, tmp_path, monkeypatch):
    import alignair_assay.workflow.diagnose as diagnose
    from alignair_lab import databand

    def unavailable(*args, **kwargs):
        raise RuntimeError("the assay is unavailable")

    evaluate = diagnose.evaluate
    monkeypatch.setattr(diagnose, "evaluate", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        run_data_band(seeds=(0, 1), steps=2, schedule="constant", name="band", parallel=2, **_small(rig, tmp_path))
    assert json.loads((tmp_path / "band" / "data_band.json").read_text())["status"] == "failed"
    monkeypatch.setattr(diagnose, "evaluate", evaluate)
    plans, train_screens = [], databand.train_screens

    def watched(plan, *args, **kwargs):
        plans.append(plan)
        return train_screens(plan, *args, **kwargs)

    monkeypatch.setattr(databand, "train_screens", watched)
    record = run_data_band(seeds=(0, 1), steps=2, schedule="constant", name="band", **_small(rig, tmp_path))
    assert plans == [[]] and record["status"] == "done"                        # both runs reused, none trained
    with pytest.raises(FileExistsError, match="done"):
        run_data_band(seeds=(0, 1), steps=2, schedule="constant", name="band", **_small(rig, tmp_path))
    with pytest.raises(FileExistsError):                                       # another request under its name
        run_data_band(seeds=(0, 1), steps=4, schedule="constant", name="band", **_small(rig, tmp_path))
