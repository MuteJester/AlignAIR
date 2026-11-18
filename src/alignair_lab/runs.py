"""A live run of the retrain recipe (``recipe.retrain_config``): trained from GenAIRR as the product run is, not from a
record cache, so any data seed and any dataconfigs serve. The lab's ``train`` command; ``databand`` runs it once per
data seed.

``train_run(out, seed=..., steps=..., schedule=...)`` trains the recipe of ``steps`` steps under ``schedule``
(``recipe.schedule``: ``cosine`` or ``constant``) from ``seed`` (its data and its initial weights, as a product run
draws them) into ``out``, without validation:

- ``out/launch.json`` is written before the run: its ``request``, its ``config`` (the ``TrainingConfig``'s repr), the
  GenAIRR it imports, the ``PYTHONPATH``, the code digests (``screen.code_digests``) and the start time;
- ``out/done.json`` after it: the trained ``model``, the ``steps``, the ``seed``, the ``request`` and the ``hours``.

Both record the run's request (``run_request``: the recipe id, seed, steps, schedule, dataconfigs and model options).
A finished run (its ``done.json``) is reused only when it is a run of the same request and its checkpoint's own
``train_args`` agree (``check_run``: recipe, seed, steps, schedule; a run is trusted by its checkpoint, as a baseline
is): its record is returned, ``reused`` True, and nothing is trained. A finished run of another request is refused
(FileExistsError, naming what differs), and so is a directory without ``done.json``, an interrupted run: remove it to
train again.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from alignair.train.trainer import RECIPE
from alignair_lab.recipe import DATACONFIGS, retrain_config, schedule as learning_schedule

LAUNCH = "launch.json"
DONE = "done.json"
DEFAULT_DATA_WORKERS = 6


def run_request(*, seed: int, steps: int, schedule: str, dataconfigs: Sequence[str] = DATACONFIGS,
                model_options: Mapping | None = None) -> dict:
    """What a run of the retrain recipe is a run of, as its ``launch.json`` and ``done.json`` record it."""
    return {"recipe": RECIPE, "seed": seed, "steps": steps, "schedule": schedule, "dataconfigs": list(dataconfigs),
            "model_options": json.loads(json.dumps(dict(model_options or {})))}


def check_run(out: Path, request: Mapping) -> dict:
    """The ``done.json`` record of the finished run in ``out``, which must be a run of ``request`` (``run_request``)
    whose checkpoint's ``train_args`` agree with it; FileExistsError naming what differs otherwise."""
    from alignair.model_file import read_metadata
    out = Path(out)
    record = json.loads((out / DONE).read_text(encoding="utf-8"))
    found = record.get("request") or {}
    differing = sorted(key for key in set(request) | set(found) if found.get(key) != request.get(key))
    if differing:
        raise FileExistsError(f"{out} holds another run (its {', '.join(differing)} differ from this request's): "
                              "choose another --out")
    learning = learning_schedule(request["schedule"], request["steps"])
    expected = {"recipe": request["recipe"], "seed": request["seed"], "steps": request["steps"],
                "schedule": None if learning is None else learning.to_dict()}
    train_args = read_metadata(record["model"])["training"]["train_args"]
    differing = [key for key, value in expected.items() if train_args.get(key) != value]
    if differing:
        raise FileExistsError(f"{out} holds another run: its checkpoint {record['model']} was trained with another "
                              f"{', '.join(differing)} than its record says; choose another --out")
    return record


def train_run(out: Path, *, seed: int, steps: int, schedule: str, dataconfigs: Sequence[str] = DATACONFIGS,
              device: str = "cuda", data_workers: int = DEFAULT_DATA_WORKERS,
              model_options: Mapping | None = None) -> dict:
    """Train the retrain recipe from ``seed`` into ``out`` (see the module docstring); returns ``done.json``'s
    record, with ``reused``: whether it was a finished run of this request."""
    import GenAIRR
    from alignair import run_training
    from alignair_lab.screen import code_digests
    out = Path(out)
    done = out / DONE
    request = run_request(seed=seed, steps=steps, schedule=schedule, dataconfigs=dataconfigs,
                          model_options=model_options)
    if done.is_file():
        return {**check_run(out, request), "reused": True}
    if out.exists():
        raise FileExistsError(f"{out} holds a run that did not finish (no {DONE}): remove it to train it again")
    config = retrain_config(dataconfigs=dataconfigs, steps=steps, seed=seed,
                            schedule=learning_schedule(schedule, steps), val_every=0, device=device,
                            data_workers=data_workers, model_options=model_options)
    out.mkdir(parents=True)
    started = time.time()
    (out / LAUNCH).write_text(json.dumps({
        "request": request, "config": repr(config), "genairr": GenAIRR.__file__,
        "pythonpath": os.environ.get("PYTHONPATH"), "code": code_digests(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))},
        indent=1) + "\n", encoding="utf-8")
    run = run_training(config, output_dir=out)
    record = {"model": str(run.model_path), "steps": run.steps, "seed": seed, "request": request,
              "hours": round((time.time() - started) / 3600, 3)}
    done.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return {**record, "reused": False}
