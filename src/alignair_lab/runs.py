"""A live run of the retrain recipe (``recipe.retrain_config``): trained from GenAIRR as the product run is, not from a
record cache, so any data seed and any dataconfigs serve. The lab's ``train`` command; ``databand`` runs it once per
data seed.

``train_run(out, seed=..., steps=..., schedule=...)`` trains the recipe of ``steps`` steps under ``schedule``
(``recipe.schedule``: ``cosine`` or ``constant``) from ``seed`` (its data and its initial weights, as a product run
draws them) into ``out``, without validation:

- ``out/launch.json`` is written before the run: its ``config`` (the ``TrainingConfig``'s repr), the GenAIRR it
  imports, the ``PYTHONPATH``, the code digests (``screen.code_digests``) and the start time;
- ``out/done.json`` after it: the trained ``model``, the ``steps``, the ``seed`` and the ``hours``.

A finished run (its ``done.json``) is reused: its record is returned and nothing is trained. A directory without one
is an interrupted run, which is refused (FileExistsError): remove it to train again.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from alignair_lab.recipe import DATACONFIGS, retrain_config, schedule as learning_schedule

LAUNCH = "launch.json"
DONE = "done.json"
DEFAULT_DATA_WORKERS = 6


def train_run(out: Path, *, seed: int, steps: int, schedule: str, dataconfigs: Sequence[str] = DATACONFIGS,
              device: str = "cuda", data_workers: int = DEFAULT_DATA_WORKERS,
              model_options: Mapping | None = None) -> dict:
    """Train the retrain recipe from ``seed`` into ``out`` (see the module docstring); returns ``done.json``'s
    record."""
    import GenAIRR
    from alignair import run_training
    from alignair_lab.screen import code_digests
    out = Path(out)
    done = out / DONE
    if done.is_file():
        return json.loads(done.read_text(encoding="utf-8"))
    if out.exists():
        raise FileExistsError(f"{out} holds a run that did not finish (no {DONE}): remove it to train it again")
    config = retrain_config(dataconfigs=dataconfigs, steps=steps, seed=seed,
                            schedule=learning_schedule(schedule, steps), val_every=0, device=device,
                            data_workers=data_workers, model_options=model_options)
    out.mkdir(parents=True)
    started = time.time()
    (out / LAUNCH).write_text(json.dumps({
        "config": repr(config), "genairr": GenAIRR.__file__, "pythonpath": os.environ.get("PYTHONPATH"),
        "code": code_digests(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))},
        indent=1) + "\n", encoding="utf-8")
    run = run_training(config, output_dir=out)
    record = {"model": str(run.model_path), "steps": run.steps, "seed": seed,
              "hours": round((time.time() - started) / 3600, 3)}
    done.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record
