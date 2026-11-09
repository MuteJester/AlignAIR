"""The data-seed band: the run-to-run noise of training the retrain recipe from another data seed.

A calibration (``calibrate``) measures init-seed noise on fixed batches: its baseline screens read one record cache. A
candidate that trains on other records (other loci, novel alleles, another draw) differs from the baseline by its data
draw too, which that band leaves out. Here the retrain recipe (``recipe.retrain_config``) is trained live from several
data seeds (``runs.train_run``: each run's records and initial weights from its seed, as a product run draws them), and
the runs are compared pairwise with the assay: their pooled Δs bound the noise of an independent run.

``run_data_band(seeds=(0, 1, 2), steps=..., schedule=..., ...)``, as ``calibrate.run_calibration`` runs:

1. **Refuses** before anything runs: fewer than two seeds, one given twice or one that is not a uint32, a parallel
   count below 1, an unknown schedule (ValueError); a band name that is not one path component (ValueError); a suite
   not built in the dev build; a band directory of that name that is not an unfinished band of this request (another
   request's, a finished one, or no band at all: FileExistsError).
2. Claims the band directory ``out_root/<name>`` (default ``data-band-<recipe>-s<steps>-<schedule>``), or resumes the
   unfinished band of this request there (one that failed or was interrupted), and records the request and its
   progress in its ``data_band.json`` (a failure is recorded before it is raised).
3. Trains each seed's run in ``<band>/run-seed<d>`` with ``python -m alignair_lab train`` under the GenAIRR and the
   code this process imports, at most ``parallel`` at once, each one's output in ``<band>/run-seed<d>.log``; a
   finished run is reused, and every run is checked to be a run of this request by its checkpoint
   (``runs.check_run``) before it is compared.
4. Compares each pair of runs, the later seed's against the earlier's, with the assay in ``<band>/assay``.
5. Writes ``data_band.json`` (kind ``alignair_lab.data_band``): the request, each run's checkpoint and assay model key
   (``runs``), each pair's pooled and per-suite Δs (``pairs``, ``noise.pair``), the band (``band``: each metric's
   largest |pooled Δ|, ``noise.null_band``), each seed's values against the first seed (``seeds``, ``relative_to``;
   ``noise.seed_values``) and their mean and SD (``baseline``), and the code digests; and prints the band of each
   headline metric.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

from alignair_lab.baselines import NAME, now, write_json
from alignair_lab.calibrate import VARIANTS, train_screens
from alignair_lab.headline import HEADLINE
from alignair_lab.noise import baseline_summary, null_band, pair, seed_values
from alignair_lab.recipe import RECIPE, schedule as learning_schedule
from alignair_lab.runs import DEFAULT_DATA_WORKERS, DONE, check_run, run_request
from alignair_lab.screen import DEV_BUILD, SCREENS, assay_layout, built_suites, code_digests, suite_builds
from alignair_lab.secondary import METRIC, cached_comparison, merged

DATA_BAND = "data_band.json"
DATA_BAND_KIND = "alignair_lab.data_band"
DATA_BAND_VERSION = 1
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_STEPS = 20_000


def data_band_name(steps: int, schedule: str) -> str:
    """The band directory's default name."""
    return f"data-band-{RECIPE}-s{steps}-{schedule}"


def check_seeds(seeds: Sequence[int], what: str = "data band") -> list[int]:
    """``seeds`` as a list: two or more distinct uint32 data seeds (ValueError otherwise, naming ``what``)."""
    found = list(seeds)
    for seed in found:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"a data seed is an integer from 0 to 2**32 - 1, got {seed!r}")
    if len(found) < 2 or len(set(found)) != len(found):
        raise ValueError(f"a {what} compares two data seeds or more, each once; got {found}")
    return found


def unfinished(directory: Path, name: str, request: Mapping, *, record: str = DATA_BAND,
               what: str = "data band") -> dict:
    """The record (the file ``record``) of the unfinished ``what`` of ``request`` in ``directory``, which a run of the
    same request resumes; FileExistsError for a finished one, another request's or a directory that holds none."""
    try:
        previous = json.loads((directory / record).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = None
    if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in request.items()):
        raise FileExistsError(f"{directory} exists and is not an unfinished {what} of this request; choose another "
                              "--name")
    if previous.get("status") == "done":
        raise FileExistsError(f"the {what} {name!r} is done ({directory / record}); choose another --name to "
                              "measure it again")
    return previous


def train_command(out: Path, seed: int, *, steps: int, schedule: str, model_options: Mapping, device: str,
                  data_workers: int, dataconfigs: Sequence[str] = ()) -> list[str]:
    """The ``python -m alignair_lab train`` command of one run of the retrain recipe (``dataconfigs``: the recipe's
    own when empty)."""
    options = [part for key, value in model_options.items()
               for part in ("--model-option", f"{key}={json.dumps(value)}")]
    loci = [part for name in dataconfigs for part in ("--dataconfig", name)]
    return [sys.executable, "-m", "alignair_lab", "train", "--out", str(out), "--seed", str(seed), "--steps",
            str(steps), "--schedule", schedule, "--device", device, "--data-workers", str(data_workers), *loci,
            *options]


def run_data_band(*, seeds: Sequence[int] = DEFAULT_SEEDS, steps: int = DEFAULT_STEPS, schedule: str = "cosine",
                  parallel: int = 1, model_options: Mapping | None = None, suites: Sequence[str] | None = None,
                  max_cases: int | None = None, device: str = "cuda", data_workers: int = DEFAULT_DATA_WORKERS,
                  eval_batch_size: int = 256, dev_build=DEV_BUILD, out_root=SCREENS, name: str | None = None,
                  echo: Callable[[str], None] | None = None) -> dict:
    """Measure the data-seed band of the retrain recipe at ``steps`` steps under ``schedule`` from ``seeds`` (see the
    module docstring); returns the band record (``data_band.json``). ``echo`` receives the progress lines and the
    summary (default: printed)."""
    if echo is None:
        def echo(line: str) -> None:
            print(line, flush=True)
    seeds = check_seeds(seeds)
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise ValueError(f"parallel is a positive integer, got {parallel!r}")
    learning_schedule(schedule, steps)                       # an unknown schedule is refused here
    model_options = json.loads(json.dumps(dict(model_options or {})))
    out_root = Path(out_root)
    name = data_band_name(steps, schedule) if name is None else name
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a data band name is 1 to 64 letters, digits, '.', '_' or '-', starting with a letter or "
                         f"digit; got {name!r}")
    directory = out_root / name
    names = built_suites(Path(dev_build), suites)
    builds = suite_builds(Path(dev_build), names)
    request = {"kind": DATA_BAND_KIND, "version": DATA_BAND_VERSION, "recipe": RECIPE, "steps": steps,
               "schedule": schedule, "data_seeds": seeds, "options": model_options, "suites": names,
               "suite_builds": builds, "max_cases": max_cases}
    previous = unfinished(directory, name, request) if directory.exists() else None
    if previous is None:
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError:
            raise FileExistsError(f"a data band named {name!r} exists ({directory}); choose another --name") from None

    runs = {str(seed): {"directory": str(directory / f"run-seed{seed}")} for seed in seeds}
    record = {**request, "name": name, "status": "preparing",
              "started": now() if previous is None else previous.get("started"),
              "variants": list(VARIANTS), "device": device, "eval_batch_size": eval_batch_size, "runs": runs}
    if previous is not None:
        record["resumed"] = now()
        echo(f"[data-band] resuming the unfinished data band {name} ({previous.get('status')}): its finished runs "
             "are reused")

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        write_json(directory / DATA_BAND, record)
        echo(f"[data-band] {status}")

    try:
        plan = [(seed, f"run-seed{seed}", train_command(directory / f"run-seed{seed}", seed, steps=steps,
                                                        schedule=schedule, model_options=model_options,
                                                        device=device, data_workers=data_workers))
                for seed in seeds if not (directory / f"run-seed{seed}" / DONE).is_file()]
        stage(f"training {len(plan)} run(s), reusing {len(seeds) - len(plan)}")
        train_screens(plan, directory, parallel, echo, label="data-band", what="run", seed_label="data seed")
        for seed in seeds:
            done = check_run(directory / f"run-seed{seed}",
                             run_request(seed=seed, steps=steps, schedule=schedule, model_options=model_options))
            runs[str(seed)]["checkpoint"] = done["model"]
        stage(f"comparing the {len(seeds) * (len(seeds) - 1) // 2} pairs of runs")
        from alignair_assay.workflow.diagnose import evaluate
        if previous is not None and (directory / "assay").exists():     # a resumed band compares afresh
            shutil.rmtree(directory / "assay")
        layout = assay_layout(directory / "assay", Path(dev_build), names)
        pairs, code = [], None
        for first, second in combinations(seeds, 2):
            earlier, later = runs[str(first)], runs[str(second)]
            result = evaluate(later["checkpoint"], layout, names, device=device, batch_size=eval_batch_size,
                              variants=VARIANTS, profile=False, max_cases=max_cases, against=earlier["checkpoint"],
                              label=f"{name}-seed{second}")
            summary = result.against["summary"]
            earlier["model_key"], later["model_key"] = summary["baseline"], result.model_key
            code = summary["code"]["this"]
            extra = cached_comparison(layout.diagnostics_root, result.model_key, summary["baseline"], METRIC)
            pairs.append({"a": first, "b": second, **pair(merged(summary, extra))})
        band = null_band(pairs)
        values = seed_values(pairs, seeds, relative_to=seeds[0])
        record.update(status="done", pairs=pairs, band=band, relative_to=seeds[0], seeds=values,
                      baseline=baseline_summary(values), code_sha256=code, code=code_digests(), finished=now())
        write_json(directory / DATA_BAND, record)
        stage("done")
        shown = [(label, band[metric]) for metric, label in HEADLINE if metric in band]
        echo(f"[data-band] the data-seed band, the largest |pooled Δ| of {len(pairs)} pairs, in pp: "
             + ", ".join(f"{label} {100 * value:.2f}" for label, value in shown)
             + f"; every metric's and each pair's Δs are in {directory / DATA_BAND}")
        return record
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", finished=now())
        stage("failed")
        raise
