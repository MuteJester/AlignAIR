"""The locus screen (decision 12): one chip trained on several loci against one chip per locus, each locus getting the
same training in both.

The arms train the retrain recipe (``recipe.retrain_config``) live (``runs.train_run``): the shared arm on several
dataconfigs, whose records are interleaved round-robin (``gym.streams.training_records``), so at ``n`` dataconfigs and
``n * k`` steps each gets ``k`` steps' worth; each per-locus arm on one dataconfig at ``k`` steps. Every arm trains
from each of the screen's seeds. Each locus is scored on its own dev build: the shared arm's run of seed ``s`` against
the locus's per-locus run of seed ``s`` (Δ = shared - per-locus), and the locus's noise is its band, the largest |pooled
Δ| among its per-locus runs' seed pairs (runs of one recipe differing by their data draw), floored at ``BAND_FLOOR``
(a metric every seed scores alike would otherwise call any shortfall worse). A metric is ``worse`` on a locus when its
mean Δ is below minus the band, ``better`` above the band, else ``within``; ``unscored`` when some seed does not
compare it or the band does not hold it. The shared arm passes when no headline metric (``headline.HEADLINE``) is
worse or unscored on any locus (a metric no seed compares on a locus, D on IGK, does not apply there). Speed (reads
per second of the product pipeline on the locus's S1 reads, the median of 3 passes after a warm-up, on the screen's
device and on the CPU with 8 threads) is measured for each arm's first seed on each of its loci; it is reported, not
gated.

``run_locus_screen(...)``, as ``databand.run_data_band`` runs:

1. **Refuses** before anything runs: seeds as the data band does, a parallel count below 1, an unknown schedule or a
   bad name (ValueError); arms that are not one shared arm (scored on several loci) and one single-dataconfig arm
   per locus it is scored on (ValueError); a locus without a dev build, or whose dev build holds a suite built on
   another reference than its per-locus arm's dataconfig (ValueError); a ``reuse`` run that is not a finished run of
   its arm's request (``runs.check_run``: FileExistsError); a screen directory that is not an unfinished screen of
   this request (FileExistsError).
2. Claims ``out_root/<name>``, or resumes the unfinished screen of this request there, and records its progress in
   ``locus_screen.json`` (a failure is recorded before it is raised).
3. Trains each arm's missing runs in ``<screen>/<arm>/run-seed<d>`` (``python -m alignair_lab train``, at most
   ``parallel`` at once, the per-locus arms first: a failure shows before the long shared runs); a finished run is
   reused, and so is every run of a ``reuse`` arm (a directory holding its
   ``run-seed<d>`` runs, such as the data band's), each checked by its checkpoint.
4. Scores each locus in ``<screen>/assay/<locus>`` (``alignair_assay.workflow.diagnose.evaluate``, as ``calibrate``
   compares), then measures speed.
5. Writes ``locus_screen.json`` (kind ``alignair_lab.locus_screen``: the request, ``runs``, ``pairs`` per locus and
   seed, ``bands``, ``verdict`` per locus and metric, ``passes``, ``speed``, the code digests) and ``locus_screen.md``.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

from alignair_lab.baselines import NAME, now, write_json
from alignair_lab.calibrate import VARIANTS, train_screens
from alignair_lab.databand import check_seeds, train_command, unfinished
from alignair_lab.headline import HEADLINE
from alignair_lab.noise import null_band, pair
from alignair_lab.recipe import RECIPE, schedule as learning_schedule
from alignair_lab.runs import DEFAULT_DATA_WORKERS, DONE, check_run, run_request
from alignair_lab.screen import SCREENS, assay_layout, built_suites, code_digests, suite_builds
from alignair_lab.secondary import METRIC, cached_comparison, merged

LOCUS_SCREEN = "locus_screen.json"
REPORT = "locus_screen.md"
LOCUS_SCREEN_KIND = "alignair_lab.locus_screen"
LOCUS_SCREEN_VERSION = 1
HUMAN_LOCI = ("HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB", "HUMAN_TCRA_IMGT", "HUMAN_TCRB_IMGT",
              "HUMAN_TCRD_IMGT", "HUMAN_TCRG_IMGT")
LOCUS_STEPS = 20_000                      # a per-locus arm's steps; the shared arm trains this per dataconfig
DEFAULT_ARMS = {"all": (HUMAN_LOCI, LOCUS_STEPS * len(HUMAN_LOCI), ("IGH", "IGK", "TRB")),
                "IGH": (("HUMAN_IGH_OGRDB",), LOCUS_STEPS, ("IGH",)),
                "IGK": (("HUMAN_IGK_OGRDB",), LOCUS_STEPS, ("IGK",)),
                "TRB": (("HUMAN_TCRB_IMGT",), LOCUS_STEPS, ("TRB",))}
DEFAULT_LOCI = {"IGH": Path("/home/thomas/Desktop/AlignAIR/.private/arch/iter_dev"),
                "IGK": Path("/media/thomas/NVMe_Data/alignair_arch/dev_builds/HUMAN_IGK_OGRDB"),
                "TRB": Path("/media/thomas/NVMe_Data/alignair_arch/dev_builds/HUMAN_TCRB_IMGT")}
DEFAULT_NAME = f"locus-screen-{RECIPE}"
SPEED_SUITE = "S1"                        # the reads speed is measured on (a build's first suite without it)
SPEED_PASSES = 3                          # timed passes per measurement; the median is kept
CPU_THREADS = 8                           # the CPU speed's torch threads
BAND_FLOOR = 0.001                        # the smallest band: a tenth of a point
HEADLINE_METRICS = tuple(metric for metric, _ in HEADLINE)


def _arms(arms: Mapping, loci: Mapping) -> tuple[str, dict[str, str]]:
    """The shared arm's name and, per locus it is scored on, that locus's per-locus arm (ValueError otherwise)."""
    shared = [arm for arm, (_, _, scored) in arms.items() if len(scored) > 1]
    if len(shared) != 1:
        raise ValueError(f"a locus screen has one shared arm, scored on several loci; got {shared or 'none'}")
    per = {}
    for arm, (dataconfigs, _, scored) in arms.items():
        if arm == shared[0]:
            continue
        if len(scored) != 1 or len(dataconfigs) != 1:
            raise ValueError(f"arm {arm}: a per-locus arm trains on one dataconfig and is scored on one locus")
        per[scored[0]] = arm
    for locus in arms[shared[0]][2]:
        if locus not in per:
            raise ValueError(f"locus {locus}: the shared arm is scored on it, but no per-locus arm is")
        if locus not in loci:
            raise ValueError(f"locus {locus}: no dev build")
    return shared[0], per


def _check_builds(arms: Mapping, per: Mapping[str, str], loci: Mapping, names: Mapping) -> None:
    """ValueError unless every suite of each locus's dev build was built on its per-locus arm's dataconfig."""
    for locus, arm in per.items():
        want = arms[arm][0][0]
        for suite in names[locus]:
            built = json.loads((Path(loci[locus]) / "suites" / suite / "build.json").read_text(encoding="utf-8"))
            if built.get("reference") != want:
                raise ValueError(f"the {locus} dev build {loci[locus]} holds {suite} built on "
                                 f"{built.get('reference')}, not {want}")


def _reads_per_second(checkpoint: str, reads: Sequence[str], device: str) -> float:
    """Reads per second of the product pipeline (``alignair.api.predict_sequences``) on ``reads`` on ``device``: the
    median of ``SPEED_PASSES`` timed passes after a warm-up on the first 64."""
    import statistics

    import torch
    from alignair import api
    model, reference = api.load_model(checkpoint, device=device)

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    api.predict_sequences(model, reference, reads[:64], device=device)
    rates = []
    for _ in range(SPEED_PASSES):
        synchronize()
        start = time.perf_counter()
        api.predict_sequences(model, reference, reads, device=device)
        synchronize()
        rates.append(len(reads) / (time.perf_counter() - start))
    return statistics.median(rates)


def _speed(checkpoint: str, build: Path, names: Sequence[str], device: str) -> dict:
    """``{"device": reads/s, "cpu8": reads/s}`` on a build's speed suite's reads: on ``device``, and on the CPU with
    ``CPU_THREADS`` threads."""
    import torch
    from alignair_assay.core.cases import read_fasta
    suite = SPEED_SUITE if SPEED_SUITE in names else names[0]
    reads = [sequence for _, sequence in read_fasta(build / "suites" / suite / "reads.fasta")]
    out = {"device": _reads_per_second(checkpoint, reads, device)}
    threads = torch.get_num_threads()
    torch.set_num_threads(CPU_THREADS)
    try:
        out["cpu8"] = _reads_per_second(checkpoint, reads, "cpu")
    finally:
        torch.set_num_threads(threads)
    return out


def _verdict(pairs: Mapping[str, Mapping], band: Mapping[str, float]) -> dict:
    """Per metric some seed compares: the seeds' pooled Δs, their mean, the band (floored at ``BAND_FLOOR``) and the
    call (see the module docstring); ``unscored`` when some seed does not compare it or the band does not hold it."""
    out = {}
    compared = sorted({metric for found in pairs.values() for metric in found["pooled"]})
    for metric in compared:
        deltas = [found["pooled"][metric] for found in pairs.values() if metric in found["pooled"]]
        mean = sum(deltas) / len(deltas)
        if len(deltas) < len(pairs) or metric not in band:
            out[metric] = {"deltas": deltas, "mean_delta": mean, "band": band.get(metric), "call": "unscored"}
            continue
        width = max(band[metric], BAND_FLOOR)
        call = "worse" if mean < -width else "better" if mean > width else "within"
        out[metric] = {"deltas": deltas, "mean_delta": mean, "band": width, "call": call}
    return out


def _passes(verdict: Mapping[str, Mapping]) -> bool:
    """Whether no headline metric is worse or unscored on any locus (``verdict``: locus -> ``_verdict``)."""
    return not any(call["call"] in ("worse", "unscored") for calls in verdict.values()
                   for metric, call in calls.items() if metric in HEADLINE_METRICS)


def training_order(arms: Mapping) -> list[str]:
    """The arms in the order their runs train: the per-locus arms (scored on one locus) first, then the shared one."""
    return sorted(arms, key=lambda arm: len(arms[arm][2]) > 1)


def _report(record: Mapping) -> str:
    """``locus_screen.md``: per locus the headline metrics' calls, then speed, then the verdict."""
    shared = next(arm for arm, spec in record["arms"].items() if len(spec["loci"]) > 1)
    lines = [f"# Locus screen {record['name']} ({record['recipe']}, seeds {', '.join(map(str, record['seeds']))})", "",
             f"Shared arm `{shared}`: {', '.join(record['arms'][shared]['dataconfigs'])}, "
             f"{record['arms'][shared]['steps']} steps. Δ = shared − per-locus, in pp; band = the largest |pooled Δ| "
             "among the per-locus arm's seed pairs.", ""]
    for locus, calls in record["verdict"].items():
        lines += [f"## {locus}", "", "| metric | gated | mean Δ | band | call |", "|---|---|---|---|---|"]
        ordered = [m for m in HEADLINE_METRICS if m in calls] + sorted(m for m in calls if m not in HEADLINE_METRICS)
        for m in ordered:
            band = "n/a" if calls[m]["band"] is None else f"{100 * calls[m]['band']:.2f}"
            lines.append(f"| {m} | {'yes' if m in HEADLINE_METRICS else ''} | {100 * calls[m]['mean_delta']:+.2f} | "
                         f"{band} | {calls[m]['call']} |")
        lines.append("")
    loci = list(record["verdict"])
    lines += ["## Speed (reads/s: the screen's device / CPU, 8 threads)", "", "| arm | " + " | ".join(loci) + " |",
              "|---|" + "---|" * len(loci)]
    for arm, by_locus in record["speed"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{by_locus[locus]['device']:.1f} / {by_locus[locus]['cpu8']:.1f}"
                                                 if locus in by_locus else "" for locus in loci) + " |")
    lines += ["", f"**Verdict:** the shared arm {'passes' if record['passes'] else 'fails'}: "
                  f"{'no' if record['passes'] else 'a'} headline metric is worse than its locus's band on "
                  f"{'any locus' if record['passes'] else 'some locus'}.", ""]
    return "\n".join(lines)


def run_locus_screen(*, arms: Mapping = DEFAULT_ARMS, loci: Mapping = DEFAULT_LOCI, seeds: Sequence[int] = (0, 1, 2),
                     schedule: str = "cosine", parallel: int = 1, model_options: Mapping | None = None,
                     device: str = "cuda", data_workers: int = DEFAULT_DATA_WORKERS, eval_batch_size: int = 256,
                     max_cases: int | None = None, out_root=SCREENS, name: str = DEFAULT_NAME,
                     reuse: Mapping | None = None, echo: Callable[[str], None] | None = None) -> dict:
    """Run the locus screen (see the module docstring); returns its record (``locus_screen.json``). ``arms``: arm ->
    ``(dataconfigs, steps, loci it is scored on)``; ``loci``: locus -> dev build; ``reuse``: arm -> a directory
    holding that arm's finished ``run-seed<d>`` runs."""
    if echo is None:
        def echo(line: str) -> None:
            print(line, flush=True)
    seeds = check_seeds(seeds, what="locus screen")
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < 1:
        raise ValueError(f"parallel is a positive integer, got {parallel!r}")
    learning_schedule(schedule, 1)                           # an unknown schedule is refused here
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a locus screen name is 1 to 64 letters, digits, '.', '_' or '-', starting with a letter "
                         f"or digit; got {name!r}")
    model_options = json.loads(json.dumps(dict(model_options or {})))
    arms = {arm: (tuple(dataconfigs), int(steps), tuple(scored)) for arm, (dataconfigs, steps, scored) in arms.items()}
    shared, per = _arms(arms, loci)
    names = {locus: built_suites(Path(loci[locus]), None) for locus in arms[shared][2]}
    builds = {locus: suite_builds(Path(loci[locus]), names[locus]) for locus in names}
    _check_builds(arms, per, loci, names)
    reuse = {arm: Path(path) for arm, path in (reuse or {}).items()}
    for arm in reuse:
        if arm not in arms:
            raise ValueError(f"reuse names arm {arm!r}, which the screen does not have")

    def request_of(arm: str, seed: int) -> dict:
        dataconfigs, steps, _ = arms[arm]
        return run_request(seed=seed, steps=steps, schedule=schedule, dataconfigs=dataconfigs,
                           model_options=model_options)

    def run_dir(arm: str, seed: int) -> Path:
        return (reuse[arm] if arm in reuse else Path(out_root) / name / arm) / f"run-seed{seed}"

    for arm in reuse:                                        # a reused run is checked before anything runs
        for seed in seeds:
            if not (run_dir(arm, seed) / DONE).is_file():
                raise FileExistsError(f"reuse {arm}: {run_dir(arm, seed)} holds no finished run")
            check_run(run_dir(arm, seed), request_of(arm, seed))

    out_root = Path(out_root)
    directory = out_root / name
    request = {"kind": LOCUS_SCREEN_KIND, "version": LOCUS_SCREEN_VERSION, "recipe": RECIPE,
               "arms": {arm: {"dataconfigs": list(dataconfigs), "steps": steps, "loci": list(scored)}
                        for arm, (dataconfigs, steps, scored) in arms.items()},
               "seeds": seeds, "schedule": schedule, "options": model_options,
               "loci": {locus: str(loci[locus]) for locus in names}, "suite_builds": builds, "max_cases": max_cases,
               "reuse": {arm: str(path) for arm, path in reuse.items()}}
    previous = (unfinished(directory, name, request, record=LOCUS_SCREEN, what="locus screen")
                if directory.exists() else None)
    if previous is None:
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError:
            raise FileExistsError(f"a locus screen named {name!r} exists ({directory}); choose another "
                                  "--name") from None
    record = {**request, "name": name, "status": "preparing",
              "started": now() if previous is None else previous.get("started"), "device": device,
              "eval_batch_size": eval_batch_size, "runs": {}}
    if previous is not None:
        record["resumed"] = now()
        echo(f"[locus-screen] resuming the unfinished locus screen {name} ({previous.get('status')}): its finished "
             "runs are reused")

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        write_json(directory / LOCUS_SCREEN, record)
        echo(f"[locus-screen] {status}")

    try:
        plan = [(seed, f"{arm}-seed{seed}", train_command(run_dir(arm, seed), seed, steps=arms[arm][1],
                                                          schedule=schedule, model_options=model_options,
                                                          device=device, data_workers=data_workers,
                                                          dataconfigs=arms[arm][0]))
                for arm in training_order(arms) if arm not in reuse for seed in seeds
                if not (run_dir(arm, seed) / DONE).is_file()]
        stage(f"training {len(plan)} run(s), reusing {len(arms) * len(seeds) - len(plan)}")
        train_screens(plan, directory, parallel, echo, label="locus-screen", what="run", seed_label="data seed")
        runs = {arm: {str(seed): {"checkpoint": check_run(run_dir(arm, seed), request_of(arm, seed))["model"]}
                      for seed in seeds} for arm in arms}
        stage("scoring each locus", runs=runs)
        from alignair_assay.workflow.diagnose import evaluate
        pairs, bands, speed = {}, {}, {arm: {} for arm in arms}
        for locus in arms[shared][2]:
            assay = directory / "assay" / locus
            if assay.exists():                               # a resumed screen compares afresh
                shutil.rmtree(assay)
            layout = assay_layout(assay, Path(loci[locus]), names[locus])

            def compare(earlier: str, later: str, label: str) -> dict:
                result = evaluate(later, layout, names[locus], device=device, batch_size=eval_batch_size,
                                  variants=VARIANTS, profile=False, max_cases=max_cases, against=earlier, label=label)
                summary = result.against["summary"]
                extra = cached_comparison(layout.diagnostics_root, result.model_key, summary["baseline"], METRIC)
                return pair(merged(summary, extra))

            local = per[locus]
            pairs[locus] = {str(seed): compare(runs[local][str(seed)]["checkpoint"],
                                               runs[shared][str(seed)]["checkpoint"],
                                               f"{name}-{shared}-{locus}-seed{seed}") for seed in seeds}
            bands[locus] = null_band([compare(runs[local][str(a)]["checkpoint"], runs[local][str(b)]["checkpoint"],
                                              f"{name}-{local}-seed{b}") for a, b in combinations(seeds, 2)])
            for arm in (shared, local):
                speed[arm][locus] = _speed(runs[arm][str(seeds[0])]["checkpoint"], Path(loci[locus]), names[locus],
                                           device)
        verdict = {locus: _verdict(pairs[locus], bands[locus]) for locus in pairs}
        passes = _passes(verdict)
        record.update(status="done", pairs=pairs, bands=bands, verdict=verdict, passes=passes, speed=speed,
                      code=code_digests(), finished=now())
        write_json(directory / LOCUS_SCREEN, record)
        (directory / REPORT).write_text(_report(record), encoding="utf-8")
        echo(f"[locus-screen] done: the shared arm {'passes' if passes else 'fails'}; {directory / REPORT}")
        return record
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", finished=now())
        write_json(directory / LOCUS_SCREEN, record)
        echo("[locus-screen] failed")
        raise
