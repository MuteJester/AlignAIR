"""Replicates: a candidate screened from several init seeds, each against the same baseline screen, and pooled into one
report that sets the candidate's seeds against the calibration's baseline seeds (``noise.decide``).

The replicate of init seed 0 is the screen ``<name>``, of init seed k ``<name>-init<k>`` (``replicate_name``), each a
screen like any other (``screen.run_screen``).

``run_replicates(name, init_seeds=(0, 1, 2), ...)``:

1. **Refuses** before anything is written (ValueError, FileExistsError, FileNotFoundError, ConfigurationError among
   them): no init seed, one given twice or one that is not a uint32; a parallel count outside 1 to ``MAX_PARALLEL``; a
   name or report name that is not one path component; a report directory in use; a baseline that is no finished
   baseline screen of the protocol (``baselines.baseline_screen``, as a screen resolves it); a suite not built in the
   dev build; a replicate under way, or of another request than this one (``check_replicates``, naming the screen and
   what differs); a replicate's directory without a screen record; and with ``reuse``, a missing replicate.
2. Claims the report directory ``out_root/<report>`` (default ``report_name``) and records its progress in its
   ``report.json`` (a failure is recorded before it is raised).
3. Reuses each finished replicate of this request and trains each missing one with ``python -m alignair_lab screen``
   against that baseline screen (``calibrate.train_screens``), at most ``parallel`` at once, each one's output in
   ``<report>/<screen>.log``.
4. Pools them. Each replicate's comparison summary is the one its ``screen.json`` holds, so nothing is diagnosed again.
   The baseline seeds are the calibration's of the protocol (``baselines.null_band``), made relative to the baseline
   screen's seed (``baselines.against_calibration``). Writes ``report.json`` (the request, the screens, the baseline,
   the calibration, each seed's pooled Δs and each metric's decision) and ``headline.txt``
   (``headline.render_pooled_headline``, with the boundary tables of the first init seed's screen), and prints the
   headline.

A request is what the replicates share (``request_of``): the architecture and its model options, the steps and the
schedule, the protocol (recipe, cache, steps, schedule), the suites and their builds, the cases, the variants and the
baseline (its kind, checkpoint and options).
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from alignair.errors import ConfigurationError
from alignair_lab.baselines import (
    BASELINE_INIT_SEED,
    NAME,
    NULL_ARCHITECTURE,
    SCREEN_RECORD,
    against_calibration,
    baseline_screen,
    calibration_checkpoint,
    check_init_seed,
    now,
    null_band,
    protocol,
    write_json,
)
from alignair_lab.calibrate import train_screens
from alignair_lab.headline import HEADLINE, render_pooled_headline
from alignair_lab.noise import decide, pooled
from alignair_lab.recipe import BASELINE_ARCHITECTURE, schedule as learning_schedule, training_config
from alignair_lab.records import CachedRecordSource
from alignair_lab.screen import (
    CACHE,
    DEFAULT_DATA_WORKERS,
    DEFAULT_SCHEDULE,
    DEFAULT_STEPS,
    DEFAULT_VARIANTS,
    DEV_BUILD,
    SCREENS,
    built_suites,
    prepare_candidate,
    suite_builds,
)
from alignair_lab.sizes import size_line

DEFAULT_INIT_SEEDS = (0, 1, 2)
MAX_PARALLEL = 2                          # replicates trained at once: the GPU holds two screens beside other runs
REPORT = "report.json"
HEADLINE_FILE = "headline.txt"
REPORT_KIND, REPORT_VERSION = "alignair_lab.replicates", 1
REQUEST_FIELDS = ("architecture", "model_options", "steps", "schedule", "protocol", "suites", "suite_builds",
                  "max_cases", "variants")
BASELINE_FIELDS = ("kind", "checkpoint", "options")
CODE_FIELDS = ("alignair_lab", "alignair_assay")      # the code digests replicates must share (``screen.code_digests``)
DECIDED = tuple(metric for metric, _ in HEADLINE)    # the metrics pooled over the replicates, paired alike in each


def replicate_name(name: str, seed: int) -> str:
    """The screen of init seed ``seed`` of the replicates ``name``: ``name`` itself for init seed 0 (the registered
    baseline screen's), ``<name>-init<seed>`` for another."""
    return name if seed == BASELINE_INIT_SEED else f"{name}-init{seed}"


def report_name(name: str, seeds: Sequence[int]) -> str:
    """The default report directory of the replicates ``name`` of ``seeds``: ``report-<name>-seeds<seeds>``, the seeds
    run together when each is one digit, else joined by ``_``."""
    joiner = "" if all(seed < 10 for seed in seeds) else "_"
    return f"report-{name}-seeds{joiner.join(map(str, seeds))}"


def _canonical(value):
    return json.loads(json.dumps(value))


def request_of(record: Mapping) -> dict:
    """What a screen record shares with the other replicates of its request (see the module docstring)."""
    baseline = record.get("baseline") or {}
    return _canonical({**{field: record.get(field) for field in REQUEST_FIELDS},
                       "baseline": {field: baseline.get(field) for field in BASELINE_FIELDS}})


def _differences(found: Mapping, expected: Mapping) -> list[str]:
    fields = [field for field in REQUEST_FIELDS if found.get(field) != expected.get(field)]
    return fields + [f"baseline.{field}" for field in BASELINE_FIELDS
                     if found["baseline"].get(field) != expected["baseline"].get(field)]


def check_replicates(records: Mapping[int, Mapping], expected: Mapping | None = None) -> dict:
    """The request of ``records`` (init seed -> a replicate's screen record). ConfigurationError, naming the screen,
    for a replicate that has not finished, is not a candidate's, was trained from another init seed than its own, or
    is of another request than ``expected`` (default: the first replicate's), naming the fields that differ."""
    want = None if expected is None else _canonical(expected)
    for seed, record in records.items():
        name = record.get("name") or "?"
        what = f"the replicate {name} (init seed {seed})"
        context = {"field": "replicates", "screen": name}
        if record.get("status") != "done":
            raise ConfigurationError(f"{what} has not finished (status {record.get('status')!r})", context=context,
                                     hint="Wait for it, or move it away to train it again.")
        if record.get("role") != "candidate":
            raise ConfigurationError(f"{what} is not a candidate's screen: its role is {record.get('role')!r}",
                                     context=context)
        if record.get("init_seed") != seed:
            raise ConfigurationError(f"{what} was trained from init seed {record.get('init_seed')!r}", context=context)
        found = request_of(record)
        if want is None:
            want = found
            continue
        differing = _differences(found, want)
        if differing:
            raise ConfigurationError(f"{what} is of another request than the replicates': {', '.join(differing)} "
                                     "differ", context={**context, "differing": differing},
                                     hint="Replicates share everything but their init seeds: move the screen away, or "
                                          "name the request it was screened with.")
    digests = [(seed, record.get("name"), _code(record)) for seed, record in records.items() if _code(record)]
    for seed, name, code in digests[1:]:
        differing = [f"code.{field}" for field in CODE_FIELDS if code[field] != digests[0][2][field]]
        if differing:
            raise ConfigurationError(f"the replicate {name} (init seed {seed}) was screened under other code than "
                                     f"{digests[0][1]}: {', '.join(differing)} differ",
                                     context={"field": "replicates", "screen": name, "differing": differing},
                                     hint="Replicates of one request are screened by one lab and one assay code.")
    return want


def _pairing(comparison: Mapping, metrics: Sequence[str]) -> dict[str, dict]:
    """Each suite of a comparison summary -> its ``cases`` and each of ``metrics``' paired reads."""
    return {name: {"cases": suite.get("cases"),
                   **{metric: (suite.get("headline") or {}).get(metric, {}).get("n_paired") for metric in metrics}}
            for name, suite in comparison["suites"].items()}


def check_pairing(records: Mapping[int, Mapping], metrics: Sequence[str]) -> None:
    """ConfigurationError, naming the replicate and what differs, unless every replicate's comparison has the first's
    suites, and in each suite its number of cases and, for each of ``metrics``, its number of paired reads: their Δs
    are pooled suite by suite, on the same reads."""
    first = next(iter(records.values()))
    want = _pairing(first["comparison"], metrics)
    for seed, record in records.items():
        found = _pairing(record["comparison"], metrics)
        what = f"the replicate {record.get('name')} (init seed {seed}) is not paired as {first.get('name')}"
        context = {"field": "replicates", "screen": record.get("name")}
        if set(found) != set(want):
            differing = sorted(set(found) ^ set(want), key=lambda name: (len(name), name))
            raise ConfigurationError(f"{what}: suites {', '.join(differing)} are not in both", context=context)
        for suite, entries in want.items():
            for field, expected in entries.items():
                if found[suite][field] != expected:
                    unit = "" if field == "cases" else " paired reads"
                    raise ConfigurationError(f"{what}: {suite} {field}: {found[suite][field]}{unit}, not "
                                             f"{expected}", context=context)


def _code(record: Mapping) -> dict | None:
    """A screen record's lab and assay code digests (``screen.code_digests``), or None when it records them not."""
    code = record.get("code") or {}
    return {field: code[field] for field in CODE_FIELDS} if all(code.get(field) for field in CODE_FIELDS) else None


def code_warning(records: Mapping[int, Mapping]) -> str | None:
    """The warning line for the replicates that record no lab or assay code digest (screened before they were kept),
    so their code cannot be checked to be the others'; None when every one records them."""
    unknown = [record.get("name") or str(seed) for seed, record in records.items() if _code(record) is None]
    if not unknown:
        return None
    records_them = "it records" if len(unknown) == 1 else "they record"
    return (f"Warning: code digest unknown for {', '.join(unknown)}: {records_them} no alignair_lab or alignair_assay "
            "code digest, so the lab's and the assay's code cannot be checked to be the same.")


def check_calibration(records: Mapping[int, Mapping], band: Mapping | None, init_seed: int | None) -> None:
    """ConfigurationError, naming the replicate and both model keys, when the calibration of ``band`` (None: there is
    none, nothing to check) cannot weigh a replicate's comparison: its seed-``init_seed`` screen is not the replicate's
    baseline checkpoint (``baselines.calibration_checkpoint``)."""
    if band is None:
        return
    for seed, record in records.items():
        refused = calibration_checkpoint(band, init_seed, record["comparison"]["baseline"])
        if refused:
            raise ConfigurationError(f"the replicate {record.get('name')} (init seed {seed}) cannot be pooled against "
                                     f"the calibration {band['path']}: {refused}",
                                     context={"field": "replicates", "screen": record.get("name")})


def expected_request(*, steps: int, architecture: str, model_options: Mapping, suites: Sequence[str] | None,
                     max_cases: int | None, variants: Sequence[str], cache, dev_build, out_root, schedule: str,
                     against_screen: str | None, baseline_options: Mapping) -> tuple[dict, dict]:
    """``(request, provenance)``: the request a screen of these arguments records (``request_of``), and the provenance
    of the baseline screen it compares with (``baselines.baseline_screen``)."""
    learning = learning_schedule(schedule, steps)
    source = CachedRecordSource(cache)
    wanted = protocol(source.manifest, steps, learning)
    provenance = baseline_screen(Path(out_root), wanted, baseline_options, name=against_screen, cache=source.path,
                                 schedule=schedule)
    names = built_suites(Path(dev_build), suites)
    request = {"architecture": architecture, "model_options": dict(model_options), "steps": steps,
               "schedule": schedule, "protocol": wanted, "suites": names,
               "suite_builds": suite_builds(Path(dev_build), names), "max_cases": max_cases, "variants": list(variants),
               "baseline": {field: provenance.get(field) for field in BASELINE_FIELDS}}
    return _canonical(request), provenance


def _option_arguments(flag: str, options: Mapping) -> list[str]:
    return [part for key, value in options.items() for part in (flag, f"{key}={json.dumps(value)}")]


def _command(name: str, seed: int, request: Mapping, *, against: str, device: str, data_workers: int,
             eval_batch_size: int, cache, dev_build, out_root) -> list[str]:
    cases = [] if request["max_cases"] is None else ["--max-cases", str(request["max_cases"])]
    return [sys.executable, "-m", "alignair_lab", "screen", "--name", name, "--steps", str(request["steps"]),
            "--architecture", request["architecture"], "--init-seed", str(seed), "--schedule", request["schedule"],
            *_option_arguments("--model-option", request["model_options"]),
            *_option_arguments("--baseline-model-option", request["baseline"]["options"]),
            "--suites", ",".join(request["suites"]), *cases, "--variants", ",".join(request["variants"]),
            "--against-screen", against, "--device", device, "--data-workers", str(data_workers),
            "--eval-batch-size", str(eval_batch_size), "--cache", str(cache), "--dev-build", str(dev_build),
            "--out-root", str(out_root)]


def _seeds(init_seeds: Sequence[int]) -> list[int]:
    seeds = [check_init_seed(seed) for seed in init_seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"replicates need one init seed or more, each once; got {seeds}")
    return seeds


def run_replicates(name: str, *, init_seeds: Sequence[int] = DEFAULT_INIT_SEEDS, parallel: int = 1,
                   reuse: bool = False, report: str | None = None, steps: int = DEFAULT_STEPS,
                   architecture: str = BASELINE_ARCHITECTURE, model_options: Mapping | None = None,
                   suites: Sequence[str] | None = None, max_cases: int | None = None,
                   variants: Sequence[str] = DEFAULT_VARIANTS, device: str = "cuda",
                   data_workers: int = DEFAULT_DATA_WORKERS, eval_batch_size: int = 256, cache=CACHE,
                   dev_build=DEV_BUILD, out_root=SCREENS, schedule: str = DEFAULT_SCHEDULE,
                   against_screen: str | None = None, baseline_options: Mapping | None = None,
                   echo: Callable[[str], None] | None = None) -> dict:
    """Screen ``architecture`` with ``model_options`` from each of ``init_seeds`` and pool the replicates (see the
    module docstring); returns the report record (``report.json``) with its printed ``headline``. ``reuse``: train
    nothing, pool the finished replicates only. ``echo`` receives the progress lines and the headline (default:
    printed)."""
    if echo is None:
        def echo(line: str) -> None:
            print(line, flush=True)
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"a replicates name is 1 to 64 letters, digits, '.', '_' or '-', starting with a letter or "
                         f"digit; got {name!r}")
    seeds = _seeds(init_seeds)
    if isinstance(parallel, bool) or not isinstance(parallel, int) or not 1 <= parallel <= MAX_PARALLEL:
        raise ValueError(f"parallel is 1 to {MAX_PARALLEL} replicates trained at once, got {parallel!r}")
    report = report_name(name, seeds) if report is None else report
    if not isinstance(report, str) or not NAME.fullmatch(report):
        raise ValueError(f"a report name is one path component, got {report!r}")
    out_root = Path(out_root)
    directory = out_root / report
    if directory.exists():
        raise FileExistsError(f"a report named {report!r} exists ({directory}); choose another --report")
    request, provenance = expected_request(steps=steps, architecture=architecture,
                                           model_options=_canonical(dict(model_options or {})), suites=suites,
                                           max_cases=max_cases, variants=variants, cache=cache, dev_build=dev_build,
                                           out_root=out_root, schedule=schedule, against_screen=against_screen,
                                           baseline_options=dict(baseline_options or {}))
    records, plan = {}, []
    for seed in seeds:
        screen = replicate_name(name, seed)
        if (out_root / screen / SCREEN_RECORD).is_file():
            records[seed] = json.loads((out_root / screen / SCREEN_RECORD).read_text(encoding="utf-8"))
        elif (out_root / screen).exists():
            raise FileExistsError(f"the replicate of init seed {seed}, {out_root / screen}, holds no screen record: "
                                  "move it away")
        elif reuse:
            raise FileNotFoundError(f"the replicate of init seed {seed}, {screen}, is not in {out_root}, and --reuse "
                                    "trains none")
        else:
            plan.append((seed, screen, _command(screen, seed, request, against=provenance["name"], device=device,
                                                data_workers=data_workers, eval_batch_size=eval_batch_size,
                                                cache=cache, dev_build=dev_build, out_root=out_root)))
    check_replicates(records, expected=request)
    if records:
        check_pairing(records, DECIDED)
    band = null_band(out_root, request["protocol"], request["baseline"]["options"], request["suites"],
                     request["max_cases"], request["suite_builds"])
    check_calibration(records, band, provenance["init_seed"])
    if plan:
        prepare_candidate(training_config(steps=steps, architecture=architecture,
                                          model_options=request["model_options"],
                                          schedule=learning_schedule(schedule, steps)))
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError:
        raise FileExistsError(f"a report named {report!r} exists ({directory}); choose another --report") from None

    record = {"kind": REPORT_KIND, "version": REPORT_VERSION, "name": report, "replicates": name, "status": "preparing",
              "started": now(), "init_seeds": seeds, "request": request,
              "screens": {str(seed): {"name": replicate_name(name, seed), "reused": seed in records} for seed in seeds},
              "baseline": {"name": provenance["name"], "init_seed": provenance["init_seed"],
                           "checkpoint": provenance["checkpoint"]}}

    def stage(status: str, **fields):
        record.update(status=status, **fields)
        write_json(directory / REPORT, record)
        echo(f"[replicates] {status}")

    try:
        stage(f"training {len(plan)} replicate(s), reusing {len(records)}")
        train_screens(plan, directory, parallel, echo, label="replicates", what="replicate")
        for seed, screen, _ in plan:
            records[seed] = json.loads((out_root / screen / SCREEN_RECORD).read_text(encoding="utf-8"))
        records = {seed: records[seed] for seed in seeds}
        check_replicates(records, expected=request)
        check_pairing(records, DECIDED)
        check_calibration(records, band, provenance["init_seed"])
        calibration = against_calibration(band, provenance["init_seed"])
        comparisons = {seed: records[seed]["comparison"] for seed in seeds}
        metrics = [metric for metric, _ in HEADLINE]
        decisions = decide(list(comparisons.values()), (calibration or {}).get("seeds"), metrics,
                           band=(calibration or {}).get("band"))
        title = _title(name, request, seeds, records, provenance, band, calibration)
        headline = render_pooled_headline(comparisons, decisions=decisions, title=title,
                                          boundaries=(seeds[0], replicate_name(name, seeds[0])))
        (directory / HEADLINE_FILE).write_text(headline + "\n", encoding="utf-8")
        stored = None if band is None else {key: value for key, value in band.items() if key != "seeds"}
        record.update(calibration=stored, decisions=decisions, headline=str(directory / HEADLINE_FILE),
                      per_seed={metric: {str(seed): pooled(comparisons[seed], metric) for seed in seeds}
                                for metric in metrics},
                      finished=now())
        stage("done")
        echo("")
        echo(headline)
        return {**record, "headline": headline}
    except BaseException as error:
        record.update(error=f"{type(error).__name__}: {error}", finished=now())
        stage("failed")
        raise


def _title(name: str, request: Mapping, seeds: Sequence[int], records: Mapping[int, Mapping], provenance: Mapping,
           band: Mapping | None, calibration: Mapping | None) -> list[str]:
    options = f" with {json.dumps(request['model_options'])}" if request["model_options"] else ""
    rates = ", ".join(f"{records[seed]['name']} (init seed {seed}, "
                      f"{(records[seed].get('candidate') or {}).get('steps_per_second', 'unrecorded')} steps/s)"
                      for seed in seeds)
    lines = [f"Replicates {name}: {request['architecture']}{options}, {request['steps']} steps ({request['schedule']} "
             f"schedule), init seeds {', '.join(map(str, seeds))}, each against the baseline screen "
             f"{provenance['name']} ({NULL_ARCHITECTURE}, init seed {provenance['init_seed']}).",
             f"Screens: {rates}."]
    size = (records[seeds[0]].get("candidate") or {}).get("size")
    if size:
        lines.append(size_line("candidate", size, records[seeds[0]]["candidate"].get("steps_per_second")))
    if band is None:
        lines.append("Calibration: none of this protocol, options, suite build and cases, so uncalibrated.")
    else:
        lines.append(f"Calibration: {band['path']} (baseline screens of init seeds "
                     f"{', '.join(map(str, band['init_seeds']))}; their Δs against init seed "
                     f"{provenance['init_seed']}'s screen).")
        if calibration is None or calibration["seeds"] is None:
            lines.append(f"Uncalibrated: the calibration has no baseline screen of init seed "
                         f"{provenance['init_seed']}, the replicates' baseline screen's.")
    codes = list(dict.fromkeys((records[seed]["comparison"].get("code") or {}).get("this") for seed in seeds))
    if len(codes) > 1:
        lines.append("Warning: the replicates were diagnosed under different AlignAIR code "
                     f"({', '.join((code or 'unknown')[:12] for code in codes)}): their Δs hold the code changes too.")
    elif band is not None and band.get("code_sha256") != codes[0]:
        lines.append(f"Warning: the calibration was measured under other AlignAIR code "
                     f"({(band.get('code_sha256') or 'unrecorded')[:12]}) than the replicates "
                     f"({(codes[0] or 'unknown')[:12]}).")
    unknown = code_warning(records)
    if unknown:
        lines.append(unknown)
    return [*lines, ""]
