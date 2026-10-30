"""The record cache: a recipe's curriculum records generated once, written to disk, and replayed by any number of runs.

``training_records`` zips independent record streams (a curriculum component interleaves several), and each stream
is one provider call ``(reference, parameters, seed)``, listed in order by ``gym.workers.component_plan`` (which calls
each a "component"). ``write_record_cache`` runs each planned stream through the default GenAIRR provider
(``gym.streams.stream_records``), one process per stream, for as many records as ``steps * batch_size`` records of
the zipped training stream (run seed, start step 0) draw from it (``component_counts``), and writes them in
compressed chunks. ``CachedRecordSource`` is a record provider (``training_records``' and ``train``'s
``record_source``) that replays the stream it is asked for: the stream it gives is the live stream, record for
record, up to ``steps * batch_size`` records.

Layout::

    PATH/manifest.json                     written last: a directory without it is not a cache
    PATH/<n>-seed<s>/chunk-<k>.pkl.zlib    stream n (seed s): CHUNK_RECORDS of its records per chunk, in order

The manifest records the GenAIRR the records come from (``genairr``: the assay's provenance, whose
``genairr_source_sha256`` is ``alignair_assay.core.provenance.tree_digest`` of the package directory), the product's
training ``recipe`` id, the dataconfig names, every field of the stream policy (``policy``), the seed, purpose and
start step, the steps and batch size the cache serves, the code that produced the records (``code``:
``PRODUCING_CODE``, provenance only), and per stream its seed, reference (name and DataConfig checksum), parameters
(``encode_parameters``), record count and chunks (file, records, bytes, SHA-256).

**Refused** (ConfigurationError), at the provider call and in whichever process makes it: a process importing another
GenAIRR than the cache's, a seed the cache has no stream of, another reference or other parameters than the cached
stream's of that seed, and a chunk whose bytes are not those it was written as. A stream read to its end raises
TrainingDataError: the stream never wraps around. ``CachedRecordSource.require`` checks a whole stream before any
record is drawn, the recipe and every policy field named, and then draws the first records of every stream live:
a change to the code that produces the records (the provider, the seed blocks, the interleave, the curriculum) leaves
the manifest as it was, so only drawing the stream again shows it.

The chunks are pickles: a cache is trusted local data, written by this module.
"""
from __future__ import annotations

import dataclasses
import datetime
import functools
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import secrets
import shutil
import signal
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from itertools import islice, repeat
from operator import index
from pathlib import Path

import alignair
from alignair.errors import ConfigurationError, TrainingDataError, error_context
from alignair.train.gym.streams import CurriculumStreamPolicy, dataconfig_name, stream_records, training_records
from alignair.train.gym.workers import component_plan
from alignair.train.trainer import RECIPE
from alignair_assay.core.provenance import files_digest, genairr_provenance, tree_digest

KIND = "alignair_lab.record_cache"
VERSION = 2
MANIFEST = "manifest.json"
# Records per chunk file. At about 600 bytes a record compressed (3.7 kB pickled), a chunk is about 2.5 MB; each of the
# 28 streams of a 40k-step cache of the recipe holds about 91,000 records, 23 chunks.
CHUNK_RECORDS = 4096
COMPRESSION_LEVEL = 6
# The product code, besides GenAIRR, that turns a stream's parameters and seed into records (paths under the alignair
# package; a directory stands for its Python files): the provider and its scoping, the seed blocks and the interleave,
# the experiment builder and the curriculum. Recorded as provenance; ``require``'s live draw is the guard.
PRODUCING_CODE = ("train/gym/streams.py", "train/gym/experiment.py", "train/gym/curriculum")
# Records of every stream ``require`` draws live and compares with the cache, at most what the cache holds per stream.
VERIFY_RECORDS = 2
_HINT = "Write a cache for this stream: python -m alignair_lab cache --out PATH --steps N."


def _count(value, field: str) -> int:
    try:
        if isinstance(value, bool) or index(value) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a positive integer", context={"field": field}) from exc
    return index(value)


@functools.lru_cache(maxsize=None)
def genairr_digest(directory: str) -> str:
    """``tree_digest`` of a GenAIRR package directory, computed once per process: the assay's
    ``genairr_source_sha256``."""
    return tree_digest(directory)


def imported_genairr() -> str:
    """The content digest of the GenAIRR package this process imports."""
    import GenAIRR
    return genairr_digest(str(Path(GenAIRR.__file__).resolve().parent))


def producing_code() -> dict:
    """The ``PRODUCING_CODE`` files as they are now: each one's SHA-256 (``files``, by path under the alignair package)
    and their ``files_digest`` (``sha256``)."""
    root = Path(alignair.__file__).resolve().parent
    paths = sorted({path for name in PRODUCING_CODE for path in
                    ((root / name).rglob("*.py") if (root / name).is_dir() else [root / name])
                    if "__pycache__" not in path.parts})
    return {"files": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in paths},
            "sha256": files_digest(root, paths)}


def dataconfig_names(dataconfigs) -> list[str]:
    """The names the manifest records for ``dataconfigs``: a name as given, else ``dataconfig_name``."""
    return [value if isinstance(value, str) else dataconfig_name(value) for value in dataconfigs]


def policy_fields(policy: CurriculumStreamPolicy) -> dict:
    """Every field of a stream policy, encoded as the manifest stores it (``encode_parameters``)."""
    return encode_parameters({field.name: getattr(policy, field.name) for field in dataclasses.fields(policy)})


def reference_sha256(reference) -> str | None:
    """A reference's content checksum (``DataConfig.compute_checksum``); None for a reference without one."""
    checksum = getattr(reference, "compute_checksum", None)
    return checksum() if callable(checksum) else None


_TUPLE = "__tuple__"


def encode_parameters(value):
    """Stream parameters as the manifest stores them, losslessly: JSON keeps ints, floats and booleans apart (1, 1.0
    and true); a tuple becomes ``{"__tuple__": [...]}``, so it stays apart from a list. ConfigurationError for a value
    JSON cannot hold exactly (NaN, infinities, other types, non-string keys, a ``__tuple__`` key)."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key != _TUPLE for key in value):
            raise ConfigurationError("cached parameters need string keys other than '__tuple__'",
                                     context={"field": "parameters"})
        return {key: encode_parameters(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {_TUPLE: [encode_parameters(item) for item in value]}
    if isinstance(value, list):
        return [encode_parameters(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ConfigurationError(f"cached parameters must be finite JSON values, tuples or mappings; got {value!r}",
                             context={"field": "parameters"})


def canonical_parameters(parameters, *, encoded: bool = False) -> str:
    """One string per distinct value of ``parameters`` (``encode_parameters``; ``encoded``: already encoded, as the
    manifest holds them): equal strings mean equal parameters, ints, floats, booleans, tuples and lists apart."""
    return json.dumps(parameters if encoded else encode_parameters(parameters), sort_keys=True, allow_nan=False)


def component_counts(dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, records: int) -> tuple[int, ...]:
    """How many records each planned component (``component_plan`` order) gives when ``records`` records of
    ``training_records(dataconfigs, policy=policy, seed=seed)`` are drawn: counted on that stream itself, so the
    zip's complete rounds are included."""
    records = _count(records, "records")
    counts = []

    def counting(reference, parameters, component_seed):
        position = len(counts)
        counts.append(0)

        def generate():
            while True:
                counts[position] += 1
                yield {}
        return generate()

    for _ in islice(training_records(dataconfigs, policy=policy, seed=seed, record_source=counting), records):
        pass
    return tuple(counts)


# A writer process's view of the shared progress: each stream's records written and the time its writing started.
_PROGRESS = None
PROGRESS_RECORDS = 256         # records a writer generates between progress counts


def _start_writer(written, started):
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # the writing process stops its workers
    global _PROGRESS
    _PROGRESS = written, started


def _write_component(item):
    """A writer process: generate one stream's records with the default provider and write its chunks."""
    position, (root, name, reference, parameters, seed, count, chunk_records, genairr) = item
    start = time.perf_counter()
    if imported_genairr() != genairr:
        raise ConfigurationError("a cache writer process imported another GenAIRR than the process that started it",
                                 context={"stage": "record_cache", "seed": seed},
                                 hint="Put the intended GenAIRR first on PYTHONPATH.")
    (Path(root) / name).mkdir()
    if _PROGRESS is not None:
        _PROGRESS[1][position] = time.time()
    records = iter(stream_records(reference, parameters, seed))
    chunks, written = [], 0
    while written < count:
        chunk, size = [], min(chunk_records, count - written)
        while len(chunk) < size:
            part = list(islice(records, min(PROGRESS_RECORDS, size - len(chunk))))
            if not part:
                raise TrainingDataError("a record stream ended before the cache was full",
                                        context={"stage": "record_cache", "seed": seed,
                                                 "records": written + len(chunk)})
            chunk.extend(part)
            if _PROGRESS is not None:
                _PROGRESS[0][position] += len(part)             # only this process writes this stream's count
        data = zlib.compress(pickle.dumps(chunk, pickle.HIGHEST_PROTOCOL), COMPRESSION_LEVEL)
        file = f"{name}/chunk-{len(chunks):06d}.pkl.zlib"
        (Path(root) / file).write_bytes(data)
        chunks.append({"file": file, "records": len(chunk), "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()})
        written += len(chunk)
    return position, {"chunks": chunks, "seconds": round(time.perf_counter() - start, 3)}


def default_writers(streams: int) -> int:
    """The writer processes used when none is asked for: one per CPU, at most one per stream."""
    return max(1, min(os.cpu_count() or 1, streams))


def writing_order(plan) -> list[int]:
    """The order streams are handed to writer processes: the selected streams (``productive_only``), whose records
    take about 20 times longer to simulate, first, so the slowest work starts first; each group in plan order."""
    selected = [position for position, stream in enumerate(plan) if stream.parameters.get("productive_only")]
    return selected + [position for position in range(len(plan)) if position not in set(selected)]


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{int(seconds // 3600)} h {int(seconds % 3600 // 60)} min"


def seconds_left(*, done, totals, running, selected, finished, workers) -> tuple[float | None, bool]:
    """The writer's time left, per stream: ``done`` of ``totals`` records after ``running`` seconds, ``selected`` or
    not, ``finished`` or not. Each unfinished stream's remaining records take their time at its own rate, or, before
    it has written any, at the mean rate of the streams of its kind that have; that work is shared among ``workers``
    (at most one per unfinished stream), and the time left is never less than the slowest running stream's own.
    Returns ``(seconds, complete)``: a stream of a kind no stream has measured yet is left out, ``complete`` is then
    False and the seconds a lower bound; ``(None, False)`` before anything is measured."""
    rates = [count / seconds if count > 0 and seconds > 0 else None for count, seconds in zip(done, running)]
    kinds = {}
    for rate, kind in zip(rates, selected):
        if rate is not None:
            kinds.setdefault(kind, []).append(rate)
    unfinished = [index for index, over in enumerate(finished) if not over and totals[index] > done[index]]
    if not unfinished:
        return 0., True
    work, slowest, complete = 0., 0., True
    for index in unfinished:
        own = rates[index]
        rate = own if own is not None else (sum(kinds[selected[index]]) / len(kinds[selected[index]])
                                            if selected[index] in kinds else None)
        if rate is None:
            complete = False
            continue
        left = (totals[index] - done[index]) / rate
        work += left
        if own is not None:
            slowest = max(slowest, left)
    if work == 0 and not complete:
        return None, False
    return max(work / min(workers, len(unfinished)), slowest), complete


def _progress_line(done: int, total: int, finished: int, streams: int, elapsed: float,
                   left: tuple[float | None, bool]) -> str:
    seconds, complete = left
    eta = "unknown" if seconds is None else (_duration(seconds) if complete else f"at least {_duration(seconds)}")
    return (f"{done}/{total} records ({100 * done // max(total, 1)}%), {finished} of {streams} streams done, "
            f"{done / elapsed if elapsed > 0 else 0.:.1f} records/s, ETA {eta}")


def write_record_cache(path, *, dataconfigs, policy: CurriculumStreamPolicy, seed: int, steps: int, batch_size: int,
                       workers: int | None = None, chunk_records: int = CHUNK_RECORDS,
                       progress: Callable[[str], None] | None = None, progress_seconds: float = 60.) -> dict:
    """Write the records ``steps`` steps of ``batch_size`` draw from ``training_records(dataconfigs, policy=policy,
    seed=seed)`` to the new directory ``path``; return the manifest.

    Each stream is generated in a spawned process of its own, at most ``workers`` at a time (default
    ``default_writers``), the selected streams first (``writing_order``). Every ``progress_seconds`` and once at the
    end, ``progress`` (when given) receives a line with the records written, the streams done, the records per second
    and the time left (``seconds_left``). The cache is written into a sibling directory and renamed to ``path`` once its
    manifest is written, so an interrupted writer leaves no cache behind; it cannot resume one: an interrupted write
    starts again. FileExistsError when ``path`` exists.
    """
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(f"{path} exists; a record cache is written once, to a new path")
    if not isinstance(policy, CurriculumStreamPolicy):
        raise ConfigurationError("supply a CurriculumStreamPolicy", context={"field": "policy"})
    steps, batch_size, chunk_records = (_count(steps, "steps"), _count(batch_size, "batch_size"),
                                        _count(chunk_records, "chunk_records"))
    if isinstance(dataconfigs, (str, bytes)):
        raise ConfigurationError("dataconfigs must be a collection of references or names",
                                 context={"field": "dataconfigs"})
    dataconfigs = tuple(dataconfigs)
    start = time.perf_counter()
    plan = component_plan(dataconfigs, policy=policy, seed=seed)
    counts = component_counts(dataconfigs, policy=policy, seed=seed, records=steps * batch_size)
    processes = default_writers(len(plan)) if workers is None else min(_count(workers, "workers"), len(plan))
    if isinstance(progress_seconds, bool) or not isinstance(progress_seconds, (int, float)) or progress_seconds <= 0:
        raise ConfigurationError("progress_seconds must be a positive number", context={"field": "progress_seconds"})
    genairr = genairr_provenance()
    if genairr["genairr_source_sha256"] != imported_genairr():
        raise ConfigurationError("GenAIRR changed while the cache was being prepared",
                                 context={"stage": "record_cache"})
    components = [{"seed": component.seed, "reference": dataconfig_name(component.reference),
                   "reference_sha256": reference_sha256(component.reference),
                   "parameters": encode_parameters(component.parameters), "records": count,
                   "directory": f"{position:03d}-seed{component.seed}"}
                  for position, (component, count) in enumerate(zip(plan, counts))]
    path.parent.mkdir(parents=True, exist_ok=True)
    # A plain mkdir gives the directory the permissions of any new one; the random suffix keeps writers apart.
    partial = path.parent / f".{path.name}.partial-{os.getpid()}-{secrets.token_hex(4)}"
    partial.mkdir()
    try:
        tasks = [(str(partial), entry["directory"], component.reference, component.parameters, component.seed,
                  entry["records"], chunk_records, genairr["genairr_source_sha256"])
                 for entry, component in zip(components, plan)]
        totals = [entry["records"] for entry in components]
        selected = [bool(component.parameters.get("productive_only")) for component in plan]
        context = multiprocessing.get_context("spawn")
        written, started = context.Array("q", len(plan)), context.Array("d", len(plan))
        writing = time.perf_counter()

        def report():
            now, done = time.time(), list(written)
            running = [entry["seconds"] if "seconds" in entry else (now - begun if begun else 0.)
                       for entry, begun in zip(components, started)]
            finished = ["seconds" in entry for entry in components]
            left = seconds_left(done=done, totals=totals, running=running, selected=selected, finished=finished,
                                workers=processes)
            progress(_progress_line(sum(done), sum(totals), sum(finished), len(plan),
                                    time.perf_counter() - writing, left))

        with context.Pool(processes, initializer=_start_writer, initargs=(written, started)) as pool:
            pending = {position: pool.apply_async(_write_component, ((position, tasks[position]),))
                       for position in writing_order(plan)}
            reported = time.monotonic()
            while pending:
                next(iter(pending.values())).wait(min(1., progress_seconds))
                for position in [position for position, result in pending.items() if result.ready()]:
                    components[position].update(pending.pop(position).get()[1])   # a writer's error is raised here
                if progress is not None and time.monotonic() - reported >= progress_seconds and pending:
                    reported = time.monotonic()
                    report()
        if progress is not None:
            report()
        manifest = {
            "kind": KIND, "version": VERSION,
            "recipe": RECIPE, "dataconfigs": dataconfig_names(dataconfigs), "policy": policy_fields(policy),
            "seed": seed, "purpose": "training", "start_step": 0,
            "steps": steps, "batch_size": batch_size, "records": steps * batch_size,
            "genairr": genairr, "code": producing_code(),
            "chunk_records": chunk_records, "compression": f"zlib-{COMPRESSION_LEVEL}",
            "bytes": sum(chunk["bytes"] for entry in components for chunk in entry["chunks"]),
            "workers": processes, "seconds": round(time.perf_counter() - start, 3),
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "components": components,
        }
        (partial / f"{MANIFEST}.tmp").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        os.replace(partial / f"{MANIFEST}.tmp", partial / MANIFEST)
        os.rename(partial, path)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


class CachedRecordSource:
    """A record provider replaying the record cache at ``path`` (see the module docstring); picklable as its path,
    so data workers can replay their components. Each process reads the manifest once, when first asked."""

    def __init__(self, path):
        self.path = os.fspath(Path(path).expanduser().resolve())
        self._manifest = None

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path, self._manifest = state["path"], None

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            try:
                manifest = json.loads((Path(self.path) / MANIFEST).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigurationError(f"{self.path} is not a complete record cache: no readable {MANIFEST}",
                                         context={"stage": "record_cache", "cache": self.path},
                                         hint="A cache is complete once write_record_cache has renamed it into "
                                              "place. " + _HINT) from exc
            if manifest.get("kind") != KIND or manifest.get("version") != VERSION:
                raise ConfigurationError(f"{self.path} is not a version {VERSION} record cache",
                                         context={"stage": "record_cache", "cache": self.path,
                                                  "kind": manifest.get("kind"), "version": manifest.get("version")})
            self._manifest = manifest
        return self._manifest

    def __call__(self, reference, parameters: Mapping, seed: int) -> Iterator[Mapping]:
        return self._replay(self.component(reference, parameters, seed))

    def component(self, reference, parameters: Mapping, seed: int) -> dict:
        """The manifest's entry of the stream this provider call asks for; refused (ConfigurationError) unless it was
        generated by this process's GenAIRR, from the same reference and parameters."""
        manifest = self.manifest
        context = {"stage": "record_cache", "cache": self.path, "seed": seed}
        imported, cached = imported_genairr(), manifest["genairr"]["genairr_source_sha256"]
        if imported != cached:
            raise ConfigurationError("the record cache was generated by another GenAIRR than this process imports",
                                     context={**context, "cache_genairr": cached, "imported_genairr": imported},
                                     hint="Put the GenAIRR the cache was written with first on PYTHONPATH. " + _HINT)
        found = [entry for entry in manifest["components"] if entry["seed"] == seed]
        if not found:
            raise ConfigurationError(f"the record cache has no component of seed {seed}",
                                     context={**context, "cached_seeds": [entry["seed"] for entry in
                                                                          manifest["components"]]},
                                     hint=f"The cache holds the training streams of run seed {manifest['seed']} from "
                                          "step 0; another run seed, stream purpose (validation, probe) or start "
                                          "step (a resumed run) draws other seed blocks. " + _HINT)
        entry = found[0]
        if reference_sha256(reference) != entry["reference_sha256"]:
            raise ConfigurationError(f"the record cache's component of seed {seed} is of another reference "
                                     f"({entry['reference']})", context=context, hint=_HINT)
        requested, cached = encode_parameters(parameters), entry["parameters"]
        if canonical_parameters(requested, encoded=True) != canonical_parameters(cached, encoded=True):
            fields = sorted(field for field in requested.keys() | cached.keys()
                            if field not in requested or field not in cached
                            or canonical_parameters(requested[field], encoded=True)
                            != canonical_parameters(cached[field], encoded=True))
            raise ConfigurationError(f"the record cache's component of seed {seed} was generated with other parameters",
                                     context={**context, "fields": fields}, hint=_HINT)
        return entry

    def _replay(self, entry: dict) -> Iterator[Mapping]:
        root = Path(self.path)
        for chunk in entry["chunks"]:
            with error_context(stage="record_cache", cache=self.path, seed=entry["seed"], chunk=chunk["file"]):
                data = (root / chunk["file"]).read_bytes()
                if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                    raise ConfigurationError(f"chunk {chunk['file']} of the record cache is not the chunk it was "
                                             "written as: its SHA-256 differs from the manifest's",
                                             context={"stage": "record_cache", "chunk": chunk["file"]}, hint=_HINT)
                records = pickle.loads(zlib.decompress(data))
            yield from records
        raise TrainingDataError(f"the record cache is exhausted: its component of seed {entry['seed']} holds "
                                f"{entry['records']} records",
                                context={"stage": "record_cache", "cache": self.path, "seed": entry["seed"],
                                         "records": entry["records"]},
                                hint="Train fewer steps, or write a cache for more. " + _HINT)

    def require(self, dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, records: int,
                verify: int = VERIFY_RECORDS) -> dict:
        """Check before any record is drawn that this cache serves the first ``records`` records of the training
        stream ``training_records(dataconfigs, policy=policy, seed=seed)`` (ConfigurationError otherwise), and return
        the manifest:

        1. the cache's ``recipe`` is the product's, and its dataconfigs, seed and every policy field are the
           request's (refused naming the ``field``);
        2. each planned stream passes ``component`` and the plan is the cache's whole stream list;
        3. ``records`` is at most the records the cache was written for (each stream then holds what that many
           records draw from it);
        4. the first ``verify`` records of every stream (at most what each holds), drawn live, are the cache's.
        """
        records = _count(records, "records")
        manifest, context = self.manifest, {"stage": "record_cache", "cache": self.path}
        requested = {"recipe": RECIPE, "dataconfigs": dataconfig_names(dataconfigs), "seed": seed,
                     **policy_fields(policy)}
        cached = {"recipe": manifest.get("recipe"), "dataconfigs": manifest.get("dataconfigs"),
                  "seed": manifest.get("seed"), **manifest.get("policy", {})}
        differing = [field for field in requested if field not in cached
                     or canonical_parameters(requested[field], encoded=True)
                     != canonical_parameters(cached[field], encoded=True)]
        if differing:
            field = differing[0]
            raise ConfigurationError(f"the record cache was written for another {field}: {cached.get(field)!r}, not "
                                     f"{requested[field]!r}",
                                     context={**context, "field": field, "fields": differing}, hint=_HINT)
        plan = component_plan(dataconfigs, policy=policy, seed=seed)
        for planned in plan:
            self.component(planned.reference, planned.parameters, planned.seed)
        if len(plan) != len(manifest["components"]):
            raise ConfigurationError(f"the record cache was written for another stream: {len(manifest['components'])} "
                                     f"streams, not {len(plan)}", context=context, hint=_HINT)
        if records > manifest["records"]:
            raise ConfigurationError(f"the record cache serves {manifest['records']} records ({manifest['steps']} "
                                     f"steps of {manifest['batch_size']}); {records} were asked for",
                                     context={**context, "needed": records, "records": manifest["records"]},
                                     hint=_HINT)
        per_stream = min([verify, *(entry["records"] for entry in manifest["components"])])
        if per_stream > 0:
            self._verify_live(dataconfigs, policy, seed, per_stream * len(plan))
        return manifest

    def _verify_live(self, dataconfigs, policy, seed, count):
        """Draw the first ``count`` records of the training stream live and from this cache, and refuse the cache at
        the first record that differs, naming the seed of the stream it comes from. Every stream is drawn at the same
        rate, so ``count`` records hold the same number of every stream's."""
        def draw(record_source):
            return islice(training_records(dataconfigs, policy=policy, seed=seed, record_source=record_source), count)

        def origin(reference, parameters, stream_seed):
            return repeat({"seed": stream_seed})

        for position, (live, cached, source) in enumerate(zip(draw(None), draw(self), draw(origin))):
            if live != cached:
                fields = sorted(field for field in live.keys() | cached.keys()
                                if field not in live or field not in cached or live[field] != cached[field])
                raise ConfigurationError(
                    f"the record cache does not reproduce the live stream: record {position}, of the stream of seed "
                    f"{source['seed']}, differs in {', '.join(fields[:8])}; the code that produces the records has "
                    "changed since the cache was written",
                    context={"stage": "record_cache", "cache": self.path, "record": position, "seed": source["seed"],
                             "fields": fields}, hint=_HINT)
