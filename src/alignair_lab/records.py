"""The record cache: a recipe's curriculum records generated once, written to disk, and replayed by any number of runs.

``training_records`` zips independent component streams, and each component is one provider call
``(reference, parameters, seed)``, listed in order by ``gym.workers.component_plan``. ``write_record_cache`` runs
each planned component's own stream through the default GenAIRR provider (``gym.streams.stream_records``), one
process per component at most, for as many records as ``steps * batch_size`` records of the zipped stream draw from
it (``component_counts``), and writes them in compressed chunks. ``CachedRecordSource`` is a record provider
(``training_records``' and ``train``'s ``record_source``) that replays the component it is asked for: the stream it
gives is the live stream, record for record, up to ``steps * batch_size`` records.

Layout::

    PATH/manifest.json                     written last: a directory without it is not a cache
    PATH/<n>-seed<s>/chunk-<k>.pkl.zlib    component n (seed s): CHUNK_RECORDS of its records per chunk, in order

The manifest records the GenAIRR the records come from (``genairr``: the assay's provenance, whose
``genairr_source_sha256`` is ``alignair_assay.core.provenance.tree_digest`` of the package directory), the SHA-256 of
the experiment builder (``gym/experiment.py``), the dataconfig names, the policy, the seed, the steps and batch size
the cache serves, and per component its seed, reference (name and DataConfig checksum), parameters, record count
and chunks (file, records, bytes, SHA-256).

**Refused**, at the provider call and in whichever process makes it (ConfigurationError): a process importing
another GenAIRR or another experiment builder than the cache's, a seed the cache has no component of, and another
reference or other parameters than the cached component's of that seed. A component read to its end raises
TrainingDataError: the stream never wraps around. ``CachedRecordSource.require`` checks a whole stream before any
record is drawn.

The chunks are pickles: a cache is trusted local data, written by this module.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import math
import json
import multiprocessing
import os
import pickle
import shutil
import signal
import tempfile
import time
import zlib
from collections.abc import Iterator, Mapping
from itertools import islice
from operator import index
from pathlib import Path

from alignair.errors import ConfigurationError, TrainingDataError, error_context
from alignair.train.gym import experiment
from alignair.train.gym.streams import CurriculumStreamPolicy, dataconfig_name, stream_records, training_records
from alignair.train.gym.workers import component_plan
from alignair_assay.core.provenance import genairr_provenance, tree_digest

KIND = "alignair_lab.record_cache"
VERSION = 2
MANIFEST = "manifest.json"
# Records per chunk file. At about 600 bytes a record compressed (3.7 kB pickled), a chunk is about 2.5 MB and one
# component of a 40k-step cache of the 1.0.0 recipe is about 90 chunks.
CHUNK_RECORDS = 4096
COMPRESSION_LEVEL = 6
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


@functools.lru_cache(maxsize=None)
def experiment_sha256() -> str:
    """SHA-256 of the experiment builder the default provider compiles each component's GenAIRR experiment with."""
    return hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest()


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


def _ignore_interrupts():
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # the writing process stops its workers


def _write_component(item):
    """A writer process: generate one component's records with the default provider and write its chunks."""
    position, (root, name, reference, parameters, seed, count, chunk_records, genairr) = item
    start = time.perf_counter()
    if imported_genairr() != genairr:
        raise ConfigurationError("a cache writer process imported another GenAIRR than the process that started it",
                                 context={"stage": "record_cache", "seed": seed},
                                 hint="Put the intended GenAIRR first on PYTHONPATH.")
    (Path(root) / name).mkdir()
    records = iter(stream_records(reference, parameters, seed))
    chunks, written = [], 0
    while written < count:
        chunk = list(islice(records, min(chunk_records, count - written)))
        if not chunk:
            raise TrainingDataError("a component stream ended before the cache was full",
                                    context={"stage": "record_cache", "seed": seed, "records": written})
        data = zlib.compress(pickle.dumps(chunk, pickle.HIGHEST_PROTOCOL), COMPRESSION_LEVEL)
        file = f"{name}/chunk-{len(chunks):06d}.pkl.zlib"
        (Path(root) / file).write_bytes(data)
        chunks.append({"file": file, "records": len(chunk), "bytes": len(data),
                       "sha256": hashlib.sha256(data).hexdigest()})
        written += len(chunk)
    return position, {"chunks": chunks, "seconds": round(time.perf_counter() - start, 3)}


def write_record_cache(path, *, dataconfigs, policy: CurriculumStreamPolicy, seed: int, steps: int, batch_size: int,
                       workers: int | None = None, chunk_records: int = CHUNK_RECORDS) -> dict:
    """Write the records ``steps`` steps of ``batch_size`` draw from ``training_records(dataconfigs, policy=policy,
    seed=seed)`` to the new directory ``path``; return the manifest.

    Each component is generated in a spawned process of its own, at most ``workers`` at a time (default: one per
    component). The cache is written into a sibling directory and renamed to ``path`` once its manifest is written, so
    an interrupted writer leaves no cache behind. FileExistsError when ``path`` exists.
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
    processes = len(plan) if workers is None else min(_count(workers, "workers"), len(plan))
    genairr = genairr_provenance()
    if genairr["genairr_source_sha256"] != imported_genairr():
        raise ConfigurationError("GenAIRR changed while the cache was being prepared", context={"stage": "record_cache"})
    components = [{"seed": component.seed, "reference": dataconfig_name(component.reference),
                   "reference_sha256": reference_sha256(component.reference),
                   "parameters": encode_parameters(component.parameters), "records": count,
                   "directory": f"{position:03d}-seed{component.seed}"}
                  for position, (component, count) in enumerate(zip(plan, counts))]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{path.name}.partial-", dir=path.parent))
    umask = os.umask(0)
    os.umask(umask)
    partial.chmod(0o777 & ~umask)                      # a new directory's permissions, not mkdtemp's private ones
    try:
        tasks = [(str(partial), entry["directory"], component.reference, component.parameters, component.seed,
                  entry["records"], chunk_records, genairr["genairr_source_sha256"])
                 for entry, component in zip(components, plan)]
        with multiprocessing.get_context("spawn").Pool(processes, initializer=_ignore_interrupts) as pool:
            for position, written in pool.imap_unordered(_write_component, enumerate(tasks)):
                components[position].update(written)
        manifest = {
            "kind": KIND, "version": VERSION,
            "dataconfigs": [value if isinstance(value, str) else dataconfig_name(value) for value in dataconfigs],
            "policy": {"progresses": list(policy.progresses), "heavy_shm": policy.heavy_shm,
                       "short_boost": policy.short_boost, "mutation_cap": policy.mutation_cap, "mode": policy.mode},
            "seed": seed, "steps": steps, "batch_size": batch_size, "records": steps * batch_size,
            "genairr": genairr, "experiment_sha256": experiment_sha256(),
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
        """The manifest's entry of the component this provider call asks for; refused (ConfigurationError) unless it
        was generated by this process's GenAIRR and experiment builder, from the same reference and parameters."""
        manifest = self.manifest
        context = {"stage": "record_cache", "cache": self.path, "seed": seed}
        imported, cached = imported_genairr(), manifest["genairr"]["genairr_source_sha256"]
        if imported != cached:
            raise ConfigurationError("the record cache was generated by another GenAIRR than this process imports",
                                     context={**context, "cache_genairr": cached, "imported_genairr": imported},
                                     hint="Put the GenAIRR the cache was written with first on PYTHONPATH. " + _HINT)
        if experiment_sha256() != manifest["experiment_sha256"]:
            raise ConfigurationError("the record cache was generated by another experiment builder "
                                     "(alignair/train/gym/experiment.py) than this process imports",
                                     context=context, hint=_HINT)
        found = [entry for entry in manifest["components"] if entry["seed"] == seed]
        if not found:
            raise ConfigurationError(f"the record cache has no component of seed {seed}",
                                     context={**context, "cached_seeds": [entry["seed"] for entry in
                                                                          manifest["components"]]},
                                     hint=f"The cache holds the components of the run seed {manifest['seed']}; "
                                          "another seed, or a resume (whose stream starts at seed + step), draws "
                                          "other components. " + _HINT)
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
                records = pickle.loads(zlib.decompress((root / chunk["file"]).read_bytes()))
            yield from records
        raise TrainingDataError(f"the record cache is exhausted: its component of seed {entry['seed']} holds "
                                f"{entry['records']} records",
                                context={"stage": "record_cache", "cache": self.path, "seed": entry["seed"],
                                         "records": entry["records"]},
                                hint="Train fewer steps, or write a cache for more. " + _HINT)

    def require(self, dataconfigs, *, policy: CurriculumStreamPolicy, seed: int, records: int) -> dict:
        """Check before any record is drawn that this cache serves the first ``records`` records of
        ``training_records(dataconfigs, policy=policy, seed=seed)`` (ConfigurationError otherwise): each planned
        component passes ``component``, the plan is the cache's whole component list, and ``records`` is at most the
        records the cache was written for (each component then holds what that many records draw from it). Returns
        the manifest."""
        records = _count(records, "records")
        plan = component_plan(dataconfigs, policy=policy, seed=seed)
        for planned in plan:
            self.component(planned.reference, planned.parameters, planned.seed)
        manifest, context = self.manifest, {"stage": "record_cache", "cache": self.path}
        if len(plan) != len(manifest["components"]):
            raise ConfigurationError(f"the record cache was written for another stream: {len(manifest['components'])} "
                                     f"components, not {len(plan)}", context=context, hint=_HINT)
        if records > manifest["records"]:
            raise ConfigurationError(f"the record cache serves {manifest['records']} records ({manifest['steps']} "
                                     f"steps of {manifest['batch_size']}); {records} were asked for",
                                     context={**context, "needed": records, "records": manifest["records"]},
                                     hint=_HINT)
        return manifest
