"""The in-process evaluator: AlignAIR's diagnostics on built ``dev`` suites, cached per suite, for model iteration.

``evaluate(model_or_checkpoint, layout, suites, ...)`` is what a training loop calls, with an in-memory
``(model, reference_set)`` pair, and what the ``diagnose`` command runs on a ``.alignair`` model file. It
orchestrates the ``diagnostics`` layer over a run directory: for each suite it replays AlignAIR's predict pipeline on
the suite's reads (all of them, or ``max_cases`` evenly spaced: ``select_cases``, the speed protocol's rule) and:

1. scores the raw heads against the truth (``heads.head_scores``, with the cases' GenAIRR records) and the
   reliability of each gene's top-1 allele probability (``heads.calibration_table``);
2. replays and scores the truth-substitution ``variants`` on the same reads (``ablation.run_variants``, with the
   replay above as its baseline, and with the donors' genotypes for a suite with donors, as ``all`` scores it), and
   compares them: the cumulative chain step by step (``ablation.stage_contributions``) and each single swap with
   the baseline (``ablation.swap_effects``), over ``compared_metrics``.

With ``profile``, the model is then profiled (``profile.profile_model``) on up to ``PROFILE_READS`` of the
diagnosed reads, evenly spaced, after every accuracy replay. The report (``report.diagnostics``) is written as
``diagnostics.json``, the one-page summary ``diagnostics.md``, each suite's ``<S>/report.md`` and, with a profile,
``profile.md``, every file atomically (``_write_text``), and returned in a ``DiagnosticsResult`` with the summary as
data (``summary``, flattened by ``scalars()``).

**Refused** before the model is loaded or anything is written (ValueError): no suite, an unknown suite or variant,
a batch size or ``max_cases`` below 1, a suite not built in ``layout``, a suite built on another split than ``dev``
(``test`` is locked for claims), suites built against different references, a device torch cannot use, and an
unverified replay: AlignAIR product code the replay no longer matches (``diagnostics.pin``, naming each changed
function), unless ``allow_unverified_replay``. A missing model file is a FileNotFoundError. A model whose germline
reference is not the truth reference is refused by ``replay.load_model_checked`` (ValueError naming the first
differing allele), an in-memory one included, and so is a model file whose bytes changed while it was loaded.

**Model key.** ``diagnostics.identity.model_key``: ``<weights>-<code>``, the first ``KEY_DIGITS`` hex digits of
``weights_sha256`` (the SHA-256 of a model file's bytes, or of an in-memory model's weights, configuration, module
tree and allele order) and of the AlignAIR code digest (``alignair_code``). Everything is written under
``layout.diagnostics_dir(model_key)``, so a new checkpoint or an AlignAIR code edit never meets another model's
results.

**Cache.** A suite's result is written to ``layout.diagnostics_suite_json(model_key, suite)`` with its ``key``,
everything the result depends on (``suite_key``): the full weights and code digests and the torch version, the
assay's own scoring code (``assay_code_sha256``), the suite's build (its recorded identity and config hash and the
SHA-256 of its cases, truth and donors files), the variants, ``max_cases``, the device, the batch size and the
``replay`` (whether it is verified, and the ``diagnostics.pin.replay_digest`` of the product code it mirrors). It is
reused only when that key matches in full; otherwise it is removed and computed again, so a suite's record holds
its latest configuration. The profile is cached the same way (``layout.diagnostics_profile_json``), keyed by the
model, the assay code (the replay it times is assay code), the device (with its name, the host and torch's CPU
threads), the batch size and the profiled reads. The report is rendered afresh from the results on every call.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import torch

from alignair_assay.core.cases import evenly_spaced, read_cases
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import files_digest, sha256_file
from alignair_assay.core.reference import Reference
from alignair_assay.diagnostics.ablation import (
    CONTRIBUTION_METRICS,
    DEFAULT_VARIANTS,
    VariantResult,
    run_variants,
    stage_contributions,
    swap_effects,
)
from alignair_assay.diagnostics.heads import calibration_table, head_scores
from alignair_assay.diagnostics.identity import ALIGNAIR_DIR, alignair_code, model_key, replay_pin, weights_sha256
from alignair_assay.diagnostics.pin import unverified_message
from alignair_assay.diagnostics.profile import profile_model
from alignair_assay.diagnostics.replay import load_model_checked, model_device, replay, unwrap_model
from alignair_assay.metrics import is_informational
from alignair_assay.report.diagnostics import (
    build_diagnostics_report,
    diagnostics_markdown,
    diagnostics_scalars,
    profile_markdown,
    suite_markdown,
)
from alignair_assay.scoring.clusters import CLUSTER_KEYS, cluster_ids
from alignair_assay.scoring.pipeline import load_truth, truth_reference
from alignair_assay.stats import summarize, to_dict
from alignair_assay.suites import get_suite
from alignair_assay.suites.donors import read_genotypes
from alignair_assay.truth.io import read_truth_header
from alignair_assay.workflow.build import build_changes

T = TypeVar("T")

# What a cached suite result and a cached profile are; bump the version when their content or meaning changes.
SUITE_KIND = "alignair_assay.diagnostics_suite"
PROFILE_KIND = "alignair_assay.diagnostics_profile"
CACHE_VERSION = "2"
# The split diagnostics run on: test is locked for claims.
SPLIT = "dev"
# The device a model file is diagnosed on unless another is asked for (an in-memory model's default is its own).
DEFAULT_DEVICE = "cuda"
# At most this many diagnosed reads, evenly spaced, are profiled.
PROFILE_READS = 1000
# This package's directory and the layers a cached result depends on; this module, which computes the results, is
# digested with them (the report is rendered afresh from the results, so it is not among them).
ASSAY_DIR = Path(__file__).resolve().parents[1]
ASSAY_LAYERS = ("core", "truth", "airr", "metrics", "stats", "scoring", "suites", "diagnostics")

# Taken once, when this module is imported: the assay code this process runs, whatever happens to the files later.
_ASSAY_CODE = files_digest(ASSAY_DIR, [*(path for layer in ASSAY_LAYERS for path in (ASSAY_DIR / layer).rglob("*.py")
                                         if "__pycache__" not in path.parts), Path(__file__).resolve()])


class UnverifiedReplayWarning(UserWarning):
    """``evaluate`` ran, with ``allow_unverified_replay``, against AlignAIR code the replay is not pinned to."""


@dataclass(frozen=True)
class DiagnosticsResult:
    """What ``evaluate`` found for one model."""

    model_key: str
    # per suite, in the order asked: its result (see ``evaluate``) and ``reused`` (True when it came from the cache)
    suites: dict[str, dict]
    # ``profile.profile_model``'s result and ``reused``; None when the model was not profiled
    profile: dict | None
    # the one-page summary as markdown, as written to ``diagnostics.md``
    report_md: str
    # the same summary as data (``report.diagnostics.diagnostics_summary``); ``scalars()`` flattens it
    summary: dict = field(default_factory=dict)
    # False when the replay ran on AlignAIR code it is not pinned to (``allow_unverified_replay``), with
    # ``replay_changes`` (``diagnostics.pin.pin_changes``: name -> new digest)
    replay_verified: bool = True
    replay_changes: dict[str, str] = field(default_factory=dict)
    # suite -> what changed, for each suite whose build is not current for this checkout (diagnosed as built)
    stale_builds: dict[str, list[str]] = field(default_factory=dict)

    def scalars(self) -> dict[str, float]:
        """The summary's numbers flattened for a training dashboard (TensorBoard, W&B), e.g.
        ``{"S1/baseline/junction.nt_exact": 0.77, "S1/delta/segmentation/junction.nt_exact": 0.21, ...}``
        (``report.diagnostics.diagnostics_scalars``)."""
        return diagnostics_scalars(self.summary)


def evaluate(model_or_checkpoint, layout: RunLayout, suites: Sequence[str], *, device: str | None = None,
             batch_size: int = 256, variants: Sequence[str] = DEFAULT_VARIANTS, profile: bool | None = None,
             max_cases: int | None = None, allow_unverified_replay: bool = False) -> DiagnosticsResult:
    """Diagnose a model on ``suites`` (suite names, or one name), built on the dev split in ``layout``, as the module
    docstring describes; reuses each cached suite result whose key matches in full.

    ``model_or_checkpoint`` is a ``.alignair`` model file, loaded on ``device`` (default ``DEFAULT_DEVICE``,
    ``"cuda"``), or an in-memory ``(model, reference_set)`` pair from a training loop. A pair's model is unwrapped
    (``replay.unwrap_model``: ``torch.compile``, ``DataParallel``, ``DistributedDataParallel``), its reference
    checked, and then run on ``device`` (default: the device it is on); a model moved to another device is moved
    back to its own afterwards, whatever happens, and every replay puts it back in its training mode, so the
    training loop's optimizer carries on. ``profile`` (default: True for a model file, False for a pair, which a
    training loop would otherwise profile and cache again on every call) adds the speed profile.

    **Replay pin.** The replay mirrors AlignAIR product code pinned by digest (``diagnostics.pin``), checked when
    ``diagnostics.identity`` is imported, with the model key's code digest (``replay_pin``). When it has changed,
    ``evaluate`` refuses to run (ValueError naming each changed function, its new digest and how to re-pin),
    unless ``allow_unverified_replay``: then it warns (``UnverifiedReplayWarning``) and stamps ``replay_verified``
    False and the ``replay_changes`` into the result, every suite's result and the report. Whether the replay is
    verified, and the digest of the product code, are in every suite's cache key, so verified and unverified results
    never mix.

    Each suite's result holds ``suite``; its cache ``key``; ``reads`` (``built``, ``diagnosed``, ``scored``,
    ``excluded``, ``cropped``: reads longer than the model's window, and ``misoriented``: scored reads the model put
    in another frame than the truth's, whose boundaries are not scored); ``clusters`` (``cluster_counts``: the
    clusters of two or more diagnosed reads, their ``keys`` of ``scoring.clusters.CLUSTER_KEYS``, and how many
    ``clusters`` and ``cases``);
    ``heads`` and each variant's ``scores`` (metric -> ``mean``, ``n``, ``n_applicable``, ``coverage``);
    ``calibration`` (gene -> ``heads.calibration_table``); ``metrics`` (the compared family, ``compared_metrics``);
    ``variants`` (name -> ``scores``, ``fallbacks``, per-stage ``seconds``, ``substituted``); ``contributions``
    (``ablation.stage_contributions``) and ``swaps`` (``ablation.swap_effects``), clustered by
    ``scoring.clusters.cluster_ids`` of the suite's truths; and the ``seconds`` it took.
    ``variants`` are run in ``ablation.DEFAULT_VARIANTS`` order (``variant_set``: the baseline is added to any
    other); none runs the heads, the calibration and the profile only. Raises as the module docstring says."""
    start = time.perf_counter()
    names = list(dict.fromkeys([suites] if isinstance(suites, str) else suites))
    if not names:
        raise ValueError("no suite to diagnose")
    for name in names:
        get_suite(name)
    variants = variant_set(variants)
    _check_count("batch_size", batch_size)
    if max_cases is not None:
        _check_count("max_cases", max_cases)
    for name in names:
        check_dev_build(layout, name)
    reference = truth_reference(layout, names)
    stale = stale_builds(layout, names)
    from_file = isinstance(model_or_checkpoint, (str, os.PathLike))
    if from_file:
        own, own_device = None, None
        device = parse_device(DEFAULT_DEVICE if device is None else device)
    else:
        own = unwrap_model(model_or_checkpoint[0])
        own_device = model_device(own)
        device = parse_device(str(own_device) if device is None else device)
    profile = from_file if profile is None else bool(profile)
    pin_digest, changes = replay_pin()
    changes = dict(changes)
    if changes and not allow_unverified_replay:
        raise ValueError(unverified_message(changes))
    if changes:
        warnings.warn(UnverifiedReplayWarning(f"diagnosing with an unverified replay: {unverified_message(changes)}"),
                      stacklevel=2)
    replay_state = {"verified": not changes, "sha256": pin_digest}
    weights = weights_sha256(model_or_checkpoint)
    try:
        model, reference_set = load_model_checked(model_or_checkpoint, reference, device=device)
        if from_file and weights_sha256(model_or_checkpoint) != weights:
            # Rewritten while it was loaded (a training job saving over it): the weights loaded may not be the ones
            # hashed, and results must never be cached under another model's key.
            raise ValueError(f"{Path(model_or_checkpoint).expanduser()} changed while it was being loaded; "
                             "diagnose it again once it is completely written")
        return _evaluate(model, reference_set, reference, layout, names, weights=weights, variants=variants,
                         device=device, batch_size=batch_size, profile=profile, max_cases=max_cases,
                         source=str(Path(model_or_checkpoint).expanduser()) if from_file else "in-memory model",
                         replay_state=replay_state, replay_changes=changes, stale=stale, start=start)
    finally:
        if own is not None:
            own.to(own_device)


def stale_builds(layout: RunLayout, names: Sequence[str]) -> dict[str, list[str]]:
    """Suite -> what changed (``workflow.build.build_changes`` with its recorded configuration), for each of
    ``names`` whose dev build is not current for this checkout: its cases and truth come from other suite, truth or
    GenAIRR code. A suite whose build record cannot be read is left out (``check_dev_build`` refuses it)."""
    stale = {}
    for name in names:
        try:
            build = json.loads(layout.build_json(name).read_text(encoding="utf-8"))
            changes = build_changes(layout, name, split=build["split"], reference_digest=build["reference_digest"],
                                    seed=build["seed"], scale=build["scale"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if changes:
            stale[name] = changes
    return stale


def _evaluate(model, reference_set, reference: Reference, layout: RunLayout, names: list[str], *, weights: str,
              variants: tuple[str, ...], device: str, batch_size: int, profile: bool, max_cases: int | None,
              source: str, replay_state: dict, replay_changes: dict[str, str], stale: dict[str, list[str]],
              start: float) -> DiagnosticsResult:
    """``evaluate`` on a checked model already on ``device``; ``replay_state`` (``verified`` and the product code's
    ``sha256``) goes into every suite's cache key, ``stale`` (``stale_builds``) and the wall time since ``start``
    into the report."""
    code = alignair_code()
    key_name = model_key(weights, code["code_sha256"])
    identity = {"weights_sha256": weights, "code_sha256": code["code_sha256"], "torch_version": code["torch_version"]}
    config = {"variants": list(variants), "max_cases": max_cases, "device": device, "batch_size": batch_size}
    verified = bool(replay_state["verified"])
    stamp = {"replay_verified": verified, "replay_changes": dict(replay_changes)}

    results: dict[str, dict] = {}
    for name in names:
        key = suite_key(layout, name, identity, {**config, "replay": replay_state})
        path = layout.diagnostics_suite_json(key_name, name)
        cached = _cached(path, key)
        if cached is None:
            path.unlink(missing_ok=True)
            cached = {**_diagnose_suite(name, key, model, reference_set, reference, layout, variants=variants,
                                        max_cases=max_cases, device=device, batch_size=batch_size), **stamp}
            _write_json(path, cached)
            results[name] = {**cached, "reused": False}
        else:
            results[name] = {**cached, "reused": True}

    measured = (_profile(model, reference_set, layout, key_name, identity, names, max_cases=max_cases, device=device,
                         batch_size=batch_size) if profile else None)
    report = build_diagnostics_report(
        model={"model_key": key_name, "source": source, **identity, "alignair_source": str(ALIGNAIR_DIR),
               "assay_code_sha256": assay_code_sha256()},
        config={"run_dir": str(layout.root), "suites": names, **config, "profile": bool(profile), **stamp,
                "stale_builds": stale, "seconds": round(time.perf_counter() - start, 3)},
        suites=results, profile=measured)
    markdown = diagnostics_markdown(report)
    for name in names:
        _write_text(layout.diagnostics_suite_md(key_name, name), suite_markdown(report, name))
    if measured is not None:
        _write_text(layout.diagnostics_profile_md(key_name), profile_markdown(report))
    _write_json(layout.diagnostics_json(key_name), report)
    _write_text(layout.diagnostics_md(key_name), markdown)
    return DiagnosticsResult(model_key=key_name, suites=results, profile=measured, report_md=markdown,
                             summary=report["summary"], replay_verified=verified,
                             replay_changes=dict(replay_changes), stale_builds=stale)


def assay_code_sha256() -> str:
    """``core.provenance.files_digest`` of the Python files of this package's ``ASSAY_LAYERS`` and of this module
    (the truth, metrics, scoring and diagnostics code a cached result was computed by), taken when this module was
    imported."""
    return _ASSAY_CODE


def variant_set(variants: Sequence[str]) -> tuple[str, ...]:
    """``variants`` without repeats in ``ablation.DEFAULT_VARIANTS`` order, with the baseline added when any other
    is asked for (every comparison is against it or chains from it). ValueError naming an unknown variant."""
    variants = [variants] if isinstance(variants, str) else list(variants)
    unknown = [name for name in variants if name not in DEFAULT_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant {unknown[0]!r}; known: {', '.join(DEFAULT_VARIANTS)}")
    wanted = set(variants) | ({"baseline"} if variants else set())
    return tuple(name for name in DEFAULT_VARIANTS if name in wanted)


def select_cases(items: Sequence[T], max_cases: int | None) -> list[T]:
    """All ``items``, or ``max_cases`` of them evenly spaced (``core.cases.evenly_spaced``: positions
    ``floor(i * N / max_cases)``, the speed protocol's rule) when there are more."""
    if max_cases is None or max_cases >= len(items):
        return list(items)
    return evenly_spaced(items, max_cases)


def check_dev_build(layout: RunLayout, suite: str) -> None:
    """ValueError unless ``suite`` is built in ``layout`` (its ``build.json``) on the dev split, as its build
    record and its truth file's header both say: the test split is locked for claims."""
    path = layout.build_json(suite)
    if not path.is_file():
        raise ValueError(f"suite {suite} is not built in {layout.root} (no {path.name}); build it on the dev split "
                         f"first: build --suites {suite} --run-dir {layout.root}")
    try:
        build = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{path} is not a readable build record ({error}); rebuild suite {suite}") from error
    header = read_truth_header(layout.truth(suite)) or {}
    splits = sorted({split for split in (build.get("split"), header.get("split")) if split is not None})
    if splits != [SPLIT]:
        raise ValueError(f"suite {suite} was built on the {' and '.join(splits) or 'unrecorded'} split; diagnostics "
                         f"run on the {SPLIT} split only (test is locked for claims): build it with --split {SPLIT}")


def suite_key(layout: RunLayout, suite: str, model: dict, config: dict) -> dict:
    """What a suite's result depends on (see the module docstring), as JSON reads it back: ``kind``, ``version``,
    the ``model`` identity, ``assay_code_sha256``, the ``suite`` build (``name``, the build record's ``identity``
    and ``config_hash``, the SHA-256 of its cases, truth and donors files; None for no donors) and ``config``'s
    ``variants``, ``max_cases``, ``device``, ``batch_size`` and ``replay`` (whether the replay is verified, and the
    ``diagnostics.pin.replay_digest`` of the product code it mirrors)."""
    build = json.loads(layout.build_json(suite).read_text(encoding="utf-8"))
    donors = layout.donors(suite)
    key = {"kind": SUITE_KIND, "version": CACHE_VERSION, "model": model, "assay_code_sha256": assay_code_sha256(),
           "suite": {"name": suite, "identity": build.get("identity"), "config_hash": build.get("config_hash"),
                     "cases_sha256": sha256_file(layout.cases(suite)), "truth_sha256": sha256_file(layout.truth(suite)),
                     "donors_sha256": sha256_file(donors) if donors.is_file() else None},
           **config}
    return json.loads(json.dumps(key))


def compared_metrics(primary: Sequence[str], results: dict[str, VariantResult]) -> list[str]:
    """The metrics the variants are compared on: the suite's ``primary`` metrics, then
    ``ablation.CONTRIBUTION_METRICS``, each once, leaving out the informational ones and those no variant scored."""
    scored = {name for result in results.values() for case in result.scores.values() for name in case}
    return [name for name in dict.fromkeys((*primary, *CONTRIBUTION_METRICS))
            if name in scored and not is_informational(name)]


def cluster_counts(truths: Iterable, clusters: dict[str, str]) -> dict:
    """The clusters that make a suite's comparisons cluster-aware, those of two or more of the diagnosed reads
    (``clusters`` is ``scoring.clusters.cluster_ids`` of ``truths``): ``keys`` (the ``CLUSTER_KEYS`` they are
    clustered by, the ``<key>`` of their ids ``"<stratum>/<key>:<value>"``, in ``CLUSTER_KEYS`` order; a read with a
    clone and a pair id is clustered by its clone only), ``clusters`` (how many) and ``cases`` (the reads in them). A
    cluster of one read is a read on its own, as ``stats.compare`` treats it: S11 with ``max_cases`` keeping only its
    ``:fwd`` reads has pair ids but no pair, and is compared read by read. ``truths`` is not read: the ids say it."""
    sizes = Counter(clusters.values())
    members = {case for case, cluster in clusters.items() if sizes[cluster] > 1}
    shared = {clusters[case] for case in members}
    return {"keys": [key for key in CLUSTER_KEYS if any(f"/{key}:" in cluster for cluster in shared)],
            "clusters": len(shared), "cases": len(members)}


def _diagnose_suite(name: str, key: dict, model, reference_set, reference: Reference, layout: RunLayout, *,
                    variants: tuple[str, ...], max_cases: int | None, device: str, batch_size: int) -> dict:
    """One suite's result (see ``evaluate``), as JSON reads it back. ValueError naming the suite when its cases
    cannot be scored (for example no scored truth among the diagnosed cases)."""
    start = time.perf_counter()
    header = read_truth_header(layout.truth(name)) or {}
    _, truths = load_truth(layout.truth(name), header.get("reference"), reference)
    built = read_cases(layout.cases(name))
    cases = select_cases(built, max_cases)
    ids, sequences = [case["case_id"] for case in cases], [case["sequence"] for case in cases]
    chosen = set(ids)
    truths = [truth for truth in truths if truth.case_id in chosen]
    genotypes = read_genotypes(layout.donors(name)) if layout.donors(name).is_file() else None
    try:
        baseline = replay(model, reference_set, sequences, ids, device=device, batch_size=batch_size)
        heads = head_scores(baseline, truths, cases)
        calibration = {gene: calibration_table(baseline, truths, gene) for gene in baseline.allele_names}
        results = (run_variants(model, reference_set, cases, truths, reference=reference, device=device,
                                batch_size=batch_size, variants=variants, genotypes=genotypes, _baseline=baseline)
                   if variants else {})
    except ValueError as error:
        raise ValueError(f"suite {name}: {error}") from error
    family = compared_metrics(get_suite(name).primary, results)
    scored = sum(truth.excluded is None for truth in truths)
    clusters = cluster_ids(truths)
    result = {
        "suite": name, "key": key,
        "reads": {"built": len(built), "diagnosed": len(cases), "scored": scored, "excluded": len(truths) - scored,
                  "cropped": sum(bool(record.get("length_cropped")) for record in baseline.records),
                  "misoriented": int(sum(case.get("head.misoriented", 0.0) for case in heads.values()))},
        "clusters": cluster_counts(truths, clusters),
        "heads": _summary(heads), "calibration": calibration, "metrics": family,
        "variants": {variant: {"scores": _summary(found.scores), "fallbacks": found.fallbacks,
                               "seconds": found.seconds, "substituted": list(found.substituted)}
                     for variant, found in results.items()},
        "contributions": stage_contributions(results, family, clusters=clusters),
        "swaps": swap_effects(results, family, clusters=clusters),
        "seconds": round(time.perf_counter() - start, 3),
    }
    return json.loads(json.dumps(result, allow_nan=False))


def _profile(model, reference_set, layout: RunLayout, key_name: str, identity: dict, names: list[str], *,
             max_cases: int | None, device: str, batch_size: int) -> dict:
    """The model's speed profile on up to ``PROFILE_READS`` of the diagnosed reads of ``names`` (each suite's
    ``select_cases`` of its cases, as ``_diagnose_suite`` takes them, in suite order, then evenly spaced), reused from
    its cache when the key matches, with ``reused``."""
    reads = [case["sequence"] for name in names for case in select_cases(read_cases(layout.cases(name)), max_cases)]
    reads = select_cases(reads, PROFILE_READS)
    key = json.loads(json.dumps({
        "kind": PROFILE_KIND, "version": CACHE_VERSION, "model": identity,
        "assay_code_sha256": assay_code_sha256(), "device": device,
        "device_name": _device_name(device), "host": platform.node(), "threads": torch.get_num_threads(),
        "batch_size": batch_size, "reads": len(reads),
        "reads_sha256": hashlib.sha256("\n".join(reads).encode()).hexdigest()}))
    path = layout.diagnostics_profile_json(key_name)
    cached = _cached(path, key)
    if cached is not None:
        return {**cached["profile"], "reused": True}
    path.unlink(missing_ok=True)
    measured = json.loads(json.dumps(profile_model(model, reference_set, reads, device=device, batch_size=batch_size),
                                     allow_nan=False))
    _write_json(path, {"key": key, "profile": measured})
    return {**measured, "reused": False}


def _summary(scores: dict) -> dict:
    """Per metric, the mean, value count, applicable count and coverage over every scored case."""
    return to_dict(summarize(scores, list(scores)))


def _cached(path: Path, key: dict) -> dict | None:
    """The record at ``path`` when it is readable JSON whose ``key`` is ``key``; None otherwise."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("key") == key else None


def _write_json(path: Path, data) -> None:
    """Write ``data`` as strict JSON, atomically (``_write_text``)."""
    _write_text(path, json.dumps(data, indent=2, allow_nan=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` through a temporary file of its own in the same directory (``tempfile.NamedTemporaryFile``),
    then rename it over ``path``, so the file is never seen half written, even with two writers (or two threads of
    one process). The temporary file is removed when the write fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.chmod(temporary, 0o644)  # a temporary file is private; the report is not
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _check_count(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def parse_device(device: str, *, fix: str = 'device="cpu"') -> str:
    """``device`` as torch names it; ValueError for one torch does not know, or a CUDA device it does not see
    (suggesting ``fix``: the API keyword by default; the ``diagnose`` command passes its own option)."""
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"device {device!r} is not a torch device: {error}") from error
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {device!r}: torch sees no CUDA device; use {fix}")
    return str(parsed)


def _device_name(device: str) -> str:
    """The GPU's name for a CUDA device; the processor's for the CPU."""
    parsed = torch.device(device)
    if parsed.type == "cuda":
        return torch.cuda.get_device_name(parsed)
    return platform.processor() or parsed.type
