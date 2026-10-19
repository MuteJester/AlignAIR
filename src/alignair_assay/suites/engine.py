"""Deterministic suite generation: the same (suite, split, seed, scale) gives the same cases for any worker count.

Every seed derives from ``(seed, split, suite, stratum, ...)``, so the ``dev`` and ``test`` splits share no
generation seed. Each stratum requests ``max(1, round(n * scale))`` cases; rejection sampling, clonal and cohort
strata never short-fill silently, they raise naming the suite, stratum and counts. A suite's donors are drawn
once from ``(seed, split, suite, "donors", tag)`` and do not scale. Strata run in parallel, one spawned process
per stratum; a worker receives only names and numbers, resolves the suite from the registry and re-draws the
donors from the same seed.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial

from alignair_assay.core.cases import make_case
from alignair_assay.core.reference import Reference, dataconfig
from alignair_assay.suites import SUITES, get_suite
from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.donors import Donor, sample_donors

SPLITS = ("dev", "test")
DEFAULT_SEED = 20261019
MAX_ROUNDS = 50
# Rejection sampling: the first round over-asks by this factor, later rounds by 1 / acceptance, capped.
_FIRST_FACTOR = 2
_MAX_FACTOR = 20
_MIN_ACCEPTANCE = 0.05


def derive_seed(seed: int, *parts) -> int:
    """A seed in [0, 2**31) for ``parts`` under ``seed``: the first 4 bytes (big-endian) of the SHA-256 of
    ``"seed:part1:part2:..."``, modulo 2**31."""
    text = ":".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % 2**31


def check_split(split: str) -> None:
    """ValueError naming the known splits when ``split`` is not one of ``SPLITS``."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; known splits: {', '.join(SPLITS)}")


def requested_count(stratum: Stratum, scale: float) -> int:
    """Cases (clonal: clones) the stratum requests at ``scale``: ``max(1, round(n * scale))``."""
    return max(1, round(stratum.n * scale))


@dataclass(frozen=True)
class GeneratedSuite:
    """A generated suite: its cases in stratum order, its donors (empty without them), and per stratum name the
    generation stats ``requested``, ``accepted``, ``generated``, ``rounds`` and ``cases`` (plus ``clones`` for a
    clonal stratum)."""

    cases: list[dict]
    donors: tuple[Donor, ...]
    stats: dict


def generate_suite(suite: Suite, *, reference_name: str, split: str, seed: int, scale: float,
                   workers: int) -> GeneratedSuite:
    """Generate every stratum of ``suite`` against GenAIRR dataconfig ``reference_name``.

    ``workers`` processes (at most one per stratum) generate the strata; with one, generation runs in-process.
    The output does not depend on ``workers``. Parallel generation resolves the suite by name, so it needs a
    registered suite. Raises ValueError for an unknown split, a scale that is not positive or a bad worker count,
    and RuntimeError when a stratum cannot be filled.
    """
    check_split(split)
    if not (scale > 0 and math.isfinite(scale)):
        raise ValueError(f"scale must be a positive number, got {scale!r}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers!r}")
    processes = min(workers, len(suite.strata))
    if processes > 1 and SUITES.get(suite.name) is not suite:
        raise ValueError(f"suite {suite.name}: parallel generation needs a registered suite (see suites.SUITES)")
    ctx = make_context(suite, reference_name, split=split, seed=seed)
    if processes > 1:
        tasks = [(suite.name, index, reference_name, split, seed, scale) for index in range(len(suite.strata))]
        # Spawned, not forked: the parent has usually started threads (torch), and forking those can deadlock.
        with ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_generate_registered, tasks))
    else:
        results = [generate_stratum(suite, stratum, ctx, split=split, seed=seed, scale=scale)
                   for stratum in suite.strata]
    cases = [case for stratum_cases, _ in results for case in stratum_cases]
    stats = {stratum.name: stratum_stats for stratum, (_, stratum_stats) in zip(suite.strata, results)}
    return GeneratedSuite(cases=cases, donors=ctx.donors, stats=stats)


def make_context(suite: Suite, reference_name: str, *, split: str, seed: int) -> RecipeContext:
    """The recipe context of ``suite``: the dataconfig ``reference_name``, its truth reference and the suite's
    donors."""
    reference = Reference.from_dataconfig(reference_name)
    return RecipeContext(config=dataconfig(reference_name), reference=reference,
                         donors=_donors(suite, reference, split=split, seed=seed))


def generate_stratum(suite: Suite, stratum: Stratum, ctx: RecipeContext, *, split: str, seed: int,
                     scale: float) -> tuple[list[dict], dict]:
    """The cases of one stratum, in order, and its generation stats (see ``GeneratedSuite``)."""
    ctx = replace(ctx, n=requested_count(stratum, scale))
    round_seed = partial(derive_seed, seed, split, suite.name, stratum.name)
    if stratum.clonal:
        records, stats = _clonal_records(suite, stratum, ctx, round_seed(0))
    elif stratum.cohort:
        records, stats = _cohort_records(suite, stratum, ctx, round_seed(0))
    else:
        records, stats = _sampled_records(suite, stratum, ctx, round_seed)
    orientations = random.Random(derive_seed(seed, split, suite.name, stratum.name, "orientation"))
    cases = []
    for index, record in enumerate(records):
        case_id = f"{suite.name}:{stratum.name}:{index:06d}"
        labels = {**stratum.labels, **(stratum.case_labels(record, ctx) if stratum.case_labels else {})}
        groups = {}
        if stratum.cohort:
            groups["subject"] = record["subject_id"]
        if stratum.clonal:
            groups["clone"] = str(record["clone_id"])
        if stratum.both_orientations:
            for suffix, orientation in ((":fwd", 0), (":rc", 1)):
                cases.append(make_case(record, case_id=case_id + suffix, suite=suite.name, stratum=stratum.name,
                                       orientation=orientation, labels=labels, groups={**groups, "pair": case_id}))
        else:
            orientation = int(orientations.random() < stratum.rc_fraction)
            cases.append(make_case(record, case_id=case_id, suite=suite.name, stratum=stratum.name,
                                   orientation=orientation, labels=labels, groups=groups))
    return cases, {**stats, "cases": len(cases)}


def _generate_registered(task: tuple) -> tuple[list[dict], dict]:
    """Worker: generate stratum ``index`` of the registered suite ``name``; takes only names and numbers so it
    runs under any process start method."""
    name, index, reference_name, split, seed, scale = task
    suite = get_suite(name)
    ctx = make_context(suite, reference_name, split=split, seed=seed)
    return generate_stratum(suite, suite.strata[index], ctx, split=split, seed=seed, scale=scale)


def _sampled_records(suite: Suite, stratum: Stratum, ctx: RecipeContext, round_seed) -> tuple[list[dict], dict]:
    """The first ``ctx.n`` accepted records, in generation order, over at most ``MAX_ROUNDS`` rounds."""
    n = ctx.n
    accepted: list[dict] = []
    generated = rounds = 0
    while len(accepted) < n:
        if rounds == MAX_ROUNDS:
            raise RuntimeError(f"{suite.name}/{stratum.name}: accepted {len(accepted)} of {n} "
                               f"after {MAX_ROUNDS} rounds")
        if stratum.accept is None:
            factor = 1
        elif generated == 0:
            factor = _FIRST_FACTOR
        else:
            factor = min(_MAX_FACTOR, 1 / max(len(accepted) / generated, _MIN_ACCEPTANCE))
        batch = stratum.recipe(ctx).run_records(n=math.ceil((n - len(accepted)) * factor),
                                                seed=round_seed(rounds)).records
        generated += len(batch)
        rounds += 1
        accepted.extend(r for r in batch if stratum.accept is None or stratum.accept(r))
    stats = {"requested": n, "accepted": len(accepted), "generated": generated, "rounds": rounds}
    return accepted[:n], stats


def _clonal_records(suite: Suite, stratum: Stratum, ctx: RecipeContext, seed: int) -> tuple[list[dict], dict]:
    """Every record of one run growing ``ctx.n`` clones; fewer distinct clones than requested raises."""
    records = list(stratum.recipe(ctx).run_records(seed=seed).records)
    clones = len({r["clone_id"] for r in records})
    if clones < ctx.n:
        raise RuntimeError(f"{suite.name}/{stratum.name}: generated {clones} of {ctx.n} clones")
    stats = {"requested": ctx.n, "accepted": len(records), "generated": len(records), "rounds": 1,
             "clones": clones}
    return records, stats


def _cohort_records(suite: Suite, stratum: Stratum, ctx: RecipeContext, seed: int) -> tuple[list[dict], dict]:
    """The first ``ctx.n`` records, in record order, of one cohort run giving each of the suite's donors
    ``ceil(ctx.n / donors)`` records; the last donors may get fewer cases, or none."""
    if not ctx.donors:
        raise ValueError(f"{suite.name}/{stratum.name}: a cohort stratum needs donors (Suite.donors)")
    genotypes = [donor.genotype(ctx.config) for donor in ctx.donors]
    records = list(stratum.recipe(ctx).run_cohort(genotypes, n_per_subject=math.ceil(ctx.n / len(genotypes)),
                                                  seed=seed).records)
    if len(records) < ctx.n:
        raise RuntimeError(f"{suite.name}/{stratum.name}: generated {len(records)} of {ctx.n} cohort records")
    stats = {"requested": ctx.n, "accepted": len(records), "generated": len(records), "rounds": 1}
    return records[:ctx.n], stats


def _donors(suite: Suite, reference: Reference, *, split: str, seed: int) -> tuple[Donor, ...]:
    """The suite's donors, sampled with ``derive_seed(seed, split, suite.name, "donors", spec.tag)``; empty for a
    suite without donors."""
    spec = suite.donors
    if spec is None:
        return ()
    donor_seed = derive_seed(seed, split, suite.name, "donors", spec.tag)
    return tuple(sample_donors(reference, spec.n, donor_seed, spec.params))
