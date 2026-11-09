"""S2 ``siblings``: full-length reads drawn only from V alleles that have a near-identical sibling, across SHM levels.

A V allele's sibling distance is its minimum Hamming distance to another V allele of equal length
(``sibling_alleles``); the suite samples V uniformly from the alleles within 5 bases of one, so SHM competes with
the few bases that tell the true allele from its sibling.
"""
from __future__ import annotations

from functools import lru_cache, partial

from GenAIRR import Experiment

from alignair_assay.core.reference import Reference
from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.recipes import experiment_on, mutated

# The largest sibling distance (Hamming, in bases) an allele may have to be sampled.
MAX_DISTANCE = 5
# Stratum name suffix -> Poisson per-base SHM rate (0 to 1) of the S5F ``hh_s5f`` kernel.
_RATES = {"0": 0.0, "0.03": 0.03, "0.08": 0.08, "0.15": 0.15, "0.25": 0.25}
_BINS = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4-5", 5: "4-5"}


def sibling_alleles(reference: Reference, letter: str = "v", max_distance: int = MAX_DISTANCE) -> dict[str, int]:
    """Per allele of segment ``letter`` (``"v"``, ``"d"`` or ``"j"``) of ``reference``, its sibling distance: the
    minimum Hamming distance, in bases, to another allele of the same segment and equal length. Only alleles whose
    distance is at most ``max_distance`` are kept; an allele alone at its length has no sibling. Names are sorted;
    the dict is the caller's own."""
    return dict(_siblings(tuple(reference.alleles.get(letter, {}).items()), max_distance))


@lru_cache(maxsize=8)
def _siblings(alleles: tuple[tuple[str, str], ...], max_distance: int) -> tuple[tuple[str, int], ...]:
    """``sibling_alleles`` over ``(name, sequence)`` pairs, as sorted ``(name, distance)`` pairs; cached because
    every record's case label needs it."""
    by_length: dict[int, list[tuple[str, str]]] = {}
    for name, sequence in alleles:
        by_length.setdefault(len(sequence), []).append((name, sequence))
    nearest: dict[str, int] = {}
    for group in by_length.values():
        for index, (name, sequence) in enumerate(group):
            for other, other_sequence in group[index + 1:]:
                distance = sum(a != b for a, b in zip(sequence, other_sequence))
                nearest[name] = min(nearest.get(name, distance), distance)
                nearest[other] = min(nearest.get(other, distance), distance)
    return tuple(sorted((name, distance) for name, distance in nearest.items() if distance <= max_distance))


def sibling_distance_bin(distance: int) -> str:
    """The ``sibling_distance`` label of a sibling distance: ``1``, ``2``, ``3`` or ``4-5`` (``0`` for an allele
    whose sequence another allele repeats under another name). ValueError outside 0 to 5."""
    try:
        return _BINS[distance]
    except KeyError:
        raise ValueError(f"sibling distance must lie in [0, 5], got {distance!r}") from None


def sibling_repertoire(ctx: RecipeContext, *, rate: float) -> Experiment:
    """Recombination with V sampled uniformly from the sibling alleles of ``ctx.reference``, then SHM at Poisson
    per-base ``rate`` (0 to 1; 0 skips it); no end loss."""
    siblings = sorted(sibling_alleles(ctx.reference, "v", MAX_DISTANCE))
    return mutated(experiment_on(ctx.config).restrict_alleles(v=siblings).recombine(), rate, config=ctx.config)


def sibling_label(record: dict, ctx: RecipeContext) -> dict[str, str]:
    """The case label ``sibling_distance``: the binned sibling distance of the record's generative V allele."""
    distance = sibling_alleles(ctx.reference, "v", MAX_DISTANCE)[record["truth_v_call"]]
    return {"sibling_distance": sibling_distance_bin(distance)}


S2 = Suite(
    name="S2",
    slug="siblings",
    title="V allele siblings",
    purpose="Measures whether a tool tells a V allele from a sibling one to five bases away as SHM rises.",
    primary=("v.top1_in_set", "v.set_precision", "v.gene_top1", "j.top1_in_set"),
    strata=tuple(Stratum(f"shm_{name}", 600, partial(sibling_repertoire, rate=rate), case_labels=sibling_label)
                 for name, rate in _RATES.items()),
)
