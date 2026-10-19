"""Suite types: a suite is a set of strata, each a declarative GenAIRR recipe plus how its cases are presented.

A recipe is a module-level function (or ``functools.partial`` of one) ``recipe(ctx) -> Experiment`` that returns
a fresh ``Experiment.on(ctx.config)`` chain; ``suites.engine`` runs it with derived seeds.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from alignair_assay.core.reference import Reference

if TYPE_CHECKING:
    from GenAIRR import Experiment


@dataclass(frozen=True)
class RecipeContext:
    """What a recipe builds from: ``config`` is the GenAIRR DataConfig the truth ``reference`` is built from
    (``Experiment.on(ctx.config)``, never a short alias); ``donors`` are the suite's donors (empty without them);
    ``n`` is the stratum's requested count at the current scale, which a clonal recipe passes as ``n_clones``."""

    config: object
    reference: Reference
    donors: tuple = ()
    n: int = 0


@dataclass(frozen=True)
class Stratum:
    """One stratum of a suite: ``n`` cases at scale 1.0 (clonal: ``n`` clones) simulated by ``recipe``.

    ``labels`` are slice dimensions added to every case, merged with ``case_labels(record, ctx)`` when given.
    ``rc_fraction`` (0 to 1) is the probability a case is presented reverse-complemented; ``both_orientations``
    instead presents every record twice (``:fwd`` and ``:rc``, grouped by ``groups["pair"]``). ``accept`` keeps
    only the records it returns True for (rejection sampling). ``cohort`` runs the recipe over the suite's donors
    (``run_cohort``); ``clonal`` marks a recipe ending in ``clonal_lineage``, whose records are all kept.
    """

    name: str
    n: int
    recipe: Callable[[RecipeContext], Experiment]
    labels: Mapping[str, str] = field(default_factory=dict)
    rc_fraction: float = 0.0
    both_orientations: bool = False
    accept: Callable[[dict], bool] | None = None
    cohort: bool = False
    clonal: bool = False
    case_labels: Callable[[dict, RecipeContext], Mapping[str, str]] | None = None

    def __post_init__(self) -> None:
        _check_name("stratum", self.name)
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"stratum {self.name}: n must be a positive integer, got {self.n!r}")
        if not 0.0 <= self.rc_fraction <= 1.0:
            raise ValueError(f"stratum {self.name}: rc_fraction must lie in [0, 1], got {self.rc_fraction!r}")
        if self.cohort and self.clonal:
            raise ValueError(f"stratum {self.name}: cohort and clonal are mutually exclusive in GenAIRR")
        if self.accept is not None and (self.cohort or self.clonal):
            raise ValueError(f"stratum {self.name}: accept applies only to strata that are neither cohort nor "
                             "clonal")


@dataclass(frozen=True)
class Suite:
    """A named set of strata. ``purpose`` is one sentence naming the mechanism the suite isolates; ``primary``
    are the metrics its verdict rests on; ``donors`` (a ``suites.donors.DonorSpec``) gives the suite simulated
    donors; ``arms`` are the reference arms tools are run in (``shared``, ``genotype``)."""

    name: str
    slug: str
    title: str
    purpose: str
    primary: tuple[str, ...]
    strata: tuple[Stratum, ...]
    donors: "DonorSpec | None" = None  # noqa: F821 (suites.donors, from P2 Task 4)
    arms: tuple[str, ...] = ("shared",)

    def __post_init__(self) -> None:
        _check_name("suite", self.name)
        if not self.strata:
            raise ValueError(f"suite {self.name} has no strata")
        names = [s.name for s in self.strata]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"suite {self.name}: duplicate stratum names {duplicates}")


def _check_name(kind: str, name: str) -> None:
    """Names are parts of case ids (``suite:stratum:index``) and directory names, so they may not contain ':'."""
    if not name or ":" in name:
        raise ValueError(f"{kind} name must be non-empty and contain no ':', got {name!r}")
