"""S0 ``toy``: a small two-stratum suite for tests and smoke runs; S0_EXHAUST: a stratum that can never fill."""
from __future__ import annotations

from GenAIRR import Experiment

from alignair_assay.suites.base import RecipeContext, Stratum, Suite


def toy_repertoire(ctx: RecipeContext) -> Experiment:
    """Recombined heavy chains with light somatic hypermutation (Poisson rate 0.03 per base)."""
    return Experiment.on(ctx.config).recombine().mutate(rate=0.03)


def has_d_call(record: dict) -> bool:
    """True when the simulator called a D allele."""
    return bool(record.get("d_call"))


def never(record: dict) -> bool:
    """Rejects every record."""
    return False


S0 = Suite(
    name="S0",
    slug="toy",
    title="Toy suite",
    purpose="Exercises the build, run, score and report machinery end to end on a few reads.",
    primary=("v.top1_in_set", "j.top1_in_set", "junction.nt_exact"),
    strata=(
        Stratum("plain", 30, toy_repertoire, rc_fraction=0.5),
        Stratum("with_d", 20, toy_repertoire, accept=has_d_call),
    ),
)

S0_EXHAUST = Suite(
    name="S0_EXHAUST",
    slug="exhaust",
    title="Exhaustion check",
    purpose="Shows that a stratum which can never fill raises instead of short-filling.",
    primary=(),
    strata=(Stratum("never", 5, toy_repertoire, accept=never),),
)
