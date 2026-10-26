"""S6 ``d_observability``: full-length, lightly mutated reads whose D is trimmed by a fixed amount or inverted.

``trim_{k}`` trims exactly ``k`` nucleotides at both D ends (``suites.recipes.symmetric_d_trim``). GenAIRR aborts
the run when the two trims exceed the D allele, so from ``k = 5`` on human IGH (shortest D allele 11 nt) D is
restricted to the alleles at least ``2k + 2`` nt long (``d_alleles_for_trim``): every record is trimmed exactly
``k`` and keeps at least 2 D bases, and the ``d_trim`` label is the real trim. ``inverted`` keeps GenAIRR's
default trims (``d_trim`` = ``default``) and inverts every D.
"""
from __future__ import annotations

from functools import partial

from GenAIRR import Experiment

from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.recipes import mutated, symmetric_d_trim

# Poisson per-base SHM rate (0 to 1) of the S5F ``hh_s5f`` kernel.
_RATE = 0.02
# Trim at each D end, in nucleotides.
_TRIMS = (0, 3, 6, 9, 12)


def trimmed_d_repertoire(ctx: RecipeContext, *, per_side: int) -> Experiment:
    """Recombination with D trimmed exactly ``per_side`` nucleotides at both ends, then SHM at Poisson per-base
    rate 0.02; no end loss."""
    return mutated(symmetric_d_trim(ctx, per_side), _RATE)


def inverted_d_repertoire(ctx: RecipeContext) -> Experiment:
    """Recombination with every D inverted (reverse-complemented) and default trims, then SHM at Poisson per-base
    rate 0.02; no end loss."""
    return mutated(Experiment.on(ctx.config).recombine().invert_d(prob=1.0), _RATE)


S6 = Suite(
    name="S6",
    slug="d_observability",
    title="D observability",
    purpose="Measures how D calls and D edges degrade as the D is trimmed shorter or inverted.",
    primary=("d.top1_in_set", "d_short.abstain_ok", "d.seq_start_exact", "d.seq_end_exact"),
    strata=(
        *(Stratum(f"trim_{k}", 400, partial(trimmed_d_repertoire, per_side=k),
                  labels={"d_trim": str(k), "inverted": "no"}) for k in _TRIMS),
        Stratum("inverted", 500, inverted_d_repertoire, labels={"d_trim": "default", "inverted": "yes"}),
    ),
)
