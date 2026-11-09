"""S9 ``clonal``: full-length reads sampled from simulated B-cell lineages, each case grouped by its clone.

``clonal_lineage`` grows one lineage tree per clone from a recombined founder and samples up to ``n_sample``
cells of it; identical genotypes are collapsed, so a clone yields at most 25 records and usually fewer. Members of
a clone share their V, D and J alleles, trims and NP regions, and differ only by the somatic mutations of their
branch. SHM is internal to ``clonal_lineage``: GenAIRR rejects ``mutate()`` before or after it. Its ``rate`` is
per base and per generation (default ``max_generations`` 10), not per read: 0.05 leaves reads about 40% mutated.
The suite uses 0.008, which puts reads at IgG level. On the dev split at scale 1.0 (default seed), the per-read V
mutation rate (truth V mismatches over observed V germline bases) has median 0.075 and IQR 0.065 to 0.088, over
60 clones of 4 to 25 cases (1130 in all).
"""
from __future__ import annotations

from GenAIRR import Experiment

from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.recipes import experiment_on

# Per-base SHM rate (0 to 1) per lineage generation.
_RATE = 0.008
# Cells sampled per clone, before identical genotypes are collapsed.
_SAMPLE = 25


def lineage_repertoire(ctx: RecipeContext) -> Experiment:
    """``ctx.n`` clonal lineages grown from recombined founders, up to 25 sampled cells each, with S5F ``hh_s5f``
    SHM at per-base rate 0.008 per generation over GenAIRR's default 10 generations; no end loss."""
    return experiment_on(ctx.config).recombine().clonal_lineage(n_clones=ctx.n, n_sample=_SAMPLE, rate=_RATE)


S9 = Suite(
    name="S9",
    slug="clonal",
    title="Clonal lineages",
    purpose="Measures whether a tool assigns every member of a clone the same V gene, J gene and junction length "
            "as the members diverge by SHM.",
    primary=("clone_key.exact", "clone.agree_modal", "junction.nt_exact", "v.top1_in_set"),
    strata=(Stratum("lineages", 60, lineage_repertoire, clonal=True, rc_fraction=0.5),),
)
