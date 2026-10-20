"""S5 ``indels``: full-length, lightly mutated reads whose only polymerase indels are k in V, or one in J.

GenAIRR's ``polymerase_indels(count=(k, k))`` puts exactly k indels in a read but places them anywhere in it
(V, D, J or the NP regions; record fields ``n_v_indels``, ``n_d_indels``, ``n_j_indels`` and their total
``n_indels``), so each stratum keeps, by rejection sampling, only the records whose k indels all fell in its
segment.
"""
from __future__ import annotations

from functools import partial

from GenAIRR import Experiment

from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

# V indel counts swept, one stratum each.
_V_COUNTS = range(1, 7)


def indel_repertoire(ctx: RecipeContext, *, count: int) -> Experiment:
    """A full-length repertoire at IgM-level SHM (Poisson per-base rate 0.03) with exactly ``count`` polymerase
    indels per read, placed by GenAIRR."""
    return repertoire(ctx, rate=SHM_CLASSES["igm"], noise=("polymerase_indels", {"count": (count, count)}))


def indels_only_in(record: dict, *, segment: str, count: int) -> bool:
    """True when the record has exactly ``count`` indels, all of them in ``segment`` (``"v"`` or ``"j"``)."""
    return record[f"n_{segment}_indels"] == count and record["n_indels"] == count


def _stratum(name: str, segment: str, count: int) -> Stratum:
    """300 reads with exactly ``count`` indels, all in ``segment``, labelled ``indels`` = ``name``."""
    return Stratum(name, 300, partial(indel_repertoire, count=count), labels={"indels": name},
                   accept=partial(indels_only_in, segment=segment, count=count))


S5 = Suite(
    name="S5",
    slug="indels",
    title="V indels",
    purpose="Measures indel detection, V edges and junction recovery as the number of indels in V rises.",
    primary=("v.indel_found", "v.indel_false", "v.seq_end_exact", "junction.nt_exact", "v.top1_in_set"),
    strata=(*(_stratum(f"v_{k}", "v", k) for k in _V_COUNTS), _stratum("j_1", "j", 1)),
)
