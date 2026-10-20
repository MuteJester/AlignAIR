"""S4 ``truncation``: full-length reads that lose an exact number of bases at the 5' end (into V, past it at 300)
or at the 3' end (into J)."""
from __future__ import annotations

from functools import partial

from GenAIRR import Experiment

from alignair_assay.suites.base import RecipeContext, Stratum, Suite
from alignair_assay.suites.recipes import repertoire

# Poisson per-base SHM rate (0 to 1) of the S5F ``hh_s5f`` kernel.
_RATE = 0.05
# Exact end losses, in nucleotides, per swept end.
_LOSSES = {"5p": tuple(range(0, 301, 25)), "3p": (0, 10, 20, 30, 40)}


def truncated_repertoire(ctx: RecipeContext, *, five_prime: int = 0, three_prime: int = 0) -> Experiment:
    """A full-length repertoire (``repertoire`` at Poisson per-base SHM rate 0.05) that then loses exactly
    ``five_prime`` bases at the 5' end and ``three_prime`` at the 3' end; a loss of 0 adds no step."""
    experiment = repertoire(ctx, rate=_RATE)
    if five_prime:
        experiment = experiment.end_loss_5prime(length=(five_prime, five_prime))
    if three_prime:
        experiment = experiment.end_loss_3prime(length=(three_prime, three_prime))
    return experiment


S4 = Suite(
    name="S4",
    slug="truncation",
    title="Read truncation",
    purpose="Measures how calls, germline edges and abstention hold as reads lose a fixed number of bases at one "
            "end.",
    primary=("v.top1_in_set", "v_short.abstain_ok", "v.germ_start_err", "j.top1_in_set", "j.germ_end_err",
             "d.spurious_call", "junction.nt_exact"),
    strata=tuple(
        Stratum(f"{end}_{loss}", 150,
                partial(truncated_repertoire, **{"five_prime" if end == "5p" else "three_prime": loss}),
                labels={"sweep": end, "loss": str(loss)})
        for end, losses in _LOSSES.items() for loss in losses
    ),
)
