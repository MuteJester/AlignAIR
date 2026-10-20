"""S10 ``heldout``: full-length IgG-level repertoires from generative settings outside GenAIRR's defaults."""
from __future__ import annotations

from functools import partial

from alignair_assay.core.headline import HEADLINE
from alignair_assay.suites.base import Stratum, Suite
from alignair_assay.suites.donors import DonorSpec
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

_RATE = SHM_CLASSES["igg"]
# Held-out setting -> its recipe: another SHM kernel, NP lengths 4 nt longer, or trims 3 nt longer.
_SETTINGS = {
    "uniform": partial(repertoire, model="uniform", rate=_RATE),
    "hkl_s5f": partial(repertoire, s5f_model="hkl_s5f", rate=_RATE),
    "np_shift": partial(repertoire, rate=_RATE, np_shift=4),
    "trim_shift": partial(repertoire, rate=_RATE, trim_shift=3),
}

S10 = Suite(
    name="S10",
    slug="heldout",
    title="Held-out generative settings",
    purpose="Measures how accuracy holds when the SHM kernel, NP lengths or trims leave the default distributions.",
    primary=HEADLINE,
    strata=tuple(Stratum(name, 600, recipe, labels={"heldout": name}, rc_fraction=0.5, cohort=True)
                 for name, recipe in _SETTINGS.items()),
    donors=DonorSpec(20, "S10"),
)
