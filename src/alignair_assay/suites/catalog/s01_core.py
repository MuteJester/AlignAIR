"""S1 ``core``: the four read designs crossed with the three SHM classes, simulated over 20 donors."""
from __future__ import annotations

from functools import partial

from alignair_assay.core.headline import HEADLINE
from alignair_assay.suites.base import Stratum, Suite, has_shm
from alignair_assay.suites.donors import DonorSpec
from alignair_assay.suites.recipes import READ_DESIGNS, SHM_CLASSES, repertoire

# Cases per SHM class at scale 1.0, in each read design.
_COUNTS = {"igm": 500, "igg": 500, "high": 250}

S1 = Suite(
    name="S1",
    slug="core",
    title="Core repertoire",
    purpose="Measures annotation accuracy on realistic donor repertoires across read designs and SHM levels.",
    primary=HEADLINE,
    strata=tuple(
        Stratum(f"{design}_{shm}", _COUNTS[shm], partial(repertoire, design=design, rate=SHM_CLASSES[shm]),
                labels={"design": design, "shm": shm}, rc_fraction=0.5, cohort=True,
                applies=None if shm == "igm" else has_shm)
        for design in READ_DESIGNS for shm in _COUNTS
    ),
    donors=DonorSpec(20, "S1"),
)
