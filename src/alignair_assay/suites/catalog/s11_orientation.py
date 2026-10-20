"""S11 ``orientation``: every read presented forward and reverse-complemented, as a pair."""
from __future__ import annotations

from functools import partial

from alignair_assay.core.headline import HEADLINE
from alignair_assay.suites.base import Stratum, Suite
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

S11 = Suite(
    name="S11",
    slug="orientation",
    title="Orientation pairs",
    purpose="Measures whether a tool detects read orientation and annotates both strands of a read identically.",
    primary=("orientation.correct", *HEADLINE),
    strata=tuple(
        Stratum(f"{design}_igg", 500, partial(repertoire, design=design, rate=SHM_CLASSES["igg"]),
                labels={"design": design}, both_orientations=True)
        for design in ("full", "fr3")
    ),
)
