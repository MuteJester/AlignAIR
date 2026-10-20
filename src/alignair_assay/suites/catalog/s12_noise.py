"""S12 ``noise``: full-length IgG-level reads with one kind of sequencing or amplification noise per stratum."""
from __future__ import annotations

from functools import partial

from alignair_assay.core.headline import HEADLINE
from alignair_assay.suites.base import Stratum, Suite
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

# Stratum -> the GenAIRR noise step and its arguments: per-base error rates (0 to 1) or per-read counts.
_NOISE = {
    "seq_err_0.005": ("sequencing_errors", {"rate": 0.005}),
    "seq_err_0.01": ("sequencing_errors", {"rate": 0.01}),
    "pcr_0.002": ("pcr_amplify", {"rate": 0.002}),
    "n_bases_3": ("ambiguous_base_calls", {"count": 3}),
    "n_bases_8": ("ambiguous_base_calls", {"count": 8}),
    "pol_indel_1": ("polymerase_indels", {"count": 1}),
    "pol_indel_2": ("polymerase_indels", {"count": 2}),
}

S12 = Suite(
    name="S12",
    slug="noise",
    title="Sequencing noise",
    purpose="Measures how accuracy degrades under sequencing errors, PCR errors, ambiguous bases and polymerase "
            "indels.",
    primary=HEADLINE,
    strata=tuple(Stratum(name, 300, partial(repertoire, rate=SHM_CLASSES["igg"], noise=noise),
                         labels={"noise": name}, rc_fraction=0.5)
                 for name, noise in _NOISE.items()),
)
