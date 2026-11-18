"""S13 ``flanks``: full-length IgG-level reads inside the non-germline sequence real libraries carry around V(D)J."""
from __future__ import annotations

from functools import partial

from alignair_assay.core.headline import HEADLINE
from alignair_assay.suites.base import Stratum, Suite
from alignair_assay.suites.flanks import Flanks
from alignair_assay.suites.recipes import SHM_CLASSES, repertoire

# Stratum -> its flanks (lengths in nt, inclusive): a 5'RACE UTR and leader, a multiplex V primer masked to N over the
# first V bases too, a 3' constant region, and a leader with a constant region.
_FLANKS = {
    "leader_5p": Flanks(prefix=(60, 130)),
    "primer_n_5p": Flanks(prefix=(5, 15), mask=(1, 12)),
    "c_region_3p": Flanks(suffix=(20, 80)),
    "leader_c_region": Flanks(prefix=(40, 100), suffix=(20, 60)),
}

S13 = Suite(
    name="S13",
    slug="flanks",
    title="Library flanks",
    purpose="Measures where each tool starts V and ends J when the read carries a 5' leader, a masked primer or a "
            "3' constant region around its V(D)J.",
    primary=(*HEADLINE, "v.seq_start_exact", "j.seq_end_exact"),
    strata=tuple(Stratum(name, 500, partial(repertoire, rate=SHM_CLASSES["igg"]), labels={"flank": name},
                         rc_fraction=0.5, flank=flanks)
                 for name, flanks in _FLANKS.items()),
)
