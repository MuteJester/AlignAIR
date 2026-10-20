"""Cluster ids for paired comparisons: the reads ``stats.compare`` must not treat as independent.

The suite engine groups cases with ``groups`` keys: ``clone`` for the members of one simulated lineage (S9's clonal
stratum) and ``pair`` for the forward and reverse-complemented presentation of one record (S11). Members of a clone
share their alleles, trims and junction, so a tool that gets one wrong tends to get them all wrong: a read-level
test would count one clone's difference many times. ``cluster_ids`` names each case's cluster for
``stats.compare(..., clusters=...)``.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from alignair_assay.truth.model import ObservableTruth

# The grouping keys a comparison clusters by, coarsest first: a clone contains its pairs.
CLUSTER_KEYS: tuple[str, ...] = ("clone", "pair")


def cluster_ids(truths: Iterable[ObservableTruth], keys: Sequence[str] = CLUSTER_KEYS) -> dict[str, str]:
    """Case id -> cluster id ``"<stratum>/<key>:<value>"`` for the first of ``keys`` the case's ``groups`` have. The
    stratum is part of the id because clone ids are numbered per GenAIRR run, which is per stratum. A case with
    none of the keys is left out: it is a cluster of its own. ``subject`` (a cohort donor) is not a default key;
    pass it in ``keys`` to cluster by donor."""
    out: dict[str, str] = {}
    for truth in truths:
        for key in keys:
            if key in truth.groups:
                out[truth.case_id] = f"{truth.stratum}/{key}:{truth.groups[key]}"
                break
    return out
