"""Stage-4 Safeguard 1 — FROZEN class semantics (ordered constants, not just output sizes).

The class NAMES and their ORDER are the contract every trained checkpoint binds (via CLASS_MAP_HASH). Two
namespaces are kept deliberately distinct:
  * ``camelid_igh`` is an HMM-bank EVIDENCE group (a germline profile identity — see profiles.RECEPTOR_GROUPS).
  * ``vhh`` is a supervised BIOLOGICAL LABEL (an input verified to be a camelid single-domain VHH).
An input can carry the biological label ``vhh`` while the guard's evidence for it is ``camelid_igh``; the two
must never be conflated in the class maps.
"""
from __future__ import annotations

import hashlib
import json

# Completeness head (5-way): is it a complete domain, and if not, how is it incomplete?
COMPLETENESS_CLASSES = ("complete", "missing_v_anchor", "missing_j_anchor", "too_short", "contaminated")

# Domain-type head (8-way): the recognized in-domain biological types (the open-set energy is over these).
# 'vhh' here is the biological label (camelid single-domain), NOT the camelid_igh HMM evidence group.
DOMAIN_TYPE_CLASSES = ("human_igh_vh", "human_igk", "human_igl", "tra", "trb", "vhh", "murine_vh", "igsf")

COMPLETENESS_INDEX = {c: i for i, c in enumerate(COMPLETENESS_CLASSES)}
DOMAIN_TYPE_INDEX = {c: i for i, c in enumerate(DOMAIN_TYPE_CLASSES)}

# The single recognized in-domain positive (a complete human IGH VH) — used to derive the applicability target.
SUPPORTED_DOMAIN_TYPE = "human_igh_vh"
COMPLETE = "complete"


def class_map_hash() -> str:
    """Stable hash of the ordered class maps; bound into every checkpoint so class semantics cannot drift."""
    payload = json.dumps({"completeness": list(COMPLETENESS_CLASSES),
                          "domain_type": list(DOMAIN_TYPE_CLASSES)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


CLASS_MAP_HASH = class_map_hash()
