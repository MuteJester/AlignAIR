"""Canonical record digest for the frozen Gate-3 cohort.

The freeze fingerprint must bind the EVALUATION RECORDS, not merely their lineage-key sets — otherwise a
sequence, expected verdict, category, source, or transformation could change without changing the frozen
hash. Each record's canonical string commits only redistribution-safe values (sequence + lineage are
committed as sha256, never as text):

    record_id | sequence_sha256 | lineage_sha256 | category | group | expected_state | expected_reason
             | source/provenance | transformation/version | prepared_by_anchor_rule | split

`digest()` hashes the sorted canonical records (per split, and globally). Any change to any bound field
changes the digest.
"""
from __future__ import annotations

import hashlib

from .cohort import MANIFEST

TRIM_TRANSFORM = "motif_fr4_junction_v1"
REBUILD_TRANSFORM = "motif_cys_wgxg_v1"

# category -> the transformation/version that produced its sequence
_TRANSFORM = {
    "pos_igh_structure": TRIM_TRANSFORM, "murine_vh": TRIM_TRANSFORM, "vhh": TRIM_TRANSFORM,
    "cdr3_only": REBUILD_TRANSFORM, "j_only": REBUILD_TRANSFORM,
    "reversed_vh": "structural_reverse_v1", "v_only": "structural_v_region_v1",
    "leader_vh": "structural_leader_prepend_v1", "vh_constant": "structural_ch1_append_v1",
}
# categories whose sequences were prepared by the motif anchor rule (circularity risk vs Stage-1 motifs)
ANCHOR_PREPARED = frozenset({"pos_igh_structure", "murine_vh", "vhh"})

_SPEC = {s.key: s for s in MANIFEST}
_SEP = "\x1f"   # ASCII unit separator (cannot occur in the bound fields)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def transformation_of(category: str) -> str:
    return _TRANSFORM.get(category, "none")


def prepared_by_anchor_rule(category: str) -> bool:
    return category in ANCHOR_PREPARED


def canonical_record(rec: dict, split: str) -> str:
    """Canonical, redistribution-safe string for one record. `rec` needs id/aa/lineage_key/category/
    group/expected_state; expected_reason and source come from the frozen manifest."""
    spec = _SPEC[rec["category"]]
    fields = [rec["id"], _sha(rec["aa"]), _sha(rec["lineage_key"]), rec["category"], rec["group"],
              rec["expected_state"], spec.expected_reason or "", spec.source,
              transformation_of(rec["category"]),
              "anchor" if prepared_by_anchor_rule(rec["category"]) else "direct", split]
    assert all(_SEP not in f for f in fields), "separator collision in a bound field"
    return _SEP.join(fields)


def digest(canonical_records) -> str:
    """sha256 over the sorted canonical record strings (order-independent binding)."""
    return hashlib.sha256("\n".join(sorted(canonical_records)).encode()).hexdigest()
