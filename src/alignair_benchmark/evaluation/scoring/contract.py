"""AIRR prediction contract scoring."""
from __future__ import annotations

from typing import Any

from ...core.schema import BenchmarkCase, GENES
from .primitives import as_float, field_presence, is_missing

_AIRR_REQUIRED = (
    "sequence_id",
    "sequence",
    "v_call",
    "j_call",
    "productive",
    "junction",
)
_AIRR_OPTIONAL = (
    "d_call",
    "c_call",
    "junction_aa",
    "vj_in_frame",
    "stop_codon",
    "v_cigar",
    "d_cigar",
    "j_cigar",
    "v_identity",
    "d_identity",
    "j_identity",
)


def score_airr_contract(pred: dict[str, Any], case: BenchmarkCase) -> dict[str, float]:
    required = list(_AIRR_REQUIRED)
    if case.genes.get("d") and case.genes["d"].calls:
        required.append("d_call")
    out = {
        "required_field_presence": field_presence(pred, tuple(required)),
        "optional_field_presence": field_presence(pred, _AIRR_OPTIONAL),
    }
    coord_keys = tuple(
        f"{gene}_{kind}_{side}"
        for gene in GENES
        for kind in ("sequence", "germline")
        for side in ("start", "end")
    )
    parseable = [
        as_float(pred.get(key)) is not None
        for key in coord_keys
        if key in pred and not is_missing(pred.get(key))
    ]
    out["parseable_airr_rate"] = 1.0 if parseable and all(parseable) else 0.0
    return out
