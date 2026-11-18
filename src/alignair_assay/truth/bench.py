"""Observable truth for bench_v2 cases: a forward GenAIRR record presented in one of four orientations."""
from __future__ import annotations

from dataclasses import replace

from alignair_assay.core.reference import Reference
from alignair_assay.core.seqops import revcomp
from alignair_assay.truth.build import truth_from_genairr_record
from alignair_assay.truth.model import ObservableTruth

NON_BIOLOGICAL_ORIENTATION = "non_biological_orientation"

# bench_v2 orientation id -> presented sequence back to the canonical (forward) frame
_TO_CANONICAL = {
    0: lambda s: s,
    1: revcomp,
    2: lambda s: revcomp(s)[::-1],  # complement only
    3: lambda s: s[::-1],  # reverse only
}


def truth_from_bench_case(case: dict, reference: Reference) -> ObservableTruth:
    """Truth for one bench_v2 JSONL row, in the canonical frame of its forward ``record``.

    Complement-only and reverse-only presentations (orientation ids 2/3) are built but marked excluded.
    """
    oid = case["orientation_id"]
    if oid not in _TO_CANONICAL:
        raise ValueError(f"{case['case_id']}: unknown orientation_id {oid!r}")
    truth = truth_from_genairr_record(case["record"], reference, case_id=case["case_id"],
                                      stratum=case["stratum"], orientation=oid)
    if _TO_CANONICAL[oid](case["sequence"]).upper() != truth.sequence:
        raise ValueError(f"{case['case_id']}: presented sequence does not match orientation_id {oid}")
    if oid in (2, 3):
        truth = replace(truth, excluded=NON_BIOLOGICAL_ORIENTATION)
    return truth
