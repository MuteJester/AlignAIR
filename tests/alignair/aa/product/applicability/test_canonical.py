"""Canonical record digest: determinism, order-independence, field-sensitivity, and (data-backed)
post-freeze reproducibility."""
from __future__ import annotations

import json
import os

import pytest

from alignair.aa.product.applicability import canonical

pytestmark = pytest.mark.component

_REC = {"id": "igk_3", "aa": "DIQMTQSPSSLSASVGDRVTITCRASQ", "lineage_key": "study::subj::IGKV1::QQ",
        "category": "igk", "group": "igk_igl", "expected_state": "unknown_ood"}


def test_canonical_is_deterministic_and_binds_the_sequence():
    a = canonical.canonical_record(_REC, "test")
    assert a == canonical.canonical_record(dict(_REC), "test")
    # the sequence is bound via its sha256 (not in the clear)
    assert _REC["aa"] not in a
    import hashlib
    assert hashlib.sha256(_REC["aa"].encode()).hexdigest() in a


def test_digest_is_order_independent():
    recs = [canonical.canonical_record({**_REC, "id": f"r{i}"}, "test") for i in range(5)]
    assert canonical.digest(recs) == canonical.digest(list(reversed(recs)))


@pytest.mark.parametrize("field,val", [("aa", "AAAAAAAAAAAAAAAAAAAA"), ("expected_state", "supported"),
                                       ("category", "igl"), ("id", "igk_9")])
def test_any_bound_field_change_changes_the_digest(field, val):
    base = canonical.digest([canonical.canonical_record(_REC, "test")])
    changed = canonical.digest([canonical.canonical_record({**_REC, field: val}, "test")])
    assert base != changed


def test_split_is_bound():
    assert (canonical.digest([canonical.canonical_record(_REC, "val")])
            != canonical.digest([canonical.canonical_record(_REC, "test")]))


def test_anchor_prepared_and_transformation_labels():
    assert canonical.prepared_by_anchor_rule("pos_igh_structure") is True
    assert canonical.prepared_by_anchor_rule("pos_igh_repertoire") is False
    assert canonical.transformation_of("cdr3_only") == canonical.REBUILD_TRANSFORM
    assert canonical.transformation_of("igk") == "none"


_STORE = ".private/aa-gate3-hydration"
_FP = "Notes and Plans/docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"


@pytest.mark.skipif(not (os.path.exists(os.path.join(_STORE, "dev_cohort.jsonl"))
                         and os.path.exists(os.path.join(_STORE, "locked_test.jsonl"))
                         and os.path.exists(_FP)),
                    reason="frozen cohort not hydrated in this environment")
@pytest.mark.local_data
def test_frozen_record_digest_is_reproducible():
    """Recompute the global canonical digest from the frozen dev+test partitions and assert it matches
    the committed fingerprint — proving the freeze binds the actual evaluation records."""
    recs = []
    for name in ("dev_cohort.jsonl", "locked_test.jsonl"):
        p = os.path.join(_STORE, name)
        if os.path.exists(p):
            recs += [json.loads(l) for l in open(p)]
    got = canonical.digest([canonical.canonical_record(r, r["split"]) for r in recs])
    want = json.load(open(_FP))["record_digest"]["global"]
    assert got == want
