import hashlib
import json
import os

import pytest

from alignair.aa.guard.disjoint import (
    Gate3DigestMismatch, audit_disjoint, build_gate3_global_denylist,
)


def _rec(id_, seq, lineage, donor="d", study="s"):
    return {"id": id_, "aa": seq, "lineage_key": lineage, "donor": donor, "study": study}


def test_flags_sequence_hash_overlap_with_gate3_presplit_denylist():
    seq = "QVQLVQSGAEVKK"
    h = hashlib.sha256(seq.encode()).hexdigest()
    rep = audit_disjoint([_rec("a", seq, "L1")],
                         gate3_denylist={"sequence_hashes": {h}, "lineages": set(),
                                         "donors": set(), "studies": set()},
                         phase3=({}, {}))
    assert not rep.clean and rep.overlaps["gate3_seqhash"] == ["a"]


def test_clean_when_no_overlap():
    rep = audit_disjoint([_rec("a", "AAAA", "L1")],
                         gate3_denylist={"sequence_hashes": set(), "lineages": set(),
                                         "donors": set(), "studies": set()},
                         phase3=({}, {}))
    assert rep.clean and all(not v for v in rep.overlaps.values())


def test_flags_lineage_donor_and_study_overlap():
    rep = audit_disjoint([_rec("a", "AAAA", "LX", donor="D9", study="S9")],
                         gate3_denylist={"sequence_hashes": set(), "lineages": {"LX"},
                                         "donors": {"D9"}, "studies": {"S9"}},
                         phase3=({}, {}))
    assert not rep.clean
    assert rep.overlaps["gate3_lineage"] == ["a"]
    assert rep.overlaps["donor"] == ["a"] and rep.overlaps["study"] == ["a"]


def test_denylist_requires_global_digest_match(tmp_path):
    # A missing/empty corpus must raise the mismatch error, NOT FileNotFoundError.
    with pytest.raises(Gate3DigestMismatch):
        build_gate3_global_denylist(tmp_path / "cleaned.jsonl",
                                    {"record_digest": {"global": "wrong"}})


def test_module_never_reads_the_seal():
    """Source-level guarantee: the auditor has no path or call capable of reading the sealed test."""
    import inspect
    import alignair.aa.guard.disjoint as d
    src = inspect.getsource(d)
    assert "locked_test" not in src           # never names the sealed partition file
    assert "unlock_locked_test" not in src    # never calls the one-shot unlock
    assert "cohort_loader" not in src          # never imports the sealed loader


# Durable on-machine proof that the replay reproduces the committed frozen digest (skips without private data).
_CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
_FP = "docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"


@pytest.mark.skipif(not os.path.exists(_CLEANED), reason="private Gate-3 corpus absent")
def test_real_corpus_reproduces_committed_global_digest():
    fp = json.load(open(_FP))
    dl = build_gate3_global_denylist(_CLEANED, fp)   # raises if the replay does not match the seal
    # 1905 pre-split records = dev(1069) + locked(836); denylisting all of them covers the seal
    assert len(dl["sequence_hashes"]) > 1000
    assert len(dl["lineages"]) > 1000
