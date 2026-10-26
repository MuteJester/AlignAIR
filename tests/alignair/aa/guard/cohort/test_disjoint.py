import hashlib
import json
import os

import pytest

from alignair.aa.guard.disjoint import (
    Gate3DigestMismatch, _replay_splits, audit_disjoint, build_gate3_global_denylist,
)

pytestmark = pytest.mark.component


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


def _legacy_freeze_fixture():
    """Synthetic inputs whose expected outputs were captured from the retired freeze writer."""
    records = []
    for group, category, lineages in [
        ("positive_real", "pos_igh_repertoire", [f"R{i}" for i in range(8)]),
        ("positive_sim", "pos_igh_simulator", [f"S{i}" for i in range(4)]),
        ("structural", "reversed_vh", ["R0", "R1", "X0", "X1", "X2"]),
        ("igk_igl", "igk", [f"N{i}" for i in range(5)]),
    ]:
        for lineage in lineages:
            records.append({"id": f"{group}-{lineage}", "aa": "A" * 20 + "C" * len(records),
                            "lineage_key": lineage, "category": category, "group": group,
                            "expected_state": "supported" if group.startswith("positive") else
                                              "insufficient_evidence" if group == "structural" else "unknown_ood"})
    return records


def test_replay_preserves_captured_legacy_splits_including_shared_parents_and_odd_counts():
    expected = {
        "positive_real": {"R0": "val", "R1": "test", "R2": "fit", "R3": "fit", "R4": "test",
                          "R5": "val", "R6": "fit", "R7": "test"},
        "positive_sim": {"S0": "test", "S1": "fit", "S2": "val", "S3": "fit"},
        "structural": {"R0": "val", "R1": "test", "X0": "val", "X1": "test", "X2": "test"},
        "igk_igl": {"N0": "test", "N1": "test", "N2": "val", "N3": "val", "N4": "test"},
    }
    assignments = _replay_splits(_legacy_freeze_fixture())
    assert assignments == {(group, lineage): split for group, values in expected.items() for lineage, split in values.items()}


def test_replay_reproduces_legacy_digest_after_phase3_exclusion_and_first_sequence_dedup(tmp_path):
    kept = _legacy_freeze_fixture()
    records = kept + [
        {**kept[0], "id": "excluded-name", "aa": "N" * 30, "lineage_key": " Therapeutic "},
        {**kept[0], "id": "excluded-hash", "aa": "D" * 30, "lineage_key": "alias"},
        {**kept[-1], "id": "duplicate", "lineage_key": "duplicate-lineage"},
    ]
    cleaned, gold = tmp_path / "cleaned.jsonl", tmp_path / "gold.tsv"
    cleaned.write_text("\n".join(json.dumps(record) for record in records))
    gold.write_text("name\texact_vh_id\ntherapeutic\t" + hashlib.sha256(b"D" * 30).hexdigest() + "\n")
    # Fixed expected value from the original writer, independent of the replay implementation.
    fingerprint = {"record_digest": {"global": "599e750dd08d79b8b6f9923f3f649eefa0c6978ff48f774c75d89d3293b7835d"}}
    denylist = build_gate3_global_denylist(cleaned, fingerprint, phase3_gold_path=gold)
    assert denylist["sequence_hashes"] == {hashlib.sha256(record["aa"].encode()).hexdigest() for record in kept}
    assert denylist["lineages"] == {record["lineage_key"] for record in kept}
    records[0]["aa"] = "W" * 30
    cleaned.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(Gate3DigestMismatch):
        build_gate3_global_denylist(cleaned, fingerprint, phase3_gold_path=gold)


# Durable on-machine proof that the replay reproduces the committed frozen digest (skips without private data).
_CLEANED = ".private/aa-gate3-hydration/cleaned_cohort.jsonl"
_FP = "Notes and Plans/docs/superpowers/artifacts/2026-07-24-aa-gate3/split_fingerprint.json"


@pytest.mark.skipif(not (os.path.exists(_CLEANED) and os.path.exists(_FP)),
                    reason="private Gate-3 corpus or frozen fingerprint absent")
@pytest.mark.local_data
def test_real_corpus_reproduces_committed_global_digest():
    fp = json.load(open(_FP))
    dl = build_gate3_global_denylist(_CLEANED, fp)   # raises if the replay does not match the seal
    # 1905 pre-split records = dev(1069) + locked(836); denylisting all of them covers the seal
    assert len(dl["sequence_hashes"]) > 1000
    assert len(dl["lineages"]) > 1000
