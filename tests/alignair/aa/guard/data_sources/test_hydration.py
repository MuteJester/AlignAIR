import json

import pytest

from alignair.aa.guard.hydration import (
    atomic_write_text, fetch_with_retry, filter_disjoint, hydrate_source, sha256_text,
)

pytestmark = pytest.mark.component


def test_fetch_with_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise IOError("transient 500")
        return b"ok"

    assert fetch_with_retry(flaky, retries=5, sleep=lambda s: None) == b"ok"
    assert calls["n"] == 3


def test_fetch_with_retry_reraises_after_exhaustion():
    def always():
        raise IOError("boom")

    with pytest.raises(IOError):
        fetch_with_retry(always, retries=3, sleep=lambda s: None)


def test_filter_disjoint_drops_denylisted():
    recs = [{"id": "a", "aa": "KEEP", "lineage_key": "L1"},
            {"id": "b", "aa": "DROP", "lineage_key": "L2"}]
    dl = {"sequence_hashes": {sha256_text("DROP")}, "lineages": set(), "donors": set(), "studies": set()}
    kept, dropped = filter_disjoint(recs, denylist=dl, phase3=({}, {}))
    assert [r["id"] for r in kept] == ["a"] and dropped["seqhash"] == 1


def test_atomic_write_replaces(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello" and not (tmp_path / "sub" / "f.txt.tmp").exists()


def test_hydrate_source_resumable_censused_and_checksummed(tmp_path):
    fetch_calls = {"n": 0}

    def fetch():
        fetch_calls["n"] += 1
        return b'[{"aa": "AAA", "lineage_key": "L1"}, {"aa": "BBB", "lineage_key": "L2"}]'

    def normalize(raw):
        return [{"id": f"r{i}", "aa": r["aa"], "group": "igsf", "category": "igsf",
                 "lineage_key": r["lineage_key"]} for i, r in enumerate(json.loads(raw))]

    dl = {"sequence_hashes": set(), "lineages": set(), "donors": set(), "studies": set()}
    kw = dict(source="test", store=tmp_path, fetch=fetch, normalize=normalize,
              denylist=dl, phase3=({}, {}), provenance={"url": "x"}, sleep=lambda s: None)

    r1 = hydrate_source(**kw)
    assert fetch_calls["n"] == 1 and len(r1.kept) == 2
    assert r1.census["kept_after_denylist"] == 2 and r1.census["raw_sha256"]
    assert (tmp_path / "test.raw.sha256").exists()          # immutable raw checksum sidecar

    # Resumability: a second call reuses the normalized cache and does NOT re-fetch.
    r2 = hydrate_source(**kw)
    assert fetch_calls["n"] == 1 and len(r2.kept) == 2


def test_hydrate_source_filters_denylist_into_census(tmp_path):
    def fetch():
        return b'[{"aa": "KEEP", "lineage_key": "L1"}, {"aa": "DROP", "lineage_key": "L2"}]'

    def normalize(raw):
        return [{"id": f"r{i}", "aa": r["aa"], "group": "igsf", "category": "igsf",
                 "lineage_key": r["lineage_key"]} for i, r in enumerate(json.loads(raw))]

    dl = {"sequence_hashes": {sha256_text("DROP")}, "lineages": set(), "donors": set(), "studies": set()}
    r = hydrate_source(source="t2", store=tmp_path, fetch=fetch, normalize=normalize,
                       denylist=dl, phase3=({}, {}), provenance={"url": "y"}, sleep=lambda s: None)
    assert len(r.kept) == 1 and r.census["dropped"]["seqhash"] == 1
    assert r.census["fetched"] == 2 and r.census["kept_after_denylist"] == 1
