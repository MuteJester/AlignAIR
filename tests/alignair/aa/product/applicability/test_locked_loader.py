"""The sealed cohort loader: dev() reaches only fit/val; the test set requires an explicit, audited,
one-shot unlock."""
from __future__ import annotations

import json

import pytest

from alignair.aa.product.applicability.cohort_loader import GateCohort, LockedTestError

pytestmark = pytest.mark.component


@pytest.fixture
def store(tmp_path):
    dev = [{"id": "a", "split": "fit", "aa": "X"}, {"id": "b", "split": "val", "aa": "Y"}]
    (tmp_path / "dev_cohort.jsonl").write_text("\n".join(json.dumps(r) for r in dev) + "\n")
    (tmp_path / "locked_test.jsonl").write_text(json.dumps({"id": "t", "split": "test", "aa": "Z"}) + "\n")
    return str(tmp_path)


def test_dev_returns_fit_and_val(store):
    gc = GateCohort(store)
    assert {r["id"] for r in gc.dev()} == {"a", "b"}
    assert [r["id"] for r in gc.dev("fit")] == ["a"]
    assert [r["id"] for r in gc.dev("val")] == ["b"]


def test_dev_refuses_test(store):
    with pytest.raises(LockedTestError):
        GateCohort(store).dev("test")


def test_unlock_requires_a_substantive_reason(store):
    with pytest.raises(ValueError):
        GateCohort(store).unlock_locked_test("too short")


def test_unlock_is_one_shot_and_audited(store):
    gc = GateCohort(store)
    recs = gc.unlock_locked_test("Task 11 final locked-test evaluation")
    assert [r["id"] for r in recs] == ["t"]
    import os
    assert os.path.exists(os.path.join(store, "locked_test_unlock_audit.jsonl"))
    # a second unlock is a hard error unless explicitly overridden
    with pytest.raises(LockedTestError):
        gc.unlock_locked_test("Task 11 final locked-test evaluation")
    assert [r["id"] for r in gc.unlock_locked_test("re-run with override", allow_reunlock=True)] == ["t"]
