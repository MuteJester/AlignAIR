import pytest

from alignair.aa.guard.mining import mine_hard_negatives

pytestmark = pytest.mark.component


def test_mining_raises_when_pool_intersects_forbidden():
    pool = [{"id": "a", "lineage_key": "L1"}]
    with pytest.raises(ValueError):
        mine_hard_negatives(lambda r: 0.0, pool, forbidden_ids={"a"})


def test_mining_returns_lineage_deduped_hardest_first():
    pool = [{"id": "a", "lineage_key": "L1"}, {"id": "b", "lineage_key": "L1"},
            {"id": "c", "lineage_key": "L2"}]
    scores = {"a": 0.5, "b": 0.1, "c": 0.9}          # lower margin = harder
    out = mine_hard_negatives(lambda r: scores[r["id"]], pool, forbidden_ids=set(), top_k=10)
    assert [r["id"] for r in out] == ["b", "c"]      # L1 deduped to its hardest (b); ordered by margin
