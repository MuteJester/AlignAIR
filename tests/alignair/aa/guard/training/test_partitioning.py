import pytest

from alignair.aa.guard.train_data import (_parent_key, build_training_stream,
                                          partition_by_parent_lineage)

pytestmark = pytest.mark.component


def _rec(rid, group, lineage, parent=None):
    r = {"id": rid, "group": group, "lineage_key": lineage, "aa": "X" * 20}
    if parent:
        r["parent_lineage"] = parent
    return r


def test_parent_lineage_split_keeps_siblings_together():
    recs = [_rec("p1", "positive_real", "P1")]
    recs += [_rec(f"s1_{i}", "structural", "P1", parent="P1") for i in range(3)]
    recs += [_rec(f"p{n}", "positive_real", f"P{n}") for n in range(2, 20)]
    train, dev, ph = partition_by_parent_lineage(recs, seed=0, dev_frac=0.3)
    fam = {r["id"] for r in recs if _parent_key(r) == "P1"}
    tr, dv = {r["id"] for r in train}, {r["id"] for r in dev}
    assert fam <= tr or fam <= dv               # all siblings + parent on one side
    assert len(ph) == 64 and not (tr & dv)


def test_stream_oe_minority_and_group_coverage():
    train = ([_rec(f"pos{i}", "positive_real", f"L{i}") for i in range(20)]
             + [_rec(f"neg{i}", "igsf", f"N{i}") for i in range(20)]
             + [_rec(f"tcr{i}", "tcr", f"T{i}") for i in range(20)]
             + [_rec(f"st{i}", "structural", f"L{i}") for i in range(20)])
    oe = [_rec(f"oe{i}", "oe", f"O{i}") for i in range(20)]
    batches = list(build_training_stream(train, oe, seed=0, oe_cap=0.15, batch_size=32, n_batches=10))
    oe_ids = {r["id"] for r in oe}
    assert all(sum(1 for r in b if r["id"] in oe_ids) / len(b) <= 0.15 + 1e-9 for b in batches)
    groups = {r["group"] for b in batches for r in b}
    assert {"positive_real", "igsf", "tcr", "structural"} <= groups


def test_stream_raises_on_forbidden_id():
    train = [_rec("x", "positive_real", "L")]
    with pytest.raises(ValueError):
        list(build_training_stream(train, [], seed=0, forbidden_ids={"x"}, n_batches=1))
