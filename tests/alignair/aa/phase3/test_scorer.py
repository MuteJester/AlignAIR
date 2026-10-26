import pytest
from types import SimpleNamespace
from alignair.aa.phase3.scorer import Prediction, score_gold, score_concordance

pytestmark = pytest.mark.component

class FakeMap:  # class0 -> IGHV3-23 (2 alleles); class1 -> IGHV1-69
    class_members = {0: ["IGHV3-23*01", "IGHV3-23*02"], 1: ["IGHV1-69*01"]}

def _r(name, lin, fam, rep, cat="true", gene=""):
    return SimpleNamespace(name=name, lineage_id=lin, lineage_representative=rep,
                           true_v_family=fam, true_v_gene=gene, in_catalog=cat)

def test_length_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        score_gold([], [], FakeMap(), 0.05, 0)
    with pytest.raises(ValueError):
        score_gold([_r("a","a","IGHV3","true")], [], FakeMap(), 0.05, 0)

def test_perfect_family_but_small_n_inconclusive_and_sizes_reported():
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "true") for i in range(15)]
    preds = [Prediction(0, {0}) for _ in rows]
    out = score_gold(rows, preds, FakeMap(), 0.05, 0)
    assert out["binding"]["top_class_family_accuracy"] == 1.0
    assert out["binding"]["decision"] in ("inconclusive", "failed")
    assert out["binding"]["decision"] != "supported"
    assert out["binding"]["n_lineages"] == 15
    assert out["set_size"]["families"]["mean"] == 1.0   # class 0 -> single family
    assert out["set_size"]["alleles"]["mean"] == 2.0    # class 0 -> two alleles
    assert "bootstrap_ci_family_accuracy" in out["descriptive"]

def test_family_miss_is_failed_and_oov_counted():
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "true") for i in range(10)]
    rows[0] = _r("m0", "L0", "IGHVX", "true", cat="false")   # OOV truth family
    preds = [Prediction(1, {1}) for _ in rows]              # predict IGHV1 != IGHV3
    out = score_gold(rows, preds, FakeMap(), 0.05, 0)
    assert out["binding"]["decision"] == "failed"
    assert out["oov"]["oov_rate"] > 0.0
    assert "in_catalog_accuracy" in out["oov"]

def test_descriptive_spans_all_deduplicated_rows_not_just_reps():
    rows = [_r("rep","L","IGHV3","true"), _r("mem","L","IGHV3","false")]
    preds = [Prediction(0, {0}), Prediction(1, {1})]
    out = score_gold(rows, preds, FakeMap(), 0.05, 0)
    assert out["binding"]["n_lineages"] == 1               # one representative
    assert out["descriptive"]["n_all_vh"] == 2             # both rows

def test_concordance_reports_agreement_and_interval():
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "true", gene="IGHV3-23") for i in range(12)]
    preds = [Prediction(0, {0}) for _ in rows]
    out = score_concordance(rows, preds, FakeMap(), 0.05)
    assert out["label_kind"].startswith("alignment-derived")
    assert out["top_class_gene_agreement"] == 1.0
    assert len(out["agreement_ci"]) == 2

def test_score_gold_no_representatives_raises_valueerror():
    """Non-empty cohort with no representatives should raise ValueError, not ZeroDivisionError."""
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "false") for i in range(5)]
    preds = [Prediction(0, {0}) for _ in rows]
    with pytest.raises(ValueError, match="no lineage-representative rows"):
        score_gold(rows, preds, FakeMap(), 0.05, 0)

def test_score_concordance_no_representatives_raises_valueerror():
    """Non-empty cohort with no representatives should raise ValueError, not IndexError."""
    rows = [_r(f"m{i}", f"L{i}", "IGHV3", "false", gene="IGHV3-23") for i in range(5)]
    preds = [Prediction(0, {0}) for _ in rows]
    with pytest.raises(ValueError, match="no lineage-representative rows"):
        score_concordance(rows, preds, FakeMap(), 0.05)
