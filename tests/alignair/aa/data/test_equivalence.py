import pytest
from alignair.aa.equivalence import (
    db_hash, build_equivalence_classes, EquivalenceClassMap,
)

pytestmark = pytest.mark.component


def _toy():
    # a*01 and a*02 translate identically (synonymous) ; b*01 differs.
    aa = {"a*01": "MKV", "a*02": "MKV", "b*01": "MKL"}
    return build_equivalence_classes("V", aa, db_hash([("a*01", "atg"), ("b*01", "atc")]))


def test_identical_translations_share_a_class():
    cmap = _toy()
    assert cmap.n_classes() == 2
    assert cmap.class_of("a*01") == cmap.class_of("a*02")
    assert cmap.class_of("b*01") != cmap.class_of("a*01")


def test_ambiguity_set_returns_all_alleles_in_class():
    cmap = _toy()
    cid = cmap.class_of("a*01")
    assert cmap.ambiguity_set(cid) == ["a*01", "a*02"]


def test_json_round_trip_is_identity():
    cmap = _toy()
    back = EquivalenceClassMap.from_json(cmap.to_json())
    assert back.gene == cmap.gene
    assert back.db_hash == cmap.db_hash
    assert back.allele_class == cmap.allele_class
    assert back.class_members == cmap.class_members


def test_db_hash_is_deterministic_and_sensitive():
    h1 = db_hash([("a*01", "atg"), ("b*01", "atc")])
    h2 = db_hash([("b*01", "atc"), ("a*01", "atg")])   # order-independent
    h3 = db_hash([("a*01", "atg"), ("b*01", "atg")])   # a seq changed
    assert h1 == h2
    assert h1 != h3 and len(h1) == 16
