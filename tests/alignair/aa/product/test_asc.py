import pytest

from alignair.aa.product.asc import asc_to_imgt_multi, class_alleles, class_genes, class_families

pytestmark = pytest.mark.component

class FakeDC:
    asc_tables = {"V": [
        {"new_allele": "IGHVF1-G1*01", "imgt_allele": "IGHV3-23*01"},
        {"new_allele": "IGHVF1-G1*01", "imgt_allele": "IGHV3-23D*01"},  # SAME asc -> two IMGT (one-to-many)
        {"new_allele": "IGHVF1-G2*01", "imgt_allele": "IGHV1-2*02"},
    ]}
class FakeMap:
    class_members = {0: ["IGHVF1-G1*01"], 1: ["IGHVF1-G1*01", "IGHVF1-G2*01"]}

def test_one_to_many_retains_all():
    m = asc_to_imgt_multi(FakeDC())
    assert m["IGHVF1-G1*01"] == ["IGHV3-23*01", "IGHV3-23D*01"]   # BOTH kept, sorted


def test_class_alleles_union_all_members():
    m = asc_to_imgt_multi(FakeDC())
    assert class_alleles(1, FakeMap(), m) == ["IGHV1-2*02", "IGHV3-23*01", "IGHV3-23D*01"]
    assert class_genes(1, FakeMap(), m) == ["IGHV1-2", "IGHV3-23", "IGHV3-23D"]
    assert class_families(0, FakeMap(), m) == ["IGHV3"]


def test_asc_to_imgt_multi_raises_on_extra_asc_keys():
    """asc_to_imgt_multi assumes only V is ASC-clustered; raise ValueError if D/J present."""
    class BadDC:
        asc_tables = {"V": [], "D": []}

    with pytest.raises(ValueError, match="only V is ASC-clustered"):
        asc_to_imgt_multi(BadDC())
