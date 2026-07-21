import pytest
from alignair.aa.taxonomy import gene_of, family_of, class_families, catalogue_families

class FakeMap:
    class_members = {0: ["IGHV3-23*01", "IGHV3-23*02"], 1: ["IGHV1-69D*01"]}

def test_parsing():
    assert gene_of("IGHV3-23*01") == "IGHV3-23"
    assert family_of("IGHV3-23*01") == "IGHV3"
    assert family_of("IGHV1-69D*01") == "IGHV1"

def test_family_parse_failure_raises():
    with pytest.raises(ValueError):
        family_of("notagene")

def test_class_and_catalogue_families():
    assert class_families(0, FakeMap()) == {"IGHV3"}
    assert catalogue_families(FakeMap()) == {"IGHV3", "IGHV1"}
