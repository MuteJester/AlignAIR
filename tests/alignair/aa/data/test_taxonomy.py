import pytest
from alignair.aa.taxonomy import (
    gene_of, family_of, class_families, catalogue_families, asc_to_imgt_map, ImgtView,
)

pytestmark = pytest.mark.component

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


class FakeDataConfig:
    """Stands in for a GenAIRR DataConfig: only needs .asc_tables["V"]."""
    asc_tables = {
        "V": [
            {"imgt_allele": "IGHV2-26*04", "new_allele": "IGHVF1-G1*01"},
            {"imgt_allele": "IGHV2-26*01", "new_allele": "IGHVF1-G1*02"},
            {"imgt_allele": "IGHV3-23*01", "new_allele": "IGHVF3-G7*01"},
        ]
    }


class FakeAscMap:
    """An ASC (OGRDB) equivalence-class map, keyed the same way as the real one from build_class_maps."""
    class_members = {0: ["IGHVF1-G1*01", "IGHVF1-G1*02"], 1: ["IGHVF3-G7*01"]}


def test_asc_to_imgt_map():
    m = asc_to_imgt_map(FakeDataConfig())
    assert m == {
        "IGHVF1-G1*01": "IGHV2-26*04",
        "IGHVF1-G1*02": "IGHV2-26*01",
        "IGHVF3-G7*01": "IGHV3-23*01",
    }


class FakeDataConfigSameFamilyCollision:
    """A new_allele appears twice, mapping to two imgt_alleles of the SAME family (gene differs)."""
    asc_tables = {
        "V": [
            {"imgt_allele": "IGHV3-23*01", "new_allele": "IGHVF10-G41*03"},
            {"imgt_allele": "IGHV3-23D*01", "new_allele": "IGHVF10-G41*03"},
        ]
    }


class FakeDataConfigDifferentFamilyCollision:
    """A new_allele appears twice, mapping to imgt_alleles of DIFFERENT families -- must hard-fail."""
    asc_tables = {
        "V": [
            {"imgt_allele": "IGHV3-23*01", "new_allele": "IGHVF10-G41*03"},
            {"imgt_allele": "IGHV2-70*04", "new_allele": "IGHVF10-G41*03"},
        ]
    }


def test_asc_to_imgt_map_same_family_collision_returns_first_occurrence():
    m = asc_to_imgt_map(FakeDataConfigSameFamilyCollision())
    assert m == {"IGHVF10-G41*03": "IGHV3-23*01"}


def test_asc_to_imgt_map_different_family_collision_raises():
    with pytest.raises(ValueError):
        asc_to_imgt_map(FakeDataConfigDifferentFamilyCollision())


def test_imgt_view_translates_class_members():
    asc2imgt = asc_to_imgt_map(FakeDataConfig())
    view = ImgtView(FakeAscMap(), asc2imgt)
    assert view.class_members == {
        0: ["IGHV2-26*04", "IGHV2-26*01"],
        1: ["IGHV3-23*01"],
    }


def test_class_families_on_imgt_view_returns_imgt_family():
    asc2imgt = asc_to_imgt_map(FakeDataConfig())
    view = ImgtView(FakeAscMap(), asc2imgt)
    # ASC gene-cluster 0 covers two IMGT alleles that are both IGHV2 -> single family
    assert class_families(0, view) == {"IGHV2"}
    assert class_families(1, view) == {"IGHV3"}


def test_class_families_on_raw_asc_map_would_fail_to_parse():
    """Sanity check for the bug this bridges: ASC names are NOT IMGT-family-parseable directly."""
    with pytest.raises(ValueError):
        class_families(0, FakeAscMap())
