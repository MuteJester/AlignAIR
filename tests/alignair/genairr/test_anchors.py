"""A V or J anchor GenAIRR stores as 0 (its legacy "not found") is no anchor unless GenAIRR's resolver confirms one;
every other stored anchor is kept, so the reference a model was trained on is unchanged."""
import json

import pytest

import GenAIRR.data as gd
from alignair.model_file import container
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component

FIXTURE = "tests/alignair/fixtures/test_chip.alignair"


def _stored(dataconfig, gene):
    pool = dataconfig.v_alleles if gene == "V" else dataconfig.j_alleles
    return {allele.name: allele.anchor for alleles in pool.values() for allele in alleles}


@pytest.mark.parametrize("name", ["HUMAN_IGH_OGRDB", "HUMAN_IGK_OGRDB", "HUMAN_IGL_OGRDB"])
def test_every_stored_anchor_of_the_clean_human_sets_is_kept(name):
    dataconfig = getattr(gd, name)
    reference = ReferenceSet.from_dataconfigs(dataconfig)
    for gene in ("V", "J"):
        assert reference.gene(gene).anchors == _stored(dataconfig, gene)


def test_a_zero_anchor_the_resolver_cannot_confirm_is_dropped():
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRB_IMGT)
    assert "TRBV17*01" not in reference.gene("V").anchors and "TRBJ2-2P*01" not in reference.gene("J").anchors
    assert reference.gene("V").anchors["TRBV20-1*01"] > 0          # a real anchor stays


def test_resolving_leaves_the_dataconfig_untouched():
    dataconfig = gd.HUMAN_TCRB_IMGT
    before = dataconfig.compute_checksum()
    ReferenceSet.from_dataconfigs(dataconfig)
    assert _stored(dataconfig, "V")["TRBV17*01"] == 0 and dataconfig.compute_checksum() == before


def test_the_trained_reference_is_unchanged():
    embedded = json.loads(container.read_section(FIXTURE, "reference_json"))
    assert ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB).to_serializable() == embedded
