"""The references a model trains on keep their bytes: for every GenAIRR dataconfig, and for the unions the lab and
the locus screen train on, ``ReferenceSet.from_dataconfigs`` gives the ``alignair.reference.v1`` JSON pinned in
``tests/alignair/fixtures/reference_v1_sha256.json`` (optional keys a v1 reader ignores aside), and the model card's
two reference hashes stay. A moved digest is a moved training target; re-pin only with a retrain or a deliberate rule
(GenAIRR G1/G2 move them by design; the owner's codon-104 rule of 2026-11-07 moved 27 dataconfigs, by 60 anchors of
V alleles whose codon 104 is not Cys, and nothing else; the owner's shared-V rule of 2026-11-09 moved the all-human
union, whose TRD locus now also lists the 9 TRAV/DV alleles of the TRA block, and nothing else)."""
import GenAIRR.data as gd
import pytest

from alignair.model_file import serialize
from alignair.reference.reference_set import ReferenceSet
from tests.support.reference_pin import PIN, v1_digest

pytestmark = pytest.mark.component


def test_the_pin_covers_every_dataconfig():
    assert set(PIN["dataconfigs"]) == set(gd.list_configs()) and len(PIN["dataconfigs"]) == 106


@pytest.mark.parametrize("name", sorted(PIN["dataconfigs"]))
def test_a_dataconfigs_reference_keeps_its_v1_bytes(name):
    assert v1_digest(ReferenceSet.from_dataconfigs(getattr(gd, name))) == PIN["dataconfigs"][name]


@pytest.mark.parametrize("key", sorted(PIN["unions"]))
def test_a_union_keeps_its_v1_bytes(key):
    reference = ReferenceSet.from_dataconfigs(*[getattr(gd, name) for name in key.split("+")])
    assert v1_digest(reference) == PIN["unions"][key]


def test_the_model_card_hashes_are_unchanged():
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert serialize.reference_fasta_sha256(reference) == PIN["card"]["reference_fasta_sha256"]
    assert serialize.allele_order_sha256(reference) == PIN["card"]["allele_order_sha256"]
