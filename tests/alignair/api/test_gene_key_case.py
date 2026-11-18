"""Gene keys are accepted in either case at every public entry that takes them (``"V"`` or ``"v"``), and give what
the upper-case keys give; outputs keep their own case."""
import json

import pytest

from alignair import Aligner, ReferenceSet
from alignair.genotype import genotype_bank, load_genotype
from tests.alignair.predict.golden_rows import TEST_CHIP

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def aligner():
    return Aligner.from_pretrained(str(TEST_CHIP), device="cpu")


def test_reference_entries_take_gene_keys_in_either_case(aligner):
    reference = aligner.reference
    v, j = reference.gene("V"), reference.gene("J")
    assert reference.gene("v") is v
    assert reference.subset({"v": v.names[:2]}).gene("V").names == reference.subset({"V": v.names[:2]}).gene("V").names
    assert (reference.locus_mask("IGH", "v") == reference.locus_mask("IGH", "V")).all()
    reference.validate(genes=("v", "j"))
    alleles = {"v": {v.names[0]: v.sequences[0]}, "j": {j.names[0]: j.sequences[0]}}
    lower = ReferenceSet.from_alleles(alleles, anchors={"v": {v.names[0]: 288}})
    upper = ReferenceSet.from_alleles({g.upper(): a for g, a in alleles.items()}, anchors={"V": {v.names[0]: 288}})
    assert lower.genes.keys() == upper.genes.keys() == {"V", "J"}
    assert lower.gene("V").anchors == upper.gene("V").anchors == {v.names[0]: 288}


def test_genotype_entries_take_gene_keys_in_either_case(aligner, tmp_path):
    reference = aligner.reference
    names = {"V": reference.gene("V").names[:3], "J": reference.gene("J").names[:1]}
    mixed = {"v": names["V"], "J": names["J"]}
    assert genotype_bank(reference, mixed).gene("V").names == genotype_bank(reference, names).gene("V").names
    assert aligner.with_genotype(mixed).reference.gene("J").names == names["J"]
    path = tmp_path / "genotype.json"
    path.write_text(json.dumps(mixed))
    assert load_genotype(str(path)) == {"v": set(names["V"]), "j": set(names["J"])}       # segments lower-case
