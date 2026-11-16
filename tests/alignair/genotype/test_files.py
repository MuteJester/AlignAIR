"""Genotype files: JSON or YAML ``{gene: [alleles]}``, checked against the reference they are applied to."""
import json

import pytest

from alignair.genotype.files import UnknownAlleleError, load_genotype

pytestmark = pytest.mark.component


def test_load_genotype_json_and_yaml(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"v": ["IGHV1-2*01", "IGHV3-23*01"], "j": ["IGHJ4*02"]}))
    g = load_genotype(str(p))
    assert g["v"] == {"IGHV1-2*01", "IGHV3-23*01"} and g["j"] == {"IGHJ4*02"}
    y = tmp_path / "g.yaml"
    y.write_text("v:\n  - IGHV1-2*01\nd:\n  - IGHD3-3*01\n")
    gy = load_genotype(str(y))
    assert gy["v"] == {"IGHV1-2*01"} and gy["d"] == {"IGHD3-3*01"}


def test_the_bank_refuses_a_genotype_naming_alleles_the_reference_lacks():
    from alignair.genotype import genotype_bank
    from alignair.reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_alleles({"V": {"IGHV1-2*01": "ACGT", "IGHV3-23*01": "ACGG"},
                                            "J": {"IGHJ4*02": "TTTT"}})
    with pytest.raises(UnknownAlleleError, match="NOT-A-REAL-ALLELE") as caught:
        genotype_bank(reference, {"v": {"IGHV1-2*01", "NOT-A-REAL-ALLELE"}})
    assert "--reference" in caught.value.hint and isinstance(caught.value, ValueError)
