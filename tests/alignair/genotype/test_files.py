"""Genotype files: JSON or YAML ``{gene: [alleles]}``, checked against the reference they are applied to."""
import json

import pytest

from alignair.genotype.files import UnknownAlleleError, load_genotype

pytestmark = pytest.mark.component


class _Gene:
    def __init__(self, names):
        self.names = names


class _Ref:
    def __init__(self, names):
        self._n = names

    def gene(self, g):
        return _Gene(self._n)


def test_load_genotype_json_and_yaml(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"v": ["IGHV1-2*01", "IGHV3-23*01"], "j": ["IGHJ4*02"]}))
    g = load_genotype(str(p))
    assert g["v"] == {"IGHV1-2*01", "IGHV3-23*01"} and g["j"] == {"IGHJ4*02"}
    y = tmp_path / "g.yaml"
    y.write_text("v:\n  - IGHV1-2*01\nd:\n  - IGHD3-3*01\n")
    gy = load_genotype(str(y))
    assert gy["v"] == {"IGHV1-2*01"} and gy["d"] == {"IGHD3-3*01"}


def test_load_genotype_validates_against_reference(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"v": ["IGHV1-2*01", "NOT-A-REAL-ALLELE"]}))
    ref = _Ref(["IGHV1-2*01", "IGHV3-23*01"])
    g, unknown = load_genotype(str(p), reference=ref, drop_unknown=True)
    assert g["v"] == {"IGHV1-2*01"} and "NOT-A-REAL-ALLELE" in unknown["v"]
    with pytest.raises(UnknownAlleleError, match="unknown") as caught:
        load_genotype(str(p), reference=ref, drop_unknown=False)
    assert "--reference" in caught.value.hint and isinstance(caught.value, ValueError)
