"""The AIRR GenotypeSet of an inferred genotype, and its schema validation."""
import pytest
from alignair.genotype.airr import to_genotype_set, validate

pytestmark = pytest.mark.component


def _calls():
    return {"IGHV1-2": ["IGHV1-2*01", "IGHV1-2*02"], "IGHJ4": ["IGHJ4*02"]}


def test_the_genotypes_alleles_are_one_locus_documented_alleles_in_order():
    gs, _ = to_genotype_set(_calls(), "IGH")
    gt = gs["genotype_class_list"][0]
    assert [d["label"] for d in gt["documented_alleles"]] == ["IGHV1-2*01", "IGHV1-2*02", "IGHJ4*02"]
    assert gt["undocumented_alleles"] == [] and gt["deleted_genes"] == []
    assert all(d["phasing"] is None for d in gt["documented_alleles"])
    assert gt["inference_process"] == "repertoire_sequencing" and gt["locus"] == "IGH"


def test_null_germline_set_ref_warns():
    _, warnings = to_genotype_set(_calls(), "IGH", germline_set_ref=None)
    assert any("germline_set_ref" in w for w in warnings)
    _, w2 = to_genotype_set(_calls(), "IGH", germline_set_ref="ogrdb:IGH")
    assert not any("germline_set_ref" in w for w in w2)


def test_validate_passes_conforming_and_catches_errors():
    gs, _ = to_genotype_set(_calls(), "IGH", germline_set_ref="ogrdb:IGH")
    assert validate(gs) == []                                                # conforms to AIRR schema
    errs = validate({"genotype_class_list": [{"locus": "XXX"}]})
    assert any("receptor_genotype_set_id" in e for e in errs)
    assert any("locus" in e for e in errs)


def test_vendored_fixture_matches_validator_fields():
    import json
    import pathlib
    fx = json.loads((pathlib.Path(__file__).parent / "fixtures" / "airr_genotype_schema.json").read_text())
    assert set(fx["Genotype"]["required"]) == {"receptor_genotype_id", "locus", "documented_alleles",
                                               "undocumented_alleles", "deleted_genes", "inference_process"}
