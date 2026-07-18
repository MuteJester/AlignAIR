"""Genotype task 6: AIRR GenotypeSet emit + schema validation."""
from alignair.genotype.airr import to_genotype_set, validate
from alignair.genotype.zygosity import GeneCall


def _calls():
    return [
        GeneCall("IGHV1-2", [{"name": "IGHV1-2*01", "support": 0.9, "evidence": True}], "homozygous",
                 False, [], novel=[{"promotable": True, "positions": [165], "sequence": "ACGT",
                                    "source_mask": ["ref", "observed"]}]),
        GeneCall("IGHD3-3", [], "deletion-candidate", True, [], []),
        GeneCall("IGHJ4", [{"name": "IGHJ4*02", "support": 0.8, "evidence": True}], "homozygous",
                 False, [], novel=[{"promotable": False, "positions": [10], "sequence": "TTTT"}]),
    ]


def test_maps_documented_undocumented_deleted():
    gs, _ = to_genotype_set(_calls(), "IGH")
    gt = gs["genotype_class_list"][0]
    labels = [d["label"] for d in gt["documented_alleles"]]
    assert "IGHV1-2*01" in labels and "IGHJ4*02" in labels
    assert any(u["sequence"] == "ACGT" for u in gt["undocumented_alleles"])   # promotable novel promoted
    assert all(u["sequence"] != "TTTT" for u in gt["undocumented_alleles"])   # partial-coverage NOT promoted
    assert {"label": "IGHD3-3"} in gt["deleted_genes"]
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
