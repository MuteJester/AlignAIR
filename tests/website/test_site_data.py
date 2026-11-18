"""The website's data (``website/src/data/*.json``) is generated from the code by ``website/scripts/site_data.py``: the
committed files are what the code produces now, so the option reference, the column dictionary, the error codes, the
Python API and the search index cannot drift. The model list comes from the seven model files, and the comparison with
IgBLAST only from a locked test-split run."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "website" / "src" / "data"
MODEL_IDS = {"IGH": "alignair4-igh-human", "IGK": "alignair4-igk-human", "IGL": "alignair4-igl-human",
             "TRA": "alignair4-tra-human", "TRB": "alignair4-trb-human", "TRD": "alignair4-trd-human",
             "TRG": "alignair4-trg-human"}
ALLELES = {"IGH": (198, 33, 7), "IGK": (168, 0, 8), "IGL": (181, 0, 10), "TRA": (76, 0, 68), "TRB": (98, 3, 16),
           "TRD": (14, 3, 4), "TRG": (20, 0, 6)}


def _site_data():
    spec = importlib.util.spec_from_file_location("site_data", ROOT / "website" / "scripts" / "site_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["cli-options", "output-columns", "error-codes", "python-api", "search-index"])
def test_generated_site_data_matches_the_code(name):
    assert json.loads((DATA / f"{name}.json").read_text()) == json.loads(_site_data().render(name))


def test_the_option_reference_covers_every_command_and_groups_predict_by_purpose():
    from alignair.cli.predict import PREDICT_GROUPS
    options = json.loads((DATA / "cli-options.json").read_text())
    assert {"predict", "train", "info", "models get", "reference export", "genotype"} <= set(options)
    assert [group["title"] for group in options["predict"]["groups"]] == list(PREDICT_GROUPS)
    flags = {flag for group in options["predict"]["groups"] for option in group["options"] for flag in option["flags"]}
    assert {"--model", "--genotype", "--reference", "--threads"} <= flags and "--help" not in flags


def test_every_error_code_has_a_meaning_and_an_action():
    entries = json.loads((DATA / "error-codes.json").read_text())
    assert {"ALIGNAIR_MODEL_NOT_FOUND", "ALIGNAIR_REGISTRY_UNAVAILABLE", "ALIGNAIR_GENOTYPE_ALLELE_NOT_FOUND"} <= {
        entry["code"] for entry in entries}
    assert [entry["code"] for entry in entries if not entry["meaning"] or not entry["action"]] == []


def test_every_error_row_has_its_own_anchor():
    # Two error classes can share a code (ALIGNAIR_CONFIG_INVALID); each row still needs a distinct link target.
    anchors = [entry["anchor"] for entry in json.loads((DATA / "error-codes.json").read_text())]
    assert sorted({anchor for anchor in anchors if anchors.count(anchor) > 1}) == []


def test_the_model_list_names_the_seven_models():
    models = json.loads((DATA / "models.json").read_text())
    assert {model["locus"]: model["id"] for model in models} == MODEL_IDS
    assert {model["locus"]: (model["alleles"]["V"], model["alleles"]["D"], model["alleles"]["J"])
            for model in models} == ALLELES
    assert {model["locus"]: model["naming"] for model in models} == {
        locus: "OGRDB" if locus.startswith("IG") else "IMGT" for locus in MODEL_IDS}
    assert {model["licence"] for model in models} == {"GPL-3.0-or-later"}


def _report(split: str) -> dict:
    row = {"a": {"mean": 0.899}, "b": {"mean": 0.974}, "delta": 0.075, "ci95": [0.065, 0.085],
           "verdict": "b_better", "p_value": 1e-9}
    return {"kind": "alignair_assay.run_report", "version": "2", "a": "igblast", "b": "alignair",
            "config": {"split": split, "reference": "HUMAN_IGH_OGRDB", "suites": ["S1"], "tools": ["igblast", "alignair"]},
            "suites": [{"name": "S1", "title": "Core repertoire", "cases": 500,
                        "arms": {"shared": {"compared": True, "rows": {"d.top1_in_set": row}}}}]}


def test_the_comparison_refuses_a_development_split_run(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report("dev")))
    with pytest.raises(ValueError, match="test split"):
        _site_data().comparison([path], None)


def test_the_comparison_reads_a_locked_test_split_run(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report("test")))
    comparison = _site_data().comparison([path], None)
    (row,) = comparison["loci"]["IGH"]["suites"][0]["rows"]
    assert comparison["status"] == "ready" and row == {"measure": "d.top1_in_set", "igblast": 0.899,
                                                        "alignair": 0.974, "delta": 0.075, "ci95": [0.065, 0.085],
                                                        "verdict": "b_better"}


def _real_report() -> dict:
    def measure(a, b, verdict):
        return {"a": {"mean": a}, "b": {"mean": b}, "mean_a_paired": a, "mean_b_paired": b, "delta": b - a,
                "ci95": [b - a - 0.001, b - a + 0.001], "verdict": verdict, "n_paired": 21290, "n_clusters": 20870}
    def protocol(donors):
        shared = {"arm": "shared", "a": "igblast", "b": "alignair-api@alignair4-igh-human", "donors": donors,
                  "overall": {"real.v.unsupported": measure(0.052, 0.051, "tie_or_inconclusive"),
                              "real.v.known_top1": measure(0.995, 0.997, "b_better"),
                              "real.v.no_call": measure(0.0, 0.0, "no_variation")}}
        return {"donors": donors, "comparisons": [shared, {**shared, "arm": "genotype"}]}
    race, multiplex = protocol(["SC-13", "SC-17"]), protocol(["W-110", "W-52"])
    return {"kind": "alignair_assay.real_summary", "config": {"split": "test"},
            "donors": {"SC-13": {"reads": "x" * 10_000}},
            "pooled": {"donors": ["SC-13", "SC-17", "W-110", "W-52"], "comparisons": race["comparisons"],
                       "protocols": {"race": race, "multiplex": multiplex}}}


def test_the_comparison_keeps_the_real_data_runs_declared_measures_only(tmp_path):
    """The real-data report is large (per-donor detail); the page shows its pre-declared measure and the known-answer
    accuracy beside it, per protocol and pooled, on the shared arm."""
    sim, real = tmp_path / "report.json", tmp_path / "real.json"
    sim.write_text(json.dumps(_report("test")))
    real.write_text(json.dumps(_real_report()))
    out = _site_data().comparison([sim], real)["real"]
    assert out["donors"] == ["SC-13", "SC-17", "W-110", "W-52"]
    assert sorted(out["scopes"]) == ["multiplex", "pooled", "race"]
    assert out["scopes"]["race"]["donors"] == ["SC-13", "SC-17"]
    assert out["scopes"]["race"]["measures"]["real.v.known_top1"] == {
        "igblast": 0.995, "alignair": 0.997, "delta": 0.997 - 0.995, "ci95": [0.997 - 0.995 - 0.001,
        0.997 - 0.995 + 0.001], "verdict": "b_better", "paired": 21290, "clonotypes": 20870}
    assert sorted(out["scopes"]["pooled"]["measures"]) == ["real.v.known_top1", "real.v.unsupported"]
    assert len(json.dumps(out)) < 5000


def test_the_committed_comparison_is_pending_or_from_a_test_split_run():
    comparison = json.loads((DATA / "comparison.json").read_text())
    assert comparison["status"] in ("pending", "ready")
    if comparison["status"] == "ready":
        assert comparison["split"] == "test"


def test_a_report_of_another_pair_of_tools_is_refused(tmp_path):
    report = copy.deepcopy(_report("test"))
    report["a"] = "oracle"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="igblast"):
        _site_data().comparison([path], None)


def test_the_search_index_holds_each_sections_prose():
    # Words that are only in a page's text (Cell Ranger, slurm, mouse) must find it: each heading carries its prose.
    site_data = _site_data()
    chunk = ('<H2 id="a">First</H2>\n<p>Cell <code>x</code> Ranger&apos;s file,{" "}\n  split</p>'
             '<CodeBlock code={`alignair predict`} />\n<H3 id="b">Second</H3><p>slurm</p>')
    assert site_data.section_texts(chunk) == {"a": "Cell x Ranger's file, split", "b": "slurm"}
