import hashlib
import json

import pytest

from alignair.aa.guard import cohort
from alignair.aa.guard.manifest import verify_manifest
from alignair.errors import ArtifactIntegrityError, DataContractError

pytestmark = pytest.mark.component


def _sources():
    sources = {name: [] for name in cohort.SOURCES}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    counter = 0

    def record(group, category, lineage, **metadata):
        nonlocal counter
        counter += 1
        code = "".join(alphabet[(counter // 20 ** power) % 20] for power in range(6))
        sequence = "QVQL" + code + "A" * 40 + code + "D" * 35 + code + "WGQGTL"
        return {"id": lineage, "aa": sequence, "group": group, "category": category,
                "lineage_key": lineage, **metadata}

    for study in range(12):
        for category in ("igk", "igl", "pos_igh_repertoire"):
            group = "igk_igl" if category in ("igk", "igl") else "positive_real"
            for index in range(40):
                sources["oas"].append(record(group, category, f"S{study}::{category}{index}",
                    repository="oas", study=f"Study{study}_2020", donor=f"D{study}-{index}"))
    for species in ("lama glama", "vicugna pacos"):
        for index in range(60):
            sources["sabdab"].append(record("vhh_nonhuman", "vhh", f"{species}-{index}", heavy_species=species))
    for index in range(80):
        sources["sabdab"].append(record("vhh_nonhuman", "murine_vh", f"mouse-{index}", heavy_species="mus musculus"))
    for source, group, category in (("igsf", "igsf", "igsf_vset"),
                                     ("genairr_sim", "positive_sim", "pos_igh_simulator")):
        for index in range(120):
            sources[source].append(record(group, category, f"{source}-{index}"))
    for study in range(4):
        for category in ("tra", "trb"):
            for index in range(60):
                sources["tenx_tcr"].append(record("tcr", category, f"T{study}::{category}{index}",
                    repository="10x", study=f"10x_donor{study}", donor=f"Donor{study}"))
    return sources


def _write_inputs(tmp_path, sources):
    store, census_dir = tmp_path / "sources", tmp_path / "census"
    store.mkdir()
    census_dir.mkdir()
    for name, records in sources.items():
        (store / f"{name}.normalized.json").write_text(json.dumps(records))
        (census_dir / f"{name}.census.json").write_text(json.dumps({"provenance": {"raw_sha256": name}}))
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text('{"record_digest":{"global":"declared-source-filter-fingerprint"}}')
    return dict(store=store, census_dir=census_dir, split_fingerprint=fingerprint,
                output_dir=tmp_path / "results")


def test_current_workflow_allocates_parents_before_structural_generation_and_freezes_on_go(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path, _sources())
    original = cohort.audit_stage1
    observed = {}

    def audit(**kwargs):
        observed.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cohort, "audit_stage1", audit)
    result = cohort.run_gate(**inputs)
    assert result["decision"] == "GO", result["failures"]
    parents = {record["lineage_key"] for record in observed["records"] if record["group"] == "positive_real"}
    derived = [record for record in observed["records"] if record["group"] == "structural"]
    assert derived
    for record in derived:
        assert record["parent_lineage"] in parents
        role = "selection" if record["inherited_role"] == "selection" else "loco"
        assert record["parent_lineage"] in observed[role]["positive_real"]
    manifest = json.loads((inputs["output_dir"] / "FROZEN_COHORT_MANIFEST.json").read_text())
    assert verify_manifest(manifest)
    raw = (inputs["store"] / "oas.normalized.json").read_bytes()
    assert manifest["sources"]["oas"]["normalized_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["seal_provenance"]["split_fingerprint_sha256"] == hashlib.sha256(inputs["split_fingerprint"].read_bytes()).hexdigest()
    assert json.loads((inputs["output_dir"] / "STAGE1_RESULT.json").read_text()) == result


def test_missing_groups_stop_without_freezing(tmp_path):
    inputs = _write_inputs(tmp_path, {"igsf": _sources()["igsf"]})
    result = cohort.run_gate(**inputs)
    assert result["decision"].startswith("STOP")
    assert any(failure.startswith("missing_groups") for failure in result["failures"])
    assert not (inputs["output_dir"] / "FROZEN_COHORT_MANIFEST.json").exists()


def test_existing_frozen_manifest_and_report_cannot_be_replaced(tmp_path):
    inputs = _write_inputs(tmp_path, _sources())
    assert cohort.run_gate(**inputs)["decision"] == "GO"
    before = {path.name: path.read_bytes() for path in inputs["output_dir"].iterdir()}
    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        cohort.run_gate(**inputs)
    assert {path.name: path.read_bytes() for path in inputs["output_dir"].iterdir()} == before


def test_changed_source_cannot_be_frozen_as_the_audited_snapshot(tmp_path, monkeypatch):
    inputs = _write_inputs(tmp_path, _sources())
    path = inputs["store"] / "oas.normalized.json"
    original_audit = cohort.audit_stage1

    def audit(**kwargs):
        path.write_text("[]")  # simulate another process changing a source after it was read
        return original_audit(**kwargs)

    monkeypatch.setattr(cohort, "audit_stage1", audit)
    with pytest.raises(ArtifactIntegrityError, match="changed during"):
        cohort.run_gate(**inputs)
    assert not inputs["output_dir"].exists()


def test_malformed_normalized_input_is_reported_before_writing_results(tmp_path):
    inputs = _write_inputs(tmp_path, {"igsf": ["invalid record"]})
    with pytest.raises(DataContractError):
        cohort.run_gate(**inputs)
    assert not inputs["output_dir"].exists()


def test_cli_returns_failure_status_for_an_underpowered_cohort(tmp_path, capsys):
    inputs = _write_inputs(tmp_path, {"igsf": []})
    result = cohort.main(["--store", str(inputs["store"]), "--census-dir", str(inputs["census_dir"]),
                          "--out", str(inputs["output_dir"]), "--fingerprint", str(inputs["split_fingerprint"])])
    assert result == 1
    assert json.loads(capsys.readouterr().out)["decision"].startswith("STOP")
