import io
import json

import pytest

from alignair.aa.guard.data_sources import igsf
from alignair.aa.guard.data_sources.http import HttpClient
from alignair.errors import ArtifactIntegrityError, DataContractError

pytestmark = pytest.mark.component


def _entry(accession, gene="GENE1", organism="Human", sequence="a" * 100):
    return {"primaryAccession": accession, "genes": [{"geneName": {"value": gene}}],
            "organism": {"scientificName": organism}, "sequence": {"value": sequence}}


def _match(accession, start=1, end=100):
    return {"metadata": {"accession": accession}, "entries": [{"entry_protein_locations": [
        {"fragments": [{"start": start, "end": end}, {"start": 500, "end": 600}]}]}]}


class Network:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.client = HttpClient(opener=self.open, sleep=lambda _: None)

    def open(self, request, **kwargs):
        self.calls.append(request.full_url)
        return io.BytesIO(json.dumps(self.pages[request.full_url]).encode())


@pytest.fixture
def source():
    return Network({
        igsf.URL: {"results": [_entry("P1"), _entry("P2", gene="GENE2"), _entry("USED"), _entry("MISSING")]},
        igsf.INTERPRO: {"results": [_match("P1")], "next": "https://source.test/page2"},
        "https://source.test/page2": {"results": [_match("P2"), _match("USED")], "next": None},
    })


def test_domain_selection_preserves_first_lineage_and_inclusive_coordinates():
    entries = [_entry("P1", sequence="XX" + "a" * 100 + "XX"), _entry("P2", sequence="C" * 100),
               _entry("USED"), _entry("MISSING")]
    records, excluded, missing = igsf.extract_domains(entries, {"P1": (3, 102), "P2": (1, 100)}, {"USED"})
    assert excluded == missing == 1
    assert len(records) == 1 and records[0]["accession"] == "P1"
    assert records[0]["aa"] == "A" * 100
    assert records[0]["domain_span"] == [3, 102]


def test_out_of_bounds_annotation_is_not_silently_shortened():
    records, _, _ = igsf.extract_domains([_entry("P1")], {"P1": (1, 110)}, set())
    assert records == []


@pytest.mark.parametrize("span", [(0, 100), (100, 1), (True, 100), ("1", 100)])
def test_invalid_coordinate_contract_is_reported(span):
    with pytest.raises(DataContractError):
        igsf.extract_domains([_entry("P1")], {"P1": span}, set())


def test_pagination_first_fragment_exclusions_and_cache_reuse(source, tmp_path):
    options = dict(store=tmp_path / "data", census_dir=tmp_path / "reports", client=source.client)
    result = igsf.collect_igsf({"lineages": {"GENE2::Human"}}, {"USED"}, **options)
    assert [record["accession"] for record in result.kept] == ["P1"]
    assert result.census["interpro_coords_available"] == 3
    assert result.census["excluded_gate3_used_accessions"] == 1
    assert result.census["no_interpro_coord"] == 1
    assert result.census["denylist_dropped"]["lineage"] == 1
    assert json.loads((tmp_path / "data" / "igsf.interpro_coords.json").read_text())["P1"] == [1, 100]
    assert (tmp_path / "reports" / "igsf.census.json").exists()
    again = igsf.collect_igsf({"lineages": {"GENE2::Human"}}, {"USED"}, **options)
    assert again.kept == result.kept and again.census == result.census
    assert source.calls == [igsf.URL, igsf.INTERPRO, "https://source.test/page2"]


@pytest.mark.parametrize("change", ["raw", "missing_checksum"])
def test_unverified_raw_cache_is_refused_without_overwriting_results(source, tmp_path, change):
    options = dict(store=tmp_path / "data", census_dir=tmp_path / "reports", client=source.client)
    igsf.collect_igsf({}, set(), **options)
    output = tmp_path / "data" / "igsf.normalized.json"
    before = output.read_bytes()
    if change == "raw":
        (tmp_path / "data" / "igsf.raw.json").write_text("{}")
    else:
        (tmp_path / "data" / "igsf.raw.json.sha256").unlink()
    with pytest.raises(ArtifactIntegrityError):
        igsf.collect_igsf({}, set(), **options)
    assert output.read_bytes() == before
    assert len(source.calls) == 3


def test_repeated_interpro_page_is_rejected_without_caching_incomplete_coordinates(tmp_path):
    source = Network({igsf.INTERPRO: {"results": [_match("P1")], "next": igsf.INTERPRO}})
    with pytest.raises(DataContractError, match="repeated"):
        igsf.fetch_interpro_coords(store=tmp_path, client=source.client)
    assert not (tmp_path / "igsf.interpro_coords.json").exists()


def test_malformed_interpro_fragment_raises_a_domain_error(tmp_path):
    source = Network({igsf.INTERPRO: {"results": [{"metadata": {"accession": "P1"}, "entries": [
        {"entry_protein_locations": [{"fragments": [7]}]}]}]}})
    with pytest.raises(DataContractError, match="fragments"):
        igsf.fetch_interpro_coords(store=tmp_path, client=source.client)
    assert not (tmp_path / "igsf.interpro_coords.json").exists()


def test_cli_uses_explicit_inputs_and_verified_cohort_digest(source, tmp_path, monkeypatch, capsys):
    from alignair.aa.guard.disjoint import _replay_splits
    from alignair.aa.product.applicability import canonical

    record = {"id": "old", "aa": "X" * 80, "lineage_key": "OLD::Human", "category": "igsf_vset",
              "group": "igsf", "expected_state": "unknown_ood"}
    splits = _replay_splits([record])
    digest = canonical.digest([canonical.canonical_record(record, splits[("igsf", "OLD::Human")])])
    cleaned, fingerprint, gold, previous = [tmp_path / name for name in ("cohort.jsonl", "fp.json", "gold.tsv", "previous.json")]
    cleaned.write_text(json.dumps(record) + "\n")
    fingerprint.write_text(json.dumps({"record_digest": {"global": digest}}))
    gold.write_text("name\texact_vh_id\n")
    previous.write_text('[{"accession":"USED"}]')
    monkeypatch.setattr(igsf, "HttpClient", lambda **kwargs: source.client)
    result = igsf.main(["--cleaned", str(cleaned), "--fingerprint", str(fingerprint), "--phase3-gold", str(gold),
                        "--previous-records", str(previous), "--store", str(tmp_path / "data"),
                        "--census-dir", str(tmp_path / "reports")])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["kept_after_denylist"] == 2
