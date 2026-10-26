import io
import json

import pytest

from alignair.aa.guard.data_sources import tenx_tcr as tenx
from alignair.aa.guard.data_sources.http import HttpClient
from alignair.aa.guard.hydration import sha256_text
from alignair.errors import ConfigurationError, DataContractError

pytestmark = pytest.mark.component


def _contig(chain="TRA", cdr3="C1", **overrides):
    return {"productive": True, "high_confidence": True, "cdr3": cdr3,
            "aa_sequence": ("A" if chain == "TRA" else "C") * 220,
            "annotations": [{"feature": {"region_type": "V-REGION", "chain": chain, "gene_name": chain + "V1"}}],
            **overrides}


def _numberer(named):
    return {name: {"chain_type": "A" if sequence.startswith("A") else "B", "query_start": 5, "query_end": 104}
            for name, sequence in named}


class Network:
    def __init__(self, raw):
        self.raw, self.calls = raw, []
        self.client = HttpClient(opener=self.open, sleep=lambda _: None, headers=tenx.HEADERS)

    def open(self, request, **kwargs):
        self.calls.append(request.full_url)
        assert request.get_header("Range") == tenx.HEADERS["Range"]
        return io.BytesIO(self.raw)


def test_complete_contigs_handle_nested_annotations_escapes_and_partial_tail():
    record = _contig(note='braces {} and an escaped quote " and slash \\')
    raw = ("[" + json.dumps(record) + ", " + json.dumps(_contig("TRB")) + ', {"unfinished":').encode()
    assert list(tenx.iter_contigs(raw)) == [record, _contig("TRB")]


def test_non_array_and_malformed_complete_records_raise_domain_errors():
    for raw in (b'{"error":"unavailable"}', b'[{"bad": }]', b'[}]'):
        with pytest.raises(DataContractError):
            list(tenx.iter_contigs(raw))


def test_candidates_preserve_confidence_clonotype_and_numbering_window_rules():
    rows = [_contig(productive=False), _contig(high_confidence=False), _contig(productive="T"),
            _contig(aa_sequence="A" * 99), _contig(), _contig(), _contig("TRB")]
    candidates, scanned = tenx.select_candidates(json.dumps(rows).encode())
    assert scanned == len(rows)
    assert [record["chain"] for record in candidates] == ["TRA", "TRB"]
    assert all(len(record["aa_full"]) == 190 for record in candidates)


def test_candidate_cap_is_per_chain_and_stops_when_both_are_full():
    rows = [_contig(chain, str(index)) for chain in ("TRA", "TRB") for index in range(121)]
    candidates, scanned = tenx.select_candidates(json.dumps(rows).encode())
    assert len(candidates) == 240
    assert scanned == 241
    assert sum(record["chain"] == "TRA" for record in candidates) == 120


@pytest.mark.parametrize("overrides", [{"annotations": "invalid"}, {"cdr3": ["invalid"]}])
def test_malformed_contig_fields_raise_domain_errors(overrides):
    with pytest.raises(DataContractError):
        tenx.select_candidates(json.dumps([_contig(**overrides)]).encode())


@pytest.mark.parametrize("start,end,chain,accepted", [(0, 94, "A", True), (0, 93, "A", False),
                                                    (-1, 100, "A", False), (0, 190, "A", False),
                                                    (0, 100, "B", False)])
def test_numbering_requires_correct_tcr_chain_complete_domain_and_valid_span(start, end, chain, accepted):
    result = tenx.verify_tcr_domains([{"chain": "TRA", "aa_full": "A" * 190}],
                                     numberer=lambda named: {"0": {"chain_type": chain, "query_start": start, "query_end": end}})
    assert bool(result) is accepted


def test_numbering_cannot_use_a_negative_index_to_select_another_candidate():
    with pytest.raises(DataContractError, match="input index"):
        tenx.verify_tcr_domains([{"chain": "TRA", "aa_full": "A" * 190}], numberer=lambda named: {"-1": {}})


def test_collection_reuses_raw_cache_but_reapplies_current_denylist(tmp_path):
    raw = json.dumps([_contig(), _contig(cdr3="C2"), _contig("TRB")]).encode()
    network = Network(raw)
    spec = {"study": "fixture-study", "donor": "donor1", "url": "https://source.test/contigs.json"}
    options = dict(store=tmp_path / "data", census_dir=tmp_path / "reports", units=[spec],
                   client=network.client, numberer=_numberer)
    first = tenx.collect_tenx({}, **options)
    assert first.census["per_unit"]["fixture-study"]["candidates"] == 3
    assert first.census["per_unit"]["fixture-study"]["after_unit_dedup"] == 2
    assert first.census["by_category"] == {"tra": 1, "trb": 1}
    # A historical normalized-unit file must not bypass the current exclusions.
    legacy = tmp_path / "data" / "tenx_units" / "fixture-study.normalized.json"
    legacy.write_text(json.dumps(first.kept))
    second = tenx.collect_tenx({"sequence_hashes": {sha256_text("A" * 100)}}, **options)
    assert second.census["by_category"] == {"tra": 0, "trb": 1}
    assert network.calls == [spec["url"]]
    assert json.loads(legacy.read_text()) == first.kept
    assert json.loads((tmp_path / "data" / "tenx_tcr.normalized.json").read_text()) == second.kept


def test_study_donor_labels_and_cross_unit_records_are_preserved(tmp_path):
    network = Network(json.dumps([_contig(), _contig("TRB")]).encode())
    result = tenx.collect_tenx({}, store=tmp_path / "data", census_dir=tmp_path / "reports",
                               client=network.client, numberer=_numberer)
    assert len(result.kept) == 8
    assert result.census["distinct_studies"] == result.census["distinct_donors"] == 4
    assert result.census["provenance"]["datasets"] == [unit["study"] for unit in tenx.UNITS_SPEC]
    assert {record["repository"] for record in result.kept} == {"10x"}


def test_duplicate_study_ids_are_rejected_before_downloading(tmp_path):
    network = Network(b"[]")
    spec = {"study": "same", "donor": "donor1", "url": "https://source.test/contigs.json"}
    with pytest.raises(ConfigurationError):
        tenx.collect_tenx({}, store=tmp_path, census_dir=tmp_path, units=[spec, spec], client=network.client)
    assert network.calls == []


def test_cli_reports_missing_replay_input(tmp_path, capsys):
    result = tenx.main(["--error-format", "json", "--cleaned", str(tmp_path / "absent.jsonl"),
                        "--fingerprint", str(tmp_path / "absent.json"), "--phase3-gold", str(tmp_path / "absent.tsv"),
                        "--store", str(tmp_path / "data"), "--census-dir", str(tmp_path / "reports")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_IO_FAILED"
