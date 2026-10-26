import csv
import io
import json
import urllib.error

import pytest

from alignair.aa.guard.data_sources import sabdab
from alignair.aa.guard.data_sources.http import HttpClient
from alignair.errors import DataContractError

pytestmark = pytest.mark.component


def _row(name, pdb="1AAA", heavy=None, species="Lama glama", kind="sd-h"):
    return {"SABDAB_ID": name, "PDB": pdb, "HEAVY_ID": heavy or name, "Hchain": "H",
            "heavy_species": species, "type": kind, "date": "2020-01-01"}


def _csv(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(_row("sample")))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _fasta(residue):
    return (">entry_1|Chain L|light chain\n" + "E" * 100 + "\n>entry_2|Chains X, H|heavy chain\n" +
            "GGG" + residue * 100 + "TTT\n").encode()


def _numberer(named):
    return {name: {"chain_type": "H", "query_start": 3, "query_end": 102} for name, sequence in named}


class Network:
    def __init__(self):
        self.calls = []
        self.single = [_row("used", "1OLD", "old-heavy"), _row("clone", "1NEW", "old-heavy"),
                       _row("used-pdb", "0OLD"), _row("lama"), _row("lama-clone", "1AAB", "lama"),
                       _row("other", "2BBB", species="Vicugna pacos"),
                       _row("human", "5EEE", species="Homo sapiens")]
        self.all_rows = [_row("mouse", "3CCC", species="Mus musculus", kind="Fab"),
                         _row("same-sequence", "4DDD", species="Mus musculus", kind="Fab")]
        self.pages = {sabdab.SINGLE: _csv(self.single), sabdab.ALLSUM: _csv(self.all_rows),
                      sabdab.RCSB + "1AAA": _fasta("A"), sabdab.RCSB + "2BBB": _fasta("C"),
                      sabdab.RCSB + "3CCC": _fasta("D"), sabdab.RCSB + "4DDD": _fasta("A")}
        self.client = HttpClient(opener=self.open, sleep=lambda _: None, retries=2)

    def open(self, request, **kwargs):
        url = request.full_url
        self.calls.append(url)
        value = self.pages.get(url, 404)
        if isinstance(value, int):
            raise urllib.error.HTTPError(url, value, "fixture error", {}, None)
        return io.BytesIO(value)


PREVIOUS = {"vhh": [{"sabdab_id": "used", "pdb": "0old"}], "murine_vh": []}


def test_structure_id_pdb_and_heavy_id_exclusions_are_preserved():
    source = Network()
    selected = sabdab.select_structures(source.single, PREVIOUS)
    assert [row["SABDAB_ID"] for row in selected] == ["lama", "other"]
    assert [row["SABDAB_ID"] for row in sabdab.select_structures(source.all_rows, PREVIOUS, single=False)] == [
        "mouse", "same-sequence"]


def test_fasta_uses_the_requested_chain_from_a_multiple_chain_header():
    assert sabdab.heavy_sequence(_fasta("A").decode(), "H") == "GGG" + "A" * 100 + "TTT"
    assert sabdab.heavy_sequence(_fasta("A").decode(), "missing") is None


@pytest.mark.parametrize("length,start,end,chain,accepted", [(90, 0, 89, "H", True), (89, 0, 88, "H", False),
    (160, 0, 159, "H", True), (161, 0, 160, "H", False), (100, -1, 98, "H", False),
    (100, 0, 100, "H", False), (100, 0, 99, "K", False)])
def test_sabdab_domain_length_chain_and_span_rules(length, start, end, chain, accepted):
    numberer = lambda named: {"sample": {"chain_type": chain, "query_start": start, "query_end": end}}
    result = sabdab.extract_heavy_domains([("sample", "A" * length)], numberer=numberer)
    assert bool(result) is accepted


def test_sampling_caps_preserve_lama_nonlama_and_murine_oversampling(monkeypatch):
    rows = ([_row(f"lama{i}") for i in range(91)] +
            [_row(f"other{i}", species="Vicugna pacos") for i in range(91)])
    monkeypatch.setattr(sabdab, "_fasta", lambda *args: _fasta("A").decode())
    options = dict(store=None, client=None, numberer=_numberer)
    records = sabdab._collect(rows, {}, "vhh", 90, single=True, **options)
    assert len(records) == 180
    assert sum(row["heavy_species"] == "lama glama" for row in records) == 90
    mice = [_row(f"mouse{i}", species="Mus musculus", kind="Fab") for i in range(91)]
    assert len(sabdab._collect(mice, {}, "murine_vh", 45, single=False, **options)) == 90


def test_collection_sequence_dedup_denylist_outputs_and_fasta_cache(tmp_path):
    source = Network()
    options = dict(store=tmp_path / "data", census_dir=tmp_path / "reports", client=source.client, numberer=_numberer)
    denylist = {"lineages": {"other"}}
    result = sabdab.collect_sabdab(denylist, PREVIOUS, **options)
    assert [row["id"] for row in result.kept] == ["vhh_lama", "murine_vh_mouse"]
    assert result.census["fetched_after_extraction"] == 4
    assert result.census["after_within_source_dedup"] == 3
    assert result.census["denylist_dropped"]["lineage"] == 1
    assert result.census["vhh_lama"] == 1 and result.census["vhh_nonlama"] == 0
    assert json.loads((tmp_path / "data" / "sabdab.normalized.json").read_text()) == result.kept
    assert json.loads((tmp_path / "reports" / "sabdab.census.json").read_text()) == result.census
    before = [url for url in source.calls if url.startswith(sabdab.RCSB)]
    assert sabdab.collect_sabdab(denylist, PREVIOUS, **options).census == result.census
    assert [url for url in source.calls if url.startswith(sabdab.RCSB)] == before


def test_failed_structure_download_does_not_drop_other_structures(tmp_path, caplog):
    source = Network()
    source.pages[sabdab.RCSB + "2BBB"] = 503
    result = sabdab.collect_sabdab({}, PREVIOUS, store=tmp_path, census_dir=tmp_path / "reports",
                                   client=source.client, numberer=_numberer)
    assert {row["id"] for row in result.kept} == {"vhh_lama", "murine_vh_mouse"}
    diagnostic = next(record.alignair_error for record in caplog.records if hasattr(record, "alignair_error"))
    assert diagnostic["context"]["url"].endswith("2BBB")


def test_malformed_previous_records_are_rejected_before_any_download(tmp_path):
    source = Network()
    with pytest.raises(DataContractError):
        sabdab.collect_sabdab({}, {"vhh": "invalid"}, store=tmp_path, census_dir=tmp_path, client=source.client)
    assert source.calls == []


def test_cli_reports_missing_explicit_input(tmp_path, capsys):
    result = sabdab.main(["--error-format", "json", "--cleaned", str(tmp_path / "absent.jsonl"),
                          "--fingerprint", str(tmp_path / "absent.json"), "--phase3-gold", str(tmp_path / "absent.tsv"),
                          "--previous-records", str(tmp_path / "previous.json"),
                          "--store", str(tmp_path / "data"), "--census-dir", str(tmp_path / "reports")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_IO_FAILED"
