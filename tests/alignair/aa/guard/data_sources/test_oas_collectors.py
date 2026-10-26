import csv
import gzip
import io
import json
import urllib.error

import pytest

from alignair.aa.guard.data_sources.http import HttpClient
from alignair.aa.guard.data_sources.oas import cli
from alignair.aa.guard.data_sources.oas.collectors import (
    DiversityPolicy, PairedPolicy, UnpairedPolicy, collect_paired, collect_unpaired,
)
from alignair.aa.guard.data_sources.oas.probe import probe_species, sample_chain
from alignair.aa.guard.data_sources.oas.records import Exclusions, PAIRED_BASE, UNPAIRED_BASE

pytestmark = pytest.mark.component


def _row(category="igk", index=0, *, paired=False, length=105):
    suffix = "_light" if paired else ""
    v_call = {"igk": "IGKV1*01", "igl": "IGLV1*01", "heavy": "IGHV1*01"}[category]
    return {key + suffix: value for key, value in {
        "productive": "T", "stop_codon": "F", "vj_in_frame": "T", "v_call": v_call,
        "sequence_alignment_aa": "A" * length, "cdr3_aa": f"C{index}",
    }.items()}


def _unit(rows, *, author="Fresh_2026", subject="D1", species="Human"):
    stream = io.StringIO()
    csv.writer(stream).writerow([json.dumps({"Author": author, "Subject": subject, "Species": species})])
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(stream.getvalue().encode())


def _links(*names):
    return "".join(f'<a href="{name}">{name}</a>' for name in names).encode()


class Network:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []
        self.client = HttpClient(opener=self.open, sleep=lambda _: None, retries=2)

    def open(self, request, **kwargs):
        url = request.full_url
        self.calls.append(url)
        value = self.payloads.get(url, 404)
        if isinstance(value, int):
            raise urllib.error.HTTPError(url, value, "fixture error", {}, None)
        return io.BytesIO(value)


def _numberer(named):
    return {name: {"chain_type": "H" if "IGHV" in name else "K" if "IGKV" in name else "L",
                   "query_start": 0, "query_end": len(sequence) - 1} for name, sequence in named}


def test_unpaired_keeps_catalog_and_donor_exclusions_and_human_download_quota(tmp_path):
    base = UNPAIRED_BASE + "Fresh_2026/csv/"
    network = Network({
        UNPAIRED_BASE: _links("Old_2020/", "Fresh_2026/"),
        base: _links("A_Light.csv.gz", "B_Light.csv.gz"),
        base + "A_Light.csv.gz": _unit([_row()], species="Mouse"),
        base + "B_Light.csv.gz": _unit([_row(index=1), _row(index=2)]),
    })
    exclusions = Exclusions.from_records([{"category": "igk", "lineage_key": "Old_2020::D0::V::C"}], {})
    catalog = json.dumps({"per_study": {"Old_2020": {"light": {"status": "eligible"}},
                                      "Fresh_2026": {"light": {"status": "eligible"}}}}).encode()
    result = collect_unpaired(exclusions, store=tmp_path / "store", census_dir=tmp_path / "reports",
                               catalog_bytes=catalog, client=network.client, numberer=_numberer,
                               policy=UnpairedPolicy(max_units=1))
    assert len(result.kept) == 2
    assert result.census["provenance"]["human_units_downloaded"] == 1
    assert result.census["rejection_reasons"]["non_human_unit"] == 1
    assert not any("Old_2020/csv" in url for url in network.calls)
    assert (tmp_path / "store" / "oas.normalized.json").is_file()
    assert (tmp_path / "reports" / "oas.census.json").is_file()
    assert "A" * 105 not in json.dumps(result.census)


def test_unpaired_excludes_previous_donors_even_when_directory_name_is_new(tmp_path):
    base = UNPAIRED_BASE + "Fresh_2026/csv/"
    network = Network({UNPAIRED_BASE: _links("Fresh_2026/"), base: _links("A_Light.csv.gz"),
                       base + "A_Light.csv.gz": _unit([_row()], author="Old_2020", subject="D0")})
    exclusions = Exclusions.from_records([{"category": "igk", "lineage_key": "Old_2020::D0::V::C"}], {})
    result = collect_unpaired(exclusions, store=tmp_path / "store", census_dir=tmp_path / "reports",
                               catalog_bytes=b'{"per_study":{"Fresh_2026":{"light":{"status":"eligible"}}}}',
                               client=network.client, numberer=_numberer)
    assert result.kept == []
    assert result.census["rejection_reasons"]["untouched_donor_audit_drop"] == 1


def test_paired_lineage_dedup_and_denylist_are_applied(tmp_path):
    base = PAIRED_BASE + "Fresh_2026/csv/"
    network = Network({PAIRED_BASE: _links("Fresh_2026/"), base: _links("data.csv.gz"),
                       base + "data.csv.gz": _unit([_row(paired=True)] * 2)})
    lineage = "Fresh_2026::D1::IGKV1*01::C0"
    exclusions = Exclusions(frozenset(), frozenset(), {"lineages": {lineage}})
    result = collect_paired(exclusions, store=tmp_path / "store", census_dir=tmp_path / "reports",
                             client=network.client, numberer=_numberer, policy=PairedPolicy(max_units=1))
    assert result.kept == []
    assert result.census["denylist_dropped"]["lineage"] == 1
    assert result.census["rejection_reasons"]["anarcii_rejected"] == 0


def test_diverse_policy_preserves_study_exclusions_donor_caps_and_directory_fallback(tmp_path):
    base = PAIRED_BASE + "Fresh_2026/csv_paired/"
    rows = [_row(cat, i, paired=True) for cat in ("igk", "igl") for i in range(3)]
    payloads = {PAIRED_BASE: _links("Eccles_2020/", "Fresh_2026/"),
                PAIRED_BASE + "Fresh_2026/csv/": 404,
                base: _links("A_pool.csv.gz", "D1.csv.gz", "D2.csv.gz", "D3.csv.gz")}
    for subject, filename in [("pooled", "A_pool"), ("D1", "D1"), ("D2", "D2"), ("D3", "D3")]:
        payloads[base + filename + ".csv.gz"] = _unit(rows, subject=subject)
    network = Network(payloads)
    result = collect_paired(Exclusions(frozenset(), frozenset(), {}), store=tmp_path / "store",
                             census_dir=tmp_path / "reports", client=network.client, numberer=_numberer,
                             policy=DiversityPolicy(per_donor_per_chain=2))
    assert len(result.kept) == 8
    assert result.source == "oas_paired_diverse"
    assert result.census["distinct_donors"] == 2
    assert result.census["by_category"] == {"igk": 4, "igl": 4}
    assert not any("Eccles_2020/csv" in url or "D3.csv.gz" in url for url in network.calls)
    assert {row["donor"] for row in result.kept} == {"D1", "D2"}


@pytest.mark.parametrize("species,length,status", [("Mouse", 105, "confirmed_ineligible"),
                                                  ("?", 105, "unknown_metadata"),
                                                  ("Human", 80, "confirmed_ineligible"),
                                                  ("Human", 105, "eligible")])
def test_probe_requires_explicit_species_and_measured_completeness(species, length, status):
    url = UNPAIRED_BASE + "Fresh_2026/csv/Heavy.csv.gz"
    network = Network({url: _unit([_row("heavy", length=length)], species=species)})
    assert sample_chain(network.client, "Fresh_2026", ["Heavy.csv.gz"], 100)["status"] == status


def test_probe_retries_failed_chain_even_when_other_chain_already_makes_study_eligible(tmp_path):
    base = UNPAIRED_BASE + "Fresh_2026/csv/"
    network = Network({UNPAIRED_BASE: _links("Fresh_2026/"), base: _links("Heavy.csv.gz", "Light.csv.gz"),
                       base + "Heavy.csv.gz": _unit([_row("heavy")]), base + "Light.csv.gz": 503})
    catalog = tmp_path / "species.json"
    first = probe_species(frozenset(), catalog_path=catalog, client=network.client)
    assert first["eligible_human_complete"] == 1
    assert json.loads(catalog.read_text())["per_study"]["Fresh_2026"]["light"]["status"] == "temporarily_inaccessible"
    network.payloads[base + "Light.csv.gz"] = _unit([_row()])
    probe_species(frozenset(), catalog_path=catalog, client=network.client)
    assert json.loads(catalog.read_text())["per_study"]["Fresh_2026"]["light"]["status"] == "eligible"
    assert network.calls.count(base + "Heavy.csv.gz") == 1
    assert network.calls.count(base + "Light.csv.gz") == 3  # two failed attempts, then success


def test_cli_reports_missing_input_with_standard_diagnostic(tmp_path, capsys):
    result = cli.main(["--error-format", "json", "probe", "--cleaned", str(tmp_path / "absent.jsonl"),
                       "--catalog", str(tmp_path / "catalog.json")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_IO_FAILED"


@pytest.mark.parametrize("record", [[1, 2], {"category": "igk", "lineage_key": 7}])
def test_cli_reports_malformed_exclusion_records_without_network(tmp_path, capsys, monkeypatch, record):
    def no_network(*args, **kwargs):
        raise AssertionError("Malformed input must be rejected before any download")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    cleaned = tmp_path / "cohort.jsonl"
    cleaned.write_text(json.dumps(record) + "\n")
    result = cli.main(["--error-format", "json", "probe", "--cleaned", str(cleaned),
                       "--catalog", str(tmp_path / "catalog.json")])
    assert result == 1
    assert json.loads(capsys.readouterr().err)["code"] == "ALIGNAIR_DATA_CONTRACT"
