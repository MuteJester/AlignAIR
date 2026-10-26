import csv
import gzip
import io
import json

import pytest

from alignair.aa.guard.data_sources.numbering import verify_antibody_domains
from alignair.aa.guard.data_sources.oas.records import Exclusions, candidates, normalize_study, parse_unit, valid_subject
from alignair.errors import DataContractError

pytestmark = pytest.mark.component


@pytest.mark.parametrize("quoted", [False, True])
def test_concatenated_gzip_metadata_and_csv_are_parsed(quoted):
    metadata = json.dumps({"Species": "Human", "Author": "Study_2026"})
    if quoted:
        stream = io.StringIO()
        csv.writer(stream).writerow([metadata])
        metadata = stream.getvalue().rstrip("\r\n")
    raw = gzip.compress((metadata + "\n").encode()) + gzip.compress(b"name,value\nfirst,1\nlast,2")
    meta, rows = parse_unit(raw)
    assert meta["Species"] == "Human"
    assert rows == [{"name": "first", "value": "1"}, {"name": "last", "value": "2"}]


def test_partial_gzip_discards_incomplete_trailing_csv_record():
    first = gzip.compress(b'{"Species":"Human"}\n') + gzip.compress(b"name,value\nfirst,1\n")
    second = gzip.compress(b"second,1234567890\n")
    raw = first + second[:len(second) // 2]
    assert parse_unit(raw)[1] == [{"name": "first", "value": "1"}]


def test_decompression_and_malformed_metadata_are_reported():
    with pytest.raises(DataContractError, match="byte limit"):
        parse_unit(gzip.compress(b"A" * 10000), max_bytes=100)
    with pytest.raises(DataContractError):
        parse_unit(gzip.compress(b"not JSON\na,b\n1,2\n"))
    with pytest.raises(DataContractError, match="compressed"):
        parse_unit(b"not gzip")


def test_probe_row_limit_does_not_parse_beyond_the_sample():
    raw = gzip.compress(b'{"Species":"Human"}\nname,value\nfirst,1\n"unfinished,2\n')
    assert parse_unit(raw, row_limit=1)[1] == [{"name": "first", "value": "1"}]


@pytest.mark.parametrize("subject", ["unknown", "?", "pooled", "D1;D2", "D1 AND D2", "D1/D2", None])
def test_diversity_requires_one_explicit_subject(subject):
    assert not valid_subject(subject)
    row = {"productive_light": "T", "stop_codon_light": "F", "vj_in_frame_light": "T",
           "v_call_light": "IGKV1*01", "sequence_alignment_aa_light": "A" * 105, "cdr3_aa_light": "C1"}
    assert candidates({"Subject": subject}, [row], paired=True, diverse=True) == []


def test_existing_study_and_donor_matching_is_preserved():
    exclusions = Exclusions.from_records([
        {"category": "igk", "lineage_key": "Bender et al., 2020::D1::IGKV1::C1"},
        {"category": "structural", "lineage_key": "Unrelated_2021::D2"},
    ], {})
    assert normalize_study("Bender_2020") == "bender_2020"
    assert exclusions.studies == frozenset({"bender_2020"})
    assert exclusions.donors == frozenset({("bender_2020", "D1")})


@pytest.mark.parametrize("start,end,chain,expected", [(0, 104, "K", 1), (-1, 104, "K", 0),
                                                    (0, 200, "K", 0), (100, 50, "K", 0),
                                                    (0, 90, "K", 0), (0, 104, "H", 0)])
def test_numbering_checks_chain_complete_domain_and_inclusive_coordinates(start, end, chain, expected):
    record = {"lineage_key": "L1", "aa": "A" * 105, "category": "igk"}
    numberer = lambda named: {"L1": {"chain_type": chain, "query_start": start, "query_end": end}}
    result = verify_antibody_domains([record], light_only=True, numberer=numberer)
    assert len(result) == expected
    if expected:
        assert len(result[0]["aa"]) == 105


def test_numbering_model_is_injected_and_duplicate_lineages_keep_first_sequence():
    seen = []

    def numberer(named):
        seen.extend(named)
        return {"L1": {"chain_type": "H", "query_start": 0, "query_end": 104}}

    first = {"lineage_key": "L1", "aa": "A" * 105, "category": "pos_igh_repertoire"}
    result = verify_antibody_domains([first, {**first, "aa": "C" * 105}], numberer=numberer)
    assert seen == [("L1", "A" * 105)]
    assert result[0]["aa"] == "A" * 105


def test_numbering_results_with_unknown_names_are_not_silently_matched():
    with pytest.raises(DataContractError, match="unknown input"):
        verify_antibody_domains([{"lineage_key": "L1", "aa": "A" * 105, "category": "igk"}],
                                numberer=lambda named: {"different": {"query_start": 0, "query_end": 104}})
