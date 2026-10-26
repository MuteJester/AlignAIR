from dataclasses import replace

import pytest

from alignair_assay.core.cases import evenly_spaced, make_case, read_cases, write_cases, write_fasta
from alignair_assay.core.jsonl import write_jsonl
from alignair_assay.core.seqops import revcomp
from alignair_assay.truth.build import truth_from_genairr_record
from alignair_assay.truth.case import truth_from_case
from alignair_assay.truth.io import read_truth_jsonl, truth_from_dict, truth_to_dict, write_truth_jsonl

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def records(genairr):
    from GenAIRR import Experiment
    from GenAIRR.data import HUMAN_IGH_OGRDB
    return Experiment.on(HUMAN_IGH_OGRDB).recombine().mutate(rate=0.03).run_records(n=20, seed=5)


def _cases(records, orientation=0):
    return [make_case(r, case_id=f"c{i}", suite="S0", stratum="s", orientation=orientation,
                      labels={"shm_class": "igm"}, groups={"subject": "d1"}) for i, r in enumerate(records)]


def test_case_round_trip(tmp_path, records):
    cases = _cases(records)
    for name in ("cases.jsonl", "cases.jsonl.gz"):
        write_cases(cases, tmp_path / name)
        assert read_cases(tmp_path / name) == cases
    assert (tmp_path / "cases.jsonl.gz").read_bytes()[:2] == b"\x1f\x8b"


def test_presented_sequence_orientation(records):
    rec = records[0]
    assert _cases(records)[0]["sequence"] == rec["sequence"].upper()
    flipped = make_case(rec, case_id="x", suite="S0", stratum="s", orientation=1)
    assert flipped["sequence"] == revcomp(rec["sequence"].upper())
    assert flipped["labels"] == {} and flipped["groups"] == {}


def test_truth_from_case_matches_record_truth(records, reference):
    for orientation in (0, 1):
        for case, rec in zip(_cases(records, orientation)[:5], records):
            got = truth_from_case(case, reference)
            want = truth_from_genairr_record(rec, reference, case_id=case["case_id"], stratum="s",
                                             orientation=orientation)
            assert got.segments == want.segments and got.junction == want.junction
            assert got.orientation == orientation and got.sequence == want.sequence
            assert (got.suite, got.labels, got.groups) == ("S0", {"shm_class": "igm"}, {"subject": "d1"})


def test_truth_from_case_rejects_wrong_sequence(records, reference):
    case = _cases(records)[0]
    case["sequence"] = revcomp(case["sequence"])
    with pytest.raises(ValueError, match="c0"):
        truth_from_case(case, reference)


def test_truth_round_trip_keeps_labels_groups(tmp_path, records, reference):
    truth = replace(truth_from_case(_cases(records)[0], reference), v_novel_snps=(3, 9))
    write_truth_jsonl([truth], tmp_path / "t.jsonl.gz")
    assert read_truth_jsonl(tmp_path / "t.jsonl.gz") == [truth]


def test_old_truth_file_loads_with_defaults(tmp_path, records, reference):
    row = truth_to_dict(truth_from_case(_cases(records)[0], reference))
    for key in ("suite", "labels", "groups", "v_novel_snps"):
        del row[key]
    write_jsonl([row], tmp_path / "old.jsonl")
    old = read_truth_jsonl(tmp_path / "old.jsonl")[0]
    assert (old.suite, old.labels, old.groups, old.v_novel_snps) == ("", {}, {}, None)
    assert truth_from_dict(row) == old


def test_fasta_ids_and_sequences(tmp_path, records):
    cases = _cases(records)
    write_fasta(cases, tmp_path / "reads.fasta")
    lines = (tmp_path / "reads.fasta").read_text().splitlines()
    assert lines[0::2] == [f">{c['case_id']}" for c in cases]
    assert lines[1::2] == [c["sequence"] for c in cases]


def test_evenly_spaced_takes_positions_i_times_n_over_k():
    items = list(range(10))
    assert evenly_spaced(items, 5) == [0, 2, 4, 6, 8]  # every (N / k)-th when that is whole
    assert evenly_spaced(items, 3) == [0, 3, 6]          # floor(i * N / k)
    assert evenly_spaced(items, 1) == [0]
    assert evenly_spaced(items, 10) == items
    assert evenly_spaced(tuple("abcd"), 2) == ["a", "c"]
    for k in (0, 11):
        with pytest.raises(ValueError, match="1 to 10"):
            evenly_spaced(items, k)
