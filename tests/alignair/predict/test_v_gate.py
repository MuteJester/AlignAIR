"""The V gate of the prediction pipeline (``predict.v_presence``): a read without a V gets an AIRR row without a V call
or V coordinates, carrying ``v_presence`` ``"absent"``, partial (``no_v``), its D and J called as any read's; a read
holding a V is not gated (the golden reads)."""
import json

import pytest

from tests.alignair.predict.golden_rows import FIXTURE, golden_model
from tests.alignair.predict.test_v_presence import _flank, _kmers

pytestmark = pytest.mark.integration

V_FIELDS = ("v_sequence_start", "v_sequence_end", "v_germline_start", "v_germline_end", "v_cigar")


@pytest.fixture(scope="module")
def golden():
    return golden_model()


def _dj_read(reference):
    """A D-J read the golden model's spans fit (V 0-290, D 300-316, J 322-360): a 300 nt flank holding no V k-mer,
    a D germline, six bases and a J germline."""
    d = reference.gene("D").sequences[reference.gene("D").index["IGHD3-10*01"]]
    j = reference.gene("J").sequences[reference.gene("J").index["IGHJ4*02"]]
    read = _flank(300) + d[:16] + "GGGAAA" + j
    v_kmers = set().union(*(_kmers(s) for s in reference.gene("V").sequences))
    assert not _kmers(read) & v_kmers
    return read


def test_a_read_without_a_v_has_no_v_call_and_keeps_its_d_and_j(golden):
    from alignair.api import predict_sequences
    model, reference = golden
    (row,) = predict_sequences(model, reference, [_dj_read(reference)], device="cpu")
    assert row["v_presence"] == "absent"
    assert (row["v_call"], row["v_calls"], row["v_likelihoods"]) == ("", [], [])
    assert all(row.get(key) is None for key in V_FIELDS)
    assert row["j_call"] and row["j_sequence_start"] is not None
    assert (row["airr_assembly_status"], row["airr_assembly_reason"]) == ("partial", "no_v")
    assert not row.get("junction") and row.get("productive") is None


def test_a_light_record_without_a_v_has_no_v_call(golden):
    from alignair.api import predict_sequences
    model, reference = golden
    (row,) = predict_sequences(model, reference, [_dj_read(reference)], device="cpu", airr=False)
    assert row["v_presence"] == "absent" and row["v_call"] == "" and row["j_call"]
    assert all(row.get(key) is None for key in V_FIELDS)


def test_only_a_read_holding_no_v_kmer_is_gated(golden):
    """On the golden reads: every read holding an exact V k-mer in its forward frame keeps its V call; the gated ones
    (CDR3 fragments, and a read the random-init model turns into a frame without its V) hold none."""
    from alignair.api import predict_sequences
    model, reference = golden
    rows = predict_sequences(model, reference, json.loads(FIXTURE.read_text())["reads"], device="cpu")
    v_kmers = set().union(*(_kmers(s) for s in reference.gene("V").sequences))
    holds = [bool(_kmers(row["sequence"].upper()) & v_kmers) for row in rows]
    assert all(row["v_call"] and "v_presence" not in row for row, held in zip(rows, holds) if held)
    gated = [row for row in rows if row.get("v_presence") == "absent"]
    assert gated and all(not held for row, held in zip(rows, holds) if row.get("v_presence") == "absent")
