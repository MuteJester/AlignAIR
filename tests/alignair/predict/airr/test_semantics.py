"""AIRR fields are semantically correct — `productive` is a DERIVED fact (not the neural
prediction), advertised per-segment identities are populated, and the validator enforces the
productivity cross-field invariant."""
from alignair.predict.airr import quality
from alignair.io.airr_validate import validate_airr_file


def test_airr_productive_is_derived_from_frame_and_stop():
    assert quality.airr_productive(True, False) is True         # in-frame, no stop -> productive
    assert quality.airr_productive(True, True) is False         # stop codon -> not productive
    assert quality.airr_productive(False, False) is False       # out of frame -> not productive
    assert quality.airr_productive(None, False) is None         # unknown -> AIRR productive stays BLANK


def test_segment_identity_generic_over_gaps():
    # 1 mismatch out of 4 non-gap germline positions; the '.' column is skipped
    assert quality.segment_identity("ACGT", "ACGA") == 0.75
    assert quality.segment_identity("AC.GT", "AC.GT") == 1.0
    assert quality.segment_identity(None, "ACGT") is None


def test_validate_airr_flags_productive_inconsistency(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("sequence_id\tsequence\tv_call\tj_call\tproductive\tvj_in_frame\tstop_codon\n"
                 "r1\tACGT\tV1\tJ1\tT\tF\tT\n")               # productive but out-of-frame AND stop
    rep = validate_airr_file(str(p))
    msgs = " ".join(m for _, m in rep["errors"])
    assert "vj_in_frame=F" in msgs and "stop_codon=T" in msgs


def test_validate_airr_accepts_consistent_productive(tmp_path):
    p = tmp_path / "ok.tsv"
    p.write_text("sequence_id\tsequence\tv_call\tj_call\tproductive\tvj_in_frame\tstop_codon\n"
                 "r1\tACGT\tV1\tJ1\tT\tT\tF\n")
    assert validate_airr_file(str(p))["errors"] == []
