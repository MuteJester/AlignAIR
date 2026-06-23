"""Unit tests for the derived AIRR fields (np1/np2, vj_in_frame, stop_codon)."""
from alignair.inference.dnalignair_infer import derived_rearrangement_fields


def test_airr_columns_include_downstream_fields():
    from alignair.io.airr import COLUMNS
    for c in ("np1", "np1_length", "np2", "np2_length", "vj_in_frame", "stop_codon"):
        assert c in COLUMNS


def test_derived_fields_serialize_through_write_airr(tmp_path):
    """The fields populate the actual TSV when coordinates are sane (wiring, not just the helper)."""
    import csv
    from alignair.io.airr import write_airr
    seq = "A" * 20 + "CCC" + "G" * 8 + "TT" + "C" * 20   # V[0:20] np1 D[23:31] np2 J[33:53]
    p = {"v_call": "IGHV1-2*02", "d_call": "IGHD3-10*01", "j_call": "IGHJ4*02", "productive": True,
         "v_sequence_start": 0, "v_sequence_end": 20, "d_sequence_start": 23, "d_sequence_end": 31,
         "j_sequence_start": 33, "j_sequence_end": 53,
         "junction_length": 33, "junction_start": 17, "sequence_alignment": seq[:53]}
    p.update(derived_rearrangement_fields(p, seq))
    out = tmp_path / "r.tsv"
    write_airr(str(out), ["r1"], [seq], [p], locus="IGH")
    row = next(csv.DictReader(out.open(), delimiter="\t"))
    assert row["np1"] == "CCC" and row["np1_length"] == "3"
    assert row["np2"] == "TT" and row["np2_length"] == "2"
    assert row["vj_in_frame"] == "True" and row["stop_codon"] == "False"


def test_np_regions_heavy_chain():
    #      0    5  7   11 13                     V ends 5, D 7-11, J starts 13
    seq = "AAAAACCGGGGTTTTAACCCCC"
    p = {"v_sequence_end": 5, "d_sequence_start": 7, "d_sequence_end": 11, "j_sequence_start": 13}
    out = derived_rearrangement_fields(p, seq)
    assert out["np1"] == seq[5:7].upper() and out["np1_length"] == 2     # V -> D
    assert out["np2"] == seq[11:13].upper() and out["np2_length"] == 2   # D -> J


def test_np_regions_light_chain_has_no_d():
    seq = "AAAAACCCGGGGG"
    p = {"v_sequence_end": 5, "j_sequence_start": 8}                     # no D coords
    out = derived_rearrangement_fields(p, seq)
    assert out["np1"] == seq[5:8].upper() and out["np1_length"] == 3
    assert out["np2"] == "" and out["np2_length"] == 0


def test_np_region_absent_when_segments_overlap():
    p = {"v_sequence_end": 10, "j_sequence_start": 6}                    # J starts before V ends
    out = derived_rearrangement_fields(p, "ACGT" * 10)
    assert "np1" not in out                                              # honest absence, not 0/empty


def test_vj_in_frame_from_junction_length():
    assert derived_rearrangement_fields({"junction_length": 33}, "A" * 60)["vj_in_frame"] is True
    assert derived_rearrangement_fields({"junction_length": 34}, "A" * 60)["vj_in_frame"] is False


def test_stop_codon_detected_in_coding_frame():
    aln = "ATGTAAATG"                          # M * M  -> stop present
    p = {"sequence_alignment": aln, "v_sequence_start": 0, "junction_start": 6}  # frame 0
    assert derived_rearrangement_fields(p, "N" * 20)["stop_codon"] is True


def test_no_stop_codon_in_clean_frame():
    aln = "ATGATGATGATG"                        # M M M M -> none
    p = {"sequence_alignment": aln, "v_sequence_start": 0, "junction_start": 9}  # frame 0
    assert derived_rearrangement_fields(p, "N" * 20)["stop_codon"] is False


def test_stop_codon_respects_frame_offset():
    # junction_start - aln_start = 1 -> frame 1; drop the leading base, then ATG TAA ATG
    aln = "GATGTAAATGG"                         # frame 1 -> ATGTAAATG -> M * M (stop is internal)
    p = {"sequence_alignment": aln, "v_sequence_start": 0, "junction_start": 4}  # (4-0)%3 == 1
    assert derived_rearrangement_fields(p, "N" * 20)["stop_codon"] is True
