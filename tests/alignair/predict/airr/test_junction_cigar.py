"""Fix A: CIGAR-aware, indel-robust junction derivation (read coordinates)."""
import pytest

from alignair.predict.airr import build_airr
from alignair.predict.airr.regions import (cigar_has_indel, compute_junction_cigar,
                                           cys_position, map_germline_to_read)


def test_cigar_has_indel():
    assert not cigar_has_indel("295M")
    assert not cigar_has_indel("")
    assert not cigar_has_indel(None)
    assert cigar_has_indel("100M1D57M")
    assert cigar_has_indel("1I294M")


def test_cys_position_subtracts_gap_dots_before_column_309():
    # 5 IMGT gap dots before column 309 -> ungapped Cys position 304
    assert cys_position("." * 5 + "A" * 310) == 304
    # no gaps -> Cys is exactly at column 309
    assert cys_position("A" * 330) == 309
    # reference too short to contain the column -> None (caller falls back)
    assert cys_position("A" * 100) is None


def test_map_germline_to_read_pure_match():
    # read[0:10] vs germline[0:10], all match -> germline pos g maps to read pos g
    assert map_germline_to_read(7, 0, 0, 10, "10M") == 7


def test_map_germline_to_read_deletion_shifts_read_back():
    # read[0:9] vs germline[0:10]; a deletion at germline pos 5 means germline pos 7 sits at read 6
    assert map_germline_to_read(7, 0, 0, 9, "5M1D4M") == 6


def test_map_germline_to_read_insertion_shifts_read_forward():
    # read[0:11] vs germline[0:10]; an inserted read base at pos 5 pushes germline pos 7 to read 8
    assert map_germline_to_read(7, 0, 0, 11, "5M1I5M") == 8


def test_map_germline_to_read_clamps_past_aligned_germline():
    # Cys trimmed off (target beyond the aligned germline) -> clamp to seq_end
    assert map_germline_to_read(50, 0, 0, 10, "10M") == 10


@pytest.mark.slow
def test_junction_survives_a_single_v_region_indel():
    """A clean rearrangement's junction must be unchanged by a single V-region deletion when the
    read carries a correct indel-aware CIGAR (the fixed-column path drops/duplicates a base here)."""
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v0, j0 = ref.gene("V").names[0], ref.gene("J").names[0]
    v_seq = ref.gene("V").sequences[ref.gene("V").index[v0]].upper()
    j_seq = ref.gene("J").sequences[ref.gene("J").index[j0]].upper()
    np_region = "GGGGGGGG"
    read = v_seq + np_region + j_seq
    jss = len(v_seq) + len(np_region)

    def rec_for(seq, v_end, v_cigar, jstart):
        return {"sequence_id": "t", "sequence": seq, "v_call": v0, "d_call": "Short-D", "j_call": j0,
                "v_sequence_start": 0, "v_sequence_end": v_end,
                "v_germline_start": 0, "v_germline_end": len(v_seq), "v_cigar": v_cigar,
                "d_sequence_start": v_end, "d_sequence_end": v_end,
                "d_germline_start": 0, "d_germline_end": 0,
                "j_sequence_start": jstart, "j_sequence_end": jstart + len(j_seq),
                "j_germline_start": 0, "j_germline_end": len(j_seq), "j_cigar": f"{len(j_seq)}M",
                "productive": True, "indel_count": 0}

    baseline = build_airr([rec_for(read, len(v_seq), f"{len(v_seq)}M", jss)], ref, chain="heavy")[0]
    assert baseline["junction"] and baseline["junction_length"] > 0

    # delete one V base at position p; downstream read coords shift by -1; germline end unchanged
    p = 100
    del_read = read[:p] + read[p + 1:]
    del_cigar = f"{p}M1D{len(v_seq) - p - 1}M"
    del_rec = rec_for(del_read, len(v_seq) - 1, del_cigar, jss - 1)
    del_rec["indel_count"] = 1
    out = build_airr([del_rec], ref, chain="heavy")[0]

    assert out["junction"].upper() == baseline["junction"].upper()
