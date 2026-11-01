"""The gapped alignment strings follow the V CIGAR and place D and J after the germline V: a V indel or a V that starts
inside its germline moves no later column, and identities count only the germline the read covers."""
import pytest

from alignair.predict.airr import build_airr
from alignair.predict.airr import quality
from alignair.predict.airr.alignment import build_sequence_alignment, compute_alignment_positions

pytestmark = pytest.mark.component

GAPPED = "AC.GT.ACGT"                     # ungapped ACGTACGT; IMGT gap columns 2 and 5


def test_sequence_alignment_puts_a_gap_where_the_read_lacks_a_germline_base():
    # the read lacks germline base 3 (T): the bases after it stay in their own germline columns
    assert build_sequence_alignment("ACGACGT", GAPPED, 0, 7, 0, 8, 7, v_cigar="3M1D4M") == "AC.G-.ACGT"


def test_sequence_alignment_drops_a_read_base_the_germline_lacks():
    # an inserted read base after germline base 3 has no germline column (IMGT's indel correction); the read after V
    # follows unchanged
    assert build_sequence_alignment("ACGTTACGTGG", GAPPED, 0, 9, 0, 8, 11, v_cigar="4M1I4M") == "AC.GT.ACGTGG"


def test_sequence_alignment_without_a_matching_cigar_lays_the_read_out_in_order():
    assert build_sequence_alignment("ACGTACGT", GAPPED, 0, 8, 0, 8, 8) == "AC.GT.ACGT"
    # a CIGAR that does not span the V window is not followed
    assert build_sequence_alignment("ACGTACGT", GAPPED, 0, 8, 0, 8, 8, v_cigar="5M") == "AC.GT.ACGT"


def test_alignment_positions_count_from_the_germline_v_end():
    # V covers germline 3..8 (read 0..5): the V part spans 8 + 2 gap columns, and the J at read 7..10 follows it
    positions = compute_alignment_positions(GAPPED, 8, 5, None, None, 7, 10, "light")
    assert positions["v_alignment_end"] == 10
    assert (positions["j_alignment_start"], positions["j_alignment_end"]) == (12, 15)


def test_alignment_positions_follow_a_v_deletion():
    # a V of 7 read bases over 8 germline bases (one deleted), then D at read 9..12 and J at read 14..20
    positions = compute_alignment_positions(GAPPED, 8, 7, 9, 12, 14, 20, "heavy")
    assert (positions["d_alignment_start"], positions["d_alignment_end"]) == (12, 15)
    assert (positions["j_alignment_start"], positions["j_alignment_end"]) == (17, 23)


def test_segment_identity_counts_only_the_germline_the_read_covers():
    assert quality.segment_identity("...GT", "ACGGT") == 1.0
    assert quality.segment_identity("..AGT", "ACGGT") == pytest.approx(2 / 3)
    assert quality.segment_identity("AC-T", "ACGT") == 0.75          # a deleted base is a mismatch


@pytest.mark.slow
def test_a_v_cut_at_its_5_end_with_a_deletion_keeps_v_and_j_identities():
    """A read whose V starts 30 bases into its germline and lacks one V base: v_identity counts only the covered
    germline (the deletion as its one mismatch), and the J alignment strings are the read's J against its germline."""
    import GenAIRR.data as gd
    from alignair.reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v0 = next(name for name in ref.gene("V").names if "." in ref.gene("V").gapped[name])
    j0 = ref.gene("J").names[0]
    v_seq = ref.gene("V").sequences[ref.gene("V").index[v0]]
    j_seq = ref.gene("J").sequences[ref.gene("J").index[j0]]
    cut, gone = 30, 150                               # germline base 150 is missing from the read
    v_read = v_seq[cut:gone] + v_seq[gone + 1:]
    read = v_read + "NNNNN" + j_seq
    v_len = len(v_read)
    rec = {"sequence_id": "t2", "sequence": read, "v_call": v0, "d_call": "Short-D", "j_call": j0,
           "v_sequence_start": 0, "v_sequence_end": v_len,
           "v_germline_start": cut, "v_germline_end": len(v_seq),
           "v_cigar": f"{gone - cut}M1D{len(v_seq) - gone - 1}M",
           "d_sequence_start": v_len, "d_sequence_end": v_len + 5,
           "d_germline_start": 0, "d_germline_end": 0,
           "j_sequence_start": v_len + 5, "j_sequence_end": len(read),
           "j_germline_start": 0, "j_germline_end": len(j_seq), "j_cigar": f"{len(j_seq)}M",
           "productive": True, "indel_count": 1}
    out = build_airr([rec], ref, chain="heavy")[0]
    covered = len(v_seq) - cut
    assert out["v_identity"] == pytest.approx((covered - 1) / covered)
    assert out["j_sequence_alignment"] == j_seq
    assert out["j_germline_alignment"] == j_seq
    assert out["j_identity"] == 1.0
