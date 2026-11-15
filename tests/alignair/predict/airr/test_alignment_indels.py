"""The gapped alignment strings follow the V CIGAR and place D and J after the germline V: a V indel or a V that starts
inside its germline moves no later column, and identities count only the germline the read covers."""
import pytest

from alignair.predict.airr import build_airr
from alignair.predict.airr import quality
from alignair.predict.airr.alignment import (GermlineMaps, build_germline_alignment, build_sequence_alignment,
                                             compute_alignment_positions)
from alignair.predict.state import GermlineAlignment

pytestmark = pytest.mark.component

GAPPED = "AC.GT.ACGT"                     # ungapped ACGTACGT; IMGT gap columns 2 and 5


def _at(seq_start, seq_end, germ_start=0, germ_end=0, cigar="", allele=""):
    return GermlineAlignment(allele, seq_start, seq_end, germ_start, germ_end, cigar)


def test_sequence_alignment_puts_a_gap_where_the_read_lacks_a_germline_base():
    # the read lacks germline base 3 (T): the bases after it stay in their own germline columns
    assert build_sequence_alignment("ACGACGT", GAPPED, _at(0, 7, 0, 8, "3M1D4M"), 7) == "AC.G-.ACGT"


def test_sequence_alignment_drops_a_read_base_the_germline_lacks():
    # an inserted read base after germline base 3 has no germline column (IMGT's indel correction); the read after V
    # follows unchanged
    assert build_sequence_alignment("ACGTTACGTGG", GAPPED, _at(0, 9, 0, 8, "4M1I4M"), 11) == "AC.GT.ACGTGG"


def test_sequence_alignment_without_a_matching_cigar_lays_the_read_out_in_order():
    assert build_sequence_alignment("ACGTACGT", GAPPED, _at(0, 8, 0, 8), 8) == "AC.GT.ACGT"
    # a CIGAR that does not span the V window is not followed
    assert build_sequence_alignment("ACGTACGT", GAPPED, _at(0, 8, 0, 8, "5M"), 8) == "AC.GT.ACGT"


def test_alignment_positions_count_from_the_germline_v_end():
    # V covers germline 3..8 (read 0..5): the V part spans 8 + 2 gap columns, and the J at read 7..10 follows it
    positions = compute_alignment_positions(GAPPED, _at(0, 5, 3, 8), None, _at(7, 10), has_d=False)
    assert positions["v_alignment_end"] == 10
    assert (positions["j_alignment_start"], positions["j_alignment_end"]) == (12, 15)


def test_alignment_positions_follow_a_v_deletion():
    # a V of 7 read bases over 8 germline bases (one deleted), then D at read 9..12 and J at read 14..20
    positions = compute_alignment_positions(GAPPED, _at(0, 7, 0, 8), _at(9, 12), _at(14, 20), has_d=True)
    assert (positions["d_alignment_start"], positions["d_alignment_end"]) == (12, 15)
    assert (positions["j_alignment_start"], positions["j_alignment_end"]) == (17, 23)


# V ACGTACGT, np1 TT, a D of 4 read bases over the 5-base germline GGACC (its A deleted), np2 A, J TTTGGG
_D_DEL_READ = "ACGTACGT" + "TT" + "GGCC" + "A" + "TTTGGG"


def _germline(seq, d_cigar, j_start=15, j_germline="TTTGGG", j_cigar="6M"):
    germlines = GermlineMaps(v_gapped={"V1": GAPPED}, j={"J1": j_germline}, d={"D1": "GGACC"})
    v, d, j = _at(0, 8, 0, 8), _at(10, j_start - 1, 0, 5, d_cigar), _at(j_start, len(seq), 0, len(j_germline), j_cigar)
    return build_germline_alignment(seq, germlines, {"v": "V1", "d": "D1", "j": "J1"}, v, d, j, "TT", "A",
                                    has_d=True)


def test_germline_alignment_lays_the_d_germline_along_its_cigar():
    # the deleted germline base has no read column, so the D germline takes the read's 4 columns and J stays in place
    germ = _germline(_D_DEL_READ, "2M1D2M")
    assert germ == "AC.GT.ACGT" + "TT" + "GGCC" + "A" + "TTTGGG"
    positions = compute_alignment_positions(GAPPED, _at(0, 8, 0, 8), _at(10, 14), _at(15, 21), has_d=True)
    assert germ[positions["j_alignment_start"]:positions["j_alignment_end"]] == "TTTGGG"


def test_germline_alignment_puts_a_gap_where_the_read_has_a_base_the_j_germline_lacks():
    seq = "ACGTACGT" + "TT" + "GGACC" + "A" + "TTTAGGG"
    assert _germline(seq, "5M", j_start=16, j_cigar="3M1I3M").endswith("A" + "TTT-GGG")


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
    from alignair.predict.state import GermlineAlignment
    from tests.support.read_results import make_read
    rec = make_read(read, calls={"v": v0, "j": j0}, genes=("v", "d", "j"), d_presence="short", indel_count=1.0,
                    alignments={"v": GermlineAlignment(v0, 0, v_len, cut, len(v_seq),
                                                       f"{gone - cut}M1D{len(v_seq) - gone - 1}M"),
                                "d": GermlineAlignment("", v_len, v_len + 5, 0, 0, ""),
                                "j": GermlineAlignment(j0, v_len + 5, len(read), 0, len(j_seq), f"{len(j_seq)}M")})
    out = build_airr([rec], ref, has_d=True)[0]
    covered = len(v_seq) - cut
    assert out["v_identity"] == pytest.approx((covered - 1) / covered)
    assert out["j_sequence_alignment"] == j_seq
    assert out["j_germline_alignment"] == j_seq
    assert out["j_identity"] == 1.0
