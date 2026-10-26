
import pytest
from alignair_assay.truth.boundaries import boundary_intervals
from alignair_assay.truth.segment import observe_segment

pytestmark = pytest.mark.component

ALLELE = "ACGTACGTAC"


def _seg(read, seq_start, germ_start, cigar):
    return observe_segment(read, ALLELE, "X*01", seq_start, germ_start, cigar)


def test_exact_when_no_ambiguity():
    # segment = read[0:5] vs germline[0:5]; next read base T != germline[5]='C'
    read = "ACGTAT" + "GGGG"
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (5, 5)
    assert bt.germ_end == (5, 5)
    assert bt.seq_start == (0, 0)
    assert bt.germ_start == (0, 0)


def test_np_base_matches_germline_extends():
    # next read base 'C' == allele[5]
    read = "ACGTACGGGG"
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (5, 7)  # read[5]='C'==allele[5], read[6]='G'==allele[6]
    assert bt.germ_end == (5, 7)


def test_np_base_matches_single():
    read = "ACGTACTTTT"
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (5, 6)
    assert bt.germ_end == (5, 6)


def test_edge_mismatch_shrinks():
    read = "ACGTTTTTTT"  # column 4: read T vs germline A mismatch
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (4, 5)
    assert bt.germ_end == (4, 5)


def test_left_edge_shrink_and_extend():
    # segment read[2:7] vs germline[2:7]; read[1] matches germline[1]
    read = "TCTTTTTTGG"
    seg = _seg(read, 2, 2, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    # read[2]='T' vs 'G' mismatch, read[3]='T' vs 'T' match -> shrink_l = 1
    # read[1]='C' vs allele[1]='C' -> extend 1; read[0]='T' vs 'A' stops
    assert bt.seq_start == (1, 3)
    assert bt.germ_start == (1, 3)


def test_extension_stops_at_neighbour():
    read = "ACGTACGGGG"
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, 6)
    assert bt.seq_end == (5, 6)
    bt = boundary_intervals(read, ALLELE, seg, 0, 5)
    assert bt.seq_end == (5, 5)


def test_left_extension_stops_at_limit():
    read = "ACGTACGTAC"
    seg = _seg(read, 5, 5, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 4, len(read))
    assert bt.seq_start == (4, 5)
    bt = boundary_intervals(read, ALLELE, seg, 5, len(read))
    assert bt.seq_start == (5, 5)


def test_extension_stops_at_germline_end():
    read = "ACGTACGTACAA"
    seg = observe_segment(read, ALLELE, "X*01", 0, 0, "10M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.germ_end == (10, 10)


def test_shrink_never_removes_all_columns():
    read = "TTTTT"
    allele = "AAAAA"
    seg = observe_segment(read, allele, "X*01", 0, 0, "5M")
    bt = boundary_intervals(read, allele, seg, 0, 5)
    # tie-break is deliberate: the right edge gets the shrink budget first
    assert bt.seq_end == (1, 5)
    assert bt.germ_end == (1, 5)
    assert bt.seq_start == (0, 0)
    assert bt.germ_start == (0, 0)


def test_read_n_at_edge_is_shrink_eligible():
    read = "ACGTN" + "TTTTT"
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (4, 5)
    read = "NCGTA" + "TTTTT"
    seg = _seg(read, 0, 0, "5M")
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_start == (0, 1)


def test_indel_stops_shrink_run():
    # trailing mismatch preceded by an insertion: run stops at I
    read = "ACGTAAT"
    seg = observe_segment(read, ALLELE, "X*01", 0, 0, "4M1I2M")
    # last two columns: read 'A' vs allele[4]='A' match; read 'T' vs allele[5]='C' mismatch
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (6, 7)


def test_generative_boundary_matches_np_oracle(genairr, reference):
    """Unmutated, error-free reads: expected extension comes from the NP sequence, not the read."""
    exp = (genairr.Experiment.on("human_igh").recombine()
           .end_loss_5prime(length=(0, 20)).end_loss_3prime(length=(0, 10)))
    recs = exp.run_records(n=300, seed=5)
    v_pos = j_pos = v_checked = j_checked = 0
    for rec in recs:
        if any(rec[k] for k in ("p_v_3_length", "p_d_5_length", "p_d_3_length", "p_j_5_length")):
            continue
        read = rec["sequence"].upper()
        segs = {}
        for g in "vdj":
            if not rec.get(f"{g}_call"):
                continue
            call = rec[f"{g}_call"].split(",")[0]
            s = observe_segment(read, reference.seq(call), call, rec[f"{g}_sequence_start"],
                                rec[f"{g}_germline_start"], rec[f"{g}_cigar"])
            if s is not None:
                segs[g] = s
        if not {"v", "d", "j"} <= set(segs):
            continue
        v, d, j = segs["v"], segs["d"], segs["j"]
        np1, np2 = rec["np1"].upper(), rec["np2"].upper()

        if d.seq_start == v.seq_end + len(np1):
            allele = reference.seq(v.allele).upper()
            assert read[v.seq_end:v.seq_end + len(np1)] == np1
            k = 0
            while (k < len(np1) and v.germ_end + k < len(allele)
                   and np1[k] == allele[v.germ_end + k]):
                k += 1
            bt = boundary_intervals(read, allele, v, 0, d.seq_start)
            assert bt.seq_end == (v.seq_end, v.seq_end + k), (rec["np1"], bt)
            assert bt.germ_end == (v.germ_end, v.germ_end + k)
            v_checked += 1
            v_pos += k > 0

        if j.seq_start == d.seq_end + len(np2):
            allele = reference.seq(j.allele).upper()
            assert read[d.seq_end:j.seq_start] == np2
            k = 0
            while (k < len(np2) and j.germ_start - 1 - k >= 0
                   and np2[len(np2) - 1 - k] == allele[j.germ_start - 1 - k]):
                k += 1
            bt = boundary_intervals(read, allele, j, d.seq_end, len(read))
            assert bt.seq_start == (j.seq_start - k, j.seq_start), (rec["np2"], bt)
            assert bt.germ_start == (j.germ_start - k, j.germ_start)
            j_checked += 1
            j_pos += k > 0

    assert v_checked > 100 and j_checked > 100
    assert v_pos >= 20, v_pos
    assert j_pos >= 20, j_pos
