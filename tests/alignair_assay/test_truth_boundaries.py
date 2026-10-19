import pytest

from alignair_assay.truth.boundaries import boundary_intervals
from alignair_assay.truth.segment import observe_segment

ga = pytest.importorskip("GenAIRR")

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
    # the most-shrunk segment still keeps at least one base
    assert bt.seq_end[0] - bt.seq_start[1] >= 1
    assert bt.germ_end[0] - bt.germ_start[1] >= 1


def test_indel_stops_shrink_run():
    # trailing mismatch preceded by an insertion: run stops at I
    read = "ACGTAAT"
    seg = observe_segment(read, ALLELE, "X*01", 0, 0, "4M1I2M")
    # last two columns: read 'A' vs allele[4]='A' match; read 'T' vs allele[5]='C' mismatch
    bt = boundary_intervals(read, ALLELE, seg, 0, len(read))
    assert bt.seq_end == (6, 7)


def test_generative_boundary_inside_interval():
    exp = (ga.Experiment.on("human_igh").recombine().mutate(rate=0.08)
           .polymerase_indels(count=(0, 4)).end_loss_5prime(length=(0, 120))
           .end_loss_3prime(length=(0, 30)))
    recs = exp.run_records(n=300, seed=3)
    from alignair_assay.reference import Reference
    ref = Reference.from_dataconfig("HUMAN_IGH_OGRDB")
    checked = nondegenerate = 0
    for rec in recs:
        read = rec["sequence"].upper()
        segs = {}
        for g in "vdj":
            if not rec.get(f"{g}_call"):
                continue
            call = rec[f"{g}_call"].split(",")[0]
            s = observe_segment(read, ref.seq(call), call, rec[f"{g}_sequence_start"],
                                rec[f"{g}_germline_start"], rec[f"{g}_cigar"])
            if s is not None:
                segs[g] = s
        order = [g for g in "vdj" if g in segs]
        for i, g in enumerate(order):
            seg = segs[g]
            left = segs[order[i - 1]].seq_end if i > 0 else 0
            right = segs[order[i + 1]].seq_start if i + 1 < len(order) else len(read)
            bt = boundary_intervals(read, ref.seq(seg.allele), seg, left, right)
            for (lo, hi), x in [(bt.seq_start, seg.seq_start), (bt.seq_end, seg.seq_end),
                                (bt.germ_start, seg.germ_start), (bt.germ_end, seg.germ_end)]:
                assert lo <= x <= hi, (g, bt, seg)
                checked += 1
                nondegenerate += lo < hi
    assert checked > 1000
    assert nondegenerate > 0
