import pytest

from alignair_assay.truth.segment import check_invariants, observe_segment

pytestmark = pytest.mark.component

ga = pytest.importorskip("GenAIRR")


def _observe(rec, g, reference):
    call = rec[f"{g}_call"].split(",")[0]
    return observe_segment(
        rec["sequence"].upper(), reference.seq(call), call,
        rec[f"{g}_sequence_start"], rec[f"{g}_germline_start"], rec[f"{g}_cigar"],
    )


def _records(n, seed, **steps):
    exp = ga.Experiment.on("human_igh").recombine()
    if "mutate" in steps:
        exp = exp.mutate(rate=steps["mutate"])
    if "indels" in steps:
        exp = exp.polymerase_indels(count=steps["indels"])
    if "e5" in steps:
        exp = exp.end_loss_5prime(length=steps["e5"])
    if "e3" in steps:
        exp = exp.end_loss_3prime(length=steps["e3"])
    return exp.run_records(n=n, seed=seed)


def test_five_prime_end_loss_unmutated(reference):
    rec = _records(1, 7, e5=(30, 30))[0]
    seg = _observe(rec, "v", reference)
    assert seg.germ_start == 30
    assert "D" not in seg.cigar
    assert seg.n_mismatch == 0
    assert seg.seq_start == 0


def test_three_prime_end_loss_j(reference):
    # Exactly 20 bases lost at the 3' end. GenAIRR up to the assembled encoding records them as trailing D columns
    # (j_germline_end still the J's end); the generating-reference encoding as missing coverage. Either way a read
    # ending inside J observes it up to 20 bases before its end, without a gap at the alignment's end.
    recs = _records(50, 7, e3=(20, 20))
    for rec in recs:
        seg = _observe(rec, "j", reference)
        if seg is not None and seg.seq_end == len(rec["sequence"]) and rec["j_trim_3"] == 0:
            break
    else:
        pytest.fail("no record with the read ending inside J")
    assert seg.germ_end == len(reference.seq(seg.allele)) - 20
    assert not seg.cigar.endswith("D")


def test_invariants_property(reference):
    recs = _records(300, 3, mutate=0.08, indels=(0, 4), e5=(0, 120), e3=(0, 30))
    checked = 0
    for rec in recs:
        read = rec["sequence"].upper()
        for g in "vdj":
            if not rec.get(f"{g}_call"):
                continue
            seg = _observe(rec, g, reference)
            if seg is None:
                continue
            checked += 1
            assert check_invariants(read, reference.seq(seg.allele), seg) == []
    assert checked > 300


def test_n_bases_not_mismatches():
    allele = "ACGTACGTAC"
    read = "ACGTNCGTAC"
    seg = observe_segment(read, allele, "X*01", 0, 0, "10M")
    assert seg.n_mismatch == 0
    assert seg.n_aligned == 10


def test_edge_gaps_stripped():
    seg = observe_segment("GGACGTAC", "ACGTACGT", "X*01", 0, 0, "2D1I4M2I")
    # leading D shifts germline start, I runs shrink the read span
    assert (seg.seq_start, seg.seq_end, seg.germ_start, seg.germ_end) == (1, 5, 2, 6)
    assert seg.cigar == "4M"


def test_fully_clipped_segment_absent(reference):
    recs = _records(5, 7, e5=(330, 330))
    assert any(_observe(rec, "v", reference) is None for rec in recs)


def test_unmutated_segments_have_zero_mismatches(reference):
    # Error-free recipe: any mismatch means the truth coordinates are wrong.
    recs = _records(300, 11, e5=(0, 120), e3=(0, 30))
    checked = 0
    for rec in recs:
        for g in "vdj":
            if not rec.get(f"{g}_call"):
                continue
            seg = _observe(rec, g, reference)
            if seg is None:
                continue
            checked += 1
            assert seg.n_mismatch == 0, (g, rec[f"{g}_cigar"], seg)
    assert checked > 300


@pytest.mark.parametrize("cigar,seq_end,germ_end", [("2D5M", 5, 7), ("5M1I", 6, 5)])
def test_invariants_flag_edge_gaps(cigar, seq_end, germ_end):
    from alignair_assay.truth.segment import ObservedSegment
    allele = "ACGTACGTAC"
    germ_start = 2 if cigar.startswith("2D") else 0
    seg = ObservedSegment("X*01", 0, seq_end, germ_start, germ_end, cigar, 0, 5, "")
    assert check_invariants("ACGTACGTAC", allele, seg)
