"""The two GenAIRR encodings of a record's segment alignments, decoded by ``truth.genairr``.

The walker encoding (GenAIRR up to 2.3.1) reports an inverted D with fills and unrecorded internal deletions; the
assembled encoding (later engines) reports it as a plain alignment against the reverse-complemented allele, and
reports read bases past the end of a shorter re-called allele as insertions. The decoder picks the inverted-D
encoding by the CIGAR's shape and keeps the assembled decoding only when it passes the invariants.
"""
import pytest

from alignair_assay.core.seqops import revcomp
from alignair_assay.truth.bench import truth_from_bench_case
from alignair_assay.truth.build import truth_from_genairr_record
from alignair_assay.truth.genairr import observe_forward, observe_inverted_d
from alignair_assay.truth.segment import check_invariants

# A 16-nt toy D allele. The read holds NP bases, revcomp(D[2:13]) (trims 2 and 3) and more NP bases.
D_ALLELE = "ACGTTGCAAGGCTTAC"
LEFT, RIGHT = "GGGGG", "CCCCC"
D_READ = LEFT + revcomp(D_ALLELE[2:13]) + RIGHT


def _record(cigar, *, germline=(2, 13), sequence=(5, 16), trims=(2, 3)):
    return {"d_cigar": cigar, "d_germline_start": germline[0], "d_germline_end": germline[1],
            "d_sequence_start": sequence[0], "d_sequence_end": sequence[1], "d_trim_5": trims[0],
            "d_trim_3": trims[1]}


def _key(segment):
    return segment.seq_start, segment.seq_end, segment.germ_start, segment.germ_end, segment.cigar, segment.n_mismatch


def test_assembled_inverted_d_is_aligned_to_the_reverse_complement():
    problems = []
    segment = observe_inverted_d(D_READ, D_ALLELE, "D*01", _record("11M"), problems)
    # Forward germline [2, 13) of a 16-nt allele is [3, 14) of its reverse complement.
    assert problems == [] and _key(segment) == (5, 16, 3, 14, "11M", 0)
    assert check_invariants(D_READ, revcomp(D_ALLELE), segment) == []


def test_walker_and_assembled_encodings_decode_alike():
    # The walker fills from the 5' trim up to the highest observed position (2..11) and from the lowest observed
    # position up to the 3' trim (3..12), walking the 11 read bases in between.
    walker, assembled = [], []
    old = observe_inverted_d(D_READ, D_ALLELE, "D*01", _record("10D11M10D"), walker)
    new = observe_inverted_d(D_READ, D_ALLELE, "D*01", _record("11M"), assembled)
    assert walker == assembled == [] and _key(old) == _key(new) == (5, 16, 3, 14, "11M", 0)


def test_assembled_records_internal_deletions():
    # Forward position 6 is deleted: the read shows revcomp of D[7:13] then revcomp of D[2:6].
    read = LEFT + revcomp(D_ALLELE[7:13]) + revcomp(D_ALLELE[2:6]) + RIGHT
    problems = []
    segment = observe_inverted_d(read, D_ALLELE, "D*01", _record("6M1D4M", sequence=(5, 15)), problems)
    assert problems == [] and _key(segment) == (5, 15, 3, 14, "6M1D4M", 0)
    # The walker encoding of the same read cannot record it, so the decoder places it by minimum mismatch.
    walker = []
    placed = observe_inverted_d(read, D_ALLELE, "D*01", _record("10D10M10D", sequence=(5, 15)), walker)
    assert _key(placed) == _key(segment)
    assert walker == ["inverted D: 1 unrecorded internal deletion(s) placed by minimum mismatch"]


def test_assembled_np_claims_are_kept_unless_recalled():
    # The NP base left of the D matches the reverse-complemented allele (forward position 13): the record claims it.
    read = LEFT[:-1] + revcomp(D_ALLELE[13]) + revcomp(D_ALLELE[2:13]) + RIGHT
    record = _record("12M", germline=(2, 14), sequence=(4, 16))
    problems = []
    claimed = observe_inverted_d(read, D_ALLELE, "D*01", record, problems)
    assert problems == [] and _key(claimed) == (4, 16, 2, 14, "12M", 0)
    # A re-called D's claims are scored against the re-called allele, so they are clipped to the generative allele's
    # structural range [trim_5, len - trim_3).
    problems = []
    clipped = observe_inverted_d(read, D_ALLELE, "D*01", record, problems, projected_length=len(D_ALLELE))
    assert problems == [] and _key(clipped) == (5, 16, 3, 14, "11M", 0)


def test_assembled_recall_overhang_becomes_generative_columns():
    # Generative allele 18 nt, trims 1 and 1: structural forward [1, 17). GenAIRR projects it onto a 16-nt re-call,
    # so forward position 16 (the first read base of the inverted D) is past the re-call's end: an insertion.
    allele = "ACGTTGCAAGGCTTACGA"
    read = LEFT + revcomp(allele[1:17]) + RIGHT
    record = _record("1I15M", germline=(1, 16), sequence=(5, 21), trims=(1, 1))
    problems = []
    segment = observe_inverted_d(read, allele, "D*01", record, problems, projected_length=16)
    assert problems == [] and _key(segment) == (5, 21, 1, 17, "16M", 0)
    # Read as it stands, the overhang base would be an NP base.
    unprojected = observe_inverted_d(read, allele, "D*01", record, [])
    assert _key(unprojected) == (6, 21, 2, 17, "15M", 0)


def test_assembled_recall_overhang_that_does_not_fit_is_dropped():
    allele = "ACGTTGCAAGGCTTACGA"
    read = LEFT + "T" + revcomp(allele[1:17]) + RIGHT  # an inserted base besides the one overhang position
    record = _record("2I15M", germline=(1, 16), sequence=(5, 22), trims=(1, 1))
    problems = []
    assert observe_inverted_d(read, allele, "D*01", record, problems, projected_length=16) is None
    assert problems == ["inverted D: re-called by the simulator, and its 2 read base(s) past the end of the 16-nt "
                        "re-called allele do not match the 1 germline position(s) of the generative allele there; "
                        "segment dropped"]


def test_assembled_shape_failing_invariants_falls_back_to_walker():
    # Consumption equals the recorded spans, but the decoded germline runs past the allele: not assembled.
    problems = []
    segment = observe_inverted_d(D_READ, D_ALLELE, "D*01", _record("11M", germline=(-4, 7)), problems)
    assert segment is None and problems and "segment dropped" in problems[0]


def test_forward_recall_overhang_becomes_generative_columns():
    # Generative V 12 nt, no trims; GenAIRR projects it onto an 11-nt re-call, so the last base is an insertion.
    allele = "ACGTACGTACGA"
    read = allele + "TTTT"
    record = {"v_cigar": "11M1I", "v_sequence_start": 0, "v_germline_start": 0, "v_trim_5": 0, "v_trim_3": 0}
    problems = []
    segment = observe_forward(read, allele, "V*01", record, "v", problems, recalled=True, projected_length=11)
    assert problems == [] and _key(segment) == (0, 12, 0, 12, "12M", 0)
    # Without the projected length the overhang base is left out, as an NP base.
    assert _key(observe_forward(read, allele, "V*01", record, "v", [], recalled=True)) == (0, 11, 0, 11, "11M", 0)


def test_forward_recall_overhang_cut_by_end_loss_or_not_fitting():
    allele = "ACGTACGTACGTAC"  # 14 nt, trim_3 0: positions 11, 12, 13 are past an 11-nt re-call
    record = {"j_cigar": "11M2I", "j_sequence_start": 0, "j_germline_start": 0, "j_trim_5": 0, "j_trim_3": 0}
    # The read ends inside the overhang (3' end loss): the two bases are positions 11 and 12.
    problems = []
    segment = observe_forward(allele[:13], allele, "J*01", record, "j", problems, recalled=True, projected_length=11)
    assert problems == [] and _key(segment) == (0, 13, 0, 13, "13M", 0)
    # The read goes on: two bases cannot be the three positions, so the alignment is undetermined.
    problems = []
    assert observe_forward(allele[:13] + "GG", allele, "J*01", record, "j", problems, recalled=True,
                           projected_length=11) is None
    assert problems == ["re-called segment: 2 read base(s) past the end of the 11-nt re-called allele do not match "
                        "the 3 germline position(s) of the generative allele there; segment dropped"]


# Bench_v2 cases simulated by GenAIRR before the assembled encoding, decoded as the walker decoder always decoded
# them: (D allele, read start, read end, germline start, germline end, CIGAR, mismatches) and the warning.
OLD_ENGINE_INVERTED_D = {
    "human_igh_ogrdb_broad:amplicon_janchor_rc:000106": (("IGHD1-26*01", 42, 58, 4, 20, "16M", 0), ()),
    "human_igh_ogrdb_broad:moderate_full:000052": (("IGHD4-11*01", 294, 304, 2, 12, "10M", 3), ()),
    "human_igh_ogrdb_broad:amplicon_janchor_rc:000150": (("IGHD1-14*01", 45, 56, 3, 13, "7M1I3M", 2), ()),
    "human_igh_ogrdb_broad:noisy_ambiguous:000064": (
        ("IGHD2-21*01", 276, 299, 4, 28, "16M1D7M", 3),
        ("d: inverted D: 1 unrecorded internal deletion(s) placed by minimum mismatch",)),
    "human_igh_ogrdb_broad:trimmed:000025": (
        ("IGHD2-8*01", 273, 300, 3, 30, "22M1I1M1D3M", 7),
        ("d: inverted D: 1 unrecorded internal deletion(s) placed by minimum mismatch",)),
    "human_igh_ogrdb_broad:amplicon_fr2:000000": (None, ()),  # the whole D lost to end loss: not observed
}


def test_old_engine_inverted_d_fixtures_decode_as_before(load_fixture, reference):
    cases = load_fixture("old_engine_inverted_d.json")
    assert [case["case_id"] for case in cases] == list(OLD_ENGINE_INVERTED_D)
    for case in cases:
        assert case["record"]["d_inverted"] and "D" in case["record"]["d_cigar"]  # the walker encoding's fills
        truth = truth_from_bench_case(case, reference)
        expected, warnings = OLD_ENGINE_INVERTED_D[case["case_id"]]
        assert truth.warnings == warnings and not truth.unscorable, case["case_id"]
        d = truth.segments.get("d")
        assert (None if d is None else (d.observed.allele, *_key(d.observed))) == expected, case["case_id"]
        if d is not None:
            assert d.inverted
            assert check_invariants(truth.sequence, revcomp(reference.seq(d.observed.allele)), d.observed) == []


@pytest.mark.parametrize("cigar", ["1M", "1D"])
def test_single_column_records_decode_alike_in_both_encodings(cigar):
    # One forward position (5) observed, or none: the walker has no fill to make, so both encodings coincide.
    read = LEFT + revcomp(D_ALLELE[5]) + RIGHT
    record = _record(cigar, germline=(5, 6), sequence=(5, 6 if cigar == "1M" else 5), trims=(5, 10))
    problems = []
    segment = observe_inverted_d(read, D_ALLELE, "D*01", record, problems)
    assert problems == []
    assert (None if segment is None else _key(segment)) == (None if cigar == "1D" else (5, 6, 10, 11, "1M", 0))


# Dev-split S1/S2 cases simulated by GenAIRR 1ededa6 (walker encoding) whose D is re-called. That engine filled a
# re-called D to the re-called allele's end, past the end of a shorter generative allele, and scored NP claims against
# the re-called allele: decoded without the clip, the first is dropped and the second misscores. Later engines stop
# the fill at the generative allele's end (no drop) but still score claims against the re-call.
OLD_ENGINE_RECALLS = {
    "S2:shm_0.15:000000": (("IGHD4-23*01", 296, 309, 3, 16, "13M", 4), "dropped"),
    "S1:fr1_igm:000007": (("IGHD2-2*02", 298, 325, 1, 28, "27M", 0), "misscored"),
}


def test_old_engine_recall_projection_is_undone(load_fixture, reference):
    cases = load_fixture("old_engine_recalls.json")
    assert [case["case_id"] for case in cases] == list(OLD_ENGINE_RECALLS)
    for case in cases:
        record = case["record"]
        generative = record["truth_d_call"]
        assert generative not in record["d_call"].split(",") and not record["d_inverted"]
        truth = truth_from_genairr_record(record, reference, case_id=case["case_id"], stratum=case["stratum"])
        expected, artefact = OLD_ENGINE_RECALLS[case["case_id"]]
        assert truth.warnings == () and not truth.unscorable
        observed = truth.segments["d"].observed
        assert (observed.allele, *_key(observed)) == expected
        problems = []
        unclipped = observe_forward(truth.sequence, reference.seq(generative), generative, record, "d", problems)
        if artefact == "dropped":
            assert unclipped is None and "past the end of IGHD4-23*01" in problems[0]
        else:
            assert unclipped.n_mismatch > observed.n_mismatch


def test_segment_without_alignment_columns_is_unobserved(reference, genairr):
    # The assembled encoding records a segment that end loss removed completely with an empty CIGAR and no germline
    # coordinates: the read shows none of it, so it is unobserved, not unscorable.
    record = genairr.Experiment.on("human_igh").recombine().run_records(n=1, seed=3)[0]
    gone = dict(record, d_cigar="", d_germline_start=None, d_germline_end=None,
                d_sequence_end=record["d_sequence_start"])
    truth = truth_from_genairr_record(gone, reference, case_id="x", stratum="test")
    assert "d" not in truth.segments and truth.unscorable == frozenset() and truth.warnings == ()
    assert set(truth.segments) == {"v", "j"}
