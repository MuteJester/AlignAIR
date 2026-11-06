import dataclasses

import pytest

from alignair_assay.core.cigar import parse_cigar
from alignair_assay.core.reference import Reference
from alignair_assay.core.seqops import revcomp, translate
from alignair_assay.truth.bench import truth_from_bench_case
from alignair_assay.truth.build import truth_from_genairr_record
from alignair_assay.truth.io import read_truth_jsonl, write_truth_jsonl
from alignair_assay.truth.segment import check_invariants

pytestmark = pytest.mark.component

ga = pytest.importorskip("GenAIRR")


def _exp(mutate=None, indels=None, e5=None, e3=None, invert_d=False):
    exp = ga.Experiment.on("human_igh").recombine()
    if invert_d:
        exp = exp.invert_d(prob=1.0)
    if mutate is not None:
        exp = exp.mutate(rate=mutate)
    if indels is not None:
        exp = exp.polymerase_indels(count=indels)
    if e5 is not None:
        exp = exp.end_loss_5prime(length=e5)
    if e3 is not None:
        exp = exp.end_loss_3prime(length=e3)
    return exp


def _truths(recs, reference):
    return [truth_from_genairr_record(rec, reference, case_id=f"rec{i}", stratum="test")
            for i, rec in enumerate(recs)]


def test_seqops():
    assert revcomp("ACGTN") == "NACGT"
    assert translate("TGTTGGTAA") == "CW*"
    assert translate("TGTNNNTG") == "CX"


def test_translate_iupac():
    # An ambiguous codon translates when every expansion gives the same amino acid.
    assert translate("CGN") == "R"
    assert translate("GCN") == "A"
    assert translate("TAR") == "*"
    assert translate("TTN") == "X"  # TTT/TTC = F, TTA/TTG = L
    assert translate("NNN") == "X"
    assert translate("cgnTT-") == "RX"


def test_junction_translated_regardless_of_productivity(reference):
    recs = _exp(mutate=0.05).run_records(n=200, seed=21)
    in_frame = nonproductive_in_frame = 0
    for t in _truths(recs, reference):
        if t.junction is None:
            continue
        if t.junction.length % 3 == 0:
            in_frame += 1
            assert t.junction.aa == translate(t.junction.nt)
            nonproductive_in_frame += t.productive is False
        else:
            assert t.junction.aa is None
            assert t.productive is False
    assert in_frame > 50
    assert nonproductive_in_frame > 0


def test_junction_absent_when_anchor_clipped(reference):
    recs = _exp(e5=(300, 300)).run_records(n=20, seed=4)
    for t in _truths(recs, reference):
        assert t.junction is None
        assert t.productive is None
        assert t.vj_in_frame is None
        assert t.stop_codon is None
        assert t.clone_key is None

    # V still observed but its Cys anchor clipped -> no junction; anchor kept -> junction.
    recs = _exp(e5=(285, 300)).run_records(n=60, seed=4)
    clipped_with_v = 0
    for t in _truths(recs, reference):
        v = t.segments.get("v")
        if v is None:
            continue
        clipped = v.observed.germ_start > reference.anchors[v.observed.allele]
        assert (t.junction is None) == clipped
        assert (t.productive is None) == clipped
        clipped_with_v += clipped
    assert clipped_with_v >= 10


def test_matches_simulator_junction_when_observable(reference):
    recs = _exp(mutate=0.05).run_records(n=200, seed=8)
    checked = 0
    for rec, t in zip(recs, _truths(recs, reference)):
        if not rec["junction"]:
            continue
        assert t.junction is not None
        assert t.junction.nt == rec["junction"].upper()
        assert (t.junction.start, t.junction.end) == (rec["junction_start"], rec["junction_end"])
        checked += 1
    assert checked > 150


def test_productivity_and_clone_key_fields(reference):
    recs = _exp(mutate=0.05).run_records(n=200, seed=9)
    for t in _truths(recs, reference):
        assert t.v_mismatches == t.segments["v"].observed.n_mismatch
        if t.junction is None:
            continue
        assert t.vj_in_frame == (t.junction.length % 3 == 0)
        assert t.v_frameshift is False and t.j_frameshift is False  # no indels in this mix
        assert t.productive == (t.vj_in_frame and not t.stop_codon)
        assert t.clone_key == (t.segments["v"].alleles.genes, t.segments["j"].alleles.genes,
                               t.junction.length)


def _net_indel(seg):
    ops = parse_cigar(seg.observed.cigar)
    return sum(n for n, op in ops if op == "I") - sum(n for n, op in ops if op == "D")


def test_frameshift_makes_read_unproductive(reference):
    recs = _exp(mutate=0.05, indels=(1, 4)).run_records(n=300, seed=19)
    shifted_in_frame = 0
    for t in _truths(recs, reference):
        for g in "vj":
            want = None if g not in t.segments else _net_indel(t.segments[g]) % 3 != 0
            assert getattr(t, f"{g}_frameshift") == want, (t.case_id, g)
        if t.junction is None:
            assert t.productive is None
            continue
        frameshift = t.v_frameshift or t.j_frameshift
        assert t.productive == (t.vj_in_frame and not t.stop_codon and not frameshift)
        assert t.frame_unambiguous == (t.vj_in_frame and not frameshift)
        shifted_in_frame += frameshift and t.vj_in_frame
    # Reads whose junction is in frame but whose V or J alignment is shifted: unproductive, frame ambiguous.
    assert shifted_in_frame >= 5


def test_stop_codon_scans_v_frame_beyond_junction(reference):
    # Unmutated read with a stop planted in FR3 (in the V-anchor frame, outside the junction).
    recs = _exp().run_records(n=20, seed=3)
    rec, t = next((r, t) for r, t in zip(recs, _truths(recs, reference)) if t.stop_codon is False)
    seq = rec["sequence"].upper()
    pos = t.junction.start - 30
    mutated = dict(rec, sequence=seq[:pos] + "TAA" + seq[pos + 3:])
    t2 = truth_from_genairr_record(mutated, reference, case_id="x", stratum="test")
    assert t2.junction.nt == t.junction.nt
    assert t2.stop_codon is True
    assert t2.productive is False


def test_simulator_alignment_past_allele_end_is_unscorable():
    # A GenAIRR defect seen after an insertion near the V end (e.g. 105D148M1I43M ending at germline 296 on the
    # 295-nt IGHVF6-G21*14): the alignment claims a germline base past the allele end, attributing an NP1 base to
    # V. Reproduced on a 10-nt toy V whose 9M1I2M alignment ends at germline 11.
    ref = Reference(alleles={"v": {"X*01": "ACGTACGTAC"}, "j": {"J*01": "TTTT"}}, anchors={})
    rec = {"sequence": "ACGTACGTAGCG" + "TTTT", "rev_comp": False,
           "v_call": "X*01", "v_sequence_start": 0, "v_germline_start": 0, "v_cigar": "9M1I2M",
           "d_call": "",
           "j_call": "J*01", "j_sequence_start": 12, "j_germline_start": 0, "j_cigar": "4M"}
    t = truth_from_genairr_record(rec, ref, case_id="x", stratum="test")
    assert "v" not in t.segments
    assert t.unscorable == frozenset({"v"})
    assert t.warnings == ("v: simulator alignment 9M1I2M from germline 0 ends at 11, past the end of X*01 (10 nt); "
                          "segment dropped",)
    assert set(t.segments) == {"j"} and t.segments["j"].observed.n_mismatch == 0


@pytest.mark.parametrize("shift_germline,shift_read,message", [
    (1000, 0, "past the end of"),
    (0, 1000, "past the read end"),
])
def test_inconsistent_simulator_alignment_dropped(shift_germline, shift_read, message, reference):
    rec = _exp().run_records(n=1, seed=3)[0]
    bad = dict(rec, v_germline_start=rec["v_germline_start"] + shift_germline,
               v_sequence_start=rec["v_sequence_start"] + shift_read)
    t = truth_from_genairr_record(bad, reference, case_id="x", stratum="test")
    assert "v" not in t.segments
    assert t.unscorable == frozenset({"v"})
    assert any(w.startswith("v: simulator alignment") and message in w for w in t.warnings), t.warnings
    assert truth_from_genairr_record(rec, reference, case_id="x", stratum="test").unscorable == frozenset()


def test_unknown_primary_allele_is_unscorable(reference):
    rec = _exp().run_records(n=1, seed=3)[0]
    t = truth_from_genairr_record(dict(rec, j_call="IGHJ99*01"), reference, case_id="x", stratum="test")
    assert "j" not in t.segments
    assert t.unscorable == frozenset({"j"})


def test_clipped_segment_is_unobserved_not_unscorable(reference):
    for t in _truths(_exp(e5=(330, 330)).run_records(n=5, seed=7), reference):
        assert t.unscorable == frozenset()
    assert any("v" not in t.segments for t in _truths(_exp(e5=(330, 330)).run_records(n=5, seed=7), reference))


def test_rev_comp_record_rejected(reference):
    rec = _exp().run_records(n=1, seed=3)[0]
    with pytest.raises(ValueError):
        truth_from_genairr_record(dict(rec, rev_comp=True), reference, case_id="x", stratum="test")


def test_property_records_build_cleanly(reference):
    recs = _exp(mutate=0.08, indels=(0, 4), e5=(0, 120), e3=(0, 30)).run_records(n=300, seed=3)
    with_junction = 0
    for t in _truths(recs, reference):
        assert t.warnings == ()
        for g, s in t.segments.items():
            assert s.observed_length == s.observed.germ_end - s.observed.germ_start
            assert not s.inverted
        if t.junction is not None:
            with_junction += 1
            assert len(t.junction.nt) == t.junction.end - t.junction.start == t.junction.length
    assert with_junction > 100


@pytest.mark.parametrize("recipe,min_internal_deletions", [
    ({}, 0),
    ({"e5": (0, 330), "e3": (0, 40)}, 0),
    # Deletions inside an inverted D: GenAIRR 2.3.1 does not record them and the builder places them by minimum
    # mismatch (pinned by test_record_encodings' old-engine fixtures); later engines record them.
    ({"indels": (2, 8)}, 5),
])
def test_inverted_d_unmutated_zero_mismatch(recipe, min_internal_deletions, reference):
    recs = _exp(invert_d=True, **recipe).run_records(n=200, seed=13)
    checked = internal_deletions = 0
    for rec, t in zip(recs, _truths(recs, reference)):
        assert not any("dropped" in w for w in t.warnings), t.warnings
        assert rec["d_inverted"]
        if "d" not in t.segments:
            continue
        d = t.segments["d"]
        rc_allele = revcomp(reference.seq(d.observed.allele))
        assert d.inverted
        assert d.observed.n_mismatch == 0, (rec["d_cigar"], d.observed)
        assert check_invariants(t.sequence, rc_allele, d.observed) == []
        assert d.observed.observed_germline == rc_allele[d.observed.germ_start:d.observed.germ_end]
        for name in d.alleles.alleles:
            assert d.observed.observed_germline in revcomp(reference.seq(name))
        internal_deletions += "D" in d.observed.cigar
        checked += 1
    assert checked >= 50
    assert internal_deletions >= min_internal_deletions


def test_bench_case_orientations(reference, load_fixture):
    fwd = truth_from_bench_case(load_fixture("amplicon_janchor_000000.json"), reference)
    assert fwd.orientation == 0 and fwd.excluded is None
    assert fwd.case_id == "human_igh_ogrdb_broad:amplicon_janchor:000000"
    assert fwd.stratum == "amplicon_janchor"

    case = load_fixture("orientation_000001.json")
    assert case["orientation_id"] == 1
    t = truth_from_bench_case(case, reference)
    assert t.orientation == 1
    assert t.excluded is None
    assert t.sequence == revcomp(case["sequence"]).upper() == case["record"]["sequence"].upper()

    case = load_fixture("orientation_000002.json")
    assert case["orientation_id"] == 2
    t = truth_from_bench_case(case, reference)
    assert t.excluded == "non_biological_orientation"
    assert t.orientation == 2
    assert t.sequence == case["record"]["sequence"].upper()

    reverse_only = dict(case, orientation_id=3, sequence=case["record"]["sequence"][::-1])
    assert truth_from_bench_case(reverse_only, reference).excluded == "non_biological_orientation"
    with pytest.raises(ValueError):  # presented sequence inconsistent with the orientation id
        truth_from_bench_case(dict(case, orientation_id=3), reference)


def test_jsonl_roundtrip(tmp_path, reference):
    recs = list(_exp(mutate=0.05, indels=(0, 2), e5=(0, 300)).run_records(n=40, seed=5))
    recs += list(_exp(invert_d=True).run_records(n=10, seed=5))
    recs += list(_exp(e5=(295, 300)).run_records(n=5, seed=5))   # the 5' loss takes the V Cys: no junction
    truths = _truths(recs, reference)
    truths.append(dataclasses.replace(truths[0], unscorable=frozenset({"v", "d"})))
    assert any(t.junction is None for t in truths) and any(t.junction for t in truths)
    assert any("d" in t.segments and t.segments["d"].inverted for t in truths)
    for name in ("truth.jsonl", "truth.jsonl.gz"):
        path = tmp_path / name
        write_truth_jsonl(truths, path)
        assert read_truth_jsonl(path) == truths
