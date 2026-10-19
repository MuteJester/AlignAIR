import json
from pathlib import Path

import pytest

from alignair_assay.reference import Reference
from alignair_assay.seqops import revcomp, translate
from alignair_assay.truth.build import truth_from_bench_case, truth_from_genairr_record
from alignair_assay.truth.io import read_truth_jsonl, write_truth_jsonl
from alignair_assay.truth.segment import check_invariants

ga = pytest.importorskip("GenAIRR")

REF = Reference.from_dataconfig("HUMAN_IGH_OGRDB")
FIXTURES = Path(__file__).parent / "fixtures"


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


def _truths(recs):
    return [truth_from_genairr_record(rec, REF, case_id=f"rec{i}", stratum="test")
            for i, rec in enumerate(recs)]


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def test_seqops():
    assert revcomp("ACGTN") == "NACGT"
    assert translate("TGTTGGTAA") == "CW*"
    assert translate("TGTNNNTG") == "CX"


def test_junction_translated_regardless_of_productivity():
    recs = _exp(mutate=0.05).run_records(n=200, seed=21)
    in_frame = nonproductive_in_frame = 0
    for t in _truths(recs):
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


def test_junction_absent_when_anchor_clipped():
    recs = _exp(e5=(300, 300)).run_records(n=20, seed=4)
    for t in _truths(recs):
        assert t.junction is None
        assert t.productive is None
        assert t.vj_in_frame is None
        assert t.stop_codon is None
        assert t.clone_key is None

    # V still observed but its Cys anchor clipped -> no junction; anchor kept -> junction.
    recs = _exp(e5=(285, 300)).run_records(n=60, seed=4)
    clipped_with_v = 0
    for t in _truths(recs):
        v = t.segments.get("v")
        if v is None:
            continue
        clipped = v.observed.germ_start > REF.anchors[v.observed.allele]
        assert (t.junction is None) == clipped
        assert (t.productive is None) == clipped
        clipped_with_v += clipped
    assert clipped_with_v >= 10


def test_matches_simulator_junction_when_observable():
    recs = _exp(mutate=0.05).run_records(n=200, seed=8)
    checked = 0
    for rec, t in zip(recs, _truths(recs)):
        if not rec["junction"]:
            continue
        assert t.junction is not None
        assert t.junction.nt == rec["junction"].upper()
        assert (t.junction.start, t.junction.end) == (rec["junction_start"], rec["junction_end"])
        checked += 1
    assert checked > 150


def test_productivity_and_clone_key_fields():
    recs = _exp(mutate=0.05).run_records(n=200, seed=9)
    for t in _truths(recs):
        assert t.v_mismatches == t.segments["v"].observed.n_mismatch
        if t.junction is None:
            continue
        assert t.vj_in_frame == (t.junction.length % 3 == 0)
        assert t.productive == (t.vj_in_frame and not t.stop_codon)
        assert t.clone_key == (t.segments["v"].alleles.genes, t.segments["j"].alleles.genes,
                               t.junction.length)


def test_stop_codon_scans_v_frame_beyond_junction():
    # Unmutated read with a stop planted in FR3 (in the V-anchor frame, outside the junction).
    recs = _exp().run_records(n=20, seed=3)
    rec, t = next((r, t) for r, t in zip(recs, _truths(recs)) if t.stop_codon is False)
    seq = rec["sequence"].upper()
    pos = t.junction.start - 30
    mutated = dict(rec, sequence=seq[:pos] + "TAA" + seq[pos + 3:])
    t2 = truth_from_genairr_record(mutated, REF, case_id="x", stratum="test")
    assert t2.junction.nt == t.junction.nt
    assert t2.stop_codon is True
    assert t2.productive is False


def test_rev_comp_record_rejected():
    rec = _exp().run_records(n=1, seed=3)[0]
    with pytest.raises(ValueError):
        truth_from_genairr_record(dict(rec, rev_comp=True), REF, case_id="x", stratum="test")


def test_property_records_build_cleanly():
    recs = _exp(mutate=0.08, indels=(0, 4), e5=(0, 120), e3=(0, 30)).run_records(n=300, seed=3)
    with_junction = 0
    for t in _truths(recs):
        assert t.warnings == ()
        for g, s in t.segments.items():
            assert s.observed_length == s.observed.germ_end - s.observed.germ_start
            assert not s.inverted
        if t.junction is not None:
            with_junction += 1
            assert len(t.junction.nt) == t.junction.end - t.junction.start == t.junction.length
    assert with_junction > 100


@pytest.mark.parametrize("recipe,min_placed", [
    ({}, 0),
    ({"e5": (0, 330), "e3": (0, 40)}, 0),
    # GenAIRR does not record deletions inside an inverted D; the builder places them.
    ({"indels": (2, 8)}, 5),
])
def test_inverted_d_unmutated_zero_mismatch(recipe, min_placed):
    recs = _exp(invert_d=True, **recipe).run_records(n=200, seed=13)
    checked = placed = 0
    for rec, t in zip(recs, _truths(recs)):
        placed += any("placed by minimum mismatch" in w for w in t.warnings)
        assert not any("dropped" in w for w in t.warnings), t.warnings
        assert rec["d_inverted"]
        if "d" not in t.segments:
            continue
        d = t.segments["d"]
        rc_allele = revcomp(REF.seq(d.observed.allele))
        assert d.inverted
        assert d.observed.n_mismatch == 0, (rec["d_cigar"], d.observed)
        assert check_invariants(t.sequence, rc_allele, d.observed) == []
        assert d.observed.observed_germline == rc_allele[d.observed.germ_start:d.observed.germ_end]
        for name in d.alleles.alleles:
            assert d.observed.observed_germline in revcomp(REF.seq(name))
        checked += 1
    assert checked >= 50
    assert placed >= min_placed


def test_bench_case_orientations():
    fwd = truth_from_bench_case(_fixture("amplicon_janchor_000000.json"), REF)
    assert fwd.orientation == 0 and fwd.excluded is None
    assert fwd.case_id == "human_igh_ogrdb_broad:amplicon_janchor:000000"
    assert fwd.stratum == "amplicon_janchor"

    case = _fixture("orientation_000001.json")
    assert case["orientation_id"] == 1
    t = truth_from_bench_case(case, REF)
    assert t.orientation == 1
    assert t.excluded is None
    assert t.sequence == revcomp(case["sequence"]).upper() == case["record"]["sequence"].upper()

    case = _fixture("orientation_000002.json")
    assert case["orientation_id"] == 2
    t = truth_from_bench_case(case, REF)
    assert t.excluded == "non_biological_orientation"
    assert t.orientation == 2
    assert t.sequence == case["record"]["sequence"].upper()

    reverse_only = dict(case, orientation_id=3, sequence=case["record"]["sequence"][::-1])
    assert truth_from_bench_case(reverse_only, REF).excluded == "non_biological_orientation"
    with pytest.raises(ValueError):  # presented sequence inconsistent with the orientation id
        truth_from_bench_case(dict(case, orientation_id=3), REF)


def test_jsonl_roundtrip(tmp_path):
    recs = list(_exp(mutate=0.05, indels=(0, 2), e5=(0, 300)).run_records(n=40, seed=5))
    recs += list(_exp(invert_d=True).run_records(n=10, seed=5))
    truths = _truths(recs)
    assert any(t.junction is None for t in truths) and any(t.junction for t in truths)
    assert any("d" in t.segments and t.segments["d"].inverted for t in truths)
    for name in ("truth.jsonl", "truth.jsonl.gz"):
        path = tmp_path / name
        write_truth_jsonl(truths, path)
        assert read_truth_jsonl(path) == truths
