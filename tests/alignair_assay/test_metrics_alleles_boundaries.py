import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from alignair_assay.ingest.airr import PredSegment, Prediction, load_airr_tsv
from alignair_assay.metrics import INFORMATIONAL, lower_is_better
from alignair_assay.metrics.boundaries import germline_boundary_metrics
from alignair_assay.metrics.case import score_case
from alignair_assay.reference import Reference
from alignair_assay.seqops import revcomp
from alignair_assay.testing import oracle_prediction
from alignair_assay.truth.alleles import AlleleTruth
from alignair_assay.truth.boundaries import BoundaryTruth
from alignair_assay.truth.build import SegmentTruth, truth_from_bench_case, truth_from_genairr_record
from alignair_assay.truth.segment import ObservedSegment

ga = pytest.importorskip("GenAIRR")

REF = Reference.from_dataconfig("HUMAN_IGH_OGRDB")
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ("amplicon_janchor_000000.json", "orientation_000001.json", "orientation_000002.json")

_ONE = ("top1_in_set", "gene_top1", "any_overlap", "_exact", "set_precision", "abstain_ok")
_ZERO = ("_err", ".no_call", ".spurious_call")


def _fixture_truth(name):
    return truth_from_bench_case(json.loads((FIXTURES / name).read_text()), REF)


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


def _genairr_truths():
    recipes = [
        (300, 3, {"mutate": 0.08, "indels": (0, 4), "e5": (0, 120), "e3": (0, 30)}),
        (60, 5, {"invert_d": True, "mutate": 0.05, "e5": (0, 300)}),
        (60, 7, {"e5": (280, 400)}),  # V, and sometimes D, clipped away
    ]
    truths = []
    for n, seed, recipe in recipes:
        recs = _exp(**recipe).run_records(n=n, seed=seed)
        truths += [truth_from_genairr_record(rec, REF, case_id=f"s{seed}-{i}", stratum="test")
                   for i, rec in enumerate(recs)]
    return truths


def _assert_perfect(scores, truth) -> int:
    checked = 0
    for name, value in scores.items():
        if name.endswith(_ONE):
            assert value == 1.0, (truth.case_id, name, value)
        elif name.endswith(_ZERO):
            assert value == 0.0, (truth.case_id, name, value)
        elif name.endswith(".set_size"):
            assert value == len(truth.segments[name.split(".")[0]].alleles.alleles), (truth.case_id, name)
        else:
            continue
        checked += 1
    return checked


def test_oracle_scores_perfect():
    checked = 0
    seen: Counter = Counter()
    inverted = 0
    for truth in _genairr_truths() + [_fixture_truth(n) for n in FIXTURE_NAMES]:
        scores = score_case(truth, oracle_prediction(truth), REF)
        if truth.excluded is not None:
            assert scores == {}
            continue
        checked += _assert_perfect(scores, truth)
        seen.update(scores.keys())
        d = truth.segments.get("d")
        if d is not None and d.inverted:
            inverted += 1
            assert not any(k.startswith("d.germ_") for k in scores)
    assert checked >= 1000
    assert inverted >= 20
    for name in ("v.germ_start_exact", "v.germ_end_err", "d.germ_start_exact", "j.germ_end_exact",
                 "d.seq_start_exact", "d_short.abstain_ok", "d_short.top1_in_set", "v.spurious_call",
                 "d.spurious_call", "v.set_precision"):
        assert seen[name] > 0, name


def _same(a, b):
    return a.keys() == b.keys() and all(a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a)


def test_missing_prediction_counts_as_wrong():
    truth = _fixture_truth("amplicon_janchor_000000.json")
    scores = score_case(truth, None, REF)
    for g in "vdj":
        assert scores[f"{g}.top1_in_set"] == 0.0
        assert scores[f"{g}.gene_top1"] == 0.0
        assert scores[f"{g}.any_overlap"] == 0.0
        assert scores[f"{g}.no_call"] == 1.0
        assert scores[f"{g}.seq_start_exact"] == 0.0
        assert scores[f"{g}.seq_end_exact"] == 0.0
        assert math.isnan(scores[f"{g}.set_precision"])
        assert math.isnan(scores[f"{g}.set_size"])
    errs = [k for k in scores if k.endswith("_err")]
    assert errs and all(math.isnan(scores[k]) for k in errs)
    assert not any(".germ_" in k for k in scores)  # germline frame needs a correct top-1 call
    # A prediction row without segments scores the same as no prediction at all, except that the row exists.
    empty = score_case(truth, Prediction(sequence_id=truth.case_id), REF)
    assert (empty.pop("airr.valid"), scores.pop("airr.valid")) == (1.0, 0.0)
    assert _same(empty, scores)


def test_excluded_truth_scores_nothing():
    truth = _fixture_truth("orientation_000002.json")
    assert truth.excluded is not None
    assert score_case(truth, oracle_prediction(truth), REF) == {}
    assert score_case(truth, None, REF) == {}


def test_igblast_correct_coordinates_on_clipped_read(tmp_path):
    case = json.loads((FIXTURES / "amplicon_janchor_000000.json").read_text())
    truth = truth_from_bench_case(case, REF)
    header = ["sequence_id", "sequence", "rev_comp", "productive", "v_call",
              "v_sequence_start", "v_sequence_end", "v_germline_start", "v_germline_end",
              "v_cigar", "j_cigar", "j_sequence_start", "j_sequence_end",
              "j_germline_start", "j_germline_end"]
    row = [case["case_id"], case["sequence"], "F", "T", "IGHVF1-G2*01,IGHVF1-G2*02", "1", "45", "256", "301",
           "255N26M1D19M77S", "60S1N52M", "61", "112", "2", "53"]
    path = tmp_path / "igblast.tsv"
    path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n")
    preds, _ = load_airr_tsv(path)
    scores = score_case(truth, preds[case["case_id"]], REF)

    assert scores["v.top1_in_set"] == 1.0
    assert scores["v.set_precision"] == 1.0
    assert scores["v.set_size"] == 2.0
    assert scores["v.germ_start_err"] == 0.0
    assert scores["v.germ_start_exact"] == 1.0
    assert scores["v.germ_end_err"] == 0.0  # 301 lies in the equivalence interval (300, 301)
    assert scores["v.seq_start_exact"] == scores["v.seq_end_exact"] == 1.0
    # No J call: read-frame edges still scored, germline frame not.
    assert scores["j.no_call"] == 1.0
    assert scores["j.top1_in_set"] == 0.0
    assert scores["j.seq_start_err"] == 12.0  # predicted 60, truth 72
    assert scores["j.seq_start_exact"] == 0.0
    assert scores["j.seq_end_err"] == 10.0  # predicted 112, truth 122
    assert not any(k.startswith("j.germ_") for k in scores)
    assert scores["d.no_call"] == 1.0


def test_spurious_call_on_unobserved_segment():
    truth = _fixture_truth("amplicon_janchor_000000.json")
    no_d = replace(truth, segments={g: s for g, s in truth.segments.items() if g != "d"})
    scores = score_case(no_d, oracle_prediction(truth), REF)
    assert scores["d.spurious_call"] == 1.0
    assert not any(k.startswith("d.") and k != "d.spurious_call" for k in scores)
    assert score_case(no_d, oracle_prediction(no_d), REF)["d.spurious_call"] == 0.0
    assert score_case(no_d, None, REF)["d.spurious_call"] == 0.0

    # A locus without D genes has no D presence metric at all.
    light = Reference(alleles={g: t for g, t in REF.alleles.items() if g != "d"}, anchors=REF.anchors)
    assert not any(k.startswith("d") for k in score_case(no_d, oracle_prediction(truth), light))


def test_unscorable_gene_emits_nothing():
    truth = _fixture_truth("amplicon_janchor_000000.json")
    oracle = oracle_prediction(truth)
    no_v = replace(truth, segments={g: s for g, s in truth.segments.items() if g != "v"},
                   unscorable=frozenset({"v"}))
    for pred in (oracle, None):
        scores = score_case(no_v, pred, REF)
        assert not any(k.startswith("v.") for k in scores)
        assert any(k.startswith("j.") for k in scores)
    # Unscorable wins even if a segment is (inconsistently) present.
    scores = score_case(replace(truth, unscorable=frozenset({"v"})), oracle, REF)
    assert not any(k.startswith("v.") for k in scores)


def test_short_d_scored_as_abstention():
    truth = _fixture_truth("orientation_000001.json")
    d = truth.segments["d"]
    assert d.observed_length < 8
    oracle = oracle_prediction(truth)

    scores = score_case(truth, oracle, REF)
    assert scores["d_short.top1_in_set"] == 1.0
    assert scores["d_short.abstain_ok"] == 1.0
    assert not any(k.startswith("d.") for k in scores)

    abstained = replace(oracle, segments={**oracle.segments, "d": PredSegment()})
    scores = score_case(truth, abstained, REF)
    assert (scores["d_short.top1_in_set"], scores["d_short.abstain_ok"]) == (0.0, 1.0)

    wrong = next(a for a in sorted(REF.alleles["d"]) if a not in d.alleles.alleles)
    called_wrong = replace(oracle, segments={**oracle.segments, "d": PredSegment(call=wrong, call_set=(wrong,))})
    scores = score_case(truth, called_wrong, REF)
    assert (scores["d_short.top1_in_set"], scores["d_short.abstain_ok"]) == (0.0, 0.0)

    scores = score_case(truth, oracle, REF, d_min_observed=d.observed_length)
    assert scores["d.top1_in_set"] == 1.0
    assert "d_short.abstain_ok" not in scores


def test_wrong_and_partial_calls():
    full = _fixture_truth("amplicon_janchor_000000.json")
    v = full.segments["v"]
    assert v.alleles.alleles == {"IGHVF1-G2*01", "IGHVF1-G2*02"}
    # Narrow the truth to *01 so *02 is a wrong allele of the right gene.
    narrowed = replace(v, alleles=replace(v.alleles, alleles=frozenset({"IGHVF1-G2*01"})))
    truth = replace(full, segments={**full.segments, "v": narrowed})
    oracle = oracle_prediction(full)
    other_gene = next(a for a in sorted(REF.alleles["v"]) if not a.startswith("IGHVF1-G2*"))
    pred = PredSegment(call="IGHVF1-G2*02", call_set=("IGHVF1-G2*02", "IGHVF1-G2*01"),
                       seq_start=0, seq_end=47, germ_start=255, germ_end=300)
    scores = score_case(truth, replace(oracle, segments={**oracle.segments, "v": pred}), REF)
    assert scores["v.top1_in_set"] == 0.0
    assert scores["v.gene_top1"] == 1.0
    assert scores["v.any_overlap"] == 1.0
    assert scores["v.set_precision"] == 0.5
    assert scores["v.no_call"] == 0.0
    assert (scores["v.seq_end_exact"], scores["v.seq_end_err"]) == (0.0, 2.0)  # interval (44, 45)
    assert not any(k.startswith("v.germ_") for k in scores)

    pred = PredSegment(call=other_gene, call_set=(other_gene,))
    scores = score_case(truth, replace(oracle, segments={**oracle.segments, "v": pred}), REF)
    assert (scores["v.gene_top1"], scores["v.any_overlap"], scores["v.set_precision"]) == (0.0, 0.0, 0.0)
    assert math.isnan(scores["v.seq_start_err"])


def _toy_segment(allele_set, *, inverted=False, germ_start=3):
    obs = ObservedSegment("X*01", 0, 6, germ_start, germ_start + 6, "6M", 0, 6, "ACGTAC")
    return SegmentTruth(
        observed=obs,
        boundaries=BoundaryTruth((0, 0), (6, 6), (germ_start, germ_start), (germ_start + 6, germ_start + 6)),
        alleles=AlleleTruth("X*01", frozenset(allele_set), frozenset({"X"}), frozenset(), False),
        observed_length=6, inverted=inverted,
    )


def test_germline_coordinates_shifted_into_truth_frame():
    # X*02 carries the observed germline 3 bases further in than X*01.
    ref = Reference(alleles={"v": {"X*01": "GGGACGTACGG", "X*02": "TTGGGGACGTACGG"}}, anchors={})
    seg = _toy_segment({"X*01", "X*02"})
    sibling = PredSegment(call="X*02", call_set=("X*02",), germ_start=6, germ_end=12)
    assert germline_boundary_metrics("v", seg, sibling, ref) == {
        "v.germ_start_exact": 1.0, "v.germ_start_err": 0.0, "v.germ_end_exact": 1.0, "v.germ_end_err": 0.0}
    unshifted = replace(sibling, germ_start=3, germ_end=None)
    scores = germline_boundary_metrics("v", seg, unshifted, ref)
    assert (scores["v.germ_start_exact"], scores["v.germ_start_err"]) == (0.0, 3.0)
    assert scores["v.germ_end_exact"] == 0.0 and math.isnan(scores["v.germ_end_err"])

    assert germline_boundary_metrics("v", _toy_segment({"X*01"}), sibling, ref) == {}  # call not in set
    assert germline_boundary_metrics("v", _toy_segment({"X*01", "X*02"}, inverted=True), sibling, ref) == {}
    absent = Reference(alleles={"v": {"X*01": "GGGACGTACGG", "X*02": "TTTTTTTTTTTT"}}, anchors={})
    assert germline_boundary_metrics("v", seg, sibling, absent) == {}  # observed germline not in the call


def test_repeated_observed_germline_uses_truth_occurrence():
    # The observed germline occurs twice in each allele; the occurrence at the truth position is the frame.
    ref = Reference(alleles={"v": {"X*01": "ACGTACACGTAC", "X*02": "TACGTACACGTAC"}}, anchors={})
    seg = _toy_segment({"X*01", "X*02"}, germ_start=6)
    primary = PredSegment(call="X*01", call_set=("X*01",), germ_start=6, germ_end=12)
    sibling = PredSegment(call="X*02", call_set=("X*02",), germ_start=7, germ_end=13)
    for pred in (primary, sibling):
        scores = germline_boundary_metrics("v", seg, pred, ref)
        assert scores["v.germ_start_exact"] == scores["v.germ_end_exact"] == 1.0, pred.call


def test_oracle_prediction_fields():
    truth = _fixture_truth("orientation_000001.json")
    pred = oracle_prediction(truth)
    assert pred.sequence_id == truth.case_id
    assert pred.rev_comp is True
    assert pred.sequence == revcomp(truth.sequence)
    assert pred.invalid == ()
    d = truth.segments["d"]
    assert pred.segments["d"].call == d.alleles.primary
    assert pred.segments["d"].call_set[0] == d.alleles.primary
    assert list(pred.segments["d"].call_set[1:]) == sorted(d.alleles.alleles - {d.alleles.primary})
    assert pred.segments["v"].cigar == truth.segments["v"].observed.cigar
    assert (pred.junction_nt, pred.junction_start, pred.junction_end) == (
        truth.junction.nt, truth.junction.start, truth.junction.end)
    assert pred.productive is truth.productive

    fwd = _fixture_truth("amplicon_janchor_000000.json")
    assert oracle_prediction(fwd).rev_comp is False
    assert oracle_prediction(fwd).sequence == fwd.sequence


def test_metric_direction():
    for name in ("v.seq_start_err", "j.germ_end_err", "d.no_call", "v.spurious_call", "v.germline_errors",
                 "v.mutation_count_err"):
        assert lower_is_better(name), name
    for name in ("v.top1_in_set", "v.seq_start_exact", "d_short.abstain_ok", "v.set_precision"):
        assert not lower_is_better(name), name
    assert INFORMATIONAL == frozenset({".set_size", "junction.start_err", "junction.end_err",
                                       "v_indel.mutation_count_err", "v_indel.germline_errors"})
