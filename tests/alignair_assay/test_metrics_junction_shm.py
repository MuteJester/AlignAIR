import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from alignair_assay.cigar import parse_cigar, walk
from alignair_assay.ingest.airr import Prediction
from alignair_assay.metrics import INFORMATIONAL, lower_is_better
from alignair_assay.metrics.case import score_case
from alignair_assay.metrics.junction import junction_metrics
from alignair_assay.reference import Reference, gene_name
from alignair_assay.testing import oracle_prediction
from alignair_assay.truth.build import JunctionTruth, truth_from_bench_case, truth_from_genairr_record

ga = pytest.importorskip("GenAIRR")

REF = Reference.from_dataconfig("HUMAN_IGH_OGRDB")
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ("amplicon_janchor_000000.json", "orientation_000001.json", "orientation_000002.json")

_ACCURACY = ("junction.nt_exact", "junction.aa_exact", "junction.length_exact", "clone_key.exact",
             "productive.correct", "vj_in_frame.correct", "stop_codon.correct", "orientation.correct",
             "airr.valid", "v.indel_exact", "j.indel_exact")
_ERRORS = ("junction.start_err", "junction.end_err", "junction.start_abs_err", "junction.end_abs_err",
           "v.mutation_count_err", "v.germline_errors")
_FAMILIES = {
    "junction": ("junction.nt_exact", "junction.aa_exact", "junction.length_exact", "junction.start_err",
                 "junction.end_err", "junction.start_abs_err", "junction.end_abs_err"),
    "clone_key": ("clone_key.exact",),
    "mutations": ("v.mutation_count_err",),
    "germline": ("v.germline_errors",),
    "indel": ("v.indel_exact", "j.indel_exact"),
    "productivity": ("productive.correct", "vj_in_frame.correct", "stop_codon.correct"),
    "orientation": ("orientation.correct",),
    "validity": ("airr.valid",),
}


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
    # The same record mix as the allele/boundary oracle test (test_metrics_alleles_boundaries.py).
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


def _unmutated_records(n=50, seed=11):
    return _exp().run_records(n=n, seed=seed)


def _m_columns(seg):
    o = seg.observed
    return [c for c in walk(parse_cigar(o.cigar), o.seq_start, o.germ_start)
            if c.read_pos is not None and c.germ_pos is not None]


def _with_segment(pred, g, **changes):
    return replace(pred, segments={**pred.segments, g: replace(pred.segments[g], **changes)})


def test_oracle_perfect_extended():
    seen: Counter = Counter()
    located = 0
    for truth in _genairr_truths() + [_fixture_truth(n) for n in FIXTURE_NAMES]:
        oracle = oracle_prediction(truth)
        scores = score_case(truth, oracle, REF)
        if truth.excluded is not None:
            assert scores == {}
            continue
        for name in _ACCURACY:
            if name in scores:
                assert scores[name] == 1.0, (truth.case_id, name, scores[name])
                seen[name] += 1
        for name in _ERRORS:
            if name in scores:
                assert scores[name] == 0.0, (truth.case_id, name, scores[name])
                seen[name] += 1
        assert ("junction.nt_exact" in scores) == (truth.junction is not None)
        assert ("clone_key.exact" in scores) == (truth.clone_key is not None)
        assert ("v.mutation_count_err" in scores) == ("v" in truth.segments)

        # A tool that reports only the junction string: aa from translation, coordinates by location.
        if truth.junction is not None:
            bare = replace(oracle, junction_aa=None, junction_start=None, junction_end=None)
            bare_scores = score_case(truth, bare, REF)
            assert bare_scores.get("junction.aa_exact", 1.0) == 1.0
            if truth.sequence.count(truth.junction.nt) == 1:
                assert bare_scores["junction.start_err"] == bare_scores["junction.end_err"] == 0.0
                located += 1

    families = {f: sum(seen[n] for n in names) for f, names in _FAMILIES.items()}
    for family, count in families.items():
        assert count >= 50, (family, count)
    assert seen["junction.nt_exact"] >= 200
    assert seen["clone_key.exact"] >= 200
    assert located >= 200


def test_junction_length_error_breaks_clone_key():
    for name in ("amplicon_janchor_000000.json", "orientation_000001.json"):
        truth = _fixture_truth(name)
        j = truth.junction
        oracle = oracle_prediction(truth)
        short = replace(oracle, junction_nt=j.nt[:-1], junction_aa=None, junction_end=j.end - 1)
        scores = score_case(truth, short, REF)
        assert scores["junction.length_exact"] == 0.0
        assert scores["clone_key.exact"] == 0.0
        assert scores["junction.end_err"] == -1.0
        assert scores["junction.end_abs_err"] == 1.0
        assert scores["junction.start_err"] == 0.0
        assert scores["junction.nt_exact"] == 0.0

        # Without coordinates the (unique) shortened string is located in the read.
        assert truth.sequence.count(j.nt[:-1]) == 1
        located = score_case(truth, replace(short, junction_start=None, junction_end=None), REF)
        assert (located["junction.start_err"], located["junction.end_err"]) == (0.0, -1.0)


def test_clone_key_uses_genes_of_the_call_set():
    truth = _fixture_truth("orientation_000001.json")
    oracle = oracle_prediction(truth)
    v_genes = truth.clone_key[0]
    same_gene = next(a for a in sorted(REF.alleles["v"])
                     if gene_name(a) in v_genes and a not in truth.segments["v"].alleles.alleles)
    other_gene = next(a for a in sorted(REF.alleles["v"]) if gene_name(a) not in v_genes)

    assert score_case(truth, _with_segment(oracle, "v", call=same_gene, call_set=(same_gene,)),
                      REF)["clone_key.exact"] == 1.0
    wrong = _with_segment(oracle, "v", call=other_gene, call_set=(other_gene,))
    assert score_case(truth, wrong, REF)["clone_key.exact"] == 0.0
    # A right gene anywhere in the predicted set is enough.
    hedged = _with_segment(oracle, "v", call=other_gene, call_set=(other_gene, same_gene))
    assert score_case(truth, hedged, REF)["clone_key.exact"] == 1.0
    no_j = replace(oracle, segments={g: s for g, s in oracle.segments.items() if g != "j"})
    assert score_case(truth, no_j, REF)["clone_key.exact"] == 0.0
    assert score_case(truth, replace(oracle, junction_nt=None), REF)["clone_key.exact"] == 0.0


def _sibling_case(min_k=2):
    """An unmutated read and a same-length sibling of its V allele differing at >= min_k observed positions."""
    for i, rec in enumerate(_unmutated_records()):
        truth = truth_from_genairr_record(rec, REF, case_id=f"u{i}", stratum="test")
        v = truth.segments.get("v")
        if v is None:
            continue
        assert truth.v_mismatches == 0
        primary = REF.seq(v.observed.allele)
        cols = _m_columns(v)
        for name in sorted(REF.alleles["v"]):
            seq = REF.seq(name)
            if gene_name(name) != gene_name(v.observed.allele) or name == v.observed.allele \
                    or len(seq) != len(primary):
                continue
            k = sum(seq[c.germ_pos] != primary[c.germ_pos] for c in cols)
            if k >= min_k:
                return truth, name, k
    raise AssertionError("no sibling found")


def test_wrong_sibling_allele_inflates_mutations():
    truth, sibling, k = _sibling_case()
    assert sibling not in truth.segments["v"].alleles.alleles
    oracle = oracle_prediction(truth)
    scores = score_case(truth, _with_segment(oracle, "v", call=sibling, call_set=(sibling,)), REF)
    assert scores["v.mutation_count_err"] == k
    assert scores["v.germline_errors"] == k
    assert scores["v.top1_in_set"] == 0.0


def test_germline_errors_count_unaligned_and_shifted_positions():
    truth, _, _ = _sibling_case()
    o = truth.segments["v"].observed
    n = o.seq_end - o.seq_start
    assert o.cigar == f"{n}M"
    oracle = oracle_prediction(truth)

    # Two read positions left unaligned at the 5' end: two germline errors, no mismatch difference.
    trimmed = _with_segment(oracle, "v", seq_start=o.seq_start + 2, germ_start=o.germ_start + 2,
                            cigar=f"{n - 2}M")
    scores = score_case(truth, trimmed, REF)
    assert (scores["v.germline_errors"], scores["v.mutation_count_err"]) == (2.0, 0.0)

    # The same alignment with its germline frame shifted back one base: the two unaligned positions plus
    # every column whose germline base differs from its neighbour's.
    primary = REF.seq(o.allele)
    shifted = _with_segment(trimmed, "v", germ_start=o.germ_start + 1)
    expected = 2 + sum(primary[g - 1] != primary[g] for g in range(o.germ_start + 2, o.germ_end))
    scores = score_case(truth, shifted, REF)
    assert scores["v.germline_errors"] == expected > 2

    # No usable alignment: no call, an unknown call, no CIGAR or a walk out of range.
    for broken in (_with_segment(oracle, "v", call=None, call_set=()),
                   _with_segment(oracle, "v", call="IGHVX*99", call_set=("IGHVX*99",)),
                   _with_segment(oracle, "v", cigar=None),
                   _with_segment(oracle, "v", seq_start=None),
                   _with_segment(oracle, "v", germ_start=len(primary) - 5)):
        scores = score_case(truth, broken, REF)
        assert math.isnan(scores["v.mutation_count_err"]) and math.isnan(scores["v.germline_errors"])


def test_n_bases_excluded():
    rec = dict(_unmutated_records()[0])
    truth = truth_from_genairr_record(rec, REF, case_id="clean", stratum="test")
    assert truth.v_mismatches == 0  # so an N counted as a mismatch would show up as an error of 1
    cols = _m_columns(truth.segments["v"])
    r = cols[len(cols) // 2].read_pos
    assert truth.sequence[r] != "N"
    rec["sequence"] = rec["sequence"][:r] + "N" + rec["sequence"][r + 1:]
    with_n = truth_from_genairr_record(rec, REF, case_id="with-n", stratum="test")
    assert with_n.sequence[r] == "N"
    assert with_n.v_mismatches == 0
    scores = score_case(with_n, oracle_prediction(with_n), REF)
    assert scores["v.mutation_count_err"] == 0.0
    assert scores["v.germline_errors"] == 0.0


def test_indel_exact():
    truth = _fixture_truth("amplicon_janchor_000000.json")
    assert truth.segments["v"].observed.cigar == "26M1D18M"
    assert truth.segments["j"].observed.cigar == "50M"
    oracle = oracle_prediction(truth)
    scores = score_case(truth, oracle, REF)
    assert scores["v.indel_exact"] == 1.0
    assert "j.indel_exact" not in scores  # no indel in the truth or the prediction

    v = oracle.segments["v"]
    no_gap = _with_segment(oracle, "v", cigar="44M", germ_end=v.germ_end - 1)
    assert score_case(truth, no_gap, REF)["v.indel_exact"] == 0.0
    assert score_case(truth, _with_segment(oracle, "v", cigar=None), REF)["v.indel_exact"] == 0.0
    spurious = _with_segment(oracle, "j", cigar="25M1I24M")
    assert score_case(truth, spurious, REF)["j.indel_exact"] == 0.0


def test_flags_orientation_and_validity():
    truth = _fixture_truth("orientation_000001.json")
    assert truth.orientation == 1 and truth.productive is False
    oracle = oracle_prediction(truth)
    scores = score_case(truth, replace(oracle, productive=True, rev_comp=False,
                                       invalid=("v_cigar: malformed CIGAR 'x'",)), REF)
    assert scores["productive.correct"] == 0.0
    assert scores["vj_in_frame.correct"] == scores["stop_codon.correct"] == 1.0
    assert scores["orientation.correct"] == 0.0
    assert scores["airr.valid"] == 0.0

    scores = score_case(truth, replace(oracle, vj_in_frame=None, rev_comp=None), REF)
    assert scores["vj_in_frame.correct"] == 0.0
    assert scores["orientation.correct"] == 0.0

    no_flags = replace(truth, productive=None, vj_in_frame=None, stop_codon=None)
    scores = score_case(no_flags, oracle, REF)
    assert not any(k.endswith(".correct") and k != "orientation.correct" for k in scores)


def test_missing_prediction_new_metrics():
    truth = _fixture_truth("amplicon_janchor_000000.json")
    for pred, valid in ((None, 0.0), (Prediction(sequence_id=truth.case_id), 1.0)):
        scores = score_case(truth, pred, REF)
        assert scores["airr.valid"] == valid
        for name in ("junction.nt_exact", "junction.length_exact", "clone_key.exact", "productive.correct",
                     "vj_in_frame.correct", "stop_codon.correct", "orientation.correct", "v.indel_exact"):
            assert scores[name] == 0.0, name
        assert "junction.aa_exact" not in scores  # truth junction is out of frame
        for name in _ERRORS:
            assert math.isnan(scores[name]), name


def _toy_junction():
    sequence = "GGTGTGCAAATGGCC"
    return JunctionTruth(start=3, end=12, nt="GTGCAAATG", aa="VQM", length=9), sequence


def test_junction_aa_and_location_rules():
    truth, sequence = _toy_junction()
    pred = Prediction(sequence_id="x", junction_nt="gtgcaaatg")
    scores = junction_metrics(truth, sequence, pred)
    assert scores["junction.nt_exact"] == scores["junction.aa_exact"] == 1.0  # case-insensitive; aa translated
    assert (scores["junction.start_err"], scores["junction.end_err"]) == (0.0, 0.0)

    assert junction_metrics(truth, sequence, replace(pred, junction_aa="VQL"))["junction.aa_exact"] == 0.0
    # A string that is not a whole number of codons has no translation.
    scores = junction_metrics(truth, sequence, Prediction(sequence_id="x", junction_nt="GTGCAAAT"))
    assert scores["junction.aa_exact"] == 0.0 and scores["junction.end_err"] == -1.0
    # A string occurring twice cannot be placed; reported coordinates win over location.
    repeated = Prediction(sequence_id="x", junction_nt="TG")
    scores = junction_metrics(truth, sequence, repeated)
    assert math.isnan(scores["junction.start_err"]) and math.isnan(scores["junction.end_abs_err"])
    scores = junction_metrics(truth, sequence, replace(repeated, junction_start=1, junction_end=13))
    assert (scores["junction.start_err"], scores["junction.start_abs_err"]) == (-2.0, 2.0)
    assert (scores["junction.end_err"], scores["junction.end_abs_err"]) == (1.0, 1.0)
    # No truth amino acid (out of frame) -> no aa metric.
    out_of_frame = replace(truth, aa=None)
    assert "junction.aa_exact" not in junction_metrics(out_of_frame, sequence, pred)


def test_metric_direction_extended():
    for name in ("junction.start_abs_err", "junction.end_abs_err", "v.mutation_count_err", "v.germline_errors"):
        assert lower_is_better(name), name
    for name in _ACCURACY:
        assert not lower_is_better(name), name
    # Signed edge errors are reported but compared through their absolute twins.
    informational = [n for n in ("junction.start_err", "junction.end_err", "junction.start_abs_err")
                     if n.endswith(tuple(INFORMATIONAL))]
    assert informational == ["junction.start_err", "junction.end_err"]
