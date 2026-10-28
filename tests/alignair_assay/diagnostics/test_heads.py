"""Head-level metrics: AlignAIR's raw head outputs, as the replay captures them, scored against observable truth.

Most tests build ``StageOutputs`` and truths by hand with a 60-position logits window; the oracle-shaped tests also
run on the truths of a built S0, and the replay tests check the decoders and the allele ranking against what the
product pipeline did with the same outputs. The test against the released checkpoint runs only when
``ALIGNAIR_ASSAY_MODEL`` names a ``.alignair`` model file.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("GenAIRR")

from alignair.calibration import ece  # noqa: E402
from alignair.predict.state import Predictions  # noqa: E402

from alignair_assay.core.cases import read_cases, read_fasta  # noqa: E402
from alignair_assay.core.cigar import parse_cigar  # noqa: E402
from alignair_assay.diagnostics.heads import (  # noqa: E402
    METRIC_NAMES,
    allele_ranking,
    calibration_table,
    decode_boundaries,
    head_scores,
)
from alignair_assay.diagnostics.replay import StageOutputs, load_model_checked, replay  # noqa: E402
from alignair_assay.metrics import is_informational, lower_is_better  # noqa: E402
from alignair_assay.metrics.case import MIN_OBSERVED  # noqa: E402
from alignair_assay.truth.alleles import AlleleTruth  # noqa: E402
from alignair_assay.truth.boundaries import BoundaryTruth  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.truth.model import ObservableTruth, SegmentTruth  # noqa: E402
from alignair_assay.truth.segment import ObservedSegment  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.genairr_build]  # these tests build GenAIRR suites
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
WIDTH = 60
NAMES = {"v": ["V1*01", "V1*02", "V2*01", "V3*01", "V4*01", "V5*01", "V6*01"],
         "d": ["D1*01", "D2*01", "D3*01"],
         "j": ["J1*01", "J2*01", "J3*01"]}
WITH_SHORT_D = {**NAMES, "d": [*NAMES["d"], "Short-D"]}
GENES = ("v", "d", "j")
DECODERS = ("soft", "argmax")


# --- hand-built truth and outputs -------------------------------------------------------------------------------

def _seg(seq_start, seq_end, primary, *, start=None, end=None, scored=(), generative=(), cigar=None, n_mismatch=0,
         inverted=False) -> SegmentTruth:
    """A segment observed at read [seq_start, seq_end) with boundary intervals ``start``/``end`` (inclusive; the
    observed edge alone by default) and CIGAR ``cigar`` (all M by default)."""
    cigar = cigar or f"{seq_end - seq_start}M"
    ops = parse_cigar(cigar)
    germ = sum(n for n, op in ops if op in "MD")
    observed = ObservedSegment(allele=primary, seq_start=seq_start, seq_end=seq_end, germ_start=0, germ_end=germ,
                               cigar=cigar, n_mismatch=n_mismatch, n_aligned=sum(n for n, op in ops if op == "M"),
                               observed_germline="A" * germ)
    boundaries = BoundaryTruth(seq_start=start or (seq_start, seq_start), seq_end=end or (seq_end, seq_end),
                               germ_start=(0, 0), germ_end=(germ, germ))
    alleles = AlleleTruth(primary=primary, alleles=frozenset({primary, *scored, *generative}), genes=frozenset(),
                          simulator_ties=frozenset(), disagrees_with_simulator=False,
                          generative=frozenset({primary, *generative}))
    return SegmentTruth(observed=observed, boundaries=boundaries, alleles=alleles, observed_length=germ,
                        inverted=inverted)


def _truth(case_id, length, segments, *, orientation=0, productive=None, v_mismatches=None, unscorable=(),
           excluded=None) -> ObservableTruth:
    if v_mismatches is None and "v" in segments:
        v_mismatches = segments["v"].observed.n_mismatch
    return ObservableTruth(case_id=case_id, stratum="s", sequence="".join("ACGT"[k % 4] for k in range(length)),
                           orientation=orientation, segments=segments, unscorable=frozenset(unscorable),
                           junction=None, vj_in_frame=None, stop_codon=None, productive=productive, clone_key=None,
                           v_mismatches=v_mismatches, v_frameshift=None, j_frameshift=None, excluded=excluded,
                           warnings=())


def _window(truth, width):
    """The canonical read bases the model sees: AlignAIR keeps the presented read's first ``width`` bases."""
    n = len(truth.sequence)
    lo = n - width if n > width and truth.orientation == 1 else 0
    return lo, truth.sequence[lo:lo + width]


def _blank(truths, *, width=WIDTH, names=NAMES, allowed=None) -> StageOutputs:
    """Outputs of zeros for ``truths`` (in order), as the replay would capture them for a model with a ``width``
    window and ``names`` alleles; ``predictions.allele`` holds the same arrays as ``raw``."""
    n = len(truths)
    raw = {}
    for g in names:
        for edge in ("start", "end"):
            raw[f"{g}_{edge}_logits"] = np.zeros((n, width), np.float32)
            raw[f"{g}_{edge}"] = np.zeros((n, 1), np.float32)
        raw[f"{g}_allele"] = np.zeros((n, len(names[g])), np.float32)
    raw["orientation_logits"] = np.zeros((n, 4), np.float32)
    for key in ("mutation_rate", "indel_count", "productive"):
        raw[key] = np.zeros((n, 1), np.float32)
    predictions = Predictions(allele={g: raw[f"{g}_allele"] for g in names},
                              start={g: raw[f"{g}_start"].reshape(-1) for g in names},
                              end={g: raw[f"{g}_end"].reshape(-1) for g in names},
                              mutation_rate=raw["mutation_rate"].reshape(-1),
                              indel_count=raw["indel_count"].reshape(-1),
                              productive=raw["productive"].reshape(-1) > 0.5, orientation=np.zeros(n, int))
    return StageOutputs(ids=[t.case_id for t in truths], raw=raw, predictions=predictions,
                        canonical=[_window(t, width)[1] for t in truths], segments=None, calls={}, germline={},
                        records=[], airr_rows=[], seconds={}, allele_names={g: list(v) for g, v in names.items()},
                        allowed=allowed)


def _edge(out, i, g, edge, *, soft, argmax):
    """Set read ``i``'s soft-argmax expectation and the peak of its logits for one edge of gene ``g``."""
    out.raw[f"{g}_{edge}"][i, 0] = soft
    out.raw[f"{g}_{edge}_logits"][i, argmax] = 10.0


def _probs(out, i, g, **probabilities):
    for name, p in probabilities.items():
        out.predictions.allele[g][i, out.allele_names[g].index(name.replace("_", "*"))] = p


def _indel_bases(cigar):
    return sum(n for n, op in parse_cigar(cigar) if op in "ID")


def _oracle(truths, names, *, width=WIDTH):
    """Oracle-shaped outputs: each truth edge's logits peak on the edge and its expectation sits a quarter base past
    it, probability 1 on the primary allele, and the metadata heads give the truth (Short-D at 1 exactly when the D
    is shorter than its minimum)."""
    out = _blank(truths, width=width, names=names)
    for i, t in enumerate(truths):
        lo, seen = _window(t, width)
        for g, seg in t.segments.items():
            for edge, position in (("start", seg.observed.seq_start), ("end", seg.observed.seq_end)):
                p = position - lo
                if 0 <= p < width:
                    _edge(out, i, g, edge, soft=p + 0.25, argmax=p)
            if seg.alleles.primary in names[g]:
                out.predictions.allele[g][i, names[g].index(seg.alleles.primary)] = 1.0
        if "Short-D" in names.get("d", ()):
            d = t.segments.get("d")
            out.predictions.allele["d"][i, -1] = float(d is None or d.observed_length < MIN_OBSERVED["d"])
        out.raw["orientation_logits"][i, t.orientation] = 5.0
        out.predictions.orientation[i] = t.orientation
        if "v" in t.segments:
            out.raw["mutation_rate"][i, 0] = t.v_mismatches / t.segments["v"].observed_length
        out.raw["indel_count"][i, 0] = sum(_indel_bases(seg.observed.cigar) for seg in t.segments.values())
        out.raw["productive"][i, 0] = float(bool(t.productive))
    return out


def _assert_perfect(scores):
    """Every compared metric at its best: 1 for an accuracy, 0 for an error (up to the float32 precision of the
    head outputs, which the mutation rate's truth ratio is stored in)."""
    for cid, case in scores.items():
        for name, value in case.items():
            assert name in METRIC_NAMES, name
            assert math.isfinite(value), (cid, name)
            if not is_informational(name):
                assert value == pytest.approx(0.0 if lower_is_better(name) else 1.0, abs=1e-6), (cid, name, value)


# --- boundaries ---------------------------------------------------------------------------------------------------

def test_boundaries_soft_and_argmax_against_intervals():
    truth = _truth("b", 50, {"v": _seg(0, 30, "V1*01", start=(0, 1), end=(28, 31)), "j": _seg(36, 50, "J1*01")})
    out = _blank([truth])
    _edge(out, 0, "v", "start", soft=1.7, argmax=3)    # soft floor 1 inside (0, 1); argmax 3 after it
    _edge(out, 0, "v", "end", soft=26.9, argmax=31)    # soft floor 26 before (28, 31); argmax inside
    _edge(out, 0, "j", "start", soft=35.99, argmax=36)  # soft floor 35, one short
    _edge(out, 0, "j", "end", soft=48.6, argmax=50)    # the J ends at the read end: end-exclusive 50
    # Logits outside the read are never decoded: a start lies on a read base (0..49), an end-exclusive end on 0..50.
    out.raw["j_start_logits"][0, 50] = 20.0
    out.raw["j_end_logits"][0, 51] = 20.0
    scores = head_scores(out, [truth])["b"]

    expected = {("v", "start"): {"soft": 0, "argmax": 2}, ("v", "end"): {"soft": -2, "argmax": 0},
                ("j", "start"): {"soft": -1, "argmax": 0}, ("j", "end"): {"soft": -2, "argmax": 0}}
    for (g, edge), by_decoder in expected.items():
        for decoder, signed in by_decoder.items():
            assert scores[f"head.{g}.{edge}_err.{decoder}"] == signed, (g, edge, decoder)
            assert scores[f"head.{g}.{edge}_abs_err.{decoder}"] == abs(signed)
            assert scores[f"head.{g}.{edge}_exact.{decoder}"] == float(signed == 0)
    assert not any(name.startswith("head.d.") for name in scores)  # no D in the read

    assert [a.tolist() for a in decode_boundaries(out, "j", "soft")] == [[35], [48]]
    assert [a.tolist() for a in decode_boundaries(out, "j", "argmax")] == [[36], [50]]
    # The soft decoder is correct_segments' floor and clip to the read; expectations outside it are clipped.
    out.raw["v_start"][0, 0], out.raw["v_end"][0, 0] = -0.5, 57.2
    assert [a.tolist() for a in decode_boundaries(out, "v", "soft")] == [[0], [50]]
    with pytest.raises(ValueError, match="decoder 'mean'"):
        decode_boundaries(out, "v", "mean")


# --- alleles ------------------------------------------------------------------------------------------------------

def test_allele_top_k_generative_and_threshold():
    truths = [_truth("a", 50, {"v": _seg(0, 40, "V1*01", scored={"V1*02"})}),
              _truth("b", 50, {"v": _seg(0, 40, "V5*01")}),
              _truth("c", 50, {"d": _seg(10, 22, "D2*01")})]
    out = _blank(truths)
    _probs(out, 0, "v", V3_01=0.9, V1_02=0.8, V2_01=0.5, V1_01=0.3)  # scored set at rank 2
    _probs(out, 1, "v", V1_01=0.7, V1_02=0.6, V2_01=0.45, V3_01=0.4, V5_01=0.3)  # primary at rank 5
    _probs(out, 2, "d", D2_01=0.95)
    scores = head_scores(out, truths)

    assert {k: scores["a"][f"head.v.allele_top{k}"] for k in (1, 3, 5)} == {1: 0.0, 3: 1.0, 5: 1.0}
    assert scores["a"]["head.v.allele_top1_generative"] == 0.0
    assert scores["a"]["head.v.n_above_threshold"] == 3.0  # 0.9, 0.8 and 0.5 (inclusive)
    assert {k: scores["b"][f"head.v.allele_top{k}"] for k in (1, 3, 5)} == {1: 0.0, 3: 0.0, 5: 1.0}
    assert scores["b"]["head.v.n_above_threshold"] == 2.0
    assert scores["c"]["head.d.allele_top1"] == scores["c"]["head.d.allele_top1_generative"] == 1.0
    assert scores["c"]["head.d.n_above_threshold"] == 1.0

    # The calibrated probabilities (what the selector sees) are scored, not the raw head output.
    out.predictions.allele["v"] = np.zeros_like(out.raw["v_allele"])
    out.predictions.allele["v"][0, 0] = 0.6  # V1*01 alone
    out.raw["v_allele"][0, 3] = 0.99
    scores = head_scores(out, truths)
    assert scores["a"]["head.v.allele_top1"] == 1.0 and scores["a"]["head.v.n_above_threshold"] == 1.0


def test_allele_ranking_ties_and_allowed_masks():
    truths = [_truth("a", 50, {"v": _seg(0, 40, "V3*01"), "d": _seg(41, 50, "D1*01")}),
              _truth("b", 50, {"v": _seg(0, 40, "V1*01"), "d": _seg(41, 50, "D1*01")})]
    # Read a may call V2..V4 and any D; read b any V and no D (a light-chain locus in a two-locus model).
    allowed = {"v": np.array([[k in (2, 3, 4) for k in range(7)], [True] * 7]),
               "d": np.array([[True] * 3, [False] * 3]),
               "j": np.ones((2, 3), bool)}
    out = _blank(truths, allowed=allowed)
    _probs(out, 0, "v", V1_01=0.99, V3_01=0.4, V2_01=0.2)
    _probs(out, 1, "v", V1_02=0.8, V1_01=0.8)  # a tie goes to the lower index, as the product's selector breaks it
    _probs(out, 0, "d", D1_01=0.7)
    _probs(out, 1, "d", D1_01=0.7)

    ranking = allele_ranking(out, "v")
    assert [NAMES["v"][k] for k in ranking[0]] == ["V3*01", "V2*01", "V4*01"]
    assert [NAMES["v"][k] for k in ranking[1][:2]] == ["V1*01", "V1*02"]
    assert allele_ranking(out, "d")[1].size == 0
    scores = head_scores(out, truths)
    assert scores["a"]["head.v.allele_top1"] == 1.0 and scores["a"]["head.v.n_above_threshold"] == 0.0
    assert scores["b"]["head.v.allele_top1"] == 1.0
    assert scores["b"]["head.d.allele_top5"] == scores["b"]["head.d.n_above_threshold"] == 0.0  # no D allowed
    assert scores["a"]["head.d.allele_top1"] == 1.0

    # A one-dimensional mask (a genotype) applies to every read.
    out = _blank(truths, allowed={"v": np.array([k != 0 for k in range(7)])})
    _probs(out, 1, "v", V1_01=0.9, V2_01=0.3)
    assert [NAMES["v"][k] for k in allele_ranking(out, "v")[1][:1]] == ["V2*01"]

    # Ties keep the head's order however many alleles tie (numpy's default sort is unstable past 16 elements).
    many = {**NAMES, "v": [f"V{k}*01" for k in range(20)]}
    out = _blank(truths, names=many)
    out.predictions.allele["v"][:] = 0.9
    assert all(ranking.tolist() == list(range(20)) for ranking in allele_ranking(out, "v"))


def test_short_d_probability_only_for_a_model_with_the_sentinel():
    d_short, d_long = _seg(30, 35, "D1*01"), _seg(30, 42, "D1*01")  # 5 and 12 observed bases (minimum 8)
    v = _seg(0, 25, "V1*01")
    truths = [_truth("short_yes", 60, {"v": v, "d": d_short}), _truth("long_yes", 60, {"v": v, "d": d_long}),
              _truth("absent_no", 60, {"v": v}), _truth("long_no", 60, {"v": v, "d": d_long}),
              _truth("unscorable", 60, {"v": v}, unscorable={"d"})]
    out = _blank(truths, names=WITH_SHORT_D)
    for i, p in enumerate((0.7, 0.6, 0.2, 0.1, 0.9)):
        _probs(out, i, "d", **{"Short-D": p})
    scores = head_scores(out, truths)
    assert {cid: case.get("head.d.short_d_prob_ok") for cid, case in scores.items()} == {
        "short_yes": 1.0, "long_yes": 0.0, "absent_no": 0.0, "long_no": 1.0, "unscorable": None}
    # A D below its minimum gets no boundary or allele head metric.
    assert not any(name.startswith("head.d.") and name != "head.d.short_d_prob_ok" for name in scores["short_yes"])
    assert "head.d.allele_top1" in scores["long_no"]

    # The released IGH model has no Short-D output: the metric does not apply.
    plain = _blank(truths)
    assert not any("head.d.short_d_prob_ok" in case for case in head_scores(plain, truths).values())


# --- metadata heads -----------------------------------------------------------------------------------------------

def test_metadata_heads():
    truths = [_truth("a", 60, {"v": _seg(0, 40, "V1*01", cigar="10M1I20M2D9M", n_mismatch=4),
                               "j": _seg(48, 57, "J1*01", cigar="5M1D4M")}, productive=True, orientation=1),
              _truth("b", 60, {"v": _seg(0, 40, "V1*01")}, productive=False),
              _truth("c", 60, {"v": _seg(0, 40, "V1*01"), "j": _seg(48, 58, "J1*01")}, unscorable={"j"}),
              _truth("d", 60, {"j": _seg(48, 58, "J1*01")})]
    out = _blank(truths)
    rows = {"mutation_rate": (0.15, 0.0, 0.05, 0.2), "indel_count": (2.5, 0.0, 1.0, 0.0),
            "productive": (0.7, 0.5, 0.2, 0.2)}
    for key, values in rows.items():
        out.raw[key][:, 0] = values
    out.raw["orientation_logits"][0, 1] = out.raw["orientation_logits"][1, 3] = 1.0
    cases = [{"case_id": "a", "record": {"mutation_rate": 0.2}}, {"case_id": "b", "record": {"mutation_rate": 0.0}},
             {"case_id": "c", "record": {}}, {"case_id": "d", "record": {"mutation_rate": 0.1}}]
    scores = head_scores(out, truths, cases)

    # V mismatches over observed V germline bases: 4 of 41 (M + D) for a, 0 of 40 for b; no V in d.
    assert scores["a"]["head.mutation_rate_abs_err.read"] == pytest.approx(abs(0.15 - 4 / 41))
    assert scores["b"]["head.mutation_rate_abs_err.read"] == 0.0
    assert "head.mutation_rate_abs_err.read" not in scores["d"]
    assert scores["a"]["head.mutation_rate_abs_err.record"] == pytest.approx(0.05)
    assert "head.mutation_rate_abs_err.record" not in scores["c"]  # the record has no mutation rate
    # Inserted plus deleted bases of the observed CIGARs: 1 + 2 + 1 for a; unknown for c (J unscorable).
    assert scores["a"]["head.indel_count_abs_err"] == pytest.approx(1.5)
    assert scores["b"]["head.indel_count_abs_err"] == 0.0 and "head.indel_count_abs_err" not in scores["c"]
    # Productive: probability >= 0.5 against the truth flag, where the truth defines it.
    assert scores["a"]["head.productive.correct"] == 1.0 and scores["b"]["head.productive.correct"] == 0.0
    assert "head.productive.correct" not in scores["c"]
    assert [scores[c]["head.orientation.correct"] for c in "abcd"] == [1.0, 0.0, 1.0, 1.0]
    assert all(case["head.cropped"] == 0.0 for case in scores.values())

    without = head_scores(out, truths)
    assert not any("head.mutation_rate_abs_err.record" in case for case in without.values())
    assert without["a"]["head.mutation_rate_abs_err.read"] == scores["a"]["head.mutation_rate_abs_err.read"]


# --- the crop ---------------------------------------------------------------------------------------------------

def test_cropped_reads_score_only_the_edges_the_model_saw():
    # Forward read of 80 bases: the model sees canonical 0..59 and the cut is at 60.
    forward = _truth("fwd", 80, {"v": _seg(0, 30, "V1*01", end=(30, 31)), "d": _seg(40, 60, "D1*01"),
                                 "j": _seg(64, 80, "J1*01")})
    # Reverse-complemented read of 80 bases: the model sees canonical 20..79 and the cut is at 20.
    reverse = _truth("rev", 80, {"v": _seg(0, 40, "V1*01", end=(40, 41)), "j": _seg(50, 79, "J1*01",
                                                                                      start=(49, 51))},
                     orientation=1)
    plain = _truth("plain", 50, {"v": _seg(0, 40, "V1*01")})
    out = _blank([forward, reverse, plain])
    _edge(out, 0, "v", "start", soft=0.5, argmax=0)
    _edge(out, 0, "v", "end", soft=31.2, argmax=30)
    _edge(out, 0, "d", "start", soft=40.1, argmax=41)
    out.predictions.orientation[1] = 1  # the model saw the reverse read in the truth's frame
    _edge(out, 1, "v", "end", soft=20.5, argmax=21)   # canonical 40 and 41
    _edge(out, 1, "j", "start", soft=27.0, argmax=31)  # canonical 47 (two before 49) and 51
    _edge(out, 1, "j", "end", soft=59.25, argmax=59)   # canonical 79
    scores = head_scores(out, [forward, reverse, plain])

    edges = {cid: sorted({name.rsplit(".", 1)[0] for name in case if name.endswith(".soft")})
             for cid, case in scores.items()}
    # The D end touches the cut and the J lies past it; the reverse read's V start lies before its window.
    assert edges["fwd"] == ["head.d.start_abs_err", "head.d.start_err", "head.d.start_exact",
                            "head.v.end_abs_err", "head.v.end_err", "head.v.end_exact",
                            "head.v.start_abs_err", "head.v.start_err", "head.v.start_exact"]
    assert edges["rev"] == ["head.j.end_abs_err", "head.j.end_err", "head.j.end_exact",
                            "head.j.start_abs_err", "head.j.start_err", "head.j.start_exact",
                            "head.v.end_abs_err", "head.v.end_err", "head.v.end_exact"]
    assert scores["fwd"]["head.v.end_exact.soft"] == scores["fwd"]["head.v.end_exact.argmax"] == 1.0
    assert scores["fwd"]["head.d.start_err.argmax"] == 1.0
    assert scores["rev"]["head.v.end_exact.soft"] == scores["rev"]["head.v.end_exact.argmax"] == 1.0
    assert (scores["rev"]["head.j.start_err.soft"], scores["rev"]["head.j.start_err.argmax"]) == (-2.0, 0.0)
    assert scores["rev"]["head.j.end_exact.soft"] == scores["rev"]["head.j.end_exact.argmax"] == 1.0
    assert "head.v.allele_top1" in scores["fwd"] and "head.j.allele_top1" in scores["fwd"]  # alleles still scored
    assert [scores[c]["head.cropped"] for c in ("fwd", "rev", "plain")] == [1.0, 1.0, 0.0]


# --- misoriented reads ------------------------------------------------------------------------------------------

def test_a_misoriented_read_has_no_boundary_score_and_is_counted():
    # The model's frame is not the truth's: its positions do not describe the truth read, so the read's boundaries are
    # left out (not scored as a miss, which a frame mismatch could also hit by chance) and head.misoriented counts it.
    truths = [_truth("fwd", 50, {"v": _seg(0, 30, "V1*01"), "j": _seg(36, 50, "J1*01")}),
              _truth("rev", 50, {"v": _seg(0, 30, "V1*01"), "j": _seg(36, 50, "J1*01")}, orientation=1),
              _truth("rev_seen", 50, {"v": _seg(0, 30, "V1*01")}, orientation=1)]
    out = _blank(truths)
    for i in range(3):
        _edge(out, i, "v", "start", soft=0.2, argmax=0)  # where the truth V starts, in any frame
    out.predictions.orientation[2] = 1  # the model says reverse-complemented: the truth's frame
    out.raw["orientation_logits"][2, 1] = 5.0
    _probs(out, 1, "v", V1_01=0.9)
    scores = head_scores(out, truths)

    assert [scores[c]["head.misoriented"] for c in ("fwd", "rev", "rev_seen")] == [0.0, 1.0, 0.0]
    assert scores["fwd"]["head.v.start_exact.soft"] == scores["rev_seen"]["head.v.start_exact.soft"] == 1.0
    assert not any(name.startswith(("head.v.start", "head.v.end", "head.j.start", "head.j.end")) for name in
                   scores["rev"])
    # The read's other heads are still scored: its alleles do not depend on the frame.
    assert scores["rev"]["head.v.allele_top1"] == 1.0 and scores["rev"]["head.orientation.correct"] == 0.0


# --- applicability and input checks -------------------------------------------------------------------------------

def test_excluded_unscorable_and_short_segments():
    truths = [_truth("x", 50, {"v": _seg(0, 40, "V1*01")}, excluded="non_biological_orientation"),
              _truth("u", 50, {"v": _seg(0, 30, "V1*01"), "j": _seg(40, 49, "J1*01")}, unscorable={"v"})]
    out = _blank(truths)
    scores = head_scores(out, truths)
    assert list(scores) == ["u"]  # an excluded case is never scored
    assert not any(name.startswith("head.v.") for name in scores["u"])  # V truth could not be built
    assert not any(name.startswith("head.j.") for name in scores["u"])  # 9 observed J bases, minimum 10
    assert "head.mutation_rate_abs_err.read" not in scores["u"] and "head.indel_count_abs_err" not in scores["u"]
    # An excluded case needs no replayed read.
    assert list(head_scores(_blank(truths[1:]), truths)) == ["u"]


def test_inputs_are_checked():
    truths = [_truth("a", 50, {"v": _seg(0, 40, "V1*01")}), _truth("b", 50, {"v": _seg(0, 40, "V1*01")})]
    with pytest.raises(ValueError, match="case 'b' has no replayed read"):
        head_scores(_blank(truths[:1]), truths)
    out = _blank(truths)
    out.canonical[1] = out.canonical[1][:45]
    with pytest.raises(ValueError, match="case 'b': the replayed read has 45 bases, its truth read 50"):
        head_scores(out, truths)
    with pytest.raises(ValueError, match="case 'b' has no case record"):
        head_scores(_blank(truths), truths, [{"case_id": "a", "record": {"mutation_rate": 0.1}}])


# --- oracle-shaped outputs and the registry ---------------------------------------------------------------------

def _hand_built_truths():
    """Every situation the head metrics distinguish: both orientations, both crops, indels, an inverted, a short and
    a missing D, and productive True / False / undefined."""
    return [
        _truth("fwd", 58, {"v": _seg(0, 31, "V1*01", cigar="10M1I20M", n_mismatch=3, end=(31, 33)),
                           "d": _seg(35, 47, "D2*01", inverted=True), "j": _seg(47, 58, "J2*01")}, productive=True),
        _truth("rev", 52, {"v": _seg(2, 30, "V5*01", scored={"V1*02"}), "d": _seg(32, 37, "D1*01"),
                           "j": _seg(40, 52, "J3*01", cigar="4M2D8M")}, orientation=1, productive=False),
        _truth("crop_fwd", 70, {"v": _seg(0, 40, "V2*01"), "d": _seg(42, 55, "D3*01"), "j": _seg(58, 70, "J1*01")}),
        _truth("crop_rev", 75, {"v": _seg(0, 40, "V3*01", start=(0, 1)), "d": _seg(44, 56, "D1*01"),
                                "j": _seg(60, 74, "J1*01")}, orientation=1, productive=True),
        _truth("no_d", 45, {"v": _seg(0, 30, "V6*01"), "j": _seg(33, 45, "J2*01")}),
    ]


def test_oracle_shaped_outputs_score_perfectly():
    truths = _hand_built_truths()
    out = _oracle(truths, WITH_SHORT_D)
    cases = [{"case_id": t.case_id, "record": {"mutation_rate": 0.05}} for t in truths]
    scores = head_scores(out, truths, cases)
    _assert_perfect(scores)
    # Every head metric is exercised, and the oracle's informational ones are as good as they can be.
    assert set().union(*scores.values()) == set(METRIC_NAMES)
    for case in scores.values():
        assert all(case[name] == 0.0 for name in case if name.endswith(("_err.soft", "_err.argmax")))
        assert all(case[name] == 1.0 for name in case if name.endswith((".allele_top1_generative",
                                                                       ".n_above_threshold")))


def test_oracle_shaped_outputs_score_perfectly_on_s0_truths(s0_truths, s0_cases, small_model):
    _, reference_set = small_model
    names = {g: list(reference_set.gene(g.upper()).names) for g in GENES}
    scored = [t for t in s0_truths if t.excluded is None]
    out = _oracle(scored, names, width=576)
    scores = head_scores(out, s0_truths, s0_cases)
    assert list(scores) == [t.case_id for t in scored]
    _assert_perfect(scores)
    counts = {name: sum(name in case for case in scores.values()) for name in METRIC_NAMES}
    assert min(counts[f"head.{g}.start_exact.argmax"] for g in GENES) >= 10
    assert counts["head.mutation_rate_abs_err.record"] == len(scored) and counts["head.d.short_d_prob_ok"] == 0


def test_registry_classifies_every_head_metric():
    informational = {name for name in METRIC_NAMES
                     if name.endswith(("_err.soft", "_err.argmax")) and "_abs_err" not in name}
    informational |= {f"head.{g}.{m}" for g in GENES for m in ("allele_top1_generative", "n_above_threshold")}
    informational |= {"head.mutation_rate_abs_err.record", "head.cropped", "head.misoriented"}
    errors = {name for name in METRIC_NAMES if "_abs_err" in name} - informational
    assert len(METRIC_NAMES) == len(set(METRIC_NAMES)) == 59 and len(informational) == 21 and len(errors) == 14
    for name in METRIC_NAMES:
        assert name.startswith("head."), name
        if name in informational:
            assert is_informational(name), name
        elif name in errors:
            assert lower_is_better(name) and not is_informational(name), name
        else:  # accuracies: exact edges, top-k, Short-D, productive, orientation
            assert not lower_is_better(name) and not is_informational(name), name


# --- calibration --------------------------------------------------------------------------------------------------

def test_calibration_table():
    v = _seg(0, 40, "V1*01")
    truths = [_truth(f"r{k}", 50, {"v": v}) for k in range(4)]
    truths += [_truth("short", 50, {"v": _seg(0, 12, "V1*01")}),
               _truth("excluded", 50, {"v": v}, excluded="non_biological_orientation")]
    out = _blank(truths)
    for i, (name, p) in enumerate((("V1*01", 0.95), ("V2*01", 0.85), ("V1*01", 0.15), ("V2*01", 0.12),
                                   ("V1*01", 0.9), ("V1*01", 0.9))):
        out.predictions.allele["v"][i, NAMES["v"].index(name)] = p
    table = calibration_table(out, truths, "v")

    assert table["gene"] == "v" and table["n"] == 4 and len(table["bins"]) == 10
    by_bin = {k: (b["n"], b["confidence"], b["accuracy"]) for k, b in enumerate(table["bins"]) if b["n"]}
    assert by_bin.keys() == {1, 8, 9}
    assert by_bin[9] == (1, pytest.approx(0.95), 1.0) and by_bin[8] == (1, pytest.approx(0.85), 0.0)
    assert by_bin[1] == (2, pytest.approx(0.135), 0.5)
    assert table["bins"][1]["lo"] == pytest.approx(0.1) and table["bins"][1]["hi"] == pytest.approx(0.2)
    assert table["bins"][0] == {"lo": 0.0, "hi": pytest.approx(0.1), "n": 0, "confidence": None, "accuracy": None}
    assert table["ece"] == pytest.approx(0.25 * 0.05 + 0.25 * 0.85 + 0.5 * 0.365)
    # The same expected calibration error as AlignAIR's own (right-closed bins, the first one closed).
    conf = np.array([0.95, 0.85, 0.15, 0.12], np.float32)
    assert table["ece"] == pytest.approx(ece(conf, [1.0, 0.0, 1.0, 0.0], 10))
    assert table["accuracy"] == 0.5 and table["confidence"] == pytest.approx(np.mean(conf))

    # Bins are closed on the right: probability 1 falls in the last bin, 0.5 in (0.25, 0.5] and 0 in the first.
    out = _blank(truths[:3])
    out.predictions.allele["v"][0, 0] = 1.0
    out.predictions.allele["v"][2, 0] = 0.5
    edges = calibration_table(out, truths[:3], "v", bins=4)
    assert [b["n"] for b in edges["bins"]] == [1, 1, 0, 1]
    assert edges["ece"] == pytest.approx(ece(np.array([1.0, 0.0, 0.5]), [1.0, 1.0, 1.0], 4))
    empty = calibration_table(_blank(truths[4:5]), truths[4:], "v")  # a short V and an excluded case
    assert (empty["n"], empty["ece"], empty["confidence"], empty["accuracy"]) == (0, None, None, None)
    assert all(b["n"] == 0 and b["confidence"] is None and b["accuracy"] is None for b in empty["bins"])
    with pytest.raises(ValueError, match="at least one bin"):
        calibration_table(out, truths, "v", bins=0)
    with pytest.raises(ValueError, match="no allele head for gene 'x'"):
        calibration_table(out, truths, "x")


# --- on the replay ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["igh", "igh_igk"])
def test_head_metrics_on_the_replay_agree_with_the_product(variant, make_shaped_model, shaped_spans, s0_reads,
                                                           s0_truths, s0_cases):
    model, reference_set = make_shaped_model(variant)
    ids, seqs = [name for name, _ in s0_reads], [seq for _, seq in s0_reads]
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=16)
    lengths = np.array([len(s) for s in out.canonical])
    long = lengths > shaped_spans["j"][1]
    assert long.sum() >= 10

    for g in GENES:
        # Soft: the product's segments wherever its correction left them alone (every span fits these reads).
        starts, ends = decode_boundaries(out, g, "soft")
        assert np.array_equal(starts[long], out.segments.start[g][long])
        assert np.array_equal(ends[long], out.segments.end[g][long])
        # Argmax: the logits' peak, which the shaped heads put on their spans.
        starts, ends = decode_boundaries(out, g, "argmax")
        assert set(starts[long]) == {shaped_spans[g][0]} and set(ends[long]) == {shaped_spans[g][1]}
        # The ranking's first allele is the product's top-1 call, among the read's allowed alleles.
        for i, ranking in enumerate(allele_ranking(out, g)):
            call = out.calls[g][i].names
            assert (out.allele_names[g][ranking[0]] if ranking.size else None) == (call[0] if call else None)

    scores = head_scores(out, s0_truths, s0_cases)
    assert list(scores) == [t.case_id for t in s0_truths if t.excluded is None]
    assert set().union(*scores.values()) <= set(METRIC_NAMES)
    if variant == "igh_igk":  # reads the model puts on IGK may call no D: their D top-k misses
        no_d = [cid for i, cid in enumerate(out.ids) if not out.allowed["d"][i].any() and "head.d.allele_top5"
                in scores.get(cid, {})]
        assert no_d and all(scores[cid]["head.d.allele_top5"] == 0.0 for cid in no_d)


@needs_model
def test_head_metrics_of_the_released_checkpoint_agree_with_the_product(reference, s1_layout):
    reads = read_fasta(s1_layout.reads("S1"))[:200]
    ids, seqs = [name for name, _ in reads], [seq for _, seq in reads]
    model, reference_set = load_model_checked(Path(MODEL).expanduser(), reference, device="cpu")
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=64)
    wanted = set(ids)
    truths = [t for t in read_truth_jsonl(s1_layout.truth("S1")) if t.case_id in wanted]
    cases = [c for c in read_cases(s1_layout.cases("S1")) if c["case_id"] in wanted]
    scores = head_scores(out, truths, cases)

    assert len(scores) == len(truths) == 200
    names = set().union(*scores.values())
    assert names <= set(METRIC_NAMES) and "head.d.short_d_prob_ok" not in names  # 33 D outputs, no Short-D
    assert "Short-D" not in out.allele_names["d"] and len(out.allele_names["d"]) == 33
    for g in GENES:
        # Saturated calibrated probabilities can tie: the ranking still starts with the selector's call.
        for ranking, call in zip(allele_ranking(out, g), out.calls[g]):
            assert out.allele_names[g][ranking[0]] == call.names[0]
    # V comes first in the ordering repair, so the soft decoder gives the product's V wherever the span is not empty.
    starts, ends = decode_boundaries(out, "v", "soft")
    kept = ends > starts
    assert kept.mean() > 0.9
    assert np.array_equal(starts[kept], out.segments.start["v"][kept])
    assert np.array_equal(ends[kept], out.segments.end["v"][kept])
