"""Truth-substitution ablations: truth converted into AlignAIR's stage types, each variant replayed and scored, and
the error each stage contributes.

``substitutions_for`` is tested on hand-built truths and replay outputs; the variants run on a built S0 with small
random-init models. The test against the released checkpoint runs only when ``ALIGNAIR_ASSAY_MODEL`` names a
``.alignair`` model file.
"""
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("GenAIRR")

from alignair import api  # noqa: E402
from alignair.io.airr import write_airr  # noqa: E402
from alignair.predict.state import GeneCall, GermlineAlignment  # noqa: E402

from alignair_assay.airr.reader import load_airr_tsv, rows_to_predictions  # noqa: E402
from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.cigar import parse_cigar  # noqa: E402
from alignair_assay.diagnostics.ablation import (  # noqa: E402
    DEFAULT_VARIANTS,
    SINGLE_SWAPS,
    VARIANTS,
    VariantResult,
    airr_text_rows,
    run_variants,
    stage_contributions,
    substitutions_for,
    swap_effects,
)
from alignair_assay.diagnostics.replay import StageOutputs, Substitutions, load_model_checked, replay  # noqa: E402
from alignair_assay.scoring.pipeline import score_predictions  # noqa: E402
from alignair_assay.stats import compare  # noqa: E402
from alignair_assay.truth.alleles import AlleleTruth  # noqa: E402
from alignair_assay.truth.boundaries import BoundaryTruth  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl  # noqa: E402
from alignair_assay.truth.model import ObservableTruth, SegmentTruth  # noqa: E402
from alignair_assay.truth.oracle import oracle_prediction  # noqa: E402
from alignair_assay.truth.segment import ObservedSegment  # noqa: E402

pytestmark = pytest.mark.genairr_build  # these tests build GenAIRR suites
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set ALIGNAIR_ASSAY_MODEL to a .alignair model file")
BATCH = 16
GENES = ("v", "d", "j")
NAMES = {"v": ["V1*01", "V2*01"], "d": ["D1*01"], "j": ["J1*01"]}


def _same_scores(a: dict, b: dict) -> bool:
    """Per-case scores equal, ``nan`` equal to ``nan``."""
    return a.keys() == b.keys() and all(
        a[c].keys() == b[c].keys()
        and all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in ((a[c][k], b[c][k]) for k in a[c]))
        for c in a)


# --- hand-built truths and replay outputs -----------------------------------------------------------------------

def _seg(start, end, allele, *, germ_start=3, cigar=None, inverted=False) -> SegmentTruth:
    cigar = cigar or f"{end - start}M"
    germ = sum(n for n, op in parse_cigar(cigar) if op in "MD")
    observed = ObservedSegment(allele=allele, seq_start=start, seq_end=end, germ_start=germ_start,
                               germ_end=germ_start + germ, cigar=cigar, n_mismatch=0, n_aligned=end - start,
                               observed_germline="A" * germ)
    boundaries = BoundaryTruth((start, start), (end, end), (germ_start, germ_start), (germ_start + germ,) * 2)
    alleles = AlleleTruth(primary=allele, alleles=frozenset({allele}), genes=frozenset({allele.split("*")[0]}),
                          simulator_ties=frozenset({allele}), disagrees_with_simulator=False,
                          generative=frozenset({allele}))
    return SegmentTruth(observed=observed, boundaries=boundaries, alleles=alleles, observed_length=germ,
                        inverted=inverted)


def _truth(case_id, length, segments, *, orientation=0, unscorable=(), excluded=None) -> ObservableTruth:
    return ObservableTruth(case_id=case_id, stratum="s", sequence="A" * length, orientation=orientation,
                           segments=segments, unscorable=frozenset(unscorable), junction=None, vj_in_frame=None,
                           stop_codon=None, productive=None, clone_key=None, v_mismatches=0, v_frameshift=False,
                           j_frameshift=False, excluded=excluded, warnings=())


def _outputs(lengths: dict[str, int], orientation: list[int], cropped=()) -> StageOutputs:
    """What substitutions_for reads of a baseline replay: the ids, the model's reads, its orientation, its genes
    and allele names, and the crop flags of its records."""
    ids = list(lengths)
    n = len(ids)
    return StageOutputs(ids=ids, raw={}, predictions=SimpleNamespace(orientation=np.array(orientation)),
                        canonical=["A" * lengths[i] for i in ids],
                        segments=SimpleNamespace(start={g: np.zeros(n, int) for g in GENES},
                                                 end={g: np.ones(n, int) for g in GENES}, low_quality=None),
                        calls={}, germline={}, airr_rows=[], seconds={}, allele_names=NAMES,
                        records=[{"length_cropped": True} if i in cropped else {} for i in ids])


def _standard():
    v, d, j = _seg(0, 200, "V1*01", cigar="100M1D100M"), _seg(205, 215, "D1*01"), _seg(220, 260, "J1*01")
    truths = [
        _truth("r0", 300, {"v": v, "d": d, "j": j}),
        # cropped reverse-complemented read: AlignAIR saw the canonical suffix 124..700; V crosses the cut
        _truth("r1", 700, {"v": _seg(50, 300, "V1*01"), "d": _seg(310, 320, "D1*01"), "j": _seg(330, 380, "J1*01")},
               orientation=1),
        _truth("r2", 300, {"v": v, "j": j}),                       # D not in the read
        _truth("r3", 300, {"v": v, "d": d}, unscorable={"j"}),     # J truth could not be built
        _truth("r4", 300, {"v": v, "d": replace(d, inverted=True), "j": j}),
        _truth("r5", 300, {"v": _seg(0, 200, "VX*01"), "d": d, "j": j}),  # a primary the model cannot call
        _truth("r6", 300, {"v": v, "d": d, "j": j}, excluded="non_biological_orientation"),
    ]
    lengths = {"r0": 300, "r1": 576, "r2": 300, "r3": 300, "r4": 300, "r5": 300, "r6": 300, "r7": 300}
    return truths, lengths


def test_variant_names():
    assert VARIANTS == ("baseline", "true_orientation", "true_spans", "true_calls", "true_germline")
    assert SINGLE_SWAPS == ("argmax_decoder", "true_classifier_spans")
    assert DEFAULT_VARIANTS == VARIANTS + SINGLE_SWAPS


def test_true_germline_substitutions_in_the_model_frame():
    truths, lengths = _standard()
    outputs = _outputs(lengths, [0, 1, 0, 0, 0, 0, 2, 0], cropped={"r1"})
    subs, fallbacks = substitutions_for("true_germline", truths, outputs)

    assert subs.active() == ("orientation", "spans", "calls", "germline", "classifier_spans")
    assert subs.segment_spans and subs.decoder == "soft"
    # every scored read gets its truth orientation; the excluded case and the read without truth get the model's own
    # (so the replay needs no second forward pass for them)
    assert subs.orientation == [0, 1, 0, 0, 0, 0, 2, 0]
    # r0: the observed coordinates, as they are (a forward read the model saw whole)
    assert [subs.spans[g][0][0] for g in GENES] == [0, 205, 220]
    assert [subs.spans[g][1][0] for g in GENES] == [200, 215, 260]
    assert subs.calls["v"][0] == GeneCall(("V1*01",), (1.0,))
    assert subs.germline["v"][0] == GermlineAlignment("V1*01", 0, 200, 3, 204, "100M1D100M")
    assert subs.germline["j"][0] == GermlineAlignment("J1*01", 220, 260, 3, 43, "40M")
    # r1: shifted into the window the model saw (canonical 124..700); the V that crosses the cut falls back
    assert subs.spans["v"][0][1] is None and subs.calls["v"][1] is None and subs.germline["v"][1] is None
    assert (subs.spans["d"][0][1], subs.spans["d"][1][1]) == (186, 196)
    assert subs.germline["j"][1] == GermlineAlignment("J1*01", 206, 256, 3, 53, "50M")
    assert subs.calls["d"][1] == GeneCall(("D1*01",), (1.0,))
    # r2 / r3: no truth for the segment
    assert all(subs.spans["d"][k][2] is None for k in (0, 1)) and subs.calls["d"][2] is None
    assert subs.germline["j"][3] is None and subs.calls["j"][3] is None
    # r4: an inverted D keeps its read span and call; its germline frame is not AlignAIR's, so the matcher aligns it
    assert subs.spans["d"][0][4] == 205 and subs.calls["d"][4] == GeneCall(("D1*01",), (1.0,))
    assert subs.germline["d"][4] is None
    # r5: the model cannot call the primary: span only
    assert subs.spans["v"][0][5] == 0 and subs.calls["v"][5] is None and subs.germline["v"][5] is None
    # r6 (excluded) and r7 (no truth): nothing substituted, nothing counted
    assert all(subs.spans[g][0][k] is None and subs.calls[g][k] is None for g in GENES for k in (6, 7))
    assert fallbacks == {
        "spans.v.cropped": 1, "calls.v.cropped": 1, "germline.v.cropped": 1,
        "spans.d.unobserved": 1, "calls.d.unobserved": 1, "germline.d.unobserved": 1,
        "spans.j.unscorable": 1, "calls.j.unscorable": 1, "germline.j.unscorable": 1,
        "germline.d.inverted": 1,
        "calls.v.not_in_model": 1, "germline.v.not_in_model": 1,
    }


def test_cumulative_variants_add_one_stage_each():
    truths, lengths = _standard()
    outputs = _outputs(lengths, [0, 1, 0, 0, 0, 0, 2, 0], cropped={"r1"})
    expected = {"baseline": (), "true_orientation": ("orientation",),
                "true_spans": ("orientation", "spans", "classifier_spans"),
                "true_calls": ("orientation", "spans", "calls", "classifier_spans"),
                "true_germline": ("orientation", "spans", "calls", "germline", "classifier_spans"),
                "argmax_decoder": ("decoder",), "true_classifier_spans": ("classifier_spans",)}
    found = {name: substitutions_for(name, truths, outputs) for name in DEFAULT_VARIANTS}
    assert {name: subs.active() for name, (subs, _) in found.items()} == expected
    assert found["baseline"] == (Substitutions(), {}) and found["true_orientation"][1] == {}
    assert found["argmax_decoder"] == (Substitutions(decoder="argmax"), {})
    assert set(found["true_spans"][1]) == {"spans.v.cropped", "spans.d.unobserved", "spans.j.unscorable"}
    assert found["true_spans"][0].spans == found["true_germline"][0].spans


def test_classifier_spans_need_the_models_frame():
    # Without the true orientation, truth spans are in the model's frame only where its orientation is right.
    truths, lengths = _standard()
    outputs = _outputs(lengths, [0, 0, 0, 3, 0, 0, 2, 0], cropped={"r1"})
    subs, fallbacks = substitutions_for("true_classifier_spans", truths, outputs)
    assert subs.orientation is None and subs.classifier_spans and not subs.segment_spans
    assert all(subs.spans[g][0][k] is None for g in GENES for k in (1, 3))
    assert subs.spans["v"][0][0] == 0 and subs.spans["d"][0][4] == 205
    assert fallbacks == {"spans.v.orientation": 2, "spans.d.orientation": 2, "spans.j.orientation": 1,
                         "spans.d.unobserved": 1, "spans.j.unscorable": 1}


def test_substitutions_for_checks_its_inputs():
    truths, lengths = _standard()
    outputs = _outputs(lengths, [0] * 8, cropped={"r1"})
    with pytest.raises(ValueError, match="unknown variant 'true_everything'; known: baseline, true_orientation"):
        substitutions_for("true_everything", truths, outputs)
    with pytest.raises(ValueError, match=re.escape("case 'r0': the replayed read has 299 bases, its truth read 300")):
        substitutions_for("true_spans", truths, _outputs({**lengths, "r0": 299}, [0] * 8, cropped={"r1"}))


# --- variants on S0 ---------------------------------------------------------------------------------------------

def _ids_seqs(cases):
    return [case["case_id"] for case in cases], [case["sequence"] for case in cases]


def test_baseline_rows_are_the_replay_and_score_like_the_product_tsv(make_shaped_model, s0_cases, s0_truths,
                                                                    reference, tmp_path):
    model, reference_set = make_shaped_model("igh_varied_orientation")
    results = run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu",
                           batch_size=BATCH, variants=("baseline",))
    ids, seqs = _ids_seqs(s0_cases)
    out = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    assert list(results) == ["baseline"]
    result = results["baseline"]
    assert isinstance(result, VariantResult) and result.variant == "baseline"
    assert result.rows == out.airr_rows
    assert result.fallbacks == {} and result.substituted == ()
    assert list(result.seconds) == list(out.seconds)

    # In memory, the rows read back exactly as the TSV AlignAIR's writer makes of them (ruling D2).
    path = tmp_path / "predictions.tsv"
    write_airr(str(path), ids, seqs, out.airr_rows, locus="IGH")
    preds, stats = load_airr_tsv(path)
    assert rows_to_predictions(airr_text_rows(ids, seqs, out.airr_rows, "IGH")) == (preds, stats)
    assert stats["invalid_rows"] == 0 and {p.rev_comp for p in preds.values()} == {True, False}
    assert _same_scores(result.scores, score_predictions(s0_truths, preds, reference))


# The fields the AIRR assembly builds from the germline alignment, the calls and the orientation.
FED = re.compile(r"(junction|clone_key)\.|[vdj]\.(seq|germ)_(start|end)_|[vdj]\.top1_in_set$|orientation\.")
# Excluded from the equality check: the V germline coordinates and the junction coordinates. AlignAIR's AIRR assembly
# (compute_alignment_positions) writes the V germline coordinates in IMGT-gapped numbering and the CDR3/junction
# coordinates as columns of the gapped sequence alignment, whatever germline alignment it is given. That is tracked on
# the AlignAIR fix-list; this test neither asserts nor depends on it. When the assembly writes them in read and
# ungapped germline coordinates, drop this exclusion.
REENCODED = re.compile(r"v\.germ_(start|end)_|junction\.(start|end)_")


def test_true_germline_scores_like_the_oracle_on_assembly_fed_fields(make_shaped_model, s0_cases, s0_truths,
                                                                    reference):
    # The "igh" shaped model predicts no indel, so the assembly's gate on the neural productive and indel
    # predictions (non-productive with more than one indel: no junction) stays open.
    model, reference_set = make_shaped_model("igh")
    results = run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu",
                           batch_size=BATCH, variants=VARIANTS)
    assert list(results) == list(VARIANTS)
    assert all(result.fallbacks == {} for result in results.values())  # S0 shows every segment of every read
    assert {row["airr_assembly_status"] for row in results["true_germline"].rows} == {"complete"}
    oracle = score_predictions(s0_truths, {t.case_id: oracle_prediction(t) for t in s0_truths}, reference)
    final = results["true_germline"].scores
    equal = 0
    for case_id, case in oracle.items():
        for name, value in case.items():
            if not FED.match(name):
                continue
            if REENCODED.match(name):
                continue
            assert final[case_id][name] == value, (case_id, name)
            equal += 1
    assert equal > 50 * 20
    # The model's own germline alignment (true calls, its matcher) is not oracle quality.
    own = results["true_calls"].scores
    assert any(own[c][name] != final[c][name] for c in oracle for name in oracle[c]
               if FED.match(name) and not REENCODED.match(name))


def test_unobserved_segments_fall_back_to_the_model_and_are_counted(make_shaped_model, s0_cases, s0_truths,
                                                                    reference):
    model, reference_set = make_shaped_model("igh")
    dropped = {t.case_id for t in s0_truths[:5]}
    truths = [replace(t, segments={g: s for g, s in t.segments.items() if g != "d"}) if t.case_id in dropped else t
              for t in s0_truths]
    results = run_variants(model, reference_set, s0_cases, truths, reference=reference, device="cpu",
                           batch_size=BATCH, variants=("true_spans", "true_calls", "true_germline"))
    assert results["true_spans"].fallbacks == {"spans.d.unobserved": 5}
    assert results["true_calls"].fallbacks == {"spans.d.unobserved": 5, "calls.d.unobserved": 5}
    assert results["true_germline"].fallbacks == {"spans.d.unobserved": 5, "calls.d.unobserved": 5,
                                                  "germline.d.unobserved": 5}
    ids, _ = _ids_seqs(s0_cases)
    for i, case_id in enumerate(ids):
        if case_id not in dropped:
            continue
        # D is left to the model under the same upstream truth: its classifier call, its matcher's alignment
        assert results["true_calls"].rows[i]["d_call"] == results["true_spans"].rows[i]["d_call"]
        for key in ("d_call", "d_sequence_start", "d_sequence_end", "d_germline_start", "d_cigar"):
            assert results["true_germline"].rows[i][key] == results["true_calls"].rows[i][key], key
        # and scored against the edited truth: the read shows no D
        assert "d.spurious_call" in results["true_germline"].scores[case_id]


def test_argmax_decoder_changes_only_coordinates(small_model, s0_cases, s0_truths, reference):
    model, reference_set = small_model
    results = run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu",
                           batch_size=BATCH, variants=("baseline", "argmax_decoder"))
    base, argmax = results["baseline"].rows, results["argmax_decoder"].rows
    for a, b in zip(base, argmax):
        for g in GENES:
            assert (a[f"{g}_call"], a[f"{g}_calls"], a[f"{g}_likelihoods"]) == \
                (b[f"{g}_call"], b[f"{g}_calls"], b[f"{g}_likelihoods"])
        assert a["mutation_rate"] == b["mutation_rate"] and a["productive_prediction"] == b["productive_prediction"]
    coordinates = [f"{g}_sequence_{edge}" for g in GENES for edge in ("start", "end")]
    assert any(a.get(key) != b.get(key) for a, b in zip(base, argmax) for key in coordinates)


def test_run_variants_checks_its_inputs(small_model, s0_cases, s0_truths, reference):
    model, reference_set = small_model
    with pytest.raises(ValueError, match="unknown variant 'oracle'"):
        run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu", batch_size=BATCH,
                     variants=("oracle",))
    with pytest.raises(ValueError, match="no truth"):
        run_variants(model, reference_set, s0_cases, [], reference=reference, device="cpu", batch_size=BATCH,
                     variants=("baseline",))
    # Truths are restricted to the replayed cases: a subset of cases is scored on its own truths.
    results = run_variants(model, reference_set, s0_cases[:4], s0_truths, reference=reference, device="cpu",
                           batch_size=BATCH, variants=("baseline",))
    assert list(results["baseline"].scores) == [case["case_id"] for case in s0_cases[:4]]


def test_run_variants_scores_only_against_the_models_own_reference(make_small_model, s0_cases, s0_truths,
                                                                    reference):
    # An in-memory model never goes through a checkpoint check: run_variants itself refuses one whose reference is
    # not the truth reference, naming the first differing allele, before anything is replayed.
    model, reference_set = make_small_model("HUMAN_IGK_OGRDB")
    calls = []
    handle = model.orientation_head.register_forward_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match=r"not the truth reference.*first difference: IGHD\S+ is missing from the "
                                             r"model reference"):
            run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu",
                         batch_size=BATCH, variants=("baseline",))
    finally:
        handle.remove()
    assert calls == []


def test_run_variants_reuses_a_given_baseline_replay(small_model, s0_cases, s0_truths, reference):
    model, reference_set = small_model
    ids, seqs = _ids_seqs(s0_cases)
    baseline = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    calls = []
    handle = model.orientation_head.register_forward_hook(lambda *_: calls.append(1))  # once per forward pass
    try:
        given = run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu",
                             batch_size=BATCH, variants=("baseline", "argmax_decoder"), _baseline=baseline)
        assert len(calls) == math.ceil(len(ids) / BATCH)  # the argmax variant's replay only
    finally:
        handle.remove()
    own = run_variants(model, reference_set, s0_cases, s0_truths, reference=reference, device="cpu",
                       batch_size=BATCH, variants=("baseline", "argmax_decoder"))
    assert given["baseline"].rows == own["baseline"].rows == baseline.airr_rows
    assert given["argmax_decoder"].rows == own["argmax_decoder"].rows
    assert _same_scores(given["argmax_decoder"].scores, own["argmax_decoder"].scores)
    with pytest.raises(ValueError, match="baseline replay is of other reads"):
        run_variants(model, reference_set, s0_cases[:4], s0_truths, reference=reference, device="cpu",
                     batch_size=BATCH, variants=("baseline",), _baseline=baseline)


# --- stage contributions ----------------------------------------------------------------------------------------

def _result(name: str, scores: dict) -> VariantResult:
    return VariantResult(variant=name, rows=[], scores=scores, fallbacks={}, seconds={}, substituted=())


def test_stage_contributions_pair_the_chain_in_order():
    ids = [f"c{i}" for i in range(20)]
    base = {c: {"m.acc": 0.0, "m.abs_err": 2.0} for c in ids}
    orient = {c: {"m.acc": float(i < 15), "m.abs_err": 2.0} for i, c in enumerate(ids)}
    spans = {c: {"m.acc": float(i < 15), "m.abs_err": 0.0} for i, c in enumerate(ids)}
    spans["c0"]["m.abs_err"] = math.nan  # not covered
    del spans["c1"]["m.abs_err"]  # not applicable
    results = {"baseline": _result("baseline", base), "true_orientation": _result("true_orientation", orient),
               "true_spans": _result("true_spans", spans), "argmax_decoder": _result("argmax_decoder", orient)}
    metrics = ["m.acc", "m.abs_err"]
    out = stage_contributions(results, metrics)
    assert [(e["from"], e["to"], e["stages"]) for e in out] == [
        ("baseline", "true_orientation", ["orientation"]), ("true_orientation", "true_spans", ["segmentation"])]

    first = out[0]["metrics"]["m.acc"]
    assert (first["mean_from"], first["mean_to"], first["delta"], first["n_paired"]) == (0.0, 0.75, 0.75, 20)
    assert (first["n_from"], first["coverage_from"], first["n_to"], first["coverage_to"]) == (20, 1.0, 20, 1.0)
    assert first["verdict"] == "to_better" and first["test"] == "mcnemar_exact"
    assert out[0]["metrics"]["m.abs_err"]["verdict"] == "tie_or_inconclusive"
    second = out[1]["metrics"]["m.abs_err"]
    assert (second["delta"], second["n_paired"], second["n_to"]) == (-2.0, 18, 18)
    assert second["coverage_to"] == pytest.approx(18 / 19) and second["verdict"] == "to_better"  # lower is better
    # the verdicts are stats.compare's, Holm-corrected over the metrics
    reference_cmp = compare(orient, spans, metrics, lower_is_better={"m.abs_err"})
    assert {m: e["p_value"] for m, e in out[1]["metrics"].items()} == {m: e["p_value"]
                                                                       for m, e in reference_cmp.items()}

    # A variant missing from the chain: the pair spans both stages.
    skipped = stage_contributions({k: results[k] for k in ("baseline", "true_spans")}, metrics)
    assert [(e["from"], e["to"], e["stages"]) for e in skipped] == [
        ("baseline", "true_spans", ["orientation", "segmentation"])]
    # Single swaps are each compared with the baseline.
    swaps = swap_effects(results, metrics)
    assert [(e["from"], e["to"]) for e in swaps] == [("baseline", "argmax_decoder")]
    assert swaps[0]["metrics"]["m.acc"]["delta"] == 0.75


def test_contributions_show_the_paired_means_beside_the_delta():
    # The step's delta is computed on the paired cases; its own means cover each variant's own cases. The table
    # shows the paired means beside the delta, so delta = to (paired) - from (paired) on every row.
    ids = [f"c{i}" for i in range(20)]
    base = {c: {"m.abs_err": 10.0 if i == 0 else 2.0} for i, c in enumerate(ids)}
    orient = {c: {"m.abs_err": 1.0} for c in ids}
    orient["c0"]["m.abs_err"] = math.nan  # c0 is not covered by the step's "to" variant
    results = {"baseline": _result("baseline", base), "true_orientation": _result("true_orientation", orient)}
    entry = stage_contributions(results, ["m.abs_err"])[0]["metrics"]["m.abs_err"]
    assert entry["mean_from"] == pytest.approx((10.0 + 19 * 2.0) / 20)  # its own mean, over its 20 cases
    assert (entry["mean_from_paired"], entry["mean_to_paired"], entry["n_paired"]) == (2.0, 1.0, 19)
    assert entry["delta"] == entry["mean_to_paired"] - entry["mean_from_paired"] == -1.0
    swap = swap_effects({"baseline": results["baseline"], "argmax_decoder": _result("argmax_decoder", orient)},
                        ["m.abs_err"])[0]["metrics"]["m.abs_err"]
    assert (swap["mean_from_paired"], swap["mean_to_paired"]) == (2.0, 1.0)


def test_comparisons_are_clustered_by_the_given_clusters():
    # 4 clones of 10 reads; the truth of one stage fixes the reads of one clone only.
    ids = [f"c{i}" for i in range(40)]
    clusters = {c: f"lineages/clone:{i // 10}" for i, c in enumerate(ids)}
    base = {c: {"m.acc": 0.0} for c in ids}
    fixed = {c: {"m.acc": float(i < 10)} for i, c in enumerate(ids)}
    results = {"baseline": _result("baseline", base), "true_orientation": _result("true_orientation", fixed),
               "argmax_decoder": _result("argmax_decoder", fixed)}
    reads = stage_contributions(results, ["m.acc"])[0]["metrics"]["m.acc"]
    assert (reads["n_paired"], reads["n_clusters"], reads["test"], reads["verdict"]) == (40, 40, "mcnemar_exact",
                                                                                        "to_better")
    for entry in (stage_contributions(results, ["m.acc"], clusters=clusters)[0]["metrics"]["m.acc"],
                  swap_effects(results, ["m.acc"], clusters=clusters)[0]["metrics"]["m.acc"]):
        assert (entry["n_paired"], entry["n_clusters"]) == (40, 4)
        assert (entry["test"], entry["p_value"], entry["verdict"]) == ("too_few_clusters", 1.0, "tie_or_inconclusive")
        assert entry["delta"] == reads["delta"] == 0.25


# --- the released checkpoint ------------------------------------------------------------------------------------

@needs_model
def test_variants_on_the_released_checkpoint(reference, s1_layout):
    cases = read_cases(s1_layout.cases("S1"))[:120]
    wanted = {case["case_id"] for case in cases}
    truths = [t for t in read_truth_jsonl(s1_layout.truth("S1")) if t.case_id in wanted]
    model, reference_set = load_model_checked(Path(MODEL).expanduser(), reference, device="cpu")
    results = run_variants(model, reference_set, cases, truths, reference=reference, device="cpu", batch_size=64)
    assert list(results) == list(DEFAULT_VARIANTS)
    ids, seqs = _ids_seqs(cases)
    assert results["baseline"].rows == api.predict_sequences(model, reference_set, seqs, device="cpu", batch_size=64)
    # the classifier seam reproduces the released model's own classifier call
    none = [None] * len(seqs)
    seam = replay(model, reference_set, seqs, ids, device="cpu", batch_size=64,
                  substitutions=Substitutions(spans={g: (none, none) for g in GENES}, classifier_spans=True))
    assert seam.airr_rows == results["baseline"].rows
    # true calls make every applicable call metric perfect
    for name in ("v.top1_in_set", "j.top1_in_set"):
        values = [s[name] for s in results["true_calls"].scores.values() if name in s]
        assert values and all(v == 1.0 for v in values)
    out = stage_contributions(results, ["junction.nt_exact", "v.top1_in_set"])
    assert [e["to"] for e in out] == list(VARIANTS[1:])


def test_variants_with_excluded_cases_run_one_forward_pass_per_batch(make_shaped_model, s0_cases, s0_truths):
    model, reference_set = make_shaped_model("igh")
    ids, seqs = _ids_seqs(s0_cases)
    truths = [replace(t, excluded="non_biological_orientation") if k % 3 == 0 else t for k, t in enumerate(s0_truths)]
    baseline = replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH)
    calls = []
    handle = model.orientation_head.register_forward_hook(lambda *_: calls.append(1))  # once per forward pass
    try:
        for name in ("true_orientation", "true_germline"):
            substitutions, _ = substitutions_for(name, truths, baseline)
            assert None not in substitutions.orientation
            calls.clear()
            replay(model, reference_set, seqs, ids, device="cpu", batch_size=BATCH, substitutions=substitutions)
            assert len(calls) == math.ceil(len(ids) / BATCH), name
    finally:
        handle.remove()
