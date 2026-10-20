"""Generative-allele truth: the primary allele is the one GenAIRR sampled (``truth_{g}_call``), not its post-SHM
re-call (``{g}_call``), and the coordinates are decoded against it."""
import pytest

ga = pytest.importorskip("GenAIRR")

from alignair_assay.core.cigar import format_cigar  # noqa: E402
from alignair_assay.core.jsonl import write_jsonl  # noqa: E402
from alignair_assay.core.cases import read_cases  # noqa: E402
from alignair_assay.core.layout import RunLayout  # noqa: E402
from alignair_assay.core.prediction import PredSegment  # noqa: E402
from alignair_assay.core.seqops import revcomp  # noqa: E402
from alignair_assay.metrics import is_informational, lower_is_better  # noqa: E402
from alignair_assay.metrics.alleles import call_metrics  # noqa: E402
from alignair_assay.suites.donors import sample_donors  # noqa: E402
from alignair_assay.truth.alleles import AlleleTruth  # noqa: E402
from alignair_assay.truth.build import truth_from_genairr_record  # noqa: E402
from alignair_assay.truth.genairr import observe_forward, observe_inverted_d  # noqa: E402
from alignair_assay.truth.io import read_truth_jsonl, truth_to_dict  # noqa: E402
from alignair_assay.truth.segment import check_invariants, observe_segment  # noqa: E402
from alignair_assay.truth.summary import truth_summary  # noqa: E402
from alignair_assay.workflow.build import build_suite  # noqa: E402

CONFIG = ga.data.HUMAN_IGH_OGRDB


def _run(rate, n, seed, *, invert_d=False, provenance=True):
    experiment = ga.Experiment.on(CONFIG).recombine()
    if invert_d:
        experiment = experiment.invert_d(prob=1.0)
    return experiment.mutate(rate=rate).run_records(n=n, seed=seed, expose_provenance=provenance)


def _truths(records, reference):
    return [truth_from_genairr_record(r, reference, case_id=f"rec{i}", stratum="test")
            for i, r in enumerate(records)]


def _calls(record, g):
    """GenAIRR's post-SHM call list: ``recalled_{g}_call`` in its generating-reference encoding, ``{g}_call``
    before it."""
    field = f"recalled_{g}_call" if f"recalled_{g}_call" in record else f"{g}_call"
    return [c for c in (record.get(field) or "").split(",") if c]


def _projecting(record) -> bool:
    """True for a record of an encoding that projects a re-called segment's alignment onto the re-call."""
    return "recalled_v_call" not in record


def _calls_in_v_call(records):
    """``records`` with GenAIRR's post-SHM call list in ``{g}_call`` and no truth or re-call columns: how an encoding
    before the generating-reference one presents a record simulated without provenance."""
    return [{**{k: v for k, v in r.items() if not k.startswith(("truth_", "recalled_"))},
             **{f"{g}_call": ",".join(_calls(r, g)) for g in "vdj"}} for r in records]


def _without_provenance(records):
    return [{k: v for k, v in r.items() if not k.startswith("truth_")} for r in records]


def _observed(seg):
    return seg.seq_start, seg.seq_end, seg.germ_start, seg.germ_end, seg.cigar


def _generative_alignment(outcome, read, allele_seq, allele, segment, *, inverted=False):
    """The segment's generative alignment straight from GenAIRR's IR: each pool position of its region with the
    source-allele position it came from (None: inserted), in the reverse-complemented allele's frame when
    ``inverted``. Valid for records without end loss, whose pool positions are read positions."""
    sim = outcome.final_simulation()
    region = next(r for r in sim.regions() if r.segment == segment)
    n = len(allele_seq)
    columns = []
    for p in range(region.start, region.end):
        position = sim.germline_position(p)
        columns.append((p, None if position is None else (n - 1 - position if inverted else position)))
    aligned = [k for k, (_, position) in enumerate(columns) if position is not None]
    ops, previous = [], None
    for p, position in columns[aligned[0]:aligned[-1] + 1]:
        if position is None:
            ops.append((1, "I"))
            continue
        if previous is not None and position > previous + 1:
            ops.append((position - previous - 1, "D"))
        ops.append((1, "M"))
        previous = position
    p0, g0 = columns[aligned[0]]
    frame = revcomp(allele_seq) if inverted else allele_seq
    return observe_segment(read, frame, allele, p0, g0, format_cigar(ops))


def test_primary_is_generative_allele(reference):
    records = _run(0.2, 600, 3).records
    recalls = 0
    for record, truth in zip(records, _truths(records, reference)):
        for g in "vdj":
            generative = record.get(f"truth_{g}_call")
            if not generative:
                continue
            if not reference.has(generative):
                assert g not in truth.segments and g in truth.unscorable
                continue
            if g in truth.segments:
                assert truth.segments[g].alleles.primary == generative
                assert truth.segments[g].observed.allele == generative
                assert not truth.segments[g].alleles.primary_fallback
                recalls += generative not in _calls(record, g)
    assert recalls > 0  # the sample has post-SHM re-calls, so the test exercises them


def test_recalled_means_first_call_is_not_generative(reference):
    records = _run(0.2, 600, 3).records
    truths = _truths(records, reference)
    recalled = {g: 0 for g in "vdj"}
    for record, truth in zip(records, truths):
        for g, segment in truth.segments.items():
            calls = _calls(record, g)
            assert segment.alleles.recalled == (bool(calls) and calls[0] != record[f"truth_{g}_call"])
            recalled[g] += segment.alleles.recalled
    counts = truth_summary(truths)["simulator_calls"]
    assert {g: counts[g]["recalled"] for g in "vdj"} == recalled and recalled["v"] > 0
    # Most call lists that differ from the generative set are ties, not re-calls.
    assert all(counts[g]["recalled"] <= counts[g]["call_list_differs"] for g in "vdj")
    assert counts["v"]["call_list_differs"] > recalled["v"]


def test_recall_in_scored_set_and_generative_in_generative_set(reference):
    records = _run(0.2, 600, 3).records
    outside = 0
    for record, truth in zip(records, _truths(records, reference)):
        for g, segment in truth.segments.items():
            calls, generative = _calls(record, g), record[f"truth_{g}_call"]
            alleles = segment.alleles
            assert alleles.generative <= alleles.alleles
            assert generative in alleles.generative
            if calls[0] == generative:
                continue
            assert calls[0] in alleles.alleles
            assert alleles.disagrees_with_simulator
            # The re-call is in the generative set only by the substring rule: it shows the same observed germline
            # (GenAIRR re-calls over its own window, which can reach past the observed span).
            recall = reference.seq(calls[0])
            frame = revcomp(recall) if segment.inverted else recall
            assert (calls[0] in alleles.generative) == (segment.observed.observed_germline in frame)
            outside += calls[0] not in alleles.generative
    assert outside > 0


def test_invariants_hold_against_generative_allele(reference):
    records = _run(0.15, 1000, 9).records
    for record, truth in zip(records, _truths(records, reference)):
        assert not truth.warnings and not truth.unscorable, (record["sequence_id"], truth.warnings)
        for g, segment in truth.segments.items():
            allele = reference.seq(segment.alleles.primary)
            frame = revcomp(allele) if segment.inverted else allele
            assert check_invariants(truth.sequence, frame, segment.observed) == []


def test_fallback_without_provenance(reference):
    records = _run(0.15, 200, 4).records
    # A record naming only GenAIRR's post-SHM calls (an encoding before the generating-reference one, simulated
    # without provenance) falls back to the first call.
    bare = _calls_in_v_call(records)
    truths = _truths(bare, reference)
    for record, truth in zip(bare, truths):
        for g, segment in truth.segments.items():
            assert segment.alleles.primary == _calls(record, g)[0]
            assert segment.alleles.primary_fallback
    summary = truth_summary(truths)
    assert summary["primary_fallbacks"] == {g: sum(g in t.segments for t in truths) for g in "vdj"}
    assert summary["primary_fallbacks"]["v"] > 0
    assert truth_summary(_truths(records, reference))["primary_fallbacks"] == {"v": 0, "d": 0, "j": 0}
    # The generating-reference encoding names the generative allele in {g}_call even without provenance.
    if not _projecting(records[0]):
        plain = _truths(_without_provenance(records), reference)
        assert truth_summary(plain)["primary_fallbacks"] == {"v": 0, "d": 0, "j": 0}
        for record, truth in zip(records, plain):
            for g, segment in truth.segments.items():
                assert segment.alleles.primary == record[f"truth_{g}_call"] == record[f"{g}_call"]


def test_provenance_does_not_change_sequences(reference):
    keys = {"truth_v_call", "truth_d_call", "truth_j_call"}
    plain = _run(0.15, 300, 5, provenance=False).records
    with_truth = _run(0.15, 300, 5).records
    assert all(set(b) - set(a) == keys for a, b in zip(plain, with_truth))
    assert _without_provenance(with_truth) == plain

    genotypes = [d.genotype(CONFIG) for d in sample_donors(reference, 2, 13)]
    cohort = ga.Experiment.on(CONFIG).recombine().mutate(rate=0.075)
    a = cohort.run_cohort(genotypes, n_per_subject=20, seed=6).records
    b = cohort.run_cohort(genotypes, n_per_subject=20, seed=6, expose_provenance=True).records
    assert _without_provenance(b) == a and all(keys <= set(r) for r in b)

    clonal = ga.Experiment.on(CONFIG).recombine().clonal_lineage(n_clones=3, n_sample=4, rate=0.05)
    a = clonal.run_records(seed=8).records
    b = clonal.run_records(seed=8, expose_provenance=True).records
    assert _without_provenance(b) == a and all(keys <= set(r) for r in b)


def test_top1_generative_metric():
    truth = AlleleTruth(primary="X*01", alleles=frozenset({"X*01", "X*02"}), genes=frozenset({"X"}),
                        simulator_ties=frozenset({"X*02"}), disagrees_with_simulator=True,
                        generative=frozenset({"X*01"}))
    recall = call_metrics("v", truth, PredSegment(call="X*02", call_set=("X*02",)))
    assert (recall["v.top1_in_set"], recall["v.top1_generative"]) == (1.0, 0.0)
    generative = call_metrics("v", truth, PredSegment(call="X*01", call_set=("X*01",)))
    assert (generative["v.top1_in_set"], generative["v.top1_generative"]) == (1.0, 1.0)
    assert call_metrics("v", truth, PredSegment(call=None))["v.top1_generative"] == 0.0
    # Reported, not compared: a perfect sequence-only aligner cannot reach the generative allele when SHM has made
    # the read a sibling's, so it stays out of the Holm family.
    assert not lower_is_better("v.top1_generative")
    assert all(is_informational(f"{g}.top1_generative") for g in "vdj") and not is_informational("v.top1_in_set")


def test_old_truth_file_generative_defaults_to_alleles(tmp_path, reference):
    records = _run(0.2, 100, 3).records
    truths = _truths(records, reference)
    rows = []
    for truth in truths:
        row = truth_to_dict(truth)
        for segment in row["segments"].values():
            assert "generative" in segment["alleles"] and "primary_fallback" in segment["alleles"]
            del segment["alleles"]["generative"], segment["alleles"]["primary_fallback"]
        rows.append(row)
    path = tmp_path / "old.jsonl"
    write_jsonl(rows, path)
    for old, new in zip(read_truth_jsonl(path), truths):
        for g, segment in old.segments.items():
            assert segment.alleles.generative == segment.alleles.alleles
            assert segment.alleles.primary_fallback  # files written before the generative primary used calls[0]
            assert segment.alleles.alleles == new.segments[g].alleles.alleles


def test_engine_records_carry_provenance(tmp_path):
    layout = RunLayout(tmp_path)
    build = build_suite("S0", layout, reference_name="HUMAN_IGH_OGRDB", split="dev", seed=11, scale=0.2,
                        workers=1)
    cases = read_cases(layout.cases("S0"))
    assert cases and all("truth_v_call" in case["record"] for case in cases)
    assert build["truth"]["primary_fallbacks"] == {"v": 0, "d": 0, "j": 0}


def test_recall_alignment_clipped_to_generative_structural_range(reference):
    """Every re-called segment decodes to its generative alignment. Encodings before the generating-reference one
    project a re-called segment's CIGAR edges onto the re-called allele (NP claims scored against it; GenAIRR 2.3.1
    also filled D to its length, the assembled encoding reports read bases past a shorter re-call's end as
    insertions) while the core alignment is the generative one: decoded unclipped, some segments score extra
    mismatches (and, on 2.3.1, some are dropped: pinned by test_record_encodings' old-engine fixture). The
    generating-reference encoding aligns against the generative allele, so it has nothing to undo."""
    result = _run(0.15, 1500, 9)
    misscored = recalls = 0
    for record, outcome, truth in zip(result.records, result.outcomes, _truths(result.records, reference)):
        read = record["sequence"].upper()
        for g in "vdj":
            generative = record[f"truth_{g}_call"]
            if not generative or generative in _calls(record, g) or not reference.has(generative):
                continue
            recalls += 1
            allele = reference.seq(generative)
            expected = _generative_alignment(outcome, read, allele, generative, g.upper())
            assert g in truth.segments, (record["sequence_id"], g, truth.warnings)
            assert _observed(truth.segments[g].observed) == _observed(expected)
            assert truth.segments[g].observed.n_mismatch == expected.n_mismatch
            problems = []
            try:
                unclipped = observe_forward(read, allele, generative, record, g, problems)
            except IndexError:
                unclipped = None
            if unclipped is not None and unclipped.n_mismatch > expected.n_mismatch:
                misscored += 1
    assert recalls > 0
    if _projecting(result.records[0]):
        assert misscored > 0
    else:
        assert misscored == 0


def test_inverted_d_recall_decoded_against_generative_allele(reference):
    result = _run(0.15, 1500, 9, invert_d=True)
    recalls = 0
    for record, outcome, truth in zip(result.records, result.outcomes, _truths(result.records, reference)):
        generative = record["truth_d_call"]
        if not record["d_inverted"] or generative in _calls(record, "d"):
            continue
        recalls += 1
        expected = _generative_alignment(outcome, record["sequence"].upper(), reference.seq(generative), generative,
                                         "D", inverted=True)
        assert "d" in truth.segments, (record["sequence_id"], truth.warnings)
        assert not truth.warnings and not truth.unscorable, (record["sequence_id"], truth.warnings, truth.unscorable)
        segment = truth.segments["d"]
        assert segment.inverted and segment.alleles.primary == generative
        assert _observed(segment.observed) == _observed(expected)
    assert recalls > 0


def test_inverted_d_recall_with_undetermined_span_is_unscorable():
    """A re-called inverted D whose fills stop at the shorter re-called allele's end does not determine its span:
    the decoder drops it with a specific problem instead of guessing."""
    allele = "ACGTACGTACGTACGTAC"  # 18 nt, forward
    read = revcomp(allele[2:16])  # forward positions 15..2, read left to right
    # GenAIRR's walker against a 12-nt re-call (trims 2/2): it fills D from the 5' trim up to the highest position
    # but stops at the re-called allele's end (2..11, 10 D), walks the 14 bases (15 down to 2), then fills from 3 up
    # to the re-call's 3' trim (3..9, 7 D). The highest position (15) is lost.
    record = {"d_trim_5": 2, "d_trim_3": 2, "d_sequence_start": 0, "d_sequence_end": 14,
              "d_germline_start": 2, "d_germline_end": 16, "d_cigar": "10D14M7D"}
    problems = []
    assert observe_inverted_d(read, allele, "X*01", record, problems, projected_length=12) is None
    assert problems == ["inverted D: re-called by the simulator, and its alignment, projected onto the re-called "
                        "allele, does not determine the observed span; segment dropped"]
    # The same read projected onto the generative allele itself (13 D, 14 M, 13 D) decodes exactly.
    problems = []
    segment = observe_inverted_d(read, allele, "X*01", dict(record, d_cigar="13D14M13D"), problems)
    assert problems == [] and segment is not None
    assert (segment.seq_start, segment.germ_start, segment.germ_end, segment.cigar, segment.n_mismatch) == \
        (0, 2, 16, "14M", 0)
