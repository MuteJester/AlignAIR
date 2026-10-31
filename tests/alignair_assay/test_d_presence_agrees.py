"""AlignAIR's D presence classes (``alignair.core.d_presence``, the target ``alignair.data.targets.read_d_presence``)
are the assay's D observability: absent when the truth has no observed D, too short when its observed length is below
``MIN_OBSERVED["d"]`` (the assay then scores only abstention, ``d_short.*``), callable otherwise (``d.*``)."""
import itertools

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.metrics.case import MIN_OBSERVED  # noqa: E402
from alignair_assay.truth.build import truth_from_genairr_record  # noqa: E402


def _truth_class(truth):
    from alignair.core.d_presence import ABSENT, CALLABLE, SHORT
    segment = truth.segments.get("d")
    if segment is None:
        return ABSENT
    return SHORT if segment.observed_length < MIN_OBSERVED["d"] else CALLABLE


@pytest.mark.component
def test_a_d_is_callable_from_the_assays_minimum_observed_length():
    from alignair.core.d_presence import D_CALLABLE_MIN
    assert D_CALLABLE_MIN == MIN_OBSERVED["d"]


@pytest.mark.component
@pytest.mark.parametrize("cigar", ["8M", "7M", "3M1D4M", "3M1I4M", "1D7M", "7M1D", "1I7M", "7M2I", "2D3M1D3M",
                                   "4M3I4M", "3I", "1M", "1D1M1D"])
def test_d_presence_is_the_truths_observed_length_on_any_d_alignment(cigar):
    """Every kind of D alignment GenAIRR can report, edge gaps, internal indels and alignments with one or no aligned
    base: the product's class is the class of the truth's own observed segment (``truth.segment.observe_segment``).
    Live records carry few D indels (19 in 3,250), and none changes a class, so these cases pin the rule."""
    from alignair.core.d_presence import ABSENT, CALLABLE, SHORT
    from alignair.data.targets import read_d_presence
    from alignair_assay.core.cigar import parse_cigar
    from alignair_assay.truth.segment import observe_segment
    ops = parse_cigar(cigar)
    read_length = sum(n for n, op in ops if op in "MI")
    germline_length = sum(n for n, op in ops if op in "MD")
    read, germline, start, germline_start = "ACGT" * 10, "ACGT" * 10, 5, 3
    record = {"sequence": read, "d_call": "D*01", "d_cigar": cigar, "d_sequence_start": start,
              "d_sequence_end": start + read_length, "d_germline_start": germline_start,
              "d_germline_end": germline_start + germline_length}
    segment = observe_segment(read, germline, "D*01", start, germline_start, cigar)
    # the truth's observed length (truth.build: SegmentTruth.observed_length) is the observed germline span
    want = (ABSENT if segment is None else
            SHORT if segment.germ_end - segment.germ_start < MIN_OBSERVED["d"] else CALLABLE)
    assert read_d_presence(record) == want


@pytest.mark.integration
@pytest.mark.parametrize("inverted,counts", [(False, (604, 89, 1507)), (True, (306, 19, 725))],
                         ids=["curriculum", "inverted_d"])
def test_d_presence_agrees_with_the_assays_observability(reference, inverted, counts):
    """On live HUMAN_IGH_OGRDB records of every component of the default training mix (300 from each first
    unselected stream and 25 from each full-length component's first selected stream: 2,200; or, with every D
    inverted, 150 from each first unselected stream: 1,050), the product's class equals the truth's on every record,
    and no record's D is unscorable. An inverted D is a present D, measured on its reverse-complemented germline, as
    the truth observes it. The counts are (absent, short, callable)."""
    import GenAIRR.data as gd
    from alignair.data.targets import read_d_presence
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    streams = []
    for component, substreams in enumerate(CurriculumStreamPolicy().substreams_for(gd.HUMAN_IGH_OGRDB)):
        for name, parameters in substreams:
            if inverted and name == "unselected-0":
                streams.append((f"{component}:{name}", {**parameters, "invert_d_prob": 1.0}, 150, 9000 + component))
            elif not inverted and name in ("unselected-0", "selected-0"):
                streams.append((f"{component}:{name}", parameters, 300 if name.startswith("unselected") else 25,
                                7000 + component))
    found, disagree = [0, 0, 0], []
    for stream, parameters, count, seed in streams:
        for k, record in enumerate(itertools.islice(stream_records(gd.HUMAN_IGH_OGRDB, parameters, seed), count)):
            truth = truth_from_genairr_record(record, reference, case_id=f"{stream}:{k}", stratum=stream)
            assert "d" not in truth.unscorable, (truth.case_id, truth.warnings)
            assert not inverted or truth.segments.get("d") is None or truth.segments["d"].inverted
            want, got = _truth_class(truth), read_d_presence(record)
            found[want] += 1
            if want != got:
                disagree.append((truth.case_id, want, got, record.get("d_cigar")))
    assert not disagree, disagree
    assert tuple(found) == counts
