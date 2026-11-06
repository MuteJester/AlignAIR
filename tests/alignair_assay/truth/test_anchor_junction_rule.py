"""AlignAIR's anchor rule (``alignair.predict.junction.derive_junction``) against GenAIRR's records and the assay's
truth, with the true anchors and spans, on live records from every component of AlignAIR's training mix.

The records: HUMAN_IGH_OGRDB, the first unselected stream of each of the default ``CurriculumStreamPolicy``'s
components (the full-length curriculum components, the three read-length amplicons, whose truncated reads miss an
anchor, and heavy SHM; 300 records each) and the first selected stream of each full-length component (25 each,
GenAIRR's ``productive_only()``). They are drawn live (no record cache) from fixed seeds, so the counts below are
exact.
"""
import itertools

import pytest

pytest.importorskip("GenAIRR")

from alignair_assay.truth.build import truth_from_genairr_record  # noqa: E402

pytestmark = pytest.mark.integration

UNSELECTED, SELECTED = 300, 25
FLAGS = ("vj_in_frame", "stop_codon", "productive")


@pytest.fixture(scope="module")
def records():
    """``(component, stream, record)`` for every drawn record."""
    import GenAIRR.data as gd
    from alignair.train.gym.streams import CurriculumStreamPolicy, stream_records
    drawn = []
    for component, streams in enumerate(CurriculumStreamPolicy().substreams_for(gd.HUMAN_IGH_OGRDB)):
        for name, parameters in streams:
            if name in ("unselected-0", "selected-0"):
                count = UNSELECTED if name.startswith("unselected") else SELECTED
                stream = stream_records(gd.HUMAN_IGH_OGRDB, parameters, 7000 + component)
                drawn += [(component, name, record) for record in itertools.islice(stream, count)]
    return drawn


def _true_anchors(record):
    start, end = record["junction_start"], record["junction_end"]
    return (None, None) if start is None else (start, end - 3)


def test_the_true_anchors_give_genairrs_junction_on_every_component(records):
    """(a) On every record the junction derived from the true anchors is GenAIRR's ``junction_start`` and
    ``junction_end``, None exactly where GenAIRR's is (an anchor end loss or a trim removed). The amplicons make the
    None case common, and GenAIRR never gives a junction coordinate outside the read."""
    from alignair.predict.junction import derive_junction
    components, unknown = set(), 0
    for component, name, record in records:
        sequence = record["sequence"]
        call = derive_junction(sequence, *_true_anchors(record),
                               coding_span=(record.get("v_sequence_start"), record.get("j_sequence_end")))
        assert (call.junction_start, call.junction_end) == (record["junction_start"], record["junction_end"]), \
            (component, name, record["sequence_id"])
        if record["junction_start"] is None:
            unknown += 1
            assert call.vj_in_frame is call.stop_codon is call.productive is None
        else:
            assert 0 <= record["junction_start"] < record["junction_end"] <= len(sequence)
        components.add(component)
    assert len(records) == 7 * UNSELECTED + 4 * SELECTED and components == set(range(7))
    assert unknown >= 600                       # the V-anchored and fragment amplicons alone miss nearly all


def test_the_flags_agree_with_the_assays_truth_given_the_true_anchors_and_spans(records, reference):
    """(b) With the true anchors and the true coding span (the observed V start and J end), the rule's frame, stop
    and productivity agree with the assay's truth on at least 99.5% of the reads where the truth's junction exists,
    and are all None where it is None. The stop is compared on every such read, though the assay scores it only on
    the frame-unambiguous ones (520 of them here).

    The true anchors are the ends of the truth's junction. They are GenAIRR's on every record but 5, where an indel
    deleted an anchor's own base (V CIGARs such as ``281M1D9M``, J CIGARs such as ``16M1D24M``): GenAIRR still places
    a junction, and the truth, which needs both anchor bases aligned, has none.

    The residue is one kind of read. The assay's truth calls a read with a V or J frameshift (a net indel that is not
    a multiple of 3) unproductive. The anchors cannot see an indel, so the rule calls such a read productive when
    its junction is in frame and stop-free. On these records the frame and the stop agree on every read (1436 of
    1436 each) and the productivity on 1432 of 1436 (99.72%). The 4 others are frameshift reads:
    ``2:unselected-0:799``, ``2:unselected-0:904`` and ``4:unselected-0:1425`` (V), ``4:unselected-0:1284`` (J)."""
    from alignair.predict.junction import derive_junction
    scored = {flag: 0 for flag in FLAGS}
    agree = {flag: 0 for flag in FLAGS}
    residue, frame_unambiguous = [], 0
    for k, (component, name, record) in enumerate(records):
        truth = truth_from_genairr_record(record, reference, case_id=f"{component}:{name}:{k}", stratum=name)
        segments = truth.segments
        span = ((segments["v"].observed.seq_start, segments["j"].observed.seq_end)
                if "v" in segments and "j" in segments else None)
        junction = truth.junction
        anchors = (None, None) if junction is None else (junction.start, junction.end - 3)
        call = derive_junction(truth.sequence, *anchors, coding_span=span)
        if junction is None:
            assert call.vj_in_frame is call.stop_codon is call.productive is None, truth.case_id
            continue
        assert (call.junction_start, call.junction_end) == (junction.start, junction.end)
        frame_unambiguous += truth.frame_unambiguous
        for flag in FLAGS:
            scored[flag] += 1
            agree[flag] += getattr(call, flag) == getattr(truth, flag)
            if getattr(call, flag) != getattr(truth, flag):
                residue.append((truth.case_id, flag, truth.v_frameshift, truth.j_frameshift,
                                getattr(call, flag), getattr(truth, flag)))
    assert scored == {"vj_in_frame": 1436, "stop_codon": 1436, "productive": 1436} and frame_unambiguous == 520
    assert agree["vj_in_frame"] == agree["stop_codon"] == 1436
    assert len(residue) == 4
    for flag in FLAGS:
        assert agree[flag] / scored[flag] >= 0.995, (flag, agree[flag], scored[flag], residue)
    # every disagreement is a frameshift read the rule calls productive and the truth does not
    assert residue and all(flag == "productive" and (v_shift or j_shift) and derived is True and want is False
                           for _, flag, v_shift, j_shift, derived, want in residue), residue


def test_the_anchor_targets_give_the_assays_truth_junction(records, reference):
    """(c) The anchor targets a model with anchor heads trains on (``alignair.data.targets.read_anchors``: each
    anchor through its own segment's alignment) give the truth's junction: where the truth has one, the anchors are its
    start and end - 3; where it has none, at most one anchor is observed, so the derived junction is None too.

    That includes the 5 records where GenAIRR places a junction but an indel deleted an anchor's own base (V CIGARs
    such as ``281M1D9M``, J CIGARs such as ``16M1D24M``): the truth has no junction there, and the deleted anchor is
    null."""
    import GenAIRR.data as gd
    from alignair.data.targets import read_anchors
    from alignair.reference.reference_set import ReferenceSet
    reference_set = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    with_junction = deleted = 0
    for k, (component, name, record) in enumerate(records):
        truth = truth_from_genairr_record(record, reference, case_id=f"{component}:{name}:{k}", stratum=name)
        anchors = read_anchors(record, reference_set)
        if truth.junction is not None:
            with_junction += 1
            assert (anchors["v"], anchors["j"]) == (truth.junction.start, truth.junction.end - 3), truth.case_id
            continue
        assert anchors["v"] is None or anchors["j"] is None, truth.case_id
        if record["junction_start"] is not None:                 # GenAIRR's junction, which the truth refuses
            deleted += 1
            null = "v" if anchors["v"] is None else "j"
            assert "D" in record[f"{null}_cigar"], (truth.case_id, record[f"{null}_cigar"])
    assert (with_junction, deleted) == (1436, 5)
