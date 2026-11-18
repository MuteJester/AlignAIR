"""The junction, frame, stop codon and productivity a read's V and J anchors give by rule (``derive_junction``),
pinned on hand-made reads: the frame comes from the anchors, the stop is searched over the coding span in the
junction's frame, and a missing, clipped or reversed anchor leaves everything unknown."""
import pytest

from alignair.predict.junction import JunctionCall, derive_junction

pytestmark = pytest.mark.component

UNKNOWN = JunctionCall(None, None, None, None, None)

# V: 9 bases, then the Cys codon at 9; N: 6 bases; the Trp codon at 18; J tail: 6 bases. Read length 27.
V, CYS, N, TRP, TAIL = "GCCGCCGCC", "TGT", "GCGAGA", "TGG", "GGCCAG"
READ = V + CYS + N + TRP + TAIL
SPAN = (0, len(READ))


def _derive(read=READ, v_anchor=9, j_anchor=18, span=SPAN):
    return derive_junction(read, v_anchor, j_anchor, coding_span=span)


def test_an_in_frame_junction_without_a_stop_is_productive():
    assert _derive() == JunctionCall(junction_start=9, junction_end=21, vj_in_frame=True, stop_codon=False,
                                     productive=True)


def test_a_stop_in_the_junction_makes_it_unproductive():
    call = _derive(V + CYS + "GCGTAG" + TRP + TAIL)
    assert (call.vj_in_frame, call.stop_codon, call.productive) == (True, True, False)


@pytest.mark.parametrize("read", [
    "GCCTAAGCC" + CYS + N + TRP + TAIL,                           # in V, in the junction's frame
    V + CYS + N + TRP + "TGAGCC",                                 # in J, after the Trp codon
], ids=["in_v", "in_j"])
def test_a_stop_outside_the_junction_but_inside_the_coding_span_is_a_stop(read):
    call = _derive(read)
    assert (call.junction_start, call.junction_end) == (9, 21)
    assert (call.vj_in_frame, call.stop_codon, call.productive) == (True, True, False)


def test_a_stop_in_another_frame_is_not_a_stop():
    assert _derive("GCCGTAAGC" + CYS + N + TRP + TAIL).productive is True    # TAA at 4: the junction's frame is 0


@pytest.mark.parametrize("read,span", [
    ("TAAGCCGCC" + CYS + N + TRP + TAIL, (3, 27)),                # before the V start
    (V + CYS + N + TRP + "TGAGCC", (0, 21)),                       # after the J end
], ids=["before_v_start", "after_j_end"])
def test_a_stop_outside_the_coding_span_is_not_a_stop(read, span):
    call = _derive(read, span=span)
    assert (call.stop_codon, call.productive) == (False, True)


@pytest.mark.parametrize("j_end,stop", [(23, False), (24, True)])
def test_a_codon_counts_only_when_it_lies_wholly_inside_the_span(j_end, stop):
    """The TGA at 21-23 is cut by a J end at 23 and complete with one at 24."""
    assert _derive(V + CYS + N + TRP + "TGAGCC", span=(0, j_end)).stop_codon is stop


def test_the_span_starts_at_the_first_codon_of_the_junctions_frame():
    """A V start at 1 is not on the junction's frame (9 mod 3 = 0): codons are read from 3, so a stop at 0 is
    outside the span and one at 3 inside it."""
    assert _derive("TAAGCCGCC" + CYS + N + TRP + TAIL, span=(1, 27)).stop_codon is False
    assert _derive("GCCTAAGCC" + CYS + N + TRP + TAIL, span=(1, 27)).stop_codon is True


def test_an_out_of_frame_junction_is_not_productive_and_its_stop_is_read_in_the_junctions_frame():
    """One more N base: the junction is 13 long, out of frame; the stop is still searched in the junction's frame
    from the V start (here, the TAA at 3 is found)."""
    read = "GCCTAAGCC" + CYS + N + "A" + TRP + TAIL
    call = _derive(read, j_anchor=19)
    assert call == JunctionCall(9, 22, vj_in_frame=False, stop_codon=True, productive=False)
    assert _derive(V + CYS + N + "A" + TRP + TAIL, j_anchor=19).stop_codon is False


@pytest.mark.parametrize("v_anchor,j_anchor", [(9, 6), (9, 5), (18, 9)], ids=["end_at_start", "before", "swapped"])
def test_reversed_anchors_give_an_unknown_junction(v_anchor, j_anchor):
    """A J anchor whose codon ends at or before the V anchor gives no junction, and no frame, stop or productivity."""
    assert _derive(v_anchor=v_anchor, j_anchor=j_anchor) == UNKNOWN


@pytest.mark.parametrize("v_anchor,j_anchor", [(None, 18), (9, None), (None, None)])
def test_one_missing_anchor_gives_an_unknown_junction(v_anchor, j_anchor):
    assert _derive(v_anchor=v_anchor, j_anchor=j_anchor) == UNKNOWN


@pytest.mark.parametrize("v_anchor,j_anchor", [(-1, 18), (9, 25), (9, 27), (26, 30)],
                         ids=["v_before_read", "j_codon_cut", "j_past_read", "both_past"])
def test_an_anchor_whose_codon_is_not_wholly_in_the_read_is_not_observed(v_anchor, j_anchor):
    """The derived junction is never a coordinate outside the read."""
    assert _derive(v_anchor=v_anchor, j_anchor=j_anchor) == UNKNOWN


def test_an_anchor_whose_codon_ends_the_read_is_observed():
    call = _derive(READ[:21], span=(0, 21))
    assert (call.junction_start, call.junction_end, call.productive) == (9, 21, True)


@pytest.mark.parametrize("span", [None, (None, 27), (0, None)])
def test_an_unknown_coding_span_leaves_the_stop_unknown(span):
    """Without the read's V start or J end the stop is unknown, so an in-frame junction's productivity is unknown;
    an out-of-frame one is unproductive whatever the stop."""
    assert _derive(span=span) == JunctionCall(9, 21, vj_in_frame=True, stop_codon=None, productive=None)
    out = _derive(V + CYS + N + "A" + TRP + TAIL, j_anchor=19, span=span)
    assert out == JunctionCall(9, 22, vj_in_frame=False, stop_codon=None, productive=False)


def test_a_coding_span_past_the_read_is_read_inside_it():
    assert _derive(V + CYS + N + TRP + "TGAGCC", span=(-5, 40)).stop_codon is True


def test_lowercase_reads_are_read_as_uppercase():
    assert _derive(("GCCTAAGCC" + CYS + N + TRP + TAIL).lower()).stop_codon is True
