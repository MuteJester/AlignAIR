"""A read longer than the model's window is cut to its first bases before it is oriented (``length_cropped``). When
the cut falls inside the J, the J the read keeps is a fragment: its call, junction and FWR4 rest on it, so the row is
partial (``j_cropped``) and the usual filter on ``airr_assembly_status`` drops it.

The cut is at the end of the read as given. In the model's forward frame it is the read's end for a forward or a
complemented read, and its start for a reverse-complemented or a reversed one (which lose bases before their V, not
their J). The J reaches the cut when its alignment runs to that end of the read.
"""
import pytest
from tests.alignair.predict.airr.test_cdr3_coordinates import (
    CYS,
    J_ANCHOR,
    J_GERM,
    NP,
    V_GERM,
    V_GERM_START,
    _reference,
)

from alignair.predict.airr import build_airr

pytestmark = pytest.mark.component

V_PART = V_GERM[V_GERM_START:]
TAIL = "ACGTTGCA" * 3


def _read(j_bases: int, *, tail: str = "", cropped: bool = True, orientation: int = 0, j_germline=None):
    """A read of V1 and the first ``j_bases`` of J1 (all of J1 for ``len(J_GERM)``), then ``tail``, its J aligned to
    its germline's first ``j_bases`` bases (or to ``j_germline``, a ``(start, end, cigar)``); ``cropped`` and
    ``orientation`` as the input policy and orientation head gave them."""
    from tests.support.read_results import make_read

    from alignair.predict.results import Anchors
    from alignair.predict.state import GermlineAlignment
    j_start = len(V_PART) + len(NP)
    read = V_PART + NP + J_GERM[:j_bases] + tail
    germ_start, germ_end, cigar = j_germline or (0, j_bases, f"{j_bases}M")
    j_anchor = j_start + J_ANCHOR if j_bases >= J_ANCHOR + 3 else None
    return make_read(read, calls={"v": "V1*01", "j": "J1*01"}, anchors=Anchors(CYS - V_GERM_START, j_anchor),
                     cropped=cropped, orientation_id=orientation,
                     alignments={"v": GermlineAlignment("V1*01", 0, len(V_PART), V_GERM_START, len(V_GERM),
                                                        f"{len(V_PART)}M"),
                                 "j": GermlineAlignment("J1*01", j_start, j_start + j_bases, germ_start, germ_end,
                                                        cigar)})


def _status(read) -> tuple[str, str | None]:
    row = build_airr([read], _reference(), has_d=False)[0]
    return row["airr_assembly_status"], row.get("airr_assembly_reason")


@pytest.mark.parametrize("orientation", [0, 2], ids=["forward", "complement"])
def test_a_read_cut_inside_its_j_is_partial(orientation):
    """The J runs to the cut at the read's end and its germline goes on: whether or not the junction survives (here
    the cut is past the J Trp codon), the row is partial."""
    assert _status(_read(J_ANCHOR + 5, orientation=orientation)) == ("partial", "j_cropped")


def test_the_same_read_uncut_is_complete():
    assert _status(_read(J_ANCHOR + 5, cropped=False)) == ("complete", None)


@pytest.mark.parametrize("orientation", [1, 3], ids=["reverse_complement", "reverse"])
def test_a_read_cut_before_its_v_keeps_its_j(orientation):
    """A reverse-complemented or reversed read loses the bases before its V, not its J: its row stays complete."""
    assert _status(_read(J_ANCHOR + 5, orientation=orientation)) == ("complete", None)


def test_a_cut_read_whose_j_ends_before_the_cut_is_complete():
    """The J ends inside the read, the cut falls after it (in the constant region of a 10x contig, say)."""
    assert _status(_read(len(J_GERM), tail=TAIL)) == ("complete", None)


def test_a_read_cut_before_its_j_where_a_scrap_aligns_as_j_is_partial():
    """The cut falls before the J; the aligner still stretches two read bases over the J germline, to its end (as on
    long-leader reads cut in their CDR3): the row is partial too, without a junction."""
    scrap = _read(2, j_germline=(0, len(J_GERM), f"1M{len(J_GERM) - 2}D1M"))
    assert _status(scrap) == ("partial", "j_cropped")
    assert _status(_read(2, j_germline=(0, len(J_GERM), f"1M{len(J_GERM) - 2}D1M"), cropped=False))[1] is None
