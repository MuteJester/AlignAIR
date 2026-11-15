"""The four orientations a read can come in, in the order of the orientation head's classes. Each is an involution:
applying a read's orientation to it again gives the forward read."""
from __future__ import annotations

ORIENTATIONS = ("forward", "reverse_complement", "complement", "reverse")
FORWARD, REVERSE_COMPLEMENT, COMPLEMENT, REVERSE = range(len(ORIENTATIONS))
NUM_ORIENTATIONS = len(ORIENTATIONS)

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def orient(sequence: str, orientation: int) -> str:
    """``sequence`` under ``orientation`` (an index of ``ORIENTATIONS``); a character that is not a base or N stays."""
    if orientation == FORWARD:
        return sequence
    if orientation == REVERSE_COMPLEMENT:
        return reverse_complement(sequence)
    if orientation == COMPLEMENT:
        return sequence.translate(_COMPLEMENT)
    if orientation == REVERSE:
        return sequence[::-1]
    raise ValueError(f"orientation {orientation!r} is not one of 0..{NUM_ORIENTATIONS - 1}")
