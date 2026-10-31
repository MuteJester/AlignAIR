"""Non-germline sequence around a read's V(D)J, as real libraries carry it.

GenAIRR simulates the rearrangement alone: its reads start in V (or a truncated V) and end in J. A 5'RACE read
starts in the 5' UTR and leader; a multiplex read starts at its V primer, masked to N; a read may end in the
constant region. ``flanked`` adds such sequence to a GenAIRR record and moves every read coordinate with the read
(``READ_COORDINATES``; germline coordinates do not move), so the truth built from the record holds for the flanked
read. ``masked`` turns the first bases of the read to N, as a primer mask does, over flank and V alike: the truth
reads N as a mismatch, so a masked V start is not observable. ``Flanks`` draws them for a stratum.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

READ_COORDINATES: tuple[str, ...] = tuple(f"{g}_{kind}_{edge}" for g in "vdj" for kind in ("sequence", "alignment")
                                          for edge in ("start", "end")) + ("junction_start", "junction_end")
BASES = "ACGT"


def flanked(record: dict, prefix: str = "", suffix: str = "") -> dict:
    """A copy of ``record`` whose read is ``prefix`` + read + ``suffix``: each read coordinate (``READ_COORDINATES``,
    those that are set) moves by ``len(prefix)``, ``sequence_length`` is the new read's, and ``flank_5_length`` /
    ``flank_3_length`` record the flanks."""
    out = dict(record)
    shift = len(prefix)
    out["sequence"] = prefix + record["sequence"] + suffix
    out["sequence_length"] = len(out["sequence"])
    for key in READ_COORDINATES:
        if out.get(key) is not None:
            out[key] = out[key] + shift
    out["flank_5_length"], out["flank_3_length"] = shift, len(suffix)
    return out


def masked(record: dict, n: int) -> dict:
    """A copy of ``record`` whose first ``n`` read bases are N (``mask_5_length``); no coordinate moves."""
    out = dict(record)
    out["sequence"] = "N" * n + record["sequence"][n:]
    out["mask_5_length"] = n
    return out


@dataclass(frozen=True)
class Flanks:
    """A stratum's flanks: a random-base ``prefix`` and ``suffix`` with lengths drawn uniformly from their inclusive
    ``(low, high)`` ranges (None: no flank on that side), then, with ``mask``, the prefix and the next 0 to ``mask``
    bases drawn from its range masked to N (a primer). Records gain ``flank_5_length``, ``flank_3_length`` and
    ``mask_5_length`` (0 when absent)."""

    prefix: tuple[int, int] | None = None
    suffix: tuple[int, int] | None = None
    mask: tuple[int, int] | None = None

    def __call__(self, record: dict, rng: random.Random) -> dict:
        def bases(span):
            return "" if span is None else "".join(rng.choice(BASES) for _ in range(rng.randint(*span)))

        prefix, suffix = bases(self.prefix), bases(self.suffix)
        out = flanked(record, prefix, suffix)
        extra = 0 if self.mask is None else rng.randint(*self.mask)
        return masked(out, len(prefix) + extra) if self.mask is not None else {**out, "mask_5_length": 0}
