"""Library flanks: the non-germline sequence real reads carry around V(D)J, added to simulated training records.

GenAIRR simulates the rearrangement alone, so its reads start in V and end in J; a model trained on them alone learns
that V starts at the read's first base and that nothing follows J. A 5'RACE read starts in the 5' UTR and leader, a
multiplex read starts at its V primer masked to N, and a read may end in the constant region. A 5'RACE consensus
read's leader also holds uncalled bases (N) here and there (about 12% of the reads of a real 5'RACE repertoire, up to 20
or more); a model that has only seen N as one masked run before V reads them as a primer and starts V among them.
``add_flanks`` puts such sequence around a record's read and moves every read coordinate with it
(``READ_COORDINATES``; germline coordinates stay), so the record's region, boundary and alignment targets hold for
the flanked read. A mask reaching into V starts V at its first unmasked base (its germline start and CIGAR move with
it), as the bases under N are not observable. ``flank_records`` draws flanks for a record stream under a
``FlankPolicy``, reproducibly from the stream's seed.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from ..core.tokens import DEFAULT_WINDOW
from ..errors import ConfigurationError

READ_COORDINATES: tuple[str, ...] = tuple(f"{g}_{kind}_{edge}" for g in "vdj" for kind in ("sequence", "alignment")
                                          for edge in ("start", "end")) + ("junction_start", "junction_end")
BASES = np.array(list("ACGT"))
_SALT = 0x464C414E                     # the flank generator's own seed part, apart from the stream's other draws
_LEADING_MATCH = re.compile(r"^(\d+)M")


def add_flanks(record: Mapping, *, prefix: str = "", suffix: str = "", mask: int = 0) -> dict:
    """A copy of ``record`` whose read is ``prefix`` + read + ``suffix``, its first ``mask`` bases then turned to N.

    Each read coordinate that is set moves by ``len(prefix)``; ``sequence_length`` is the new read's. A mask reaching
    past V's start starts V at the first unmasked base (``v_sequence_start``, ``v_alignment_start`` and
    ``v_germline_start`` move, the CIGAR's leading match shortens) when V opens with a match longer than the masked
    part; otherwise the mask stops at V's start. The record gains ``flank_5_length``, ``flank_3_length`` and
    ``mask_5_length``."""
    out = dict(record)
    shift = len(prefix)
    out["sequence"] = prefix + record["sequence"] + suffix
    out["sequence_length"] = len(out["sequence"])
    for key in READ_COORDINATES:
        if out.get(key) is not None:
            out[key] = out[key] + shift
    v_start = out.get("v_sequence_start")
    if mask and v_start is not None and mask > v_start:
        into_v = mask - v_start
        leading = _LEADING_MATCH.match(out.get("v_cigar") or "")
        if leading and int(leading.group(1)) > into_v:
            out["v_cigar"] = f"{int(leading.group(1)) - into_v}M" + out["v_cigar"][leading.end():]
            for key in ("v_sequence_start", "v_alignment_start", "v_germline_start"):
                if out.get(key) is not None:
                    out[key] = out[key] + into_v
        else:
            mask = v_start
    out["sequence"] = "N" * mask + out["sequence"][mask:]
    out["flank_5_length"], out["flank_3_length"], out["mask_5_length"] = shift, len(suffix), mask
    return out


def _span(value, field: str) -> tuple[int, int]:
    try:
        low, high = (int(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a (low, high) pair of lengths", context={"field": field}) from exc
    if not 0 <= low <= high:
        raise ConfigurationError(f"{field} must satisfy 0 <= low <= high", context={"field": field})
    return low, high


@dataclass(frozen=True)
class FlankPolicy:
    """Which reads get flanks and how long: a random-base 5' flank on ``p_prefix`` of the reads (length uniform in
    ``prefix``), masked to N on ``p_mask`` of those together with the next 0 to ``mask_extra`` read bases (a primer
    over the first V bases), and a random-base 3' flank on ``p_suffix`` of the reads (length in ``suffix``). On
    ``p_flank_n`` of the unmasked 5' flanks each flank base turns to N at a rate uniform in ``flank_n_rate``
    (uncalled bases in a leader; the read and its coordinates are untouched). Lengths are inclusive; a read keeps its
    own bases whole and grows to at most ``max_length``, its flanks shortened to fit (3' first)."""

    p_prefix: float = 0.4
    prefix: tuple[int, int] = (1, 140)
    p_mask: float = 0.4
    mask_extra: tuple[int, int] = (0, 12)
    p_suffix: float = 0.3
    suffix: tuple[int, int] = (1, 100)
    max_length: int = DEFAULT_WINDOW
    p_flank_n: float = 0.3
    flank_n_rate: tuple[float, float] = (0.0, 0.15)

    def __post_init__(self):
        for field in ("p_prefix", "p_mask", "p_suffix", "p_flank_n"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{field} must be a probability in [0, 1]", context={"field": field})
            object.__setattr__(self, field, float(value))
        for field in ("prefix", "mask_extra", "suffix"):
            object.__setattr__(self, field, _span(getattr(self, field), field))
        try:
            low, high = (float(x) for x in self.flank_n_rate)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("flank_n_rate must be a (low, high) pair of rates",
                                     context={"field": "flank_n_rate"}) from exc
        if not 0.0 <= low <= high <= 1.0:
            raise ConfigurationError("flank_n_rate must satisfy 0 <= low <= high <= 1",
                                     context={"field": "flank_n_rate"})
        object.__setattr__(self, "flank_n_rate", (low, high))
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length < 1:
            raise ConfigurationError("max_length must be a positive integer", context={"field": "max_length"})

    def to_dict(self) -> dict:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping) -> FlankPolicy:
        return cls(**{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})


def flank_records(records: Iterable[Mapping], policy: FlankPolicy, seed: int) -> Iterator[dict]:
    """``records`` with flanks drawn under ``policy`` from a generator of ``seed`` (the same seed, the same flanks);
    every record gains ``flank_5_length``, ``flank_3_length`` and ``mask_5_length``, 0 when it has none."""
    rng = np.random.default_rng([int(seed), _SALT])
    for record in records:
        prefix = int(rng.integers(policy.prefix[0], policy.prefix[1] + 1)) if rng.random() < policy.p_prefix else 0
        masked = bool(prefix) and rng.random() < policy.p_mask
        extra = int(rng.integers(policy.mask_extra[0], policy.mask_extra[1] + 1)) if masked else 0
        suffix = int(rng.integers(policy.suffix[0], policy.suffix[1] + 1)) if rng.random() < policy.p_suffix else 0
        room = max(0, policy.max_length - len(record["sequence"]))
        prefix = min(prefix, room)
        suffix = min(suffix, room - prefix)
        head = BASES[rng.integers(0, 4, prefix)]
        if prefix and not masked and rng.random() < policy.p_flank_n:
            rate = rng.uniform(*policy.flank_n_rate)
            head = np.where(rng.random(prefix) < rate, "N", head)
        yield add_flanks(record, prefix="".join(head), suffix="".join(BASES[rng.integers(0, 4, suffix)]),
                         mask=prefix + extra if masked and prefix else 0)
