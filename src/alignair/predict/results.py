"""Each read's prediction, typed (``ReadResult``): per gene its call, alignment, listed alleles and status
(``GeneResult``), and the read's own heads, anchors, flags and locus.

A gate sets a gene's status instead of removing its evidence: a read without a V has its V ``ABSENT``; the D presence
head's class makes a D ``ABSENT`` (no D), ``ABSTAINED`` (a span too short to call: kept, without a call) or
``CALLED``. The call and the alignment stay on the result whatever the status; the reported views apply it:
``GeneResult.call_field``/``names``/``likelihoods`` are empty unless ``CALLED``, and ``placed`` is the alignment
unless ``ABSENT``. ``light_record`` is a read's light record, the dict ``predict_sequences`` returns without the AIRR
assembly, with its keys in their order.
"""
from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..core.anchors import ANCHOR_GENES, ANCHOR_KEYS
from ..core.d_presence import ABSENT, CALLABLE, D_CLASSES, D_PRESENCE
from .clean import NOT_OBSERVED
from .junction import JunctionCall, derive_junction
from .state import GeneCall, GermlineAlignment, PredictionState
from .v_presence import V_ABSENT, V_PRESENCE

_PLACEMENT = ("sequence_start", "sequence_end", "germline_start", "germline_end", "cigar")


class GeneStatus(enum.Enum):
    """What a gate made of a gene's call."""
    CALLED = "called"
    ABSTAINED = "abstained"          # its span is kept, without a call
    ABSENT = "absent"                # the read holds none: no call, no span


@dataclass(frozen=True)
class GeneResult:
    """One gene of one read."""
    call: GeneCall                           # the selector's (or a substituted) set, kept whatever the gate says
    alignment: GermlineAlignment | None      # the germline reader's (V: after better_v), kept likewise
    listed: tuple[str, ...]                  # V: the call and its ties when any, else the call's first name
    status: GeneStatus

    @property
    def call_field(self) -> str:
        """The reported call: the listed alleles, comma-joined, when called; else empty."""
        return ",".join(self.listed) if self.status is GeneStatus.CALLED else ""

    @property
    def names(self) -> tuple[str, ...]:
        """The selector's set when called; else none."""
        return self.call.names if self.status is GeneStatus.CALLED else ()

    @property
    def likelihoods(self) -> tuple[float, ...]:
        """The set's likelihoods when called; else none."""
        return self.call.likelihoods if self.status is GeneStatus.CALLED else ()

    @property
    def placed(self) -> GermlineAlignment | None:
        """The alignment, unless the gene is absent."""
        return None if self.status is GeneStatus.ABSENT else self.alignment


@dataclass(frozen=True)
class Anchors:
    """The read's V Cys and J Trp/Phe anchors (forward-frame read positions; None: not observed)."""
    v: int | None
    j: int | None


@dataclass(frozen=True)
class ReadResult:
    """One read's prediction."""
    sequence: str                            # the read in the model's forward frame
    input_sequence: str                      # the post-crop, pre-orientation read
    orientation_id: int
    mutation_rate: float
    indel_count: float
    genes: Mapping[str, GeneResult]          # v, (d,) j
    anchors: Anchors | None                  # None without anchor heads
    d_presence: str | None                   # the D presence class (D_CLASSES); None without the head or a D
    low_quality: bool | None                 # its V or J collapsed: no feasible segmentation (None: not known)
    cropped: bool                            # cut to the model's window
    locus: str | None                        # None: the read carries no locus label

    @property
    def v_present(self) -> bool:
        return self.genes["v"].status is not GeneStatus.ABSENT

    def junction(self) -> JunctionCall | None:
        """The junction the anchors open over the coding span (the V's start to the J's end), or None without anchor
        heads; a read without a V has no V anchor."""
        if self.anchors is None:
            return None
        v, j = self.genes["v"].placed, self.genes["j"].placed
        return derive_junction(self.sequence, self.anchors.v if self.v_present else None, self.anchors.j,
                               coding_span=(None if v is None else v.seq_start, None if j is None else j.seq_end))


def _anchor(position) -> int | None:
    return None if position == NOT_OBSERVED else int(position)


def _d_status(klass: int) -> GeneStatus:
    return GeneStatus.CALLED if klass == CALLABLE else GeneStatus.ABSENT if klass == ABSENT else GeneStatus.ABSTAINED


def read_results(state: PredictionState, loci: Sequence[str | None]) -> list[ReadResult]:
    """Each read's ``ReadResult`` from the stages' outputs on ``state`` (each gene's calls and alignments in the calls'
    gene order, the segmentation's low-quality flags, the cropped flags, the reads without a V and the V ties) and each
    read's locus (``loci``). The D is gated on the D presence head when the model has one and calls a D."""
    predictions, calls, alignments = state.predictions, state.calls, state.alignments
    low_quality, no_v, v_ties = state.segments.low_quality, state.no_v, state.v_ties
    genes = list(calls)
    anchors, d_presence = predictions.anchors, predictions.d_presence
    gated_d = d_presence is not None and "d" in genes
    out = []
    for i, sequence in enumerate(state.canonical):
        results = {}
        for g in genes:
            call = calls[g][i]
            listed = v_ties[i] if g == "v" and v_ties[i] else tuple(call.names[:1])
            if g == "v" and no_v[i]:
                status = GeneStatus.ABSENT
            elif g == "d" and gated_d:
                status = _d_status(int(d_presence[i]))
            else:
                status = GeneStatus.CALLED
            results[g] = GeneResult(call, alignments[g][i], tuple(listed), status)
        out.append(ReadResult(
            sequence=sequence, input_sequence=state.reads[i],
            orientation_id=int(predictions.orientation[i]),
            mutation_rate=float(predictions.mutation_rate[i]), indel_count=float(predictions.indel_count[i]),
            genes=results,
            anchors=None if anchors is None else Anchors(*(_anchor(anchors[g][i]) for g in ANCHOR_GENES)),
            d_presence=D_CLASSES[int(d_presence[i])] if gated_d else None,
            low_quality=None if low_quality is None else bool(low_quality[i]),
            cropped=bool(state.cropped[i]), locus=loci[i]))
    return out


def light_record(read: ReadResult) -> dict:
    """A read's light record, keys in their order: the read and its heads, the anchors and the productivity they give,
    the input read and the flags, the locus, each gene's call, set, likelihoods and placement, then the D presence
    class and, for a read without a V, its V presence."""
    record = {"sequence": read.sequence, "orientation_id": read.orientation_id, "mutation_rate": read.mutation_rate,
              "indel_count": read.indel_count, "productive": None}
    if read.anchors is not None:
        record[ANCHOR_KEYS["v"]] = read.anchors.v if read.v_present else None
        record[ANCHOR_KEYS["j"]] = read.anchors.j
        record["productive"] = read.junction().productive
    record["input_sequence"] = read.input_sequence
    if read.low_quality is not None:
        record["segmentation_low_quality"] = read.low_quality
    if read.cropped:
        record["length_cropped"] = True
    if read.locus is not None:
        record["locus"] = read.locus
    for g, gene in read.genes.items():
        record[f"{g}_call"] = gene.call_field
        record[f"{g}_calls"] = list(gene.names)
        record[f"{g}_likelihoods"] = list(gene.likelihoods)
        placed = gene.placed
        if placed is not None:
            record.update({f"{g}_sequence_start": placed.seq_start, f"{g}_sequence_end": placed.seq_end,
                           f"{g}_germline_start": placed.germ_start, f"{g}_germline_end": placed.germ_end,
                           f"{g}_cigar": placed.cigar})
    if read.d_presence is not None:
        record[D_PRESENCE] = read.d_presence
    if not read.v_present:
        record[V_PRESENCE] = V_ABSENT
    return record
