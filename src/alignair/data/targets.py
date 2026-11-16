"""Derive the per-sample ground-truth bundle (``TargetBundle``) from a GenAIRR stream_records dict
plus the ReferenceSet germline sequences (forward-orientation frame)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.anchors import ANCHOR_GENES, CODON
from ..core.d_presence import ABSENT, CALLABLE, D_CALLABLE_MIN, SHORT
from ..core.edit_states import EDIT_INDEX
from ..core.regions import REGION_INDEX
from ..core.tokens import encode
from ..errors import AlignAIRError, TrainingDataError, error_context
from ..reference.genes import GENES
from .alignment import RecordAlignment, parse_cigar

# What ``read_d_presence`` and ``read_anchors`` give for a segment the record places in the read whose alignment cannot
# be mapped (no CIGAR, or one that disagrees with the segment's read and germline spans): the segment is unscorable, so
# the adapter trains no term on it, rather than calling it absent or its anchor not observed.
UNSCORABLE = -1


def read_d_presence(record: dict, width: int | None = None) -> int:
    """The read's D class (``d_presence_class``) from the record alone."""
    with error_context(stage="build_targets", record_id=record.get("sequence_id")):
        return d_presence_class(RecordAlignment.of(record), width)


def d_presence_class(alignment: RecordAlignment, width: int | None = None) -> int:
    """The read's D class (``core.d_presence``): ``ABSENT`` when no D base inside the first ``width`` read bases (the
    window the model sees; the whole read for None) is aligned to the D germline, else ``SHORT`` or ``CALLABLE`` by
    its observed germline length: from the first to the last aligned germline base, so an internal deletion counts
    and an edge gap or an insertion does not. An inverted D is a D like any other (its alignment spans the
    reverse-complemented germline). ``UNSCORABLE`` when the record places a D in the read but its alignment cannot be
    mapped. Only a model with a D presence head needs it."""
    with error_context(stage="build_targets", record_id=alignment.record.get("sequence_id")):
        if "d" not in alignment.query_spans:
            return ABSENT
        path = alignment.path("d")
        if path.reason is not None:
            return UNSCORABLE
        aligned = [germline for offset, germline in enumerate(path.positions)
                   if germline is not None and (width is None or path.query_start + offset < width)]
        if not aligned:
            return ABSENT
        return SHORT if max(aligned) - min(aligned) + 1 < D_CALLABLE_MIN else CALLABLE


def read_anchors(record: dict, reference_set) -> dict[str, int | None]:
    """Each anchor's read position (``anchor_positions``) from the record alone."""
    with error_context(stage="build_targets", record_id=record.get("sequence_id")):
        return anchor_positions(RecordAlignment.of(record), reference_set)


def anchor_positions(alignment: RecordAlignment, reference_set) -> dict[str, int | None]:
    """``{"v": ..., "j": ...}``: the read position of each anchor codon's first base, None when the read does not show
    it, ``UNSCORABLE`` when the segment's alignment cannot be mapped (``_anchor``). Only a model with anchor heads needs
    them, so ``build_targets`` does not compute them."""
    with error_context(stage="build_targets", record_id=alignment.record.get("sequence_id")):
        return {gene: _anchor(alignment, gene, reference_set) for gene in ANCHOR_GENES}


def _anchor(alignment: RecordAlignment, gene: str, reference_set) -> int | None:
    """The read position of the first base of ``gene``'s conserved anchor codon (V Cys, J Trp/Phe), or None when the
    read does not show it.

    The anchor is the read's primary allele's annotated germline anchor (``GeneReference.anchors``, the annotation the
    AIRR assembly reads), mapped to the read through the segment's alignment (``query_reference_map``: its CIGAR, read
    and germline spans), so an indel before the anchor moves it with the read. It is observed when the segment is in
    the read, the codon's defining base is an aligned base (not deleted, not trimmed away), and the codon lies wholly
    inside the read. The defining base is the V codon's first base and the J codon's last, as the junction runs from
    the V anchor through the J codon's last base (``predict.junction``), so a codon a recombination trim cut keeps its
    anchor where GenAIRR places the junction. Each anchor is found whether or not the other is. ``UNSCORABLE`` when the
    segment is in the read but its alignment cannot be mapped."""
    if gene not in alignment.query_spans:
        return None
    anchor = (reference_set.gene(gene.upper()).anchors or {}).get(alignment.primary(gene))
    if anchor is None:
        return None
    path = alignment.path(gene)
    if path.reason is not None:
        return UNSCORABLE
    defining = anchor if gene == "v" else anchor + CODON - 1
    for offset, germline in enumerate(path.positions):
        if germline == defining:
            position = path.query_start + offset - (0 if gene == "v" else CODON - 1)
            return position if 0 <= position and position + CODON <= len(alignment.sequence) else None
    return None


def _label_segment_states(state, seq, ss, gref, gs, cigar):
    """Walk a segment's CIGAR to assign exact per-observed-position edit states.

    M/=/X consume an observed base and a germline base -> germline (match) or
    substitution (mismatch). I/S consume an observed base only -> insertion.
    D/N consume a germline base only (no observed position) -> the next observed
    base is tagged ``deletion`` (a learnable 'a gap precedes me' signal, since a
    deletion has no observed token of its own). Returns the deletion count."""
    obs, germ = ss, gs
    pending_del = False
    n_del = 0
    for count, op in parse_cigar(cigar):
        if op in ("M", "=", "X"):
            for _ in range(count):
                if obs >= len(seq):
                    break
                if pending_del:
                    state[obs] = EDIT_INDEX["deletion"]
                    pending_del = False
                elif germ < len(gref) and seq[obs] != gref[germ]:
                    state[obs] = EDIT_INDEX["substitution"]
                # else leaves germline (0)
                obs += 1
                germ += 1
        elif op in ("I", "S"):
            for _ in range(count):
                if obs >= len(seq):
                    break
                state[obs] = EDIT_INDEX["insertion"]
                obs += 1
        elif op in ("D", "N"):
            germ += count
            n_del += count
            pending_del = True
    return n_del


def _names(value) -> set[str]:
    return {name.strip() for name in str(value or "").split(",") if name.strip()}


def allele_set(record: dict, gene: str, reference_set) -> set[str]:
    """The record's allele-set target of ``gene``: its call (the generating allele) and the alleles GenAIRR re-calls
    from the read (``recalled_{gene}_call``) that ``reference_set`` holds. A recalled name outside the reference is
    dropped: in a novel batch the reference is the batch bank's, so a novel allele its bank aliases to a reference
    allele drops out and that allele is the call. The call itself is not filtered (a call the reference lacks is refused
    where the target is encoded). A record without recalled calls (GenAIRR before 2.4) targets its call alone."""
    index = reference_set.gene(gene.upper()).index
    recalled = {name for name in _names(record.get(f"recalled_{gene}_call")) if name in index}
    return _names(record[f"{gene}_call"]) | recalled


def observed_mutation_rate(record: dict, germline: dict) -> float:
    """The record's mutation-rate target: its observable mutations (``n_observable_mutations``, the hypermutation
    substitutions still visible in the read) per observed germline base, ``germline_end - germline_start`` summed over
    the read's segments (``germline``: gene -> (start, end), the observed length). GenAIRR 2.4's own ``mutation_rate``
    is a mutation burden per input-molecule base, which counts mutations the read does not show. 0.0 for a read with
    no observed germline; a record without the count (GenAIRR before 2.4, whose ``mutation_rate`` counted
    mutations per read base) keeps its ``mutation_rate``. The chip's loss reads no rate target (its ``mutation_rate``
    is the reader's, ``network.chip``): this is the batch's target and the ``mutation_mae`` training metric."""
    if record.get("n_observable_mutations") is None:
        return float(record["mutation_rate"])
    observed = sum(end - start for start, end in germline.values())
    return float(record["n_observable_mutations"]) / observed if observed > 0 else 0.0


@dataclass(frozen=True)
class TargetBundle:
    """One training record's targets (``build_targets``) and the alignment they come from (``RecordAlignment``)."""
    alignment: RecordAlignment
    tokens: np.ndarray                     # the read's tokens
    region_labels: np.ndarray              # per read base, its region (``REGION_INDEX``)
    state_labels: np.ndarray               # per read base, its edit state (``EDIT_INDEX``)
    calls: dict[str, set[str]]             # gene (upper case) -> its allele-set target (``allele_set``)
    mutation_rate: float                   # ``observed_mutation_rate``
    indel_count: float

    @property
    def query_spans(self) -> dict[str, tuple[int, int]]:
        """Each gene the read shows -> its (start, end) in the read."""
        return self.alignment.query_spans

    @property
    def germline_spans(self) -> dict[str, tuple[int, int]]:
        """Each gene the read shows -> its (start, end) in its germline."""
        return self.alignment.germline_spans

    @property
    def d_inverted(self) -> bool:
        """Whether the read's D is inverted."""
        return self.alignment.d_inverted


def build_targets(record: dict, reference_set, has_d: bool) -> TargetBundle:
    """The ``TargetBundle`` of a GenAIRR training ``record`` against ``reference_set``; a TrainingDataError for a
    record that does not hold the fields the targets need."""
    with error_context(stage="build_targets", record_id=record.get("sequence_id")):
        try:
            return _build_targets(record, reference_set, has_d)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AlignAIRError):
                raise
            raise TrainingDataError("Training record does not satisfy the target schema.",
                                    context={"field": exc.args[0] if isinstance(exc, KeyError) else None},
                                    hint="Check simulator annotations and training-record coordinate fields.") from exc


def _build_targets(record: dict, reference_set, has_d: bool) -> TargetBundle:
    alignment = RecordAlignment.of(record)
    region = _region_labels(len(alignment.sequence), alignment.query_spans, has_d)
    state = _edit_states(alignment, reference_set)
    calls = {g.upper(): allele_set(record, g, reference_set) for g in GENES if record.get(f"{g}_call")}
    return TargetBundle(alignment=alignment, tokens=encode(alignment.sequence), region_labels=region,
                        state_labels=state, calls=calls,
                        mutation_rate=observed_mutation_rate(record, alignment.germline_spans),
                        indel_count=float(record["n_indels"]))


def _region_labels(length: int, coords: dict[str, tuple[int, int]], has_d: bool) -> np.ndarray:
    """Each read base's region (presence-aware: one-sided or adaptive crops can drop V and/or D)."""
    region = np.full(length, REGION_INDEX["pre"], dtype=np.int64)
    has_v = "v" in coords
    if has_v:
        vs, ve = coords["v"]
        region[vs:ve] = REGION_INDEX["V"]
    js, je = coords.get("j", (length, length))
    left = ve if has_v else 0                       # start of the inter-segment (np/D) region
    if has_d and "d" in coords:
        ds, de = coords["d"]
        region[left:ds] = REGION_INDEX["N1"]
        region[ds:de] = REGION_INDEX["D"]
        region[de:js] = REGION_INDEX["N2"]
    else:
        region[left:js] = REGION_INDEX["N1"]
    region[js:je] = REGION_INDEX["J"]
    region[je:length] = REGION_INDEX["post"]
    return region


def _edit_states(alignment: RecordAlignment, reference_set) -> np.ndarray:
    """Each read base's edit state from its segment's CIGAR (exact, indel-aware; germline where no segment's primary
    allele is in the reference)."""
    seq, coords, germ = alignment.sequence, alignment.query_spans, alignment.germline_spans
    state = np.zeros(len(seq), dtype=np.int64)  # germline = 0
    for g in GENES:
        if g not in coords:
            continue
        allele = alignment.primary(g)
        ref = reference_set.gene(g.upper())
        idx = ref.index.get(allele)
        if idx is None:
            continue
        _label_segment_states(state, seq, coords[g][0], ref.sequences[idx], germ[g][0],
                              alignment.record.get(f"{g}_cigar"))
    return state
