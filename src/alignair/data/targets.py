"""Derive the per-sample ground-truth bundle from a GenAIRR stream_records dict
plus the ReferenceSet germline sequences (forward-orientation frame)."""
import numpy as np

from ..errors import AlignAIRError, TrainingDataError, error_context

from .tokenizer import TOKEN_DICT
from ..core.anchors import ANCHOR_GENES, CODON
from ..core.d_presence import ABSENT, CALLABLE, D_CALLABLE_MIN, SHORT
from ..neural.heads.region import REGION_INDEX
from ..neural.heads.state import STATE_INDEX
from .coordinates import segment_spans
from .alignment import parse_cigar, query_reference_map

_GENES = ("v", "d", "j")
# What ``read_d_presence`` and ``read_anchors`` give for a segment the record places in the read whose alignment cannot
# be mapped (no CIGAR, or one that disagrees with the segment's read and germline spans): the assay's truth calls such a
# segment unscorable, so the adapter trains no term on it, rather than calling it absent or its anchor not observed.
UNSCORABLE = -1


def read_d_presence(record: dict, width: int | None = None) -> int:
    """The read's D class (``core.d_presence``): ``ABSENT`` when no D base inside the first ``width`` read bases (the
    window the model sees; the whole read for None) is aligned to the D germline, else ``SHORT`` or ``CALLABLE`` by
    its observed germline length: from the first to the last aligned germline base, so an internal deletion counts
    and an edge gap or an insertion does not, as the AlignAIR assay's truth measures a segment's observed length. An
    inverted D is a D like any other (its alignment spans the reverse-complemented germline). ``UNSCORABLE`` when the
    record places a D in the read but its alignment cannot be mapped. Only a model with a D presence head needs it."""
    with error_context(stage="build_targets", record_id=record.get("sequence_id")):
        coords, germ = segment_spans(record)
        if "d" not in coords:
            return ABSENT
        path = query_reference_map(record.get("d_cigar"), coords["d"], germ["d"])
        if path.reason is not None:
            return UNSCORABLE
        aligned = [germline for offset, germline in enumerate(path.positions)
                   if germline is not None and (width is None or path.query_start + offset < width)]
        if not aligned:
            return ABSENT
        return SHORT if max(aligned) - min(aligned) + 1 < D_CALLABLE_MIN else CALLABLE


def read_anchors(record: dict, reference_set) -> dict[str, int | None]:
    """``{"v": ..., "j": ...}``: the read position of each anchor codon's first base, None when the read does not show
    it, ``UNSCORABLE`` when the segment's alignment cannot be mapped (``_anchor``). Only a model with anchor heads needs
    them, so ``build_targets`` does not compute them."""
    with error_context(stage="build_targets", record_id=record.get("sequence_id")):
        coords, germ = segment_spans(record)
        length = len(str(record["sequence"]))
        return {gene: _anchor(record, gene, reference_set, coords, germ, length) for gene in ANCHOR_GENES}


def _anchor(record, gene: str, reference_set, coords, germ, length: int) -> int | None:
    """The read position of the first base of ``gene``'s conserved anchor codon (V Cys, J Trp/Phe), or None when the
    read does not show it.

    The anchor is the read's primary allele's annotated germline anchor (``GeneReference.anchors``, the annotation the
    AIRR assembly reads), mapped to the read through the segment's alignment (``query_reference_map``: its CIGAR, read
    and germline spans), so an indel before the anchor moves it with the read. It is observed when the segment is in
    the read, the codon's defining base is an aligned base (not deleted, not trimmed away), and the codon lies wholly
    inside the read. The defining base is the V codon's first base and the J codon's last, as the assay's truth defines
    the junction (``[V anchor, J codon's last base + 1)``), so a codon a recombination trim cut keeps its anchor where
    GenAIRR places the junction. Each anchor is found whether or not the other is. ``UNSCORABLE`` when the segment is
    in the read but its alignment cannot be mapped."""
    if gene not in coords:
        return None
    allele = str(record[f"{gene}_call"]).split(",")[0]          # the allele the germline coordinates belong to
    anchor = (reference_set.gene(gene.upper()).anchors or {}).get(allele)
    if anchor is None:
        return None
    path = query_reference_map(record.get(f"{gene}_cigar"), coords[gene], germ[gene])
    if path.reason is not None:
        return UNSCORABLE
    defining = anchor if gene == "v" else anchor + CODON - 1
    for offset, germline in enumerate(path.positions):
        if germline == defining:
            position = path.query_start + offset - (0 if gene == "v" else CODON - 1)
            return position if 0 <= position and position + CODON <= length else None
    return None


def _tok(seq: str) -> np.ndarray:
    n = TOKEN_DICT["N"]
    return np.array([TOKEN_DICT.get(c, n) for c in seq.upper()], dtype=np.int64)


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
                    state[obs] = STATE_INDEX["deletion"]
                    pending_del = False
                elif germ < len(gref) and seq[obs] != gref[germ]:
                    state[obs] = STATE_INDEX["substitution"]
                # else leaves germline (0)
                obs += 1
                germ += 1
        elif op in ("I", "S"):
            for _ in range(count):
                if obs >= len(seq):
                    break
                state[obs] = STATE_INDEX["insertion"]
                obs += 1
        elif op in ("D", "N"):
            germ += count
            n_del += count
            pending_del = True
    return n_del


def build_targets(record: dict, reference_set, has_d: bool) -> dict:
    with error_context(stage="build_targets", record_id=record.get("sequence_id")):
        try:
            return _build_targets(record, reference_set, has_d)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AlignAIRError):
                raise
            raise TrainingDataError("Training record does not satisfy the target schema.",
                                    context={"field": exc.args[0] if isinstance(exc, KeyError) else None},
                                    hint="Check simulator annotations and training-record coordinate fields.") from exc


def _build_targets(record: dict, reference_set, has_d: bool) -> dict:
    seq = str(record["sequence"]).upper()
    L = len(seq)
    coords, germ = segment_spans(record)

    # ---- region labels (presence-aware: one-sided/adaptive crops can drop V and/or D) ----
    region = np.full(L, REGION_INDEX["pre"], dtype=np.int64)
    has_v = "v" in coords
    if has_v:
        vs, ve = coords["v"]
        region[vs:ve] = REGION_INDEX["V"]
    js, je = coords.get("j", (L, L))
    left = ve if has_v else 0                       # start of the inter-segment (np/D) region
    if has_d and "d" in coords:
        ds, de = coords["d"]
        region[left:ds] = REGION_INDEX["N1"]
        region[ds:de] = REGION_INDEX["D"]
        region[de:js] = REGION_INDEX["N2"]
    else:
        region[left:js] = REGION_INDEX["N1"]
    region[js:je] = REGION_INDEX["J"]
    region[je:L] = REGION_INDEX["post"]

    # ---- per-position edit state from each segment's CIGAR (exact, indel-aware) ----
    state = np.zeros(L, dtype=np.int64)  # germline = 0
    for g in _GENES:
        if g not in coords:
            continue
        call = str(record[f"{g}_call"]).split(",")[0]
        ref = reference_set.gene(g.upper())
        idx = ref.index.get(call)
        if idx is None:
            continue
        ss, _ = coords[g]
        gs, _ = germ[g]
        cigar = record.get(f"{g}_cigar")
        _label_segment_states(state, seq, ss, ref.sequences[idx], gs, cigar)

    calls = {g.upper(): set(str(record[f"{g}_call"]).split(","))
             for g in _GENES if record.get(f"{g}_call")}
    # the allele the germline coordinates actually belong to (GenAIRR lists it
    # first); teacher-forced germline alignment must use THIS allele's reps, not an
    # arbitrary co-listed one whose germline length differs.
    primary = {g.upper(): str(record[f"{g}_call"]).split(",")[0]
               for g in _GENES if record.get(f"{g}_call")}

    return {
        "tokens": _tok(seq),
        "region_labels": region,
        "state_labels": state,
        "germline": germ,
        "inseq": coords,
        "calls": calls,
        "primary": primary,
        # inverted-D reads contain RC(germline) at the D locus; we cannot yet
        # supervise D-match/D-germline against the forward reference, so flag them
        # and mask those terms (RC reference modelling is deferred).
        "d_inverted": bool(record.get("d_inverted", False)),
        "orientation_id": 0,  # forward-only gym for now
        # locus index for the multi-chain chain_type head; the mixed-locus stream tags each record
        # with its dataconfig position (== AlignAIRConfig chain_type order). Single-chain -> 0.
        "chain_type": int(record.get("chain_type", 0)),
        "noise_count": float(record["n_quality_errors"] + record.get("n_pcr_errors", 0)),
        "mutation_rate": float(record["mutation_rate"]),
        "indel_count": float(record["n_indels"]),
        "productive": 1.0 if record["productive"] else 0.0,
        # GenAIRR leaves productivity unknown (None) when the read does not show it; the label above is
        # then 0.0, and this flag lets the masked objective leave the read out of the productive loss.
        "productive_known": record["productive"] is not None,
    }
