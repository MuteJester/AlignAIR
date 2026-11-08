"""The prediction pipeline's stages (``predict.pipeline.PIPELINE``) and their helpers.

Each stage is a function ``PredictionState -> PredictionState`` that sets its own fields (``dataclasses.replace``) and
leaves the others as they are. A stage calls its helpers through this module's globals, so a test that patches
``stages.clean``, ``stages.align_germline`` or ``stages.select_alleles`` reaches the stage that calls it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.anchors import ANCHOR_GENES, ANCHOR_KEYS
from ..core.d_presence import ABSENT, CALLABLE, D_CLASSES, D_PRESENCE
from ..data.inputs import prepare_nucleotide_inputs
from ..errors import InputError, NumericalError
from .clean import NOT_OBSERVED, clean
from .forward import run_model
from .germline import align_germline, realign_v
from .junction import derive_junction
from .outputs import ModelOutputs
from .segment import correct_segments
from .state import PredictionState
from .threshold import select_alleles
from .ties import better_v_alleles, tied_alleles
from .v_presence import V_ABSENT, V_PRESENCE, v_absent


def prepare_reads(state: PredictionState) -> PredictionState:
    """``input_policy``: the reads tokenization, orientation, alignment and reported coordinates share, cropped to the
    model's window."""
    reads, cropped = prepare_nucleotide_inputs(state.sequences, state.context.cfg.window)
    return replace(state, reads=reads, cropped=cropped)


def run_forward(state: PredictionState) -> PredictionState:
    """``forward``: the model's batched forward pass over the reads, with the given orientations teacher-forced, each
    batch's outputs typed (``ModelOutputs``).
    InputError for orientations that are not one per read, or an id that is not one of ``ORIENTATION_IDS``."""
    context, given = state.context, state.orientation_given
    if given is not None:
        if len(given) != len(state.reads):
            raise InputError(f"{len(given)} orientations for {len(state.reads)} reads",
                             context={"field": "orientation"})
        bad = next((i for i, o in enumerate(given) if o is not None and int(o) not in ORIENTATION_IDS), None)
        if bad is not None:
            raise InputError(f"orientation id {given[bad]} of read {bad} is not one of 0, 1, 2, 3",
                             context={"field": "orientation", "read": bad})
    batches = [ModelOutputs.from_network(batch, context.cfg.genes)
               for batch in run_model(context.model, state.reads, context.cfg, context.device, orientation=given)]
    return replace(state, batches=batches)


def decode_outputs(state: PredictionState) -> PredictionState:
    """``clean``: the batches decoded into ``Predictions``, which must be finite before any use; a teacher-forced
    orientation is the read's orientation (the model's own where none was given)."""
    predictions = clean(state.batches, state.context.cfg.genes)
    validate_predictions(predictions)
    if state.orientation_given is not None:
        own = predictions.orientation
        predictions = replace(predictions, orientation=np.array(
            [int(p) if o is None else int(o) for o, p in zip(state.orientation_given, own)], dtype=int))
    return replace(state, predictions=predictions)


def canonicalize_reads(state: PredictionState) -> PredictionState:
    """``canonicalize``: each read in the model's forward frame (its predicted orientation), so the coordinates, the
    germline and the AIRR assembly all agree."""
    orient = state.predictions.orientation
    canonical = [canonicalize(s, int(orient[i]) if orient is not None else 0) for i, s in enumerate(state.reads)]
    return replace(state, canonical=canonical)


def correct_spans(state: PredictionState) -> PredictionState:
    """``segments``: the decoded boundaries corrected into ``Segments`` on the forward-frame reads."""
    cfg = state.context.cfg
    seq_lens = np.array([len(s) for s in state.canonical])
    segments = correct_segments(state.predictions.start, state.predictions.end, seq_lens, cfg.max_seq_length,
                                cfg.pad_mode)
    return replace(state, segments=segments)


def select_calls(state: PredictionState) -> PredictionState:
    """``alleles``: each gene's call set from the allele scores."""
    context, cfg = state.context, state.context.cfg
    names = {g: list(context.reference.gene(g.upper()).names) for g in cfg.genes}
    calls = select_alleles(state.predictions.allele, names, cfg.threshold, cfg.cap, cfg.selector)
    return replace(state, calls=calls)


def align_reads(state: PredictionState) -> PredictionState:
    """``germline``: each called allele placed in its germline (read and germline coordinates, CIGAR)."""
    alignments = align_germline(state.canonical, state.segments, state.calls, state.context.reference,
                                indel_counts=state.predictions.indel_count)
    return replace(state, alignments=alignments)


def promote_better_v(state: PredictionState) -> PredictionState:
    """``better_v``: an allele the read matches better than the model's V call (a bound allele it never trained on)
    becomes the call, V realigned to it; a new alignments dict, the germline stage's is left as it was."""
    reference = state.context.reference
    better = better_v_alleles(state.canonical, state.alignments["v"], reference,
                              [call.names for call in state.calls["v"]])
    v = realign_v(state.canonical, state.segments, state.alignments["v"], better, reference,
                  indel_counts=state.predictions.indel_count)
    return replace(state, alignments={**state.alignments, "v": v})


def find_absent_v(state: PredictionState) -> PredictionState:
    """``v_presence``: the reads without a V (a D-J rearrangement, a read starting inside the junction)."""
    return replace(state, no_v=v_absent(state.canonical, state.alignments["v"], state.context.reference))


def list_v_ties(state: PredictionState) -> PredictionState:
    """``ties``: the V alleles each read cannot tell from its call, among those it may be called with."""
    return replace(state, v_ties=tied_alleles(state.canonical, state.alignments["v"], state.context.reference))


def build_records(state: PredictionState) -> PredictionState:
    """``records``: one record per read. ``reads`` is the post-crop, pre-orientation read and ``canonical`` its forward
    frame: the record owns the pre-orientation read, so the AIRR writer emits the right ``sequence``/``rev_comp`` for
    the Python API and the CLI alike."""
    context, cfg = state.context, state.context.cfg
    records = to_records(state.canonical, state.calls, state.alignments, cfg.genes, state.predictions,
                         cfg.chain_types, state.segments.low_quality, state.cropped, input_sequences=state.reads,
                         no_v=state.no_v, v_ties=state.v_ties,
                         read_loci=called_loci(state.alignments["v"], context.reference, state.no_v))
    return replace(state, records=records)


def build_rows(state: PredictionState) -> PredictionState:
    """``airr``: the full AIRR rows (``cfg.airr``), else the records themselves."""
    context, cfg = state.context, state.context.cfg
    if not cfg.airr:
        return replace(state, rows=state.records)
    from .airr import build_airr
    return replace(state, rows=build_airr(state.records, context.reference, chain=("heavy" if cfg.has_d else "light")))


def assert_finite(allele: dict, stage: str) -> None:
    """Fail loudly if any post-transform allele probability is non-finite (NaN/Inf) — a silently
    NaN'd probability would otherwise propagate into calls/confidence."""
    for g, a in allele.items():
        if not np.all(np.isfinite(a)):
            raise NumericalError(f"non-finite allele probabilities after {stage} (gene {g!r})",
                                 context={"stage": stage, "gene": g, "field": "allele_probabilities"})


def validate_predictions(preds, stage: str = "model forward") -> None:
    """Fail loudly if ANY raw model output is non-finite: a damaged checkpoint emitting NaN
    would otherwise be silently turned into a plausible call (argmax fallback) and clipped coordinates."""
    assert_finite(preds.allele, stage)
    for name, d in (("start", preds.start), ("end", preds.end)):
        for g, a in d.items():
            if not np.all(np.isfinite(a)):
                raise NumericalError(f"non-finite {name} coordinates after {stage} (gene {g!r})",
                                     context={"stage": stage, "gene": g, "field": name})
    for name in ("mutation_rate", "indel_count"):
        a = getattr(preds, name)
        if a is not None and not np.all(np.isfinite(a)):
            raise NumericalError(f"non-finite {name} after {stage}", context={"stage": stage, "field": name})


def called_loci(v_alignments, reference, no_v) -> list[str] | None:
    """Per read, the locus of its V call (the allele its V is aligned to) in a reference of several loci, ``""`` for a
    read without a V call; None for a reference of one locus or none (its records are labelled otherwise)."""
    loci = reference.locus_names() if hasattr(reference, "locus_names") else ()
    if len(loci) < 2:
        return None
    names = reference.gene("V").names
    owner = {names[k]: locus for locus in loci for k in np.flatnonzero(reference.locus_mask(locus, "V"))}
    return ["" if (no_v is not None and no_v[i]) or aln is None else owner.get(aln.allele, "")
            for i, aln in enumerate(v_alignments)]


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
ORIENTATION_IDS = range(4)          # canonicalize's transforms: identity, reverse complement, complement, reverse


def canonicalize(seq: str, orientation_id: int) -> str:
    """Re-orient a read into the model's forward frame using the predicted orientation (the transforms
    are involutions, so re-applying recovers forward): 0=identity, 1=revcomp, 2=complement, 3=reverse.
    The model predicts coordinates in the forward frame, so the germline reader, coords, and AIRR
    assembly must all operate on this canonical sequence."""
    if orientation_id == 1:
        return seq.translate(_COMPLEMENT)[::-1]
    if orientation_id == 2:
        return seq.translate(_COMPLEMENT)
    if orientation_id == 3:
        return seq[::-1]
    return seq


def _anchor(position) -> int | None:
    """A decoded anchor as a record holds it: the read position, or None when it is not observed."""
    return None if position == NOT_OBSERVED else int(position)


def to_records(sequences, calls, alignments, genes, preds, chain_types=None, low_quality=None,
                was_cropped=None, input_sequences=None, no_v=None, v_ties=None, read_loci=None) -> list:
    orientation = preds.orientation
    anchors = getattr(preds, "anchors", None)
    records = []
    for i, seq in enumerate(sequences):
        absent_v = no_v is not None and no_v[i]
        oid = int(orientation[i]) if orientation is not None else 0
        rec = {"sequence": seq, "orientation_id": oid,
               "mutation_rate": float(preds.mutation_rate[i]),
               "indel_count": float(preds.indel_count[i]),
               "productive": None}
        if anchors is not None:
            # a model with anchor heads: its record carries the anchors (forward frame, like `seq`) and the
            # productivity they give over the read's V start and J end (predict.junction), not a learned call
            rec.update({ANCHOR_KEYS[g]: _anchor(anchors[g][i]) for g in ANCHOR_GENES})
            if absent_v:                                       # no V, so no V anchor to open a junction
                rec[ANCHOR_KEYS["v"]] = None
            v, j = (None if absent_v else alignments["v"][i]), alignments["j"][i]
            rec["productive"] = derive_junction(seq, *(rec[ANCHOR_KEYS[g]] for g in ANCHOR_GENES), coding_span=(
                None if v is None else v.seq_start, None if j is None else j.seq_end)).productive
        if input_sequences is not None:
            # the post-crop, pre-orientation read (== `seq` when the read was already forward). The AIRR
            # writer uses THIS as the emitted `sequence` for a reverse-complement read, so `RC(sequence)`
            # equals the coordinate frame (`seq`) — regardless of any external sequence list.
            rec["input_sequence"] = input_sequences[i]
        if low_quality is not None:
            rec["segmentation_low_quality"] = bool(low_quality[i])
        if was_cropped is not None and was_cropped[i]:
            rec["length_cropped"] = True
        if read_loci is not None:                              # several loci: the called V's
            rec["locus"] = read_loci[i]
        elif chain_types is not None and len(chain_types) == 1:  # the reference's one locus
            rec["locus"] = chain_types[0]
        for g in genes:
            call, aln = calls[g][i], alignments[g][i]
            # the V call is listed with the alleles the read cannot tell from it (predict.ties)
            listed = v_ties[i] if g == "v" and v_ties is not None and v_ties[i] else call.names[:1]
            rec[f"{g}_call"] = ",".join(listed)
            rec[f"{g}_calls"] = list(call.names)
            rec[f"{g}_likelihoods"] = list(call.likelihoods)
            if aln is not None:
                rec[f"{g}_sequence_start"] = aln.seq_start
                rec[f"{g}_sequence_end"] = aln.seq_end
                rec[f"{g}_germline_start"] = aln.germ_start
                rec[f"{g}_germline_end"] = aln.germ_end
                rec[f"{g}_cigar"] = aln.cigar
        d_presence = getattr(preds, "d_presence", None)
        if d_presence is not None and "d" in genes:
            gate_d(rec, int(d_presence[i]))
        if absent_v:
            gate_v(rec)
        records.append(rec)
    return records


def gate_v(rec: dict) -> None:
    """Drop the V of a read that holds none (``predict.v_presence``) and record it: no call, no coordinates; the AIRR
    assembly then gives the read no junction or alignment, and its D and J calls stand."""
    rec[V_PRESENCE] = V_ABSENT
    rec["v_call"], rec["v_calls"], rec["v_likelihoods"] = "", [], []
    for key in ("v_sequence_start", "v_sequence_end", "v_germline_start", "v_germline_end", "v_cigar"):
        rec.pop(key, None)


def gate_d(rec: dict, klass: int) -> None:
    """Gate a record's D on the D presence head's class (``core.d_presence``) and record the class: an absent D is
    dropped (no call, no coordinates: the AIRR assembly then treats the read as one without a D), a short one keeps
    its span and alignment without a call (an abstention), a callable one is left as it is."""
    rec[D_PRESENCE] = D_CLASSES[klass]
    if klass == CALLABLE:
        return
    rec["d_call"], rec["d_calls"], rec["d_likelihoods"] = "", [], []
    if klass == ABSENT:
        for key in ("d_sequence_start", "d_sequence_end", "d_germline_start", "d_germline_end", "d_cigar"):
            rec.pop(key, None)
