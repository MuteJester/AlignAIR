"""The prediction pipeline's stages (``predict.pipeline.PIPELINE``) and their helpers.

Each stage is a function ``PredictionState -> PredictionState`` that sets its own fields (``dataclasses.replace``) and
leaves the others as they are. A stage calls its helpers through this module's globals, so a test that patches
``stages.clean``, ``stages.align_germline`` or ``stages.select_alleles`` reaches the stage that calls it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.orientations import NUM_ORIENTATIONS, orient
from ..data.inputs import prepare_nucleotide_inputs
from ..errors import InputError
from .clean import clean
from .forward import run_model
from .germline import align_germline, realign_v
from .outputs import ModelOutputs
from .results import light_record, read_results
from .segment import correct_segments
from .state import PredictionState
from .threshold import select_alleles
from .ties import better_v_alleles, tied_alleles
from .v_presence import v_absent


def prepare_reads(state: PredictionState) -> PredictionState:
    """``input_policy``: the reads tokenization, orientation, alignment and reported coordinates share, cropped to the
    model's window."""
    reads, cropped = prepare_nucleotide_inputs(state.sequences, state.context.cfg.window)
    return replace(state, reads=reads, cropped=cropped)


def run_forward(state: PredictionState) -> PredictionState:
    """``forward``: the model's batched forward pass over the reads, with the given orientations teacher-forced, each
    batch's outputs typed (``ModelOutputs``).
    InputError for orientations that are not one per read, or an id that is not an index of ``ORIENTATIONS``."""
    context, given = state.context, state.orientation_given
    if given is not None:
        if len(given) != len(state.reads):
            raise InputError(f"{len(given)} orientations for {len(state.reads)} reads",
                             context={"field": "orientation"})
        bad = next((i for i, o in enumerate(given) if o is not None and int(o) not in range(NUM_ORIENTATIONS)), None)
        if bad is not None:
            raise InputError(f"orientation id {given[bad]} of read {bad} is not one of 0, 1, 2, 3",
                             context={"field": "orientation", "read": bad})
    batches = [ModelOutputs.from_network(batch, context.cfg.genes)
               for batch in run_model(context.model, state.reads, context.cfg, context.device, orientation=given)]
    return replace(state, batches=batches)


def decode_outputs(state: PredictionState) -> PredictionState:
    """``clean``: the batches decoded into ``Predictions`` (the forward refused any non-finite output); a teacher-forced
    orientation is the read's orientation (the model's own where none was given)."""
    predictions = clean(state.batches, state.context.cfg.genes)
    if state.orientation_given is not None:
        own = predictions.orientation
        predictions = replace(predictions, orientation=np.array(
            [int(p) if o is None else int(o) for o, p in zip(state.orientation_given, own)], dtype=int))
    return replace(state, predictions=predictions)


def canonicalize_reads(state: PredictionState) -> PredictionState:
    """``canonicalize``: each read in the model's forward frame (its predicted orientation), so the coordinates, the
    germline and the AIRR assembly all agree."""
    orientation = state.predictions.orientation
    canonical = [orient(s, int(orientation[i])) for i, s in enumerate(state.reads)]
    return replace(state, canonical=canonical)


def correct_spans(state: PredictionState) -> PredictionState:
    """``segments``: the decoded boundaries corrected into ``Segments`` on the forward-frame reads."""
    seq_lens = np.array([len(s) for s in state.canonical])
    segments = correct_segments(state.predictions.start, state.predictions.end, seq_lens)
    return replace(state, segments=segments)


def select_calls(state: PredictionState) -> PredictionState:
    """``alleles``: each gene's call set from the allele scores."""
    context, cfg = state.context, state.context.cfg
    names = {g: list(context.reference.gene(g.upper()).names) for g in cfg.genes}
    calls = select_alleles(state.predictions.allele, names, cfg.threshold, cfg.cap)
    return replace(state, calls=calls)


def align_reads(state: PredictionState) -> PredictionState:
    """``germline``: each called allele placed in its germline (read and germline coordinates, CIGAR)."""
    alignments = align_germline(state.canonical, state.segments, state.calls, state.context.reference,
                                indel_counts=state.predictions.indel_count)
    return replace(state, alignments=alignments)


def locus_alleles(state: PredictionState) -> list | None:
    """Per read, a bool array over the V alleles of its locus (``Predictions.read_locus``, ``refine.locus``), or None
    for a model bound to one locus."""
    loci = state.predictions.read_locus
    if loci is None:
        return None
    reference = state.context.reference
    masks = [reference.locus_mask(locus, "V") for locus in reference.locus_names()]
    return [masks[int(k)] for k in loci]


def promote_better_v(state: PredictionState) -> PredictionState:
    """``better_v``: an allele the read matches better than the model's V call (a bound allele it never trained on)
    becomes the call, V realigned to it; a new alignments dict, the germline stage's is left as it was."""
    reference = state.context.reference
    better = better_v_alleles(state.canonical, state.alignments["v"], reference,
                              [call.names for call in state.calls["v"]], allowed=locus_alleles(state))
    v = realign_v(state.canonical, state.segments, state.alignments["v"], better, reference,
                  indel_counts=state.predictions.indel_count)
    return replace(state, alignments={**state.alignments, "v": v})


def find_absent_v(state: PredictionState) -> PredictionState:
    """``v_presence``: the reads without a V (a D-J rearrangement, a read starting inside the junction)."""
    return replace(state, no_v=v_absent(state.canonical, state.alignments["v"], state.context.reference))


def list_v_ties(state: PredictionState) -> PredictionState:
    """``ties``: the V alleles each read cannot tell from its call, among those it may be called with."""
    return replace(state, v_ties=tied_alleles(state.canonical, state.alignments["v"], state.context.reference,
                                              allowed=locus_alleles(state)))


def build_records(state: PredictionState) -> PredictionState:
    """``records``: each read's typed result (``predict.results.ReadResult``). ``reads`` is the post-crop,
    pre-orientation read and ``canonical`` its forward frame: the result owns the pre-orientation read, so the AIRR
    writer emits the right ``sequence``/``rev_comp`` for the Python API and the CLI alike."""
    context, cfg = state.context, state.context.cfg
    loci = record_loci(state.alignments["v"], context.reference, state.no_v, cfg.chain_types,
                       read_locus=state.predictions.read_locus)
    results = read_results(state.canonical, state.reads, state.predictions, state.calls, state.alignments,
                           state.segments.low_quality, state.cropped, state.no_v, state.v_ties, loci)
    return replace(state, results=results)


def build_rows(state: PredictionState) -> PredictionState:
    """``airr``: the full AIRR rows of the reads' results (``cfg.airr``), else each read's light record."""
    context, cfg = state.context, state.context.cfg
    if not cfg.airr:
        return replace(state, rows=[light_record(result) for result in state.results])
    from .airr import build_airr
    return replace(state, rows=build_airr(state.results, context.reference,
                                          chain=("heavy" if cfg.has_d else "light")))


def called_loci(v_alignments, reference, no_v) -> list[str] | None:
    """Per read, the locus of its V call (the allele its V is aligned to) in a reference of several loci, ``""`` for a
    read without a V call; None for a reference of one locus or none (its records are labelled otherwise). Its J and D
    calls are of the same locus: the network keeps each read to one locus (``refine.locus``), and better-V and the
    ties keep to it (``locus_alleles``)."""
    loci = reference.locus_names() if hasattr(reference, "locus_names") else ()
    if len(loci) < 2:
        return None
    names = reference.gene("V").names
    owner = {}                       # a V allele two loci list (TRAV/DV) is the first's: the one whose block holds it
    for locus in loci:
        for k in np.flatnonzero(reference.locus_mask(locus, "V")):
            owner.setdefault(names[k], locus)
    return ["" if (no_v is not None and no_v[i]) or aln is None else owner.get(aln.allele, "")
            for i, aln in enumerate(v_alignments)]


def record_loci(v_alignments, reference, no_v, chain_types, read_locus=None) -> list[str | None]:
    """Per read, its locus label: in a reference of several loci, the read's locus (``read_locus``, the network's
    ``refine.locus``: its J and D are of it too; a V allele two loci share, TRAV/DV, is called in either) for a read
    with a V call and ``""`` without one, or its V call's locus without ``read_locus`` (``called_loci``); else the
    one locus of ``chain_types`` when there is exactly one, else None (no label)."""
    loci = called_loci(v_alignments, reference, no_v)
    if loci is not None and read_locus is not None:
        names = reference.locus_names()
        loci = [names[int(k)] if label else "" for label, k in zip(loci, read_locus)]
    if loci is not None:
        return loci
    one = chain_types[0] if chain_types is not None and len(chain_types) == 1 else None
    return [one] * len(v_alignments)


