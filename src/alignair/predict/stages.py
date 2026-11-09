"""The prediction pipeline's stages (``predict.pipeline.PIPELINE``) and their helpers.

Each stage is a function ``PredictionState -> PredictionState`` that sets its own fields (``dataclasses.replace``) and
leaves the others as they are. A stage calls its helpers through this module's globals, so a test that patches
``stages.clean``, ``stages.align_germline`` or ``stages.select_alleles`` reaches the stage that calls it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..data.inputs import prepare_nucleotide_inputs
from ..errors import InputError, NumericalError
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
    loci = record_loci(state.alignments["v"], context.reference, state.no_v, cfg.chain_types)
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


def record_loci(v_alignments, reference, no_v, chain_types) -> list[str | None]:
    """Per read, its locus label: its V call's locus in a reference of several loci (``called_loci``), else the one
    locus of ``chain_types`` when there is exactly one, else None (no label)."""
    loci = called_loci(v_alignments, reference, no_v)
    if loci is not None:
        return loci
    one = chain_types[0] if chain_types is not None and len(chain_types) == 1 else None
    return [one] * len(v_alignments)


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
