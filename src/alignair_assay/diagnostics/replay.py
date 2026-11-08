"""AlignAIR's predict pipeline replayed stage by stage, with every stage's output captured.

``replay`` does what ``alignair.api.predict_sequences`` does (``alignair.predict.pipeline.predict`` under the
``PredictConfig`` that ``predict_sequences`` builds), one stage at a time:

1. ``input_policy``: shared nucleotide normalization and cropping through ``prepare_nucleotide_inputs``.
2. ``forward``: the batched forward pass of ``alignair.predict.forward.run_model``, with each batch dict built here
   (``SequenceInputAdapter``, right-padded), which is where a teacher-forced ``batch["orientation"]`` goes.
3. ``clean``: ``stages.clean`` and ``stages.validate_predictions``.
4. ``canonicalize``: ``stages.canonicalize`` into the model's forward frame.
5. ``segments``: ``stages.correct_segments``.
6. ``alleles``: ``stages.select_alleles``.
7. ``germline``: ``stages.align_germline``.
8. ``better_v``: ``stages.better_v_alleles`` and ``stages.realign_v``, an allele the read matches better than
   its V call becoming the call (``alignair.predict.ties``); a read whose V call is substituted keeps it.
9. ``v_presence``: ``stages.v_absent``, the reads that hold no V (``alignair.predict.v_presence``).
10. ``ties``: ``stages.tied_alleles``, the V alleles each read cannot tell from its call (``alignair.predict.ties``).
11. ``records``: ``stages.to_records``, which drops the V of the reads without one, lists each V call's ties and,
    for a reference of several loci, labels each read with its V call's locus (``stages.called_loci``).
12. ``airr``: ``alignair.predict.airr.build_airr``.

A genotype is the bank the model is bound to (``predict_sequences``' ``genotype``): the replay takes the model as it
is bound, so a genotype run is replayed by binding its bank first.

Every stage is AlignAIR's own function, looked up where ``pipeline.predict`` looks it up; none is re-implemented.
Substitutions (``Substitutions``) happen only between stage calls. The private AlignAIR names called here
(``_validate_predictions``, ``_canonicalize``, ``_to_records``) are pinned by the replay's equivalence tests, which also pin the code of
the product functions the replay mirrors.

**Substitutions**, each fed in between two stages (``Substitutions`` documents the fields):

- ``orientation``: teacher-forced in the forward pass (``batch["orientation"]``, the label AlignAIR trains with), so
  the network after the orientation head runs on the read in the given frame, and set as the predicted orientation
  before ``canonicalize`` (the orientation head's own output is unchanged), so the read, the coordinates and the
  records are in that frame.
- ``decoder="soft"``: the boundaries ``correct_segments`` receives are the model's soft-argmax expectations (the
  decoder the product used before it decoded by argmax) instead of the argmax of the position logits that
  ``stages.clean`` reports.
- ``spans``: replace the corrected ``Segments`` of the given reads and genes; the chip reads the whole read for its
  calls, so its allele probabilities stay its own. ``low_quality`` keeps ``correct_segments``' meaning: a V or J of zero length. The given spans are not
  re-ordered against the model's spans of other genes, so a truth span is never moved.
- ``calls``: replace the selector's calls, before the germline alignment.
- ``germline``: replace the matcher's alignments; the matcher runs only for the reads with a gene left to it. A
  substituted V alignment is the matcher's output, so ``better_v`` may still move the read to a better allele unless
  its V call is substituted too.

Every per-read entry may be None, which keeps the model's value for that read (a fallback). Substituted values are
checked (read range, model alleles, an alignment on the read's call); a bad one is a ValueError.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from alignair.api import load_model
from alignair.core.runtime import evaluation_mode
from alignair.data import SequenceInputAdapter, prepare_nucleotide_inputs
from alignair.data.tokenizer import NucleotideTokenizer
from alignair.network.orientation import NUM_ORIENTATIONS
from alignair.predict import PredictConfig, forward, stages
from alignair.predict.airr import build_airr
from alignair.predict.state import Segments

from alignair_assay.core.reference import Reference

# The replayed stages, in pipeline order: the keys of ``StageOutputs.seconds``.
STAGES = ("input_policy", "forward", "clean", "canonicalize", "segments", "alleles", "germline", "better_v",
          "v_presence", "ties", "records", "airr")
# "argmax": the logits' argmax, which the product decodes boundaries with; "soft": the soft-argmax expectation.
DECODERS = ("soft", "argmax")


@dataclass(frozen=True)
class StageOutputs:
    """Every stage's output for one replay, one entry per input read, in input order."""

    ids: list[str]
    # every AlignAIR.forward output key (logits included), concatenated over the batches
    raw: dict[str, np.ndarray]
    # alignair.predict.state.Predictions after clean
    predictions: object
    # the reads in the model's forward frame (after the input policy and canonicalization)
    canonical: list[str]
    # alignair.predict.state.Segments after correct_segments (read frame, 0-based, end-exclusive)
    segments: object
    # gene -> one alignair.predict.state.GeneCall per read
    calls: dict[str, list]
    # gene -> one alignair.predict.state.GermlineAlignment (or None) per read
    germline: dict[str, list]
    # _to_records output
    records: list[dict]
    # build_airr output: the records predict_sequences returns
    airr_rows: list[dict]
    # wall time per stage, keyed by STAGES in order
    seconds: dict[str, float]
    # gene -> allele names, index-aligned to the model's allele head ({g}_allele columns)
    allele_names: dict[str, list[str]]
    # per read: whether it holds no V (stages.v_absent), so its record has no V call; None when not replayed
    no_v: list[bool] | None = None
    # per read: its V call and the alleles tied with it (stages.tied_alleles), () without a V alignment
    v_ties: list[tuple] | None = None


@dataclass(frozen=True)
class Substitutions:
    """Values fed to the pipeline between stage calls in place of the model's output (see the module docstring).
    Each per-read list has one entry per input read, in input order; a None entry keeps the model's value for that
    read. The defaults substitute nothing, which is a pure replay. ValueError for an unknown decoder."""

    # per read: the orientation id (0 identity, 1 reverse complement, 2 complement, 3 reverse)
    orientation: list[int | None] | None = None
    # gene -> (starts, ends) per read, on the read in the model's forward frame (after the crop and the orientation),
    # 0-based, end-exclusive
    spans: dict[str, tuple[list[int | None], list[int | None]]] | None = None
    # gene -> alignair.predict.state.GeneCall per read
    calls: dict[str, list] | None = None
    # gene -> alignair.predict.state.GermlineAlignment per read: forward-frame read coordinates, ungapped germline
    # coordinates of the read's top-1 call
    germline: dict[str, list] | None = None
    decoder: str = "argmax"

    def __post_init__(self) -> None:
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder {self.decoder!r} is not one of {', '.join(DECODERS)}")

    def active(self) -> tuple[str, ...]:
        """Names of the fields that substitute something; empty for a pure replay."""
        names = [name for name in ("orientation", "spans", "calls", "germline") if getattr(self, name) is not None]
        if self.decoder != "argmax":
            names.append("decoder")
        return tuple(names)


def load_model_checked(checkpoint_or_model, truth_reference: Reference, *, device: str) -> tuple[object, object]:
    """``(model, reference_set)`` on ``device`` whose germline reference is the truth reference.

    A path (``str`` or path-like) is loaded with ``alignair.api.load_model`` (a ``.alignair`` model file; the model
    comes back in eval mode). A ``(model, reference_set)`` pair, for example from a training loop, is passed through:
    the model is unwrapped (``unwrap_model``), its reference checked, and only then moved to ``device`` in place;
    moving it back is the caller's (``workflow.diagnose.evaluate`` does). ValueError when the reference does not
    hold exactly the truth reference's alleles and sequences (equal digests), naming both references' allele counts
    and the first allele, in name order, that is missing from the model's reference, extra in it, or has a different
    sequence there."""
    if isinstance(checkpoint_or_model, (str, os.PathLike)):
        model, reference_set = load_model(str(Path(checkpoint_or_model).expanduser()), device=device)
        check_reference(reference_set, truth_reference)
        return model, reference_set
    model, reference_set = checkpoint_or_model
    model = unwrap_model(model)
    check_reference(reference_set, truth_reference)
    model.to(device)
    return model, reference_set


def unwrap_model(model):
    """The AlignAIR module a training loop's model wraps: ``torch.compile``'s ``_orig_mod`` and ``DataParallel``'s
    or ``DistributedDataParallel``'s ``module``, unwrapped repeatedly; the model itself when it wraps nothing. The
    replay reads the model's own attributes (its ``cfg``), which a wrapper does not expose. The
    wrapper shares the module's parameters, so moving the module moves the wrapper's weights too."""
    parallel = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    while True:
        if isinstance(model, torch.nn.Module) and isinstance(getattr(model, "_orig_mod", None), torch.nn.Module):
            model = model._orig_mod
        elif isinstance(model, parallel):
            model = model.module
        else:
            return model


def model_device(model) -> torch.device:
    """The device of the model's first parameter (the CPU for a model without parameters)."""
    parameter = next(iter(model.parameters()), None)
    return torch.device("cpu") if parameter is None else parameter.device


def check_reference(reference_set, truth_reference: Reference) -> None:
    """ValueError unless the ``alignair`` ReferenceSet ``reference_set`` holds exactly the truth reference's
    alleles and sequences (equal digests), naming both references' allele counts and the first allele, in name
    order, that is missing from the model's reference, extra in it, or has a different sequence there."""
    model_reference = Reference.from_reference_set(reference_set)
    if model_reference.digest() != truth_reference.digest():
        raise ValueError(f"the model's reference is not the truth reference: it has {model_reference.counts_text()}, "
                         f"the truth reference has {truth_reference.counts_text()}; first difference: "
                         f"{model_reference.first_difference(truth_reference, 'model')}")


def predict_config(model, reference_set, *, batch_size: int, **overrides) -> PredictConfig:
    """The ``PredictConfig`` that ``alignair.api.predict_sequences`` builds for ``model``, ``reference_set``,
    ``batch_size`` and ``overrides`` (``PredictConfig.for_model``)."""
    return PredictConfig.for_model(model, reference_set, batch_size=batch_size, **overrides)


def replay(model, reference_set, sequences: list[str], ids: list[str], *, device: str, batch_size: int,
           substitutions: Substitutions = Substitutions()) -> StageOutputs:
    """Run AlignAIR's predict pipeline on ``sequences`` (named by ``ids``) stage by stage and return every stage's
    output, with ``substitutions`` fed in between the stages (see the module docstring). Without substitutions,
    ``airr_rows`` equals what ``predict_sequences(model, reference_set, sequences, device=device,
    batch_size=batch_size)`` returns. The model must be on ``device``; it runs in eval mode and is put back in its
    previous mode. ValueError when the ids and sequences differ in number, there is no sequence, or a substitution
    does not fit the reads or the model (a list of another length, a gene the model lacks, an orientation id outside
    0-3, a span or an alignment outside its read or an empty span, a call naming no allele or one the model cannot
    call, an alignment on another allele than the read's call)."""
    if len(ids) != len(sequences):
        raise ValueError(f"{len(ids)} ids for {len(sequences)} sequences")
    if not sequences:
        raise ValueError("no sequences to replay")
    subs = substitutions
    cfg = predict_config(model, reference_set, batch_size=batch_size)
    genes = cfg.genes
    names = {g: list(reference_set.gene(g.upper()).names) for g in genes}
    _check_substitutions(subs, len(sequences), genes, names)
    seconds: dict[str, float] = {}

    with _timed(seconds, "input_policy"):
        reads, was_cropped = prepare_nucleotide_inputs(sequences, cfg.window)
    _check_spans(subs.spans, [len(read) for read in reads])  # orientation keeps a read's length
    with _timed(seconds, "forward"):
        batches = _forward(model, reads, cfg, device, orientation=subs.orientation)
    raw = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    with _timed(seconds, "clean"):
        preds = stages.clean(batches, genes)
        stages.validate_predictions(preds)
        if subs.orientation is not None:
            preds = replace(preds, orientation=np.array([int(p) if o is None else int(o) for o, p
                                                         in zip(subs.orientation, preds.orientation)], dtype=int))
    with _timed(seconds, "canonicalize"):
        orient = preds.orientation
        canonical = [stages.canonicalize(s, int(orient[i]) if orient is not None else 0)
                     for i, s in enumerate(reads)]
        seq_lens = np.array([len(s) for s in canonical])
    with _timed(seconds, "segments"):
        start, end = preds.start, preds.end
        if subs.decoder == "soft":
            start = {g: raw[f"{g}_start"].reshape(-1) for g in genes}
            end = {g: raw[f"{g}_end"].reshape(-1) for g in genes}
        segments = stages.correct_segments(start, end, seq_lens, cfg.max_seq_length, cfg.pad_mode)
        if subs.spans is not None:
            segments = _substitute_spans(segments, subs.spans)
    with _timed(seconds, "alleles"):
        calls = stages.select_alleles(preds.allele, names, cfg.threshold, cfg.cap, cfg.selector)
        calls = _substitute(calls, subs.calls)
    with _timed(seconds, "germline"):
        germline = _align(canonical, segments, calls, reference_set, cfg, preds.indel_count, subs.germline)
    with _timed(seconds, "better_v"):
        given = [subs.calls is not None and subs.calls.get("v") is not None and subs.calls["v"][i] is not None
                 for i in range(len(canonical))]
        better = stages.better_v_alleles(canonical, germline["v"], reference_set, [call.names for call in calls["v"]])
        better = [None if held else allele for held, allele in zip(given, better)]
        germline["v"] = stages.realign_v(canonical, segments, germline["v"], better, reference_set,
                                           indel_counts=preds.indel_count)
    with _timed(seconds, "v_presence"):
        no_v = stages.v_absent(canonical, germline["v"], reference_set)
    with _timed(seconds, "ties"):
        v_ties = stages.tied_alleles(canonical, germline["v"], reference_set)
    with _timed(seconds, "records"):
        records = stages.to_records(canonical, calls, germline, genes, preds, cfg.chain_types,
                                       segments.low_quality, was_cropped, input_sequences=reads, no_v=no_v,
                                       v_ties=v_ties, read_loci=stages.called_loci(germline["v"], reference_set, no_v))
    with _timed(seconds, "airr"):
        airr_rows = (build_airr(records, reference_set, chain=("heavy" if cfg.has_d else "light")) if cfg.airr
                     else records)

    return StageOutputs(ids=list(ids), raw=raw, predictions=preds, canonical=canonical, segments=segments,
                        calls=calls, germline=germline, records=records, airr_rows=airr_rows, seconds=seconds,
                        allele_names=names, no_v=no_v, v_ties=v_ties)


@contextmanager
def _timed(seconds: dict[str, float], stage: str):
    start = time.perf_counter()
    yield
    seconds[stage] = time.perf_counter() - start


@torch.no_grad()
def _forward(model, reads: list[str], cfg: PredictConfig, device: str, *, orientation: list | None = None) -> list[dict]:
    """``run_model``'s batched forward pass: the same batches and tokens, each batch's outputs as numpy arrays.
    ``orientation`` (per read, None for the model's own) is teacher-forced."""
    adapter = SequenceInputAdapter(NucleotideTokenizer(), cfg.max_seq_length, device, vocab_size=6)
    with evaluation_mode(model):
        outputs = []
        for start in range(0, len(reads), cfg.batch_size):
            stop = min(start + cfg.batch_size, len(reads))
            batch = adapter(reads[start:stop])
            tokens = batch["tokenized_sequence"]
            if orientation is not None:
                batch["orientation"] = _forced_orientation(model, tokens, orientation[start:stop])
            out = forward.forward_batch(model, batch, expected_rows=stop - start)
            outputs.append({key: forward.output_numpy(value) for key, value in out.items()})
        return outputs


def _forced_orientation(model, tokens: torch.Tensor, given: list) -> torch.Tensor:
    """The orientation ids to teacher-force for one batch: the given ones, and the model's own prediction (from a
    forward pass without forcing) for a None entry."""
    if any(o is None for o in given):
        own = model({"tokenized_sequence": tokens})["orientation"].tolist()
        given = [own[k] if o is None else o for k, o in enumerate(given)]
    return torch.tensor([int(o) for o in given], dtype=torch.long, device=tokens.device)


def _check_substitutions(subs: Substitutions, n: int, genes, names: dict[str, list[str]]) -> None:
    """The checks that need no read: one entry per read, the model's genes, orientation ids, span pairs and
    callable alleles."""
    if subs.orientation is not None:
        _check_length("orientation", subs.orientation, n)
        for i, o in enumerate(subs.orientation):
            if o is not None and int(o) not in range(NUM_ORIENTATIONS):
                raise ValueError(f"orientation id {o} of read {i} is not one of 0, 1, 2, 3")
    for field in ("spans", "calls", "germline"):
        for g in getattr(subs, field) or {}:
            if g not in genes:
                raise ValueError(f"{field} name gene {g!r}, which the model does not have (its genes: "
                                 f"{', '.join(genes)})")
    for g, (starts, ends) in (subs.spans or {}).items():
        _check_length(f"spans of {g}", starts, n)
        _check_length(f"spans of {g}", ends, n)
        for i, (s, e) in enumerate(zip(starts, ends)):
            if (s is None) != (e is None):
                raise ValueError(f"{g} span of read {i}: its start and end must both be given or both be None")
    for g, calls in (subs.calls or {}).items():
        _check_length(f"calls of {g}", calls, n)
        known = set(names[g])
        for i, call in enumerate(calls):
            if call is None:
                continue
            if not call.names:
                raise ValueError(f"the {g} call of read {i} names no allele")
            unknown = [name for name in call.names if name not in known]
            if unknown:
                raise ValueError(f"{unknown[0]} of read {i} is not a {g} allele of the model")
    for g, alignments in (subs.germline or {}).items():
        _check_length(f"germline of {g}", alignments, n)


def _check_length(label: str, values: list, n: int) -> None:
    if len(values) != n:
        raise ValueError(f"{label}: {len(values)} entries for {n} reads")


def _check_spans(spans: dict | None, lengths: list[int]) -> None:
    """Every given span is non-empty and inside its read."""
    for g, (starts, ends) in (spans or {}).items():
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s is None:
                continue
            if not s < e:
                raise ValueError(f"{g} span {s}..{e} of read {i} is empty")
            if s < 0 or e > lengths[i]:
                raise ValueError(f"{g} span {s}..{e} of read {i} lies outside read {i} ({lengths[i]} bases)")


def _substitute(values: dict[str, list], given: dict[str, list] | None) -> dict[str, list]:
    """``values`` with each gene's non-None ``given`` entries in place of the read's value."""
    if not given:
        return values
    out = {g: list(entries) for g, entries in values.items()}
    for g, entries in given.items():
        for i, value in enumerate(entries):
            if value is not None:
                out[g][i] = value
    return out


def _substitute_spans(segments: Segments, spans: dict) -> Segments:
    """The corrected segments with the given spans in place, and ``low_quality`` as ``correct_segments`` defines it
    (a V or J of zero length) on the result."""
    start = {g: np.array(values, copy=True) for g, values in segments.start.items()}
    end = {g: np.array(values, copy=True) for g, values in segments.end.items()}
    for g, (starts, ends) in spans.items():
        for i, (s, e) in enumerate(zip(starts, ends)):
            if s is not None:
                start[g][i], end[g][i] = s, e
    low_quality = ((end["v"] - start["v"]) <= 0) | ((end["j"] - start["j"]) <= 0)
    return Segments(start, end, low_quality=low_quality)


def _align(canonical: list[str], segments: Segments, calls: dict, reference_set, cfg: PredictConfig, indel_counts,
           given: dict | None) -> dict[str, list]:
    """``align_germline``'s alignments, with the given ones in place; the matcher runs only on the reads that have a
    gene without a given alignment (each read is aligned on its own, so a subset gives the same alignments)."""
    if not given:
        return stages.align_germline(canonical, segments, calls, reference_set, indel_counts=indel_counts)
    genes = list(segments.start)
    for g, alignments in given.items():
        for i, alignment in enumerate(alignments):
            if alignment is None:
                continue
            call = calls[g][i].names[0] if calls[g][i].names else None
            if alignment.allele != call:
                raise ValueError(f"the {g} alignment of read {i} is on {alignment.allele}, which is not the read's "
                                 f"{g} call ({call or 'no call'})")
            if not 0 <= alignment.seq_start <= alignment.seq_end <= len(canonical[i]):
                raise ValueError(f"the {g} alignment {alignment.seq_start}..{alignment.seq_end} of read {i} lies "
                                 f"outside read {i} ({len(canonical[i])} bases)")
    left = [i for i in range(len(canonical)) if any(g not in given or given[g][i] is None for g in genes)]
    aligned: dict[str, list] = {g: [None] * len(canonical) for g in genes}
    if left:
        rows = np.array(left)
        part = Segments({g: segments.start[g][rows] for g in genes}, {g: segments.end[g][rows] for g in genes},
                        None if segments.low_quality is None else segments.low_quality[rows])
        matched = stages.align_germline([canonical[i] for i in left], part,
                                          {g: [calls[g][i] for i in left] for g in genes}, reference_set,
                                          indel_counts=None if indel_counts is None else indel_counts[rows])
        for k, i in enumerate(left):
            for g in genes:
                aligned[g][i] = matched[g][k]
    return _substitute(aligned, given)
