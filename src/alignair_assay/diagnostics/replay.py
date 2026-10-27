"""AlignAIR's predict pipeline replayed stage by stage, with every stage's output captured.

``replay`` does what ``alignair.api.predict_sequences`` does (``alignair.predict.pipeline.predict`` under the
``PredictConfig`` that ``predict_sequences`` builds), one stage at a time:

1. ``input_policy``: shared nucleotide normalization and cropping through ``prepare_nucleotide_inputs``.
2. ``forward``: the batched forward pass of ``alignair.predict.forward.run_model``, with each batch dict built here
   (``SequenceInputAdapter``, right-padded), which is where a teacher-forced ``batch["orientation"]`` goes.
3. ``clean``: ``pipeline.clean`` and ``pipeline._validate_predictions``.
4. ``masks``: the allele temperature calibration (``calibrate.apply_temperature``) and the per-read locus masks of a
   multi-locus model (``pipeline._locus_allowed``). The genotype constraint needs a ``genotype`` override, which the
   replay never sets.
5. ``canonicalize``: ``pipeline._canonicalize`` into the model's forward frame.
6. ``segments``: ``pipeline.correct_segments``.
7. ``alleles``: ``pipeline.select_alleles``.
8. ``germline``: ``pipeline.align_germline``.
9. ``records``: ``pipeline._to_records``.
10. ``airr``: ``alignair.predict.airr.build_airr``.

Every stage is AlignAIR's own function, looked up where ``pipeline.predict`` looks it up; none is re-implemented.
Substitutions (``Substitutions``) happen only between stage calls. The private AlignAIR names called here
(``pipeline._genes``, ``_validate_predictions``, ``_assert_finite``, ``_locus_allowed``, ``_canonicalize``,
``_to_records``) are pinned by the replay's equivalence tests, which also pin the code of
the product functions the replay mirrors.

**Substitutions**, each fed in between two stages (``Substitutions`` documents the fields):

- ``orientation``: teacher-forced in the forward pass (``batch["orientation"]``, the label AlignAIR trains with), so
  the network after the orientation head runs on the read in the given frame, and set as the predicted orientation
  before ``canonicalize`` (the orientation head's own output is unchanged), so the read, the coordinates and the
  records are in that frame.
- ``decoder="argmax"``: the boundaries ``correct_segments`` receives are the argmax of the position logits
  (``argmax_boundaries``) instead of the soft-argmax expectations. The classifier still reads the expectations.
- ``spans``: replace the corrected ``Segments`` of the given reads and genes (``segment_spans``), and with
  ``classifier_spans`` are fed to each gene's classifier (``model.branches[g].classify`` on the re-embedded read, the
  public seam of ``AlignAIR.forward``) in place of the soft-argmax expectations; the new allele probabilities then
  go through the calibration, the masks and the selector. ``low_quality`` keeps ``correct_segments``' meaning: a V or
  J of zero length. The given spans are not re-ordered against the model's spans of other genes, so a truth span is
  never moved.
- ``calls``: replace the selector's calls, before the germline alignment.
- ``germline``: replace the matcher's alignments; the matcher runs only for the reads with a gene left to it.

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
from alignair.data.tokenizer import CenterPaddedTokenizer
from alignair.neural.heads.orientation import NUM_ORIENTATIONS, apply_orientation
from alignair.predict import PredictConfig, forward, pipeline
from alignair.predict.airr import build_airr
from alignair.predict.calibrate import apply_temperature
from alignair.predict.state import Segments

from alignair_assay.core.reference import Reference

# The replayed stages, in pipeline order: the keys of ``StageOutputs.seconds``.
STAGES = ("input_policy", "forward", "clean", "masks", "canonicalize", "segments", "alleles", "germline", "records",
          "airr")
# "soft": the soft-argmax expectation the product decodes boundaries with; "argmax": the logits' argmax.
DECODERS = ("soft", "argmax")


@dataclass(frozen=True)
class StageOutputs:
    """Every stage's output for one replay, one entry per input read, in input order."""

    ids: list[str]
    # every AlignAIR.forward output key (logits included), concatenated over the batches; with classifier_spans,
    # {g}_allele is what the classifier gave on the substituted spans
    raw: dict[str, np.ndarray]
    # alignair.predict.state.Predictions after clean and the masks stage (calibrated allele probabilities)
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
    # gene -> (reads, alleles) bool: the alleles each read may be called with (the per-read locus masks of a
    # multi-locus model, which select_alleles applied); None when the selector ran unmasked
    allowed: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class Substitutions:
    """Values fed to the pipeline between stage calls in place of the model's output (see the module docstring).
    Each per-read list has one entry per input read, in input order; a None entry keeps the model's value for that
    read. The defaults substitute nothing, which is a pure replay. ValueError for an unknown decoder,
    ``classifier_spans`` without ``spans``, or ``spans`` fed nowhere."""

    # per read: the orientation id (0 identity, 1 reverse complement, 2 complement, 3 reverse)
    orientation: list[int | None] | None = None
    # gene -> (starts, ends) per read, on the read in the model's forward frame (after the crop and the orientation),
    # 0-based, end-exclusive
    spans: dict[str, tuple[list[int | None], list[int | None]]] | None = None
    # use the spans as the corrected segments (False: they go to the classifier only)
    segment_spans: bool = True
    # feed the spans into each gene's classifier instead of the soft-argmax expectations
    classifier_spans: bool = False
    # gene -> alignair.predict.state.GeneCall per read
    calls: dict[str, list] | None = None
    # gene -> alignair.predict.state.GermlineAlignment per read: forward-frame read coordinates, ungapped germline
    # coordinates of the read's top-1 call
    germline: dict[str, list] | None = None
    decoder: str = "soft"

    def __post_init__(self) -> None:
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder {self.decoder!r} is not one of {', '.join(DECODERS)}")
        if self.classifier_spans and self.spans is None:
            raise ValueError("classifier_spans needs spans to feed the classifier")
        if self.spans is not None and not (self.segment_spans or self.classifier_spans):
            raise ValueError("spans are fed nowhere: set segment_spans or classifier_spans")

    def active(self) -> tuple[str, ...]:
        """Names of the fields that substitute something (``spans`` only when they replace the segments); empty for
        a pure replay."""
        names = [name for name in ("orientation", "spans", "calls", "germline") if getattr(self, name) is not None
                 and (name != "spans" or self.segment_spans)]
        if self.classifier_spans:
            names.append("classifier_spans")
        if self.decoder != "soft":
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
    replay calls the model's own submodules (``embedding``, ``branches``), which a wrapper does not expose. The
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
    ``batch_size`` and ``overrides``: the model's window, D presence and allele temperatures, and the reference's
    loci as the chain types unless ``overrides`` names them."""
    cfg = model.cfg
    chain_types = (reference_set.locus_names() or None) if hasattr(reference_set, "locus_names") else None
    overrides.setdefault("chain_types", chain_types)
    return PredictConfig(max_seq_length=cfg.max_seq_length, has_d=cfg.has_d, batch_size=batch_size,
                         allele_temperatures=getattr(cfg, "allele_temperatures", None), **overrides)


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
    genes = pipeline._genes(cfg)
    names = {g: list(reference_set.gene(g.upper()).names) for g in genes}
    _check_substitutions(subs, len(sequences), genes, names)
    seconds: dict[str, float] = {}

    with _timed(seconds, "input_policy"):
        reads, was_cropped = prepare_nucleotide_inputs(sequences, cfg.max_seq_length)
    _check_spans(subs.spans, [len(read) for read in reads])  # orientation keeps a read's length
    with _timed(seconds, "forward"):
        batches = _forward(model, reads, cfg, device, orientation=subs.orientation,
                           spans=subs.spans if subs.classifier_spans else None)
    raw = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    with _timed(seconds, "clean"):
        preds = pipeline.clean(batches, genes)
        pipeline._validate_predictions(preds)
        if subs.orientation is not None:
            preds = replace(preds, orientation=np.array([int(p) if o is None else int(o) for o, p
                                                         in zip(subs.orientation, preds.orientation)], dtype=int))
    with _timed(seconds, "masks"):
        allowed = _masks(preds, reference_set, cfg, genes)
    with _timed(seconds, "canonicalize"):
        orient = preds.orientation
        canonical = [pipeline._canonicalize(s, int(orient[i]) if orient is not None else 0)
                     for i, s in enumerate(reads)]
        seq_lens = np.array([len(s) for s in canonical])
    with _timed(seconds, "segments"):
        start, end = preds.start, preds.end
        if subs.decoder == "argmax":
            decoded = {g: argmax_boundaries(raw[f"{g}_start_logits"], raw[f"{g}_end_logits"], seq_lens)
                       for g in genes}
            start, end = {g: decoded[g][0] for g in genes}, {g: decoded[g][1] for g in genes}
        segments = pipeline.correct_segments(start, end, seq_lens, cfg.max_seq_length, cfg.pad_mode)
        if subs.spans is not None and subs.segment_spans:
            segments = _substitute_spans(segments, subs.spans)
    with _timed(seconds, "alleles"):
        calls = pipeline.select_alleles(preds.allele, names, cfg.threshold, cfg.cap, cfg.selector, allowed=allowed)
        calls = _substitute(calls, subs.calls)
    with _timed(seconds, "germline"):
        germline = _align(canonical, segments, calls, reference_set, cfg, preds.indel_count, subs.germline)
    with _timed(seconds, "records"):
        records = pipeline._to_records(canonical, calls, germline, genes, preds, cfg.chain_types,
                                       segments.low_quality, was_cropped, input_sequences=reads)
    with _timed(seconds, "airr"):
        airr_rows = (build_airr(records, reference_set, chain=("heavy" if cfg.has_d else "light")) if cfg.airr
                     else records)

    return StageOutputs(ids=list(ids), raw=raw, predictions=preds, canonical=canonical, segments=segments,
                        calls=calls, germline=germline, records=records, airr_rows=airr_rows, seconds=seconds,
                        allele_names=names, allowed=allowed)


def argmax_boundaries(start_logits, end_logits, lengths) -> tuple[np.ndarray, np.ndarray]:
    """The argmax decoder: per read, ``(starts, ends)`` (0-based, end-exclusive) as the argmax of the position logits
    over the read's coordinates. A start lies on a read base (``0..n-1``); an end-exclusive end on ``0..n``, where
    position ``n`` (one past the last base) exists when the read is shorter than the logits window, so a segment that
    runs to the read end can be decoded. Logits past the read are never decoded. The model's own soft-argmax is
    masked to ``0..n-1`` for both."""
    lengths = np.asarray(lengths)
    return _masked_argmax(start_logits, lengths), _masked_argmax(end_logits, lengths + 1)


@contextmanager
def _timed(seconds: dict[str, float], stage: str):
    start = time.perf_counter()
    yield
    seconds[stage] = time.perf_counter() - start


def _masked_argmax(logits, limits: np.ndarray) -> np.ndarray:
    """Per row, the argmax over positions ``0..limit-1``."""
    logits = np.asarray(logits)
    positions = np.arange(logits.shape[1])
    return np.where(positions[None, :] < limits[:, None], logits, -np.inf).argmax(axis=1)


@torch.no_grad()
def _forward(model, reads: list[str], cfg: PredictConfig, device: str, *, orientation: list | None = None,
             spans: dict | None = None) -> list[dict]:
    """``run_model``'s batched forward pass: the same batches and tokens, each batch's outputs as numpy arrays.
    ``orientation`` (per read, None for the model's own) is teacher-forced; with ``spans``, each gene's allele
    probabilities are the classifier's on those spans (``_classify``)."""
    adapter = SequenceInputAdapter(CenterPaddedTokenizer(), cfg.max_seq_length, device, vocab_size=6)
    with evaluation_mode(model):
        outputs = []
        for start in range(0, len(reads), cfg.batch_size):
            stop = min(start + cfg.batch_size, len(reads))
            batch = adapter(reads[start:stop])
            tokens = batch["tokenized_sequence"]
            if orientation is not None:
                batch["orientation"] = _forced_orientation(model, tokens, orientation[start:stop])
            out = forward.forward_batch(model, batch, expected_rows=stop - start)
            if spans is not None:
                _classify(model, tokens, batch.get("orientation", out["orientation"]), out, spans, start)
            outputs.append({key: forward.output_numpy(value) for key, value in out.items()})
        return outputs


def _forced_orientation(model, tokens: torch.Tensor, given: list) -> torch.Tensor:
    """The orientation ids to teacher-force for one batch: the given ones, and the model's own prediction (from a
    forward pass without forcing) for a None entry."""
    if any(o is None for o in given):
        own = model({"tokenized_sequence": tokens})["orientation"].tolist()
        given = [own[k] if o is None else o for k, o in enumerate(given)]
    return torch.tensor([int(o) for o in given], dtype=torch.long, device=tokens.device)


def _classify(model, tokens: torch.Tensor, orientation: torch.Tensor, out: dict, spans: dict, offset: int) -> None:
    """Replace the batch's ``{g}_allele`` with ``model.branches[g].classify`` on the given spans (the batch's
    soft-argmax expectations for a None entry), on the read re-embedded in the frame of ``orientation``, as
    ``AlignAIR.forward`` embeds it for its own classifier call."""
    emb = model.embedding(apply_orientation(tokens, out["position_mask"], orientation))
    for g, (starts, ends) in spans.items():
        s, e = out[f"{g}_start"].clone(), out[f"{g}_end"].clone()
        for k in range(tokens.shape[0]):
            if starts[offset + k] is not None:
                s[k, 0], e[k, 0] = float(starts[offset + k]), float(ends[offset + k])
        out[f"{g}_allele"] = model.branches[g].classify(emb, s, e)


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
        return pipeline.align_germline(canonical, segments, calls, reference_set, None,
                                       reader=cfg.germline_reader, indel_counts=indel_counts)
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
        matched = pipeline.align_germline([canonical[i] for i in left], part,
                                          {g: [calls[g][i] for i in left] for g in genes}, reference_set, None,
                                          reader=cfg.germline_reader,
                                          indel_counts=None if indel_counts is None else indel_counts[rows])
        for k, i in enumerate(left):
            for g in genes:
                aligned[g][i] = matched[g][k]
    return _substitute(aligned, given)


def _masks(preds, reference_set, cfg: PredictConfig, genes) -> dict | None:
    """``pipeline.predict``'s calibration and locus masking: calibrates ``preds.allele`` in place and returns the
    per-read allowed alleles of a multi-locus model (None otherwise)."""
    allowed = None
    if cfg.allele_temperatures:
        for g, temperature in cfg.allele_temperatures.items():
            if g in preds.allele:
                preds.allele[g] = apply_temperature(preds.allele[g], temperature)
        pipeline._assert_finite(preds.allele, "temperature calibration")
    if preds.chain_type is not None and getattr(reference_set, "loci", None):
        allowed = pipeline._locus_allowed(preds.chain_type, reference_set, genes, allowed)
    return allowed
