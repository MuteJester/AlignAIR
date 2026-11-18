"""AlignAIR's predict pipeline run stage by stage, with every stage's output captured.

``replay`` runs the product's own pipeline (``alignair.predict.pipeline.PIPELINE``, ``run_stages``), the one
``alignair.Aligner.predict`` runs, under the ``PredictConfig`` it builds (``PredictConfig.for_model``), with two
hooks: ``Trace``, which keeps each stage's output and seconds, and ``_SubstitutionHook``, which feeds the substitutions
in. Nothing is re-implemented here, so a change to a product stage reaches the replay as it reaches the product. The
stages, in order (``STAGES``): ``input_policy``, ``forward``, ``clean``, ``canonicalize``, ``segments``, ``alleles``,
``germline``, ``better_v`` (an allele the read matches better than its V call becomes the call), ``v_presence``,
``ties``, ``records``, ``airr``.

A genotype is the bank the model is bound to (``Aligner.with_genotype``): the replay takes the model as it is bound,
so a genotype run is replayed by binding its bank first.

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
  its V call is substituted too (a read whose V call is substituted keeps it).

Every per-read entry may be None, which keeps the model's value for that read (a fallback). Substituted values are
checked (read range, model alleles, an alignment on the read's call); a bad one is a ValueError.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from alignair.model_file import load_model
from alignair.core.orientation import NUM_ORIENTATIONS
from alignair.predict import PredictConfig
from alignair.predict.clean import expectations
from alignair.predict.pipeline import PIPELINE, STAGE_NAMES, Trace, run_stages
from alignair.predict.results import light_record
from alignair.predict.state import PredictionState, Segments

from alignair_assay.core.reference import Reference

# The product's stages, in pipeline order: the keys of ``StageOutputs.seconds``.
STAGES = STAGE_NAMES
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
    # each read's light record (alignair.predict.results.light_record)
    records: list[dict]
    # build_airr output: the records Aligner.predict returns
    airr_rows: list[dict]
    # wall time per stage, keyed by STAGES in order
    seconds: dict[str, float]
    # gene -> allele names, index-aligned to the model's allele head ({g}_allele columns)
    allele_names: dict[str, list[str]]
    # per read: whether it holds no V (stages.v_absent), so its record has no V call; None when not replayed
    no_v: list[bool] | None = None
    # per read: its V call and the alleles tied with it (stages.tied_alleles), () without a V alignment
    v_ties: list[tuple] | None = None
    # per read: its typed result (alignair.predict.results.ReadResult); None when not replayed
    results: list | None = None


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

    A path (``str`` or path-like) is loaded with ``alignair.model_file.load_model`` (a ``.alignair`` model file;
    the model comes back in eval mode). A ``(model, reference_set)`` pair, for example from a training loop, is
    passed through: the model is unwrapped (``unwrap_model``), its reference checked, and only then moved to
    ``device`` in place; moving it back is the caller's (``workflow.diagnose.evaluate`` does). ValueError when the
    reference does not hold the truth reference (``check_reference``: every truth allele with its sequence and, in
    the truth's loci, no other), naming both references' allele counts and the first allele, in name order, that is
    missing from the model's reference, extra in a truth locus, or has a different sequence there."""
    if isinstance(checkpoint_or_model, (str, os.PathLike)):
        loaded = load_model(str(Path(checkpoint_or_model).expanduser()), device=device)
        model, reference_set = loaded.model, loaded.reference
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


def _extra_in_truth_loci(reference_set, truth_reference: Reference) -> str | None:
    """The first allele, in name order, that the model's reference holds in a locus of the truth's and the truth
    lacks (``"<name> is extra in the model reference"``), or None. A locus is the truth's when one of its J alleles is
    a truth allele (J alleles are never shared between loci); a reference of one locus (no locus schema) has only
    the truth's. An extra allele of the truth's own locus would be scored by name and count as a wrong call (an
    OGRDB+EXTENDED model on the OGRDB truth holds the same sequences under other names), so it is refused; the
    alleles of other loci (a model of several loci on a one-locus suite) are not."""
    truth = truth_reference.sequences()
    loci = reference_set.locus_names() if hasattr(reference_set, "locus_names") else ()
    genes = [g for g in ("V", "D", "J") if g in reference_set.genes]
    if len(loci) < 2:
        own = [np.ones(len(reference_set.gene(g)), dtype=bool) for g in genes]
    else:
        j_names = np.asarray(reference_set.gene("J").names)
        theirs = [locus for locus in loci if any(n in truth for n in j_names[reference_set.locus_mask(locus, "J")])]
        own = []
        for g in genes:
            mask = np.zeros(len(reference_set.gene(g)), dtype=bool)
            for locus in theirs:
                mask |= reference_set.locus_mask(locus, g)
            own.append(mask)
    extra = sorted(name for g, mask in zip(genes, own)
                   for name, keep in zip(reference_set.gene(g).names, mask) if keep and name not in truth)
    return f"{extra[0]} is extra in the model reference" if extra else None


def check_reference(reference_set, truth_reference: Reference) -> None:
    """ValueError unless the ``alignair`` ReferenceSet ``reference_set`` holds every allele of the truth reference
    with its sequence and, in the truth's loci, no other (``_extra_in_truth_loci``), naming both references' allele
    counts and the first allele, in name order, that is missing from the model's reference, has a different sequence
    there or is extra in a truth locus. A model of several loci is scored on a one-locus suite: its other loci's
    alleles are accepted (its calls of them are simply wrong)."""
    model_reference = Reference.from_reference_set(reference_set)
    if model_reference.digest() == truth_reference.digest():
        return
    problem = (model_reference.first_missing(truth_reference, "model")
               or _extra_in_truth_loci(reference_set, truth_reference))
    if problem is not None:
        raise ValueError(f"the model's reference is not the truth reference: it has {model_reference.counts_text()}, "
                         f"the truth reference has {truth_reference.counts_text()}; first difference: {problem}")


def predict_config(model, reference_set, *, batch_size: int, **overrides) -> PredictConfig:
    """The ``PredictConfig`` that ``alignair.Aligner.predict`` builds for ``model``, ``reference_set``,
    ``batch_size`` and ``overrides`` (``PredictConfig.for_model``)."""
    return PredictConfig.for_model(model, reference_set, batch_size=batch_size, **overrides)


def replay(model, reference_set, sequences: list[str], ids: list[str], *, device: str, batch_size: int,
           substitutions: Substitutions = Substitutions()) -> StageOutputs:
    """Run AlignAIR's predict pipeline on ``sequences`` (named by ``ids``) stage by stage and return every stage's
    output, with ``substitutions`` fed in between the stages (see the module docstring). Without substitutions,
    ``airr_rows`` equals the records ``Aligner.from_model(model, reference_set, device=device).predict(sequences,
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
    names = {g: list(reference_set.gene(g.upper()).names) for g in cfg.genes}
    _check_substitutions(subs, len(sequences), cfg.genes, names)
    trace = Trace()
    start = PredictionState.start(model, reference_set, cfg, device, sequences, orientation=subs.orientation)
    final = run_stages(start, PIPELINE, hooks=(trace, _SubstitutionHook(subs)))
    batches = [batch.network_dict() for batch in final.batches]
    raw = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    return StageOutputs(ids=list(ids), raw=raw, predictions=trace.states["clean"].predictions,
                        canonical=final.canonical, segments=final.segments, calls=final.calls,
                        germline=final.alignments, records=[light_record(r) for r in final.results],
                        airr_rows=final.rows, seconds=dict(trace.seconds), allele_names=names, no_v=final.no_v,
                        v_ties=final.v_ties, results=final.results)


class _SubstitutionHook:
    """The hook that feeds ``Substitutions`` in between the product's stages (the module docstring): it checks the
    spans once the reads are known, runs ``segments`` on the soft expectations and puts the given spans in, puts the
    given calls in after ``alleles``, runs ``germline`` only on the reads with a gene left to the matcher and puts the
    given alignments in, and holds a read whose V call is given out of ``better_v``. Every other stage runs as it is."""

    def __init__(self, substitutions: Substitutions):
        self.subs = substitutions

    def __call__(self, stage, state, run):
        handle = getattr(self, f"_{stage.name}", None)
        return run(state) if handle is None else handle(state, run)

    def _input_policy(self, state, run):
        out = run(state)
        _check_spans(self.subs.spans, [len(read) for read in out.reads])   # orientation keeps a read's length
        return out

    def _segments(self, state, run):
        subs = self.subs
        if subs.decoder == "soft":
            genes = state.context.cfg.genes
            soft_spans = {g: expectations(state.batches, g) for g in genes}
            soft = replace(state.predictions, start={g: soft_spans[g][0] for g in genes},
                           end={g: soft_spans[g][1] for g in genes})
            out = replace(run(replace(state, predictions=soft)), predictions=state.predictions)
        else:
            out = run(state)
        if subs.spans is not None:
            out = replace(out, segments=_substitute_spans(out.segments, subs.spans))
        return out

    def _alleles(self, state, run):
        out = run(state)
        return out if not self.subs.calls else replace(out, calls=_substitute(out.calls, self.subs.calls))

    def _germline(self, state, run):
        given = self.subs.germline
        if not given:
            return run(state)
        canonical, calls = state.canonical, state.calls
        for g, alignments in given.items():
            for i, alignment in enumerate(alignments):
                if alignment is None:
                    continue
                call = calls[g][i].names[0] if calls[g][i].names else None
                if alignment.allele != call:
                    raise ValueError(f"the {g} alignment of read {i} is on {alignment.allele}, which is not the "
                                     f"read's {g} call ({call or 'no call'})")
                if not 0 <= alignment.seq_start <= alignment.seq_end <= len(canonical[i]):
                    raise ValueError(f"the {g} alignment {alignment.seq_start}..{alignment.seq_end} of read {i} "
                                     f"lies outside read {i} ({len(canonical[i])} bases)")
        genes = list(state.segments.start)
        n = len(canonical)
        left = [i for i in range(n) if any(g not in given or given[g][i] is None for g in genes)]
        aligned: dict[str, list] = {g: [None] * n for g in genes}
        if left:   # each read is aligned on its own, so a subset gives the same alignments
            part = run(state.take(left)).alignments
            for k, i in enumerate(left):
                for g in genes:
                    aligned[g][i] = part[g][k]
        return replace(state, alignments=_substitute(aligned, given))

    def _better_v(self, state, run):
        given = (self.subs.calls or {}).get("v")
        if given is None:
            return run(state)
        free = [i for i, call in enumerate(given) if call is None]
        v = list(state.alignments["v"])
        if free:   # better_v reads each read on its own
            part = run(state.take(free)).alignments["v"]
            for k, i in enumerate(free):
                v[i] = part[k]
        return replace(state, alignments={**state.alignments, "v": v})


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
