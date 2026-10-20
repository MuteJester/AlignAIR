"""AlignAIR's predict pipeline replayed stage by stage, with every stage's output captured.

``replay`` does what ``alignair.api.predict_sequences`` does (``alignair.predict.pipeline.predict`` under the
``PredictConfig`` that ``predict_sequences`` builds), one stage at a time:

1. ``input_policy``: ``pipeline.apply_input_policy`` (upper case, reads cropped to the model's window).
2. ``forward``: the batched forward pass of ``alignair.predict.forward.run_model``, with each batch dict built here
   (``forward._tokenize``, right-padded), which is where a teacher-forced ``batch["orientation"]`` goes.
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
``_to_records`` and ``forward._tokenize``) are pinned by the replay's equivalence tests, which also pin the code of
the product functions the replay mirrors.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from alignair.api import load_model
from alignair.predict import PredictConfig, forward, pipeline
from alignair.predict.airr import build_airr
from alignair.predict.calibrate import apply_temperature

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
    # every AlignAIR.forward output key (logits included), concatenated over the batches
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


@dataclass(frozen=True)
class Substitutions:
    """Truth fed to the pipeline between stage calls in place of the model's output. The defaults substitute nothing,
    which is a pure replay; ``replay`` does not implement any substitution yet."""

    orientation: list[int] | None = None
    # gene -> (starts, ends): read frame, 0-based, end-exclusive
    spans: dict[str, tuple[list[int], list[int]]] | None = None
    # feed the spans into branch.classify instead of the soft-argmax expectations
    classifier_spans: bool = False
    calls: dict[str, list] | None = None
    germline: dict[str, list] | None = None
    decoder: str = "soft"

    def __post_init__(self) -> None:
        if self.decoder not in DECODERS:
            raise ValueError(f"decoder {self.decoder!r} is not one of {', '.join(DECODERS)}")

    def active(self) -> tuple[str, ...]:
        """Names of the fields that substitute something; empty for a pure replay."""
        names = [name for name in ("orientation", "spans", "calls", "germline") if getattr(self, name) is not None]
        if self.classifier_spans:
            names.append("classifier_spans")
        if self.decoder != "soft":
            names.append("decoder")
        return tuple(names)


def load_model_checked(checkpoint_or_model, truth_reference: Reference, *, device: str) -> tuple[object, object]:
    """``(model, reference_set)`` on ``device`` whose germline reference is the truth reference.

    A path (``str`` or path-like) is loaded with ``alignair.api.load_model`` (a ``.alignair`` model file; the model
    comes back in eval mode). A ``(model, reference_set)`` pair, for example from a training loop, is passed through,
    with the model moved to ``device`` in place. ValueError when the reference does not hold exactly the truth
    reference's alleles and sequences (equal digests), naming both references' allele counts and the first allele,
    in name order, that is missing from the model's reference, extra in it, or has a different sequence there."""
    if isinstance(checkpoint_or_model, (str, os.PathLike)):
        model, reference_set = load_model(str(Path(checkpoint_or_model).expanduser()), device=device)
    else:
        model, reference_set = checkpoint_or_model
        model = model.to(device)
    model_reference = Reference.from_reference_set(reference_set)
    if model_reference.digest() != truth_reference.digest():
        raise ValueError(f"the model's reference is not the truth reference: it has {model_reference.counts_text()}, "
                         f"the truth reference has {truth_reference.counts_text()}; first difference: "
                         f"{model_reference.first_difference(truth_reference, 'model')}")
    return model, reference_set


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
    output. ``airr_rows`` equals what ``predict_sequences(model, reference_set, sequences, device=device,
    batch_size=batch_size)`` returns. The model must be on ``device``; it runs in eval mode and is put back in its
    previous mode. ValueError when the ids and sequences differ in number or there is no sequence;
    NotImplementedError for any substitution."""
    if len(ids) != len(sequences):
        raise ValueError(f"{len(ids)} ids for {len(sequences)} sequences")
    if not sequences:
        raise ValueError("no sequences to replay")
    active = substitutions.active()
    if active:
        raise NotImplementedError(f"substitution of {', '.join(active)} is not implemented yet; only "
                                  f"Substitutions() (a pure replay) runs")
    cfg = predict_config(model, reference_set, batch_size=batch_size)
    genes = pipeline._genes(cfg)
    seconds: dict[str, float] = {}

    with _timed(seconds, "input_policy"):
        reads, was_cropped = pipeline.apply_input_policy(sequences, cfg.max_seq_length)
    with _timed(seconds, "forward"):
        batches = _forward(model, reads, cfg, device)
    with _timed(seconds, "clean"):
        preds = pipeline.clean(batches, genes)
        pipeline._validate_predictions(preds)
    with _timed(seconds, "masks"):
        allowed = _masks(preds, reference_set, cfg, genes)
    with _timed(seconds, "canonicalize"):
        orient = preds.orientation
        canonical = [pipeline._canonicalize(s, int(orient[i]) if orient is not None else 0)
                     for i, s in enumerate(reads)]
        seq_lens = np.array([len(s) for s in canonical])
    with _timed(seconds, "segments"):
        segments = pipeline.correct_segments(preds.start, preds.end, seq_lens, cfg.max_seq_length, cfg.pad_mode)
    with _timed(seconds, "alleles"):
        names = {g: list(reference_set.gene(g.upper()).names) for g in genes}
        calls = pipeline.select_alleles(preds.allele, names, cfg.threshold, cfg.cap, cfg.selector, allowed=allowed)
    with _timed(seconds, "germline"):
        germline = pipeline.align_germline(canonical, segments, calls, reference_set, None,
                                           reader=cfg.germline_reader, indel_counts=preds.indel_count)
    with _timed(seconds, "records"):
        records = pipeline._to_records(canonical, calls, germline, genes, preds, cfg.chain_types,
                                       segments.low_quality, was_cropped, input_sequences=reads)
    with _timed(seconds, "airr"):
        airr_rows = (build_airr(records, reference_set, chain=("heavy" if cfg.has_d else "light")) if cfg.airr
                     else records)

    raw = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    return StageOutputs(ids=list(ids), raw=raw, predictions=preds, canonical=canonical, segments=segments,
                        calls=calls, germline=germline, records=records, airr_rows=airr_rows, seconds=seconds)


@contextmanager
def _timed(seconds: dict[str, float], stage: str):
    start = time.perf_counter()
    yield
    seconds[stage] = time.perf_counter() - start


@torch.no_grad()
def _forward(model, reads: list[str], cfg: PredictConfig, device: str) -> list[dict]:
    """``run_model``'s batched forward pass: the same batches and tokens, each batch's outputs as numpy arrays."""
    training = model.training
    model.eval()
    try:
        outputs = []
        for start in range(0, len(reads), cfg.batch_size):
            tokens = forward._tokenize(reads[start:start + cfg.batch_size], cfg.max_seq_length).to(device)
            out = model({"tokenized_sequence": tokens})
            outputs.append({key: value.detach().cpu().numpy() for key, value in out.items()})
        return outputs
    finally:
        model.train(training)


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
