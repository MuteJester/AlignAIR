"""Shared mode-safe model execution and explicit batched prediction mechanics."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from operator import index
from typing import NamedTuple

import numpy as np
import torch

from ..core.boundaries import width_bucket
from ..core.runtime import evaluation_mode
from ..data.inference import SequenceInputAdapter
from ..data.inputs import prepare_nucleotide_inputs
from ..data.tokenizer import NucleotideTokenizer
from ..errors import ConfigurationError, InferenceError, NumericalError, error_context


def _positive(value, field):
    try:
        if isinstance(value, bool) or index(value) <= 0:
            raise ValueError
        return index(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a positive integer", context={"field": field}) from exc


def forward_batch(model, inputs, *, expected_rows: int) -> dict[str, torch.Tensor]:
    """Run one prepared batch and return its detached, validated outputs on their model devices."""
    count = _positive(expected_rows, "expected_rows")
    if not isinstance(model, torch.nn.Module) or not isinstance(inputs, Mapping):
        raise ConfigurationError("model inference requires a torch module and an input mapping")
    with error_context(stage="model_inference"), evaluation_mode(model):
        outputs = model(inputs)
        if not isinstance(outputs, Mapping) or not outputs:
            raise InferenceError("model outputs must be a nonempty mapping")
        result = {}
        for key in outputs:
            value = outputs.get(key)
            if (not isinstance(key, str) or not key or not isinstance(value, torch.Tensor)
                    or value.ndim == 0 or value.shape[0] != count or value.is_complex()):
                raise InferenceError("model output is missing or has an invalid tensor shape/type",
                                     context={"field": key, "expected_rows": count})
            if not torch.isfinite(value).all():
                raise NumericalError("non-finite model output", context={"field": key})
            result[key] = value.detach()
        return result


class PredictionBatch(NamedTuple):
    record_offset: int
    size: int
    outputs: dict[str, torch.Tensor]


def prediction_batches(model, records, input_adapter, *, batch_size: int):
    """Lazily predict ordered batches, including the tail; caller mode/context is restored before yield."""
    batch_size = _positive(batch_size, "batch_size")
    if not isinstance(model, torch.nn.Module) or not callable(input_adapter):
        raise ConfigurationError("prediction batches require a torch module and callable input adapter")

    def iterate():
        stream, offset = iter(records), 0
        while chunk := tuple(islice(stream, batch_size)):
            with error_context(stage="prediction_batch", record_offset=offset, batch_size=len(chunk)):
                outputs = forward_batch(model, input_adapter(chunk), expected_rows=len(chunk))
            yield PredictionBatch(offset, len(chunk), outputs)
            offset += len(chunk)

    return iterate()


def output_numpy(value: torch.Tensor):
    """Snapshot an output on CPU; float32 represents bfloat16 where NumPy has no native dtype."""
    value = value.detach().float() if value.dtype == torch.bfloat16 else value.detach()
    return value.to(device="cpu", copy=True).numpy()


class _TeacherForced:
    """An input adapter over ``(read, orientation id)`` records: the reads tokenized as ``adapter`` tokenizes them,
    and each id teacher-forced as ``inputs["orientation"]``; a None id is the model's own, from an unforced forward in
    evaluation mode (run only for a batch that holds one)."""

    def __init__(self, adapter, model):
        self.adapter, self.model = adapter, model

    def __call__(self, chunk):
        inputs = self.adapter([read for read, _ in chunk])
        tokens = inputs["tokenized_sequence"]
        given = [orientation for _, orientation in chunk]
        if any(orientation is None for orientation in given):
            with evaluation_mode(self.model):
                own = self.model({"tokenized_sequence": tokens})["orientation"].tolist()
            given = [own[k] if orientation is None else orientation for k, orientation in enumerate(given)]
        inputs["orientation"] = torch.tensor([int(o) for o in given], dtype=torch.long, device=tokens.device)
        return inputs


def run_model(model, sequences, cfg, device: str = "cpu", *, orientation=None) -> list:
    """The network's outputs for ``sequences``, one dict of numpy arrays per ``cfg.batch_size`` reads, in input order.
    Nucleotide forward inputs use the same content/window policy as prediction and reporting. ``orientation`` (one id
    per read, None for the model's own) is teacher-forced; each id travels with its read into its batch.

    Reads are batched with reads of their width bucket (``core.boundaries.width_bucket`` of the read and its end
    slot), in input order, so a read is computed at its own bucket's width: the reads outside its bucket never change
    its outputs. Which and how many reads of its bucket share its batch (the bucket's last batch may be part-full)
    can still move their last float bits."""
    reads, _ = prepare_nucleotide_inputs(sequences, cfg.window)
    adapter = SequenceInputAdapter(NucleotideTokenizer(), cfg.max_seq_length, device,
                                    vocab_size=6)
    records = reads
    if orientation is not None:
        records, adapter = list(zip(reads, orientation, strict=True)), _TeacherForced(adapter, model)
    if not records:
        return []
    size = _positive(cfg.batch_size, "batch_size")
    collected: dict[str, np.ndarray] = {}
    for rows in bucket_batches(reads, max_seq_length=cfg.max_seq_length, batch_size=size):
        for batch in prediction_batches(model, [records[i] for i in rows], adapter, batch_size=size):
            if collected and batch.outputs.keys() != collected.keys():
                raise InferenceError("prediction output keys changed between batches",
                                     context={"expected_keys": list(collected), "actual_keys": list(batch.outputs)})
            for key, value in batch.outputs.items():
                value = output_numpy(value)
                if key not in collected:
                    collected[key] = np.empty((len(records), *value.shape[1:]), dtype=value.dtype)
                collected[key][rows] = value
    return [{key: value[start:start + size] for key, value in collected.items()}
            for start in range(0, len(records), size)]


def bucket_batches(reads, *, max_seq_length: int, batch_size: int) -> list[list[int]]:
    """The batches ``run_model`` runs ``reads`` (prepared reads) in, as read indices: each width bucket's reads (the
    read and its end slot, ``core.boundaries.width_bucket``), in input order, in chunks of ``batch_size``; the buckets
    in the order their first read comes. A bucket's last chunk may be part-full; chunks of one bucket keep their order,
    so any subsequence of one bucket's chunks, its part-full one last, batches into the same chunks again."""
    groups: dict[int, list[int]] = {}
    for i, read in enumerate(reads):
        groups.setdefault(width_bucket(len(read) + 1, max_seq_length), []).append(i)
    return [members[start:start + batch_size] for members in groups.values()
            for start in range(0, len(members), batch_size)]
