"""Shared mode-safe model execution and explicit batched prediction mechanics."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from operator import index
from typing import NamedTuple

import torch

from ..core.runtime import evaluation_mode
from ..data.inference import SequenceInputAdapter
from ..data.tokenizer import CenterPaddedTokenizer
from ..errors import ConfigurationError, InferenceError, NumericalError, error_context


def _positive(value, field):
    try:
        if isinstance(value, bool) or index(value) <= 0:
            raise ValueError
        return index(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a positive integer", context={"field": field}) from exc


def _output_keys(keys):
    if keys is None:
        return None
    try:
        selected = tuple(keys)
        if (isinstance(keys, str) or not selected or any(not isinstance(key, str) or not key for key in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("output keys must be a nonempty sequence of distinct names") from exc
    return selected


def forward_batch(model, inputs, *, expected_rows: int, output_keys=None) -> dict[str, torch.Tensor]:
    """Run one prepared batch and return detached, validated outputs on their model devices."""
    count = _positive(expected_rows, "expected_rows")
    keys = _output_keys(output_keys)
    if not isinstance(model, torch.nn.Module) or not isinstance(inputs, Mapping):
        raise ConfigurationError("model inference requires a torch module and an input mapping")
    with error_context(stage="model_inference"), evaluation_mode(model):
        outputs = model(inputs)
        if not isinstance(outputs, Mapping) or not outputs:
            raise InferenceError("model outputs must be a nonempty mapping")
        selected = tuple(outputs) if keys is None else keys
        result = {}
        for key in selected:
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


def prediction_batches(model, records, input_adapter, *, batch_size: int, output_keys=None):
    """Lazily predict ordered batches, including the tail; caller mode/context is restored before yield."""
    batch_size = _positive(batch_size, "batch_size")
    keys = _output_keys(output_keys)
    if not isinstance(model, torch.nn.Module) or not callable(input_adapter):
        raise ConfigurationError("prediction batches require a torch module and callable input adapter")

    def iterate():
        stream, offset = iter(records), 0
        while chunk := tuple(islice(stream, batch_size)):
            with error_context(stage="prediction_batch", record_offset=offset, batch_size=len(chunk)):
                outputs = forward_batch(model, input_adapter(chunk), expected_rows=len(chunk), output_keys=keys)
            yield PredictionBatch(offset, len(chunk), outputs)
            offset += len(chunk)

    return iterate()


def output_numpy(value: torch.Tensor):
    """Snapshot an output on CPU; float32 represents bfloat16 where NumPy has no native dtype."""
    value = value.detach().float() if value.dtype == torch.bfloat16 else value.detach()
    return value.to(device="cpu", copy=True).numpy()


def run_model(model, sequences, cfg, device: str = "cpu") -> list:
    """Historical nucleotide interface: uppercase, unknown-to-N, prefix truncation, right padding."""
    adapter = SequenceInputAdapter(CenterPaddedTokenizer(), cfg.max_seq_length, device,
                                    overflow="prefix", allow_empty=True, vocab_size=6)
    records = (str(sequence).upper() for sequence in sequences)
    return [{key: output_numpy(value) for key, value in batch.outputs.items()}
            for batch in prediction_batches(model, records, adapter, batch_size=cfg.batch_size)]
