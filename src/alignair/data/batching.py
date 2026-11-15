"""Shared, dtype-preserving assembly of prepared training samples."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index
from typing import Protocol

import torch

from ..errors import ConfigurationError, DataContractError, TrainingDataError, error_context


@dataclass(frozen=True)
class PositionLabels:
    """A variable-length, per-position target and its padding/ignore value."""
    values: object
    padding: int | float = -100


@dataclass(frozen=True)
class TrainingSample:
    """Encoded CPU tokens plus explicitly typed targets in the same coordinate frame."""
    tokens: object
    targets: Mapping[str, object]
    record_id: str | None = None


@dataclass(frozen=True)
class TrainingBatch:
    """Model inputs and targets; unpacking retains the existing (inputs, targets) interface."""
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]

    def __iter__(self):
        yield self.inputs
        yield self.targets

    @property
    def token_mask(self) -> torch.Tensor:
        return self.inputs["tokenized_sequence"] != 0

    @property
    def lengths(self) -> torch.Tensor:
        return self.token_mask.sum(-1)

    def to(self, device, *, non_blocking=False) -> "TrainingBatch":
        return TrainingBatch(
            {key: value.to(device, non_blocking=non_blocking) for key, value in self.inputs.items()},
            {key: value.to(device, non_blocking=non_blocking) for key, value in self.targets.items()},
        )

    def pin_memory(self) -> "TrainingBatch":
        """Let DataLoader pin a custom batch before a GPU transfer."""
        return TrainingBatch({key: value.pin_memory() for key, value in self.inputs.items()},
                             {key: value.pin_memory() for key, value in self.targets.items()})


class BatchAdapter(Protocol):
    """A source-specific adapter prepares one shared model batch."""
    def __call__(self, samples: Sequence) -> TrainingBatch: ...


def _width(value) -> int:
    try:
        if isinstance(value, bool) or index(value) <= 0:
            raise ValueError
        return index(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("batch window must be a positive integer") from exc


def pad_vectors(vectors, *, width=None, dtype=torch.long, padding=0) -> torch.Tensor:
    """Right-pad CPU vectors once; an explicit window never silently truncates a vector."""
    values = [torch.as_tensor(value) for value in vectors]
    for i, value in enumerate(values):
        if value.ndim != 1 or value.device.type != "cpu":
            raise DataContractError("batch vectors must be one-dimensional CPU tensors",
                                    context={"record_index": i})
    length = max((value.numel() for value in values), default=0) if width is None else _width(width)
    result = torch.full((len(values), length), padding, dtype=dtype)
    for i, value in enumerate(values):
        if value.numel() > length:
            raise DataContractError("encoded vector exceeds the batch window",
                                    context={"record_index": i, "length": value.numel(), "max_length": length})
        result[i, :value.numel()] = value.to(dtype=dtype)
    return result


def collate_samples(samples: Sequence[TrainingSample], *, max_length=None, device="cpu",
                    reserve_terminal_slot=False) -> TrainingBatch:
    """Stack dense targets and pad positional targets, preserving each field's declared dtype."""
    samples = list(samples)
    if not samples:
        raise TrainingDataError("cannot collate an empty training batch")
    width = _width(max_length) if max_length is not None else None
    rows = []
    fields = set(samples[0].targets)
    for i, sample in enumerate(samples):
        with error_context(record_index=i, record_id=sample.record_id):
            tokens = torch.as_tensor(sample.tokens)
            if tokens.ndim != 1 or tokens.dtype == torch.bool or tokens.is_floating_point() or tokens.is_complex():
                raise TrainingDataError("tokens must be a one-dimensional integer vector", context={"field": "tokens"})
            if (tokens < 0).any():
                raise TrainingDataError("training tokens must be nonnegative", context={"field": "tokens"})
            if set(sample.targets) != fields:
                raise TrainingDataError("training samples have different target fields",
                                        context={"missing": sorted(fields - set(sample.targets)),
                                                 "extra": sorted(set(sample.targets) - fields)})
            if width is not None and tokens.numel() > width - int(reserve_terminal_slot):
                raise TrainingDataError("encoded read exceeds the available model slots",
                                        context={"field": "tokens", "length": tokens.numel(),
                                                 "max_length": width, "reserved_endpoint": reserve_terminal_slot})
            rows.append(tokens)
    if width is None and reserve_terminal_slot:
        width = max(row.numel() for row in rows) + 1
    tokens = pad_vectors(rows, width=width)
    targets = {}
    for key in samples[0].targets:
        with error_context(field=key):
            values = [sample.targets[key] for sample in samples]
            positional = isinstance(values[0], PositionLabels)
            if any(isinstance(value, PositionLabels) != positional for value in values):
                raise TrainingDataError("target mixes positional and dense values")
            tensors = [torch.as_tensor(value.values if positional else value) for value in values]
            if any(value.dtype != tensors[0].dtype for value in tensors):
                raise TrainingDataError("target dtypes differ between samples")
            if positional:
                if any(value.padding != values[0].padding for value in values):
                    raise TrainingDataError("positional targets use different padding values")
                if any(value.ndim != 1 or value.numel() != row.numel() for value, row in zip(tensors, rows)):
                    raise TrainingDataError("positional labels must align with their encoded tokens")
                targets[key] = pad_vectors(tensors, width=tokens.shape[1], dtype=tensors[0].dtype,
                                           padding=values[0].padding)
            else:
                try:
                    targets[key] = torch.stack(tensors)
                except RuntimeError as exc:
                    raise TrainingDataError("target shapes differ between samples") from exc
    return TrainingBatch({"tokenized_sequence": tokens}, targets).to(device)
