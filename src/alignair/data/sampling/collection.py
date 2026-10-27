"""Bounded source-independent collection with preparation and selection accounting."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping
from itertools import islice
from operator import index
from typing import NamedTuple

from ...errors import ConfigurationError, TrainingDataError, TrainingError, error_context
from .contracts import RecordType, SampleBuilder, SampleSelector

_SELECTION_FIELDS = ("selected", "selection_skipped", "selection_untracked", "selection_errors")


class SampleBudgetExhausted(TrainingError):
    """A requested collection could not be filled within its source-record budget."""
    code = "ALIGNAIR_SEED_BUDGET_EXHAUSTED"


class SampleCollection(NamedTuple):
    samples: list[Mapping]
    generated: int


def _increment(counts, key):
    counts[key] = counts.get(key, 0) + 1


def _positive(value, field):
    try:
        result = index(value)
        if isinstance(value, bool) or result <= 0:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field} must be a positive integer", context={"field": field}) from exc


def _rejection_reasons(values):
    if isinstance(values, str):
        raise ConfigurationError("rejection reasons must be a sequence of complete names")
    try:
        reasons = tuple(values)
    except TypeError as exc:
        raise ConfigurationError("rejection reasons must be an iterable") from exc
    reserved = {"generated", "usable", "selected", "selection_skipped", "selection_untracked",
                "preparation_errors", "selection_errors"}
    if (not all(isinstance(reason, str) and reason for reason in reasons)
            or len(set(reasons)) != len(reasons) or reserved.intersection(reasons)):
        raise ConfigurationError("rejection reasons must be distinct names separate from collection counters")
    return reasons


def collect_samples(records: Iterable[RecordType], builder: SampleBuilder[RecordType], *, want, budget, counter=None,
                    select: SampleSelector | None = None, rejection_reasons=()) -> SampleCollection:
    """Consume at most budget records. Usable rows are counted even when selection declines them."""
    want, budget = _positive(want, "want"), _positive(budget, "budget")
    if not callable(builder) or select is not None and not callable(select):
        raise ConfigurationError("builder and selector must be callable")
    reasons = _rejection_reasons(rejection_reasons)
    counts = counter if counter is not None else Counter()
    if not isinstance(counts, MutableMapping):
        raise ConfigurationError("collection counters must be a mutable mapping")
    check_collection_counts(counts, reasons)
    if counts.get("usable", 0) and not any(key in counts for key in _SELECTION_FIELDS):
        counts["selection_untracked"] = counts["usable"]
    selected, generated = [], 0
    for record_index, record in enumerate(islice(records, budget)):
        generated += 1
        _increment(counts, "generated")
        with error_context(stage="sample_collection", record_index=record_index,
                           record_id=record.get("sequence_id") if isinstance(record, Mapping) else None):
            try:
                sample, reason = builder(record)
                if sample is None and reason not in reasons:
                    raise TrainingDataError("sample builder returned an unregistered rejection reason",
                                            context={"reason": reason})
                if sample is not None and (reason is not None or not isinstance(sample, Mapping)):
                    raise TrainingDataError("accepted samples must be mappings without a rejection reason")
            except Exception:
                _increment(counts, "preparation_errors")
                raise
            if sample is None:
                _increment(counts, reason)
                continue
            _increment(counts, "usable")
            try:
                take = select is None or select(sample, selected)
            except Exception:
                _increment(counts, "selection_errors")
                raise
            if not take:
                _increment(counts, "selection_skipped")
                continue
            _increment(counts, "selected")
            selected.append(sample)
            if len(selected) == want:
                return SampleCollection(selected, generated)
    raise SampleBudgetExhausted(
        f"collected {len(selected)}/{want} samples after consuming {generated} records (budget {budget})",
        context={"stage": "sample_collection", "collected": len(selected), "want": want,
                 "generated": generated, "budget": budget},
    )


def check_collection_counts(counter, rejection_reasons=()) -> None:
    """Check preparation and, when present, selection identities; historical ledgers remain readable."""
    if not isinstance(counter, Mapping):
        raise TrainingDataError("sample counts must be a mapping")
    rejection_reasons = _rejection_reasons(rejection_reasons)
    keys = ("generated", "usable", *rejection_reasons)
    values = (counter.get(key, 0) for key in (*keys, *_SELECTION_FIELDS, "preparation_errors"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise TrainingDataError("sample counts must be nonnegative integers")
    rejected = sum(counter.get(reason, 0) for reason in rejection_reasons) + counter.get("preparation_errors", 0)
    if counter.get("generated", 0) != counter.get("usable", 0) + rejected:
        raise TrainingDataError("generated samples must equal usable plus rejections and preparation errors",
                                context={"generated": counter.get("generated", 0),
                                         "usable": counter.get("usable", 0), "rejected": rejected})
    if any(key in counter for key in _SELECTION_FIELDS) and counter.get("usable", 0) != (
            counter.get("selected", 0) + counter.get("selection_skipped", 0)
            + counter.get("selection_errors", 0) + counter.get("selection_untracked", 0)):
        raise TrainingDataError("usable samples must equal recorded selection outcomes")
