"""Deterministic parent-lineage partitions for nucleotide or protein record collections."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral, Real

from ..errors import ConfigurationError, DataContractError, error_context


def _group_records(records):
    if isinstance(records, (str, bytes, Mapping)):
        raise DataContractError("Partition records must be an iterable of record mappings.")
    try:
        iterator = iter(records)
    except TypeError as exc:
        raise DataContractError("Partition records must be iterable.") from exc
    groups, seen = defaultdict(list), {}
    for index, record in enumerate(iterator):
        with error_context(record_index=index):
            if not isinstance(record, Mapping):
                raise DataContractError("Partition records must be mappings.")
            ident = record.get("id")
            if not isinstance(ident, str) or not ident:
                raise DataContractError("Partition record ID must be nonempty text.", context={"field": "id"})
            if ident in seen:
                raise DataContractError("Partition record IDs must be unique.",
                                        context={"field": "id", "record_id": ident, "first_record_index": seen[ident]})
            parent = record.get("parent_lineage")
            if parent is not None and not isinstance(parent, str):
                raise DataContractError("Parent lineage must be text or None.", context={"field": "parent_lineage"})
            lineage = parent or record.get("lineage_key")
            if not isinstance(lineage, str) or not lineage:
                raise DataContractError("Partition records require a nonempty parent or fallback lineage.",
                                        context={"field": "lineage_key"})
            groups[lineage].append(record)
            seen[ident] = index
    return groups


def partition_by_parent_lineage(records, *, seed, dev_frac=0.2):
    """Return (train, dev, assignment_sha256), keeping each parent family together.

    Group by nonempty parent_lineage, otherwise lineage_key. Sort groups by SHA-256 of
    ``seed:key`` and assign the first round(group_count * dev_frac) groups to development.
    The fraction counts families, not records; no class stratification or minimum-size clamp
    is applied. Preserve record objects and within-family input order. The digest binds sorted
    record-ID -> partition assignments, not sequence contents or within-family order.
    """
    with error_context(stage="lineage_partition"):
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ConfigurationError("Partition seed must be an integer.", context={"field": "seed"})
        try:
            valid = (isinstance(dev_frac, Real) and not isinstance(dev_frac, bool)
                     and math.isfinite(dev_frac) and 0 <= dev_frac <= 1)
        except OverflowError:
            valid = False
        if not valid:
            raise ConfigurationError("Development fraction must be finite and within [0, 1].",
                                     context={"field": "dev_frac"})
        groups = _group_records(records)
        ordered = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
        n_dev = round(len(ordered) * dev_frac)
        dev_keys = set(ordered[:n_dev])
        train, dev, id_part = [], [], {}
        for k in ordered:
            part = "dev" if k in dev_keys else "train"
            for r in groups[k]:
                (dev if part == "dev" else train).append(r)
                id_part[r["id"]] = part
        partition_hash = hashlib.sha256(json.dumps(id_part, sort_keys=True).encode()).hexdigest()
        return train, dev, partition_hash
