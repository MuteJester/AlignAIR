"""Benchmark transforms applied to GenAIRR records after simulation."""
from __future__ import annotations

from alignair.train.gym.crop import anchor_c0, crop_one_sided, crop_record
from ...core.schema import StratumSpec


def apply_benchmark_crop(record: dict, stratum: StratumSpec) -> dict:
    """Apply the benchmark crop policy for a GenAIRR record.

    One-sided adaptive anchor crops take precedence over symmetric
    junction-centered crops so all generation paths model amplicons identically.
    """

    if stratum.anchor is not None:
        return crop_one_sided(record, anchor_c0(record, stratum.anchor))
    if stratum.crop_to is not None:
        return crop_record(record, stratum.crop_to)
    return record
