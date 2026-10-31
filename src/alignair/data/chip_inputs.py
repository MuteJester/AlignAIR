"""The chip's training inputs (``AlignAIRConfig.chip_inputs``): per-base targets from a GenAIRR record, and per-batch
bank masks and teacher inputs.

Per record (``chip_record_targets``):

- ``coord_{gene}``: each base's column in its gene's frame (``reference.frames``), for the bases the gene's CIGAR maps
  to its primary allele; ``IGNORE`` for every other base (insertions, other regions, an alignment that cannot be
  mapped, an inverted D, whose germline the forward frame does not hold).
- ``region``: each base's recombination region as a ``READ_REGIONS`` index.
- ``region_allowed``: a bitmask over ``READ_REGIONS`` per base: its region, plus the neighbouring gene where the read
  cannot place the boundary. A V or D end may move right over the bases that continue its germline (N bases that
  happen to equal the trimmed germline), and a D or J start may move left over the bases that precede its germline,
  up to the neighbouring gene: the boundary the read can identify is any of those.

Per batch (``chip_batch_inputs``), into the model's inputs:

- ``bank_mask_{gene}`` (B, n alleles): the genotype the batch is aligned against, one mask for the whole batch: the
  batch's true alleles plus random others, its size drawn log-uniformly from 1 to every allele.
- ``teacher_coord_{gene}`` and ``teacher_region``: the true columns and regions, which the chip may read in place of
  its own predictions while it trains.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..neural.heads.region import READ_REGIONS, REGION_INDEX
from .alignment import query_reference_map
from .batching import PositionLabels
from .coordinates import segment_spans

IGNORE = -100
GENES = ("v", "d", "j")
_BIT = {gene: 1 << READ_REGIONS.index(gene.upper()) for gene in GENES}


def _primary(record, gene: str) -> str:
    """The allele the record's germline coordinates belong to (GenAIRR lists it first)."""
    return str(record[f"{gene}_call"]).split(",")[0]


def _germline(reference, gene: str, allele: str) -> str | None:
    alleles = reference.gene(gene.upper())
    index = alleles.index.get(allele)
    return None if index is None else alleles.sequences[index]


def chip_record_targets(record, reference, frames, region_labels) -> dict:
    """The chip's per-base targets of one record; ``region_labels`` are ``data.targets``' (``REGION_INDEX`` values)."""
    sequence = str(record["sequence"]).upper()
    coords, germ = segment_spans(record)
    inverted_d = bool(record.get("d_inverted", False))
    region = np.asarray(region_labels, dtype=np.int64) - REGION_INDEX[READ_REGIONS[0]]
    targets = {}
    for gene in frames:
        column = np.full(len(sequence), IGNORE, dtype=np.int64)
        if gene in coords and not (gene == "d" and inverted_d):
            columns = frames[gene].columns.get(_primary(record, gene))
            path = query_reference_map(record.get(f"{gene}_cigar"), coords[gene], germ[gene])
            if columns is not None and path.reason is None:
                for offset, base in enumerate(path.positions):
                    if base is not None and base < len(columns) and columns[base] >= 0:
                        column[path.query_start + offset] = columns[base]
        targets[f"coord_{gene}"] = PositionLabels(column, IGNORE)
    allowed = np.left_shift(1, region).astype(np.int64)
    _movable_boundaries(allowed, sequence, record, reference, coords, germ, inverted_d)
    targets["region"] = PositionLabels(region, IGNORE)
    targets["region_allowed"] = PositionLabels(allowed, 0)
    return targets


def _movable_boundaries(allowed, sequence, record, reference, coords, germ, inverted_d) -> None:
    """Add each gene's bit on the bases its boundary may move over (see the module docstring)."""
    order = [gene for gene in GENES if gene in coords]
    for i, gene in enumerate(order):
        if gene == "d" and inverted_d:
            continue
        germline = _germline(reference, gene, _primary(record, gene))
        if germline is None:
            continue
        (start, end), (g_start, g_end) = coords[gene], germ[gene]
        following = coords[order[i + 1]][0] if i + 1 < len(order) else len(sequence)
        preceding = coords[order[i - 1]][1] if i > 0 else 0
        if gene in ("v", "d"):
            t = 0
            while end + t < following and g_end + t < len(germline) and sequence[end + t] == germline[g_end + t]:
                allowed[end + t] |= _BIT[gene]
                t += 1
        if gene in ("d", "j"):
            t = 1
            while start - t >= preceding and g_start - t >= 0 and sequence[start - t] == germline[g_start - t]:
                allowed[start - t] |= _BIT[gene]
                t += 1


def chip_batch_inputs(targets: dict, reference, generator: torch.Generator | None = None) -> dict:
    """The batch's bank masks and teacher inputs (see the module docstring)."""
    inputs = {}
    for gene in GENES:
        if f"{gene}_allele" not in targets:
            continue
        alleles = targets[f"{gene}_allele"]
        rows, n = alleles.shape
        supervise = targets.get(f"{gene}_supervise", torch.ones(rows, dtype=torch.bool)).reshape(-1, 1).bool()
        mask = ((alleles > 0.5) & supervise).any(0)
        size = min(n, int(math.exp(torch.rand((), generator=generator).item() * math.log(n + 1))))
        extra = max(size, 1) - int(mask.sum())
        if extra > 0:
            others = (~mask).nonzero().flatten()
            mask[others[torch.randperm(len(others), generator=generator)[:extra]]] = True
        inputs[f"bank_mask_{gene}"] = mask.expand(rows, n).clone()
        if f"coord_{gene}" in targets:
            inputs[f"teacher_coord_{gene}"] = targets[f"coord_{gene}"]
    if "region" in targets:
        inputs["teacher_region"] = targets["region"]
    return inputs
