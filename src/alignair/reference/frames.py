"""Gene coordinate frames: every allele of a gene laid out on one shared, fixed set of columns.

A frame gives each ungapped germline base of each allele a column, so the same column means the same germline position
across a gene's alleles, and across references: the frames do not depend on which reference is loaded, so a model that
predicts columns keeps its weights when the reference changes. V uses the IMGT-gapped form (``GeneReference.gapped``),
IMGT's own shared numbering; an allele without one uses its own positions. J is shifted so that every allele's
conserved W/F codon (``GeneReference.anchors``) starts in column ``J_ANCHOR_COLUMN``; a J without an anchor starts at
column 0. D uses its own positions. Each gene has ``FRAME_WIDTHS`` columns; a base that falls outside them has column
``NO_COLUMN``. The chip's per-base coordinate targets and its bank of allele rows are both laid out on these frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP = "."
FRAME_WIDTHS = {"v": 384, "d": 48, "j": 96}
J_ANCHOR_COLUMN = 48
NO_COLUMN = -1


@dataclass(frozen=True)
class GeneFrame:
    """``width`` columns; ``columns[allele][k]`` is the column of the allele's ungapped base k (``NO_COLUMN`` outside
    the frame)."""
    width: int
    columns: dict[str, np.ndarray]


def _v_columns(sequence: str, gapped: str | None) -> np.ndarray:
    """A V allele's columns: its gapped form's when that form is the allele's sequence with gaps, else its own
    positions."""
    if gapped and gapped.replace(GAP, "").upper() == sequence:
        return np.array([i for i, base in enumerate(gapped) if base != GAP], dtype=np.int64)
    return np.arange(len(sequence), dtype=np.int64)


def gene_frames(reference) -> dict[str, GeneFrame]:
    """The frames of the reference's genes, keyed ``"v"``, ``"d"`` (with a D) and ``"j"``."""
    frames = {}
    genes = ("v", "d", "j") if reference.has_d else ("v", "j")
    for gene in genes:
        alleles = reference.gene(gene.upper())
        anchors = alleles.anchors or {}
        width = FRAME_WIDTHS[gene]
        columns = {}
        for name, sequence in zip(alleles.names, alleles.sequences):
            if gene == "v":
                cols = _v_columns(sequence, (alleles.gapped or {}).get(name))
            else:
                shift = J_ANCHOR_COLUMN - anchors[name] if gene == "j" and name in anchors else 0
                cols = np.arange(len(sequence), dtype=np.int64) + shift
            columns[name] = np.where((cols >= 0) & (cols < width), cols, NO_COLUMN)
        frames[gene] = GeneFrame(width, columns)
    return frames
