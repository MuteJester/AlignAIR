"""Gene coordinate frames: every allele of a gene laid out on one shared set of columns.

A frame gives each ungapped germline base of each allele a column, so the same column means the same germline position
across a gene's alleles. V uses the IMGT-gapped form (``GeneReference.gapped``), where the gaps align the alleles; an
allele without one uses its own positions. J is shifted so that every allele's conserved W/F codon
(``GeneReference.anchors``) starts in the same column, the largest anchor of the gene. D uses its own positions. The
chip's per-base coordinate targets and its bank of allele rows are both laid out on these frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP = "."


@dataclass(frozen=True)
class GeneFrame:
    """``width`` columns; ``columns[allele][k]`` is the column of the allele's ungapped base k."""
    width: int
    columns: dict[str, np.ndarray]


def _v_columns(sequence: str, gapped: str | None) -> tuple[np.ndarray, int]:
    """The columns of a V allele's bases and the columns it spans: its gapped form's when that form is the allele's
    sequence with gaps, else its own positions."""
    if gapped and gapped.replace(GAP, "").upper() == sequence:
        return np.array([i for i, base in enumerate(gapped) if base != GAP], dtype=np.int64), len(gapped)
    return np.arange(len(sequence), dtype=np.int64), len(sequence)


def gene_frames(reference) -> dict[str, GeneFrame]:
    """The frames of the reference's genes, keyed ``"v"``, ``"d"`` (with a D) and ``"j"``."""
    frames = {}
    genes = ("v", "d", "j") if reference.has_d else ("v", "j")
    for gene in genes:
        alleles = reference.gene(gene.upper())
        columns, width = {}, 0
        anchors = alleles.anchors or {}
        shift_to = max(anchors.values(), default=0)
        for name, sequence in zip(alleles.names, alleles.sequences):
            if gene == "v":
                cols, extent = _v_columns(sequence, (alleles.gapped or {}).get(name))
            elif gene == "j":
                shift = shift_to - anchors[name] if name in anchors else 0
                cols = np.arange(len(sequence), dtype=np.int64) + shift
                extent = len(sequence) + shift
            else:
                cols = np.arange(len(sequence), dtype=np.int64)
                extent = len(sequence)
            columns[name] = cols
            width = max(width, extent)
        frames[gene] = GeneFrame(width, columns)
    return frames
