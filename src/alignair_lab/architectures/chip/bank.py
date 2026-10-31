"""The bank compiler: the reference as rows of one-hot bases on each gene's frame columns (``alignair.reference.frames``).

The bank is the chip's input, not its weights: the buffers are rebuilt from whichever reference the model is bound to
(``AlignAIR.bind_reference``), are not saved with the weights, and a genotype is a mask over the rows. Per gene g:

- ``table_g`` (R, W·4): row r's base at column c as a one-hot in slots ``4c .. 4c + 3`` (A, T, G, C: the token ids
  1..4 minus 1); zero where the row has no base (a gap, an N, past its end).
- ``cover_g`` (R, W): 1 where the row has a base.
- ``row_allele_g`` (R,): the reference allele each row is. V and J have one row per allele; D has two, its forward
  rows first and then its reverse complements (``row_inverted_d``), so an inverted D finds its germline in the bank.
- ``sequence_g`` (R, longest allele) and ``anchor_base_g`` (R,), V and J: each row's ungapped germline as tokens (0
  past its end and for a non-A/C/G/T base), and the base its conserved codon starts at (V Cys, J W/F; -1 when none is
  annotated): they line a read up with the row from the read's own anchor, gaps of the frame aside.
"""
from __future__ import annotations

import torch
from torch import nn

from alignair.reference.frames import gene_frames

BASES = "ATGC"
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


class BankCompiler(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.genes = []
        for gene, frame in gene_frames(reference).items():
            alleles = reference.gene(gene.upper())
            rows = [(index, sequence, frame.columns[name])
                    for index, (name, sequence) in enumerate(zip(alleles.names, alleles.sequences))]
            if gene == "d":           # a reverse complement row reads its own positions, as a forward D row does
                rows += [(index, reverse_complement(sequence), frame.columns[name])
                         for index, (name, sequence) in enumerate(zip(alleles.names, alleles.sequences))]
            table = torch.zeros(len(rows), frame.width * len(BASES))
            cover = torch.zeros(len(rows), frame.width)
            for row, (_, sequence, columns) in enumerate(rows):
                for k, base in enumerate(sequence.upper()):
                    if base in BASES and columns[k] >= 0:
                        table[row, int(columns[k]) * len(BASES) + BASES.index(base)] = 1.
                        cover[row, int(columns[k])] = 1.
            self.register_buffer(f"table_{gene}", table, persistent=False)
            self.register_buffer(f"cover_{gene}", cover, persistent=False)
            self.register_buffer(f"row_allele_{gene}", torch.tensor([index for index, _, _ in rows]), persistent=False)
            if gene == "d":
                self.register_buffer("row_inverted_d", torch.arange(len(rows)) >= len(alleles), persistent=False)
            else:
                anchors = alleles.anchors or {}
                longest = max(len(sequence) for sequence in alleles.sequences)
                sequence_tokens = torch.zeros(len(alleles), longest, dtype=torch.long)
                for row, sequence in enumerate(alleles.sequences):
                    sequence_tokens[row, :len(sequence)] = torch.tensor(
                        [BASES.index(base) + 1 if base in BASES else 0 for base in sequence.upper()])
                anchor_base = [anchors[name] if 0 <= anchors.get(name, -1) < len(sequence) else -1
                               for name, sequence in zip(alleles.names, alleles.sequences)]
                self.register_buffer(f"sequence_{gene}", sequence_tokens, persistent=False)
                self.register_buffer(f"anchor_base_{gene}", torch.tensor(anchor_base), persistent=False)
            self.genes.append(gene)

    def width(self, gene: str) -> int:
        return getattr(self, f"cover_{gene}").shape[1]

    def rows(self, gene: str) -> int:
        return getattr(self, f"cover_{gene}").shape[0]

    def row_mask(self, gene: str, allele_mask: torch.Tensor) -> torch.Tensor:
        """``allele_mask`` (B, n alleles) bool -> (B, R): each row allowed when its allele is."""
        return allele_mask[:, getattr(self, f"row_allele_{gene}")]
