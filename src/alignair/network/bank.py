"""The bank compiler: the reference as rows of one-hot bases on each gene's frame columns (``alignair.reference.frames``).

The bank is the chip's input, not its weights: the buffers are rebuilt from whichever reference the model is bound to
(``AlignAIR.bind_reference``), are not saved with the weights, and a genotype is a mask over the rows. Per gene g:

- ``table_g`` (R, W·4): row r's base at column c as a one-hot in slots ``4c .. 4c + 3`` (A, T, G, C: the token ids
  1..4 minus 1); zero where the row has no base (a gap, an N, past its end).
- ``cover_g`` (R, W): 1 where the row has a base.
- ``row_allele_g`` (R,): the reference allele each row is. V and J have one row per allele; D has two, its forward
  rows first and then its reverse complements (``row_inverted_d``), so an inverted D finds its germline in the bank.
- ``sequence_g`` (R, longest row): each row's ungapped germline as tokens (0 past its end and for a non-A/C/G/T
  base), for the units that align a read to the rows base by base, gaps of the frame aside.
- ``anchor_base_g`` (R,), V and J: the base each row's conserved codon starts at (V Cys, J W/F; -1 when none is
  annotated), which lines a read up with the row from the read's own anchor.

A training batch may carry novel rows (``extension_inputs``: a batch's novel alleles, compiled on the same frames): then
the chip reads ``BankCompiler.for_batch(batch)``, a ``BankView`` whose rows are the bound bank's plus the batch's, in
the batch bank's order (``bank_order_g``), which is the bank of the reference extended with those alleles.
"""
from __future__ import annotations

import torch
from torch import nn

from ..core.orientation import reverse_complement
from ..core.tokens import BASES
from ..reference.frames import gene_frames


def compile_rows(rows, width: int):
    """``(table (R, 4·width), cover (R, width), sequence (R, longest))`` of ``rows``, each ``(sequence, columns)``: the
    one-hot bases on their frame columns, where a base is, and the ungapped tokens."""
    row_of, column_of, base_of = [], [], []        # every base's (row, column, base) as plain ints, then one write each
    for row, (sequence, columns) in enumerate(rows):
        for k, base in enumerate(sequence.upper()):
            if base in BASES and columns[k] >= 0:
                row_of.append(row), column_of.append(int(columns[k])), base_of.append(BASES.index(base))
    row_of, column_of = torch.tensor(row_of, dtype=torch.long), torch.tensor(column_of, dtype=torch.long)
    table = torch.zeros(len(rows), width * len(BASES))
    table[row_of, column_of * len(BASES) + torch.tensor(base_of, dtype=torch.long)] = 1.
    cover = torch.zeros(len(rows), width)
    cover[row_of, column_of] = 1.
    longest = max(len(sequence) for sequence, _ in rows)
    sequence_tokens = torch.tensor([[BASES.index(base) + 1 if base in BASES else 0 for base in sequence.upper()]
                                    + [0] * (longest - len(sequence)) for sequence, _ in rows], dtype=torch.long)
    return table, cover, sequence_tokens


def _anchor_bases(alleles, names) -> list[int]:
    anchors = alleles.anchors or {}
    return [anchors[name] if 0 <= anchors.get(name, -1) < len(alleles.sequences[alleles.index[name]]) else -1
            for name in names]


def extension_inputs(base, extended, frames: dict, gene: str) -> dict:
    """The batch inputs that make the chip read ``extended``'s bank for ``gene`` while bound to ``base``: the rows of
    the alleles ``base`` lacks, in ``extended``'s order, compiled on ``frames`` (``gene_frames(extended)``), and
    ``bank_order_{gene}``, each of ``extended``'s alleles as a row of the bound bank followed by the new rows.
    ValueError for a D (its rows come in forward and reverse-complement pairs), for an allele of ``base`` that
    ``extended`` lacks, and for a frame wider than the bound bank's."""
    if gene == "d":
        raise ValueError("a batch extends the V and J banks only (a D's rows are forward and reverse-complement pairs)")
    mine, theirs = base.gene(gene.upper()), extended.gene(gene.upper())
    lost = [name for name in mine.names if name not in theirs.index]
    if lost:
        raise ValueError(f"the extended reference lacks alleles of the bound one: {', '.join(lost[:5])}")
    if frames[gene].width != gene_frames(base)[gene].width:
        raise ValueError(f"the extended {gene.upper()} frame is {frames[gene].width} columns wide, the bound one "
                         f"{gene_frames(base)[gene].width}: a novel allele must keep its base's columns")
    new = [name for name in theirs.names if name not in mine.index]
    order = [mine.index[name] if name in mine.index else len(mine) + new.index(name) for name in theirs.names]
    table, cover, sequence = compile_rows([(theirs.sequences[theirs.index[n]], frames[gene].columns[n]) for n in new],
                                          frames[gene].width)
    return {f"bank_order_{gene}": torch.tensor(order, dtype=torch.long), f"bank_extension_table_{gene}": table,
            f"bank_extension_cover_{gene}": cover, f"bank_extension_sequence_{gene}": sequence,
            f"bank_extension_anchor_{gene}": torch.tensor(_anchor_bases(theirs, new), dtype=torch.long)}


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
            table, cover, sequence_tokens = compile_rows([(sequence, columns) for _, sequence, columns in rows],
                                                         frame.width)
            self.register_buffer(f"table_{gene}", table, persistent=False)
            self.register_buffer(f"cover_{gene}", cover, persistent=False)
            self.register_buffer(f"sequence_{gene}", sequence_tokens, persistent=False)
            self.register_buffer(f"row_allele_{gene}", torch.tensor([index for index, _, _ in rows]), persistent=False)
            if gene == "d":
                self.register_buffer("row_inverted_d", torch.arange(len(rows)) >= len(alleles), persistent=False)
            else:
                self.register_buffer(f"anchor_base_{gene}", torch.tensor(_anchor_bases(alleles, alleles.names)),
                                     persistent=False)
            self.genes.append(gene)
        # A reference of several loci: each locus's alleles per gene, (L, n alleles), and whether it has a D; the chip
        # restricts a read's calls to one locus at inference (``refine.locus``). 0 loci for a one-locus reference.
        loci = reference.locus_names() if hasattr(reference, "locus_names") else ()
        self.locus_count = len(loci) if len(loci) > 1 else 0
        if self.locus_count:
            for gene in self.genes:
                masks = [torch.from_numpy(reference.locus_mask(locus, gene.upper())) for locus in loci]
                self.register_buffer(f"locus_alleles_{gene}", torch.stack(masks), persistent=False)
            self.register_buffer("locus_has_d", torch.tensor(
                [bool(reference.locus_mask(locus, "D").any()) if "d" in self.genes else False for locus in loci]),
                persistent=False)

    def width(self, gene: str) -> int:
        return getattr(self, f"cover_{gene}").shape[1]

    def rows(self, gene: str) -> int:
        return getattr(self, f"cover_{gene}").shape[0]

    def row_mask(self, gene: str, allele_mask: torch.Tensor) -> torch.Tensor:
        """``allele_mask`` (B, n alleles) bool -> (B, R): each row allowed when its allele is."""
        return allele_mask[:, getattr(self, f"row_allele_{gene}")]

    def for_batch(self, batch):
        """This bank, or the ``BankView`` of a batch that carries novel rows (``bank_order_*``)."""
        extended = [gene for gene in self.genes if f"bank_order_{gene}" in batch]
        return BankView(self, batch, extended) if extended else self


class BankView:
    """The bound bank with a batch's novel rows: for each extended gene, its rows are the bound rows followed by the
    batch's, taken in the batch bank's order (``bank_order_g``), each its own allele; every other buffer is the bound
    bank's."""

    def __init__(self, bank: BankCompiler, batch, genes):
        self._bank = bank
        self.genes = bank.genes
        for gene in genes:
            order = batch[f"bank_order_{gene}"]
            bound, extra = getattr(bank, f"sequence_{gene}"), batch[f"bank_extension_sequence_{gene}"]
            longest = max(bound.shape[1], extra.shape[1])
            sequences = torch.cat([_pad(bound, longest), _pad(extra.to(bound.device), longest)])
            for name, extension in (("table", "table"), ("cover", "cover"), ("anchor_base", "anchor")):
                rows = torch.cat([getattr(bank, f"{name}_{gene}"),
                                  batch[f"bank_extension_{extension}_{gene}"].to(getattr(bank, f"{name}_{gene}"))])
                setattr(self, f"{name}_{gene}", rows[order])
            setattr(self, f"sequence_{gene}", sequences[order])
            setattr(self, f"row_allele_{gene}", torch.arange(order.numel(), device=order.device))

    def __getattr__(self, name):
        return getattr(self._bank, name)

    def width(self, gene: str) -> int:
        return getattr(self, f"cover_{gene}").shape[1]

    def rows(self, gene: str) -> int:
        return getattr(self, f"cover_{gene}").shape[0]

    def row_mask(self, gene: str, allele_mask: torch.Tensor) -> torch.Tensor:
        return allele_mask[:, getattr(self, f"row_allele_{gene}")]


def _pad(tokens: torch.Tensor, width: int) -> torch.Tensor:
    """``tokens`` (R, L) padded with 0 to ``width`` columns."""
    return torch.nn.functional.pad(tokens, (0, width - tokens.shape[1]))
