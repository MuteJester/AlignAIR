"""lab-chip-v1's core: the units wired as the design's chip-v3 and chip-v4-training diagrams draw them.

After the product's orientation correction: the trunk (``DilatedTrunk``) gives every base its features once; the
parser reads the region path; the anchor heads (``AnchorHead``) find the V Cys and J Trp/Phe, each only inside its
region (the log of the parser's region marginal is added to its position logits, so anchors and parser train each
other); the SHM reader gives each base its mutation and insertion probability and its columns; per gene, the router
weights each base by its soft region weight (through a gradient valve: the scans move the parser's boundaries at
``ROUTE_VALVE`` of their full pull) and by 1 - P(inserted), and the bank scan scores every row of the bound bank
(``bind_reference``) that the batch's genotype (``bank_mask_{gene}``, every allele when absent) allows. At inference
a gene's columns are read only over each read's window of bases whose region marginal for the gene is over
``SUPPORT`` (``ShmReader.top_columns``): a base outside it would weigh at most that in the scan.

Outputs, under the product's keys: the parser's boundary logits and their expectations, the V end and J start refined
by the called alleles' germlines from the anchors (``refiner``, with the anchor heads); at inference, the D posterior
times the junction's local alignment to every D row (``dlocal``), the D start and end refined on the called D's
germline (``drefine``), and the V posterior times the banded gapped alignment of the V span to its top rows
(``vrescore``); ``{gene}_allele``, the
allele's likelihood relative to the best (1 for the best, 0 out of the bank) so the product's set thresholds keep
near ties; the anchor logits; ``d_presence_logits`` from detached evidence (how much of the read the parser puts in
D, how concentrated the D posterior is), so presence reads the evidence without reshaping it; ``mutation_rate`` and
``indel_count`` derived from the reader over the V, D and J bases (no loss). A batch from the chip's adapter
(``teacher_region`` present) also gets the tensors the objective reads, under ``chip_*`` keys; a prediction batch does
not, so prediction moves no per-base logits to the CPU.

Teacher forcing: while training, each read uses the true regions and columns (``teacher_*`` inputs) in place of the
parser's and the reader's with probability ``P_TEACHER``.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from alignair.core.anchors import ANCHOR_GENES, ANCHOR_LOGITS
from alignair.core.d_presence import D_CLASSES, D_PRESENCE_LOGITS
from alignair.errors import ConfigurationError

from ..trunk import TRUNK_CHANNELS, AnchorHead, DilatedTrunk
from .bank import BankCompiler
from .dlocal import local_d_log_posterior
from .drefine import refine_d_boundaries
from .parser import GENES as STATE_OF, IMPOSSIBLE, Parser
from .reader import ShmReader, top_coordinates
from .refiner import refine_boundaries
from .scan import BankScan, allele_log_posterior
from .vrescore import rescore_v

ROUTE_VALVE = 0.1
P_TEACHER = 0.5
TOP_K = 3
SUPPORT = 1e-8            # at inference, the region marginal over which a base's columns are read for the gene's scan
_COUNT ={"v": "v_allele_count", "d": "d_allele_count", "j": "j_allele_count"}


class _Valve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def valve(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forward; the gradient flows back multiplied by ``scale``."""
    return _Valve.apply(x, scale)


class DPresenceEvidence(nn.Module):
    """(B, 4) evidence -> (B, 3) logits over ``D_CLASSES``: Linear(4, 16), GELU, Linear(16, 3)."""

    def __init__(self):
        super().__init__()
        self.hidden, self.classes = nn.Linear(4, 16), nn.Linear(16, len(D_CLASSES))

    def forward(self, evidence):
        return self.classes(F.gelu(self.hidden(evidence)))


def trimmed_width(mask: torch.Tensor) -> int:
    """The positions a forward needs: the batch's longest read and the end slot after it, at most the window."""
    return min(mask.shape[1], int(mask.sum(1).max()) + 1)


def _expectation(logits: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The expected slot (B, 1) of the boundary distribution over the allowed slots."""
    probs = logits.masked_fill(~slots, IMPOSSIBLE).softmax(-1)
    return (probs * torch.arange(logits.shape[-1], device=logits.device)).sum(-1, keepdim=True)


class ChipCore(nn.Module):
    """The chip's units on a trunk ``channels`` wide (``TRUNK_CHANNELS``; lab-chipw64-v1 is 64 wide)."""

    def __init__(self, cfg, channels: int = TRUNK_CHANNELS):
        super().__init__()
        self.genes = ("v", "d", "j") if cfg.has_d else ("v", "j")
        self.counts = {gene: getattr(cfg, _COUNT[gene]) for gene in self.genes}
        self.trunk = DilatedTrunk(cfg.embed_dim, channels)
        self.parser = Parser(channels)
        self.anchors = (nn.ModuleDict({gene: AnchorHead(channels) for gene in ANCHOR_GENES}) if cfg.anchor_heads
                        else None)
        self.reader = ShmReader(channels, genes=self.genes)
        self.scans = nn.ModuleDict({gene: BankScan() for gene in self.genes})
        self.d_presence = DPresenceEvidence() if cfg.d_presence_head else None
        self.bank = None

    def bind_reference(self, reference) -> None:
        """Bind the bank: any reference with the chip's genes, alleles it never trained on included (the bank is an
        input, not weights); the chip's allele outputs are then the bound reference's alleles, in its order."""
        missing = [gene.upper() for gene in self.genes if gene.upper() not in reference.genes]
        if missing:
            raise ConfigurationError(f"the bound reference has no {', '.join(missing)} alleles for the chip's genes",
                                     context={"missing": missing, "genes": list(self.genes)})
        self.counts = {gene: len(reference.gene(gene.upper())) for gene in self.genes}
        device = next(self.parameters()).device
        self.bank = BankCompiler(reference).to(device)

    def forward(self, emb, mask, end_mask, *, tokens, batch):
        if self.bank is None:
            raise ConfigurationError("lab-chip-v1 reads a reference bank: bind one with model.bind_reference")
        whole, width = mask.shape[1], trimmed_width(mask)
        if width == whole:
            return self._forward(emb, mask, end_mask, tokens=tokens, batch=batch)
        batch = {key: value[:, :width] if key.startswith("teacher_") else value for key, value in batch.items()}
        out = self._forward(emb[:, :width], mask[:, :width], end_mask[:, :width], tokens=tokens[:, :width],
                            batch=batch)
        return self._padded(out, whole, width)

    def _padded(self, out: dict, whole: int, width: int) -> dict:
        """The outputs of a forward over the first ``width`` positions, padded to the window's ``whole``: each slot
        past the reads holds what a forward over the whole window gives it."""
        extra = whole - width
        for key, value in out.items():
            if key.endswith(("_start_logits", "_end_logits")):
                out[key] = F.pad(value, (0, extra), value=IMPOSSIBLE)
            elif key.startswith("chip_") and value.dim() > 1:          # per-position internals: masked by the losses
                out[key] = torch.cat([value, value.new_zeros(value.shape[0], extra, *value.shape[2:])], 1)
        for gene, head in (self.anchors or {}).items():
            logits = out[ANCHOR_LOGITS[gene]]                           # a pad slot: the bias plus log(no region)
            pad = (head.position.bias + math.log(1e-6)).expand(logits.shape[0], extra)
            out[ANCHOR_LOGITS[gene]] = torch.cat([logits[:, :width], pad, logits[:, width:]], -1)
        return out

    def _forward(self, emb, mask, end_mask, *, tokens, batch):
        chip_batch = "teacher_region" in batch
        windowed = not (self.training or chip_batch)     # inference: each gene's columns only where the parse puts it
        features = self.trunk(emb, mask)
        parsed = self.parser(features, mask)
        reading = self.reader(features, mask, coordinates=not windowed)
        out = dict(parsed.boundary_logits)
        for gene in self.genes:
            out[f"{gene}_start"] = _expectation(out[f"{gene}_start_logits"], mask)
            out[f"{gene}_end"] = _expectation(out[f"{gene}_end_logits"], end_mask)

        length = mask.shape[1]
        log_region = torch.log(parsed.marginals + 1e-6)
        if self.anchors is not None:
            for gene, head in self.anchors.items():
                logits = head(features, mask)
                out[ANCHOR_LOGITS[gene]] = torch.cat(
                    [logits[:, :length] + log_region[..., STATE_OF[gene]], logits[:, length:]], -1)

        teach = (torch.rand(mask.shape[0], device=mask.device) < P_TEACHER
                 if self.training and chip_batch else None)
        keep = (1 - reading.inserted) * mask
        for gene in self.genes:
            weights = valve(parsed.marginals[..., STATE_OF[gene]], ROUTE_VALVE)
            if windowed:
                columns, probs = self.reader.top_columns(features, gene, parsed.marginals[..., STATE_OF[gene]] > SUPPORT,
                                                         TOP_K)
            else:
                columns, probs = top_coordinates(reading.coord_logits[gene], TOP_K)
            if teach is not None:
                true_region = (batch["teacher_region"] == STATE_OF[gene]).to(weights.dtype)
                true_column = batch[f"teacher_coord_{gene}"]
                taught_columns = torch.zeros_like(columns)
                taught_columns[..., 0] = true_column.clamp_min(0)
                taught_probs = torch.zeros_like(probs)
                taught_probs[..., 0] = (true_column >= 0).to(probs.dtype)
                weights = torch.where(teach[:, None], true_region, weights)
                columns = torch.where(teach[:, None, None], taught_columns, columns)
                probs = torch.where(teach[:, None, None], taught_probs, probs)
            allele_mask = batch.get(f"bank_mask_{gene}")
            if allele_mask is None:
                allele_mask = torch.ones(mask.shape[0], self.counts[gene], dtype=torch.bool, device=mask.device)
            row_logits = self.scans[gene](weights * keep, reading.m, tokens, columns, probs,
                                          getattr(self.bank, f"table_{gene}"), getattr(self.bank, f"cover_{gene}"),
                                          self.bank.row_mask(gene, allele_mask))
            log_posterior = allele_log_posterior(row_logits, getattr(self.bank, f"row_allele_{gene}"),
                                                 self.counts[gene])
            out[f"{gene}_log_posterior"] = log_posterior
            out[f"{gene}_allele"] = torch.exp(log_posterior - log_posterior.max(-1, keepdim=True).values)

        if self.d_presence is not None:
            in_d = parsed.marginals[..., STATE_OF["d"]].detach()
            posterior = out["d_log_posterior"].detach().exp()
            evidence = torch.stack([in_d.max(1).values, in_d.sum(1) / 50., posterior.max(-1).values,
                                    -torch.xlogy(posterior, posterior).sum(-1)], -1)
            out[D_PRESENCE_LOGITS] = self.d_presence(evidence)

        in_vdj = sum(parsed.marginals[..., STATE_OF[gene]] for gene in self.genes)
        out["mutation_rate"] = (reading.m * in_vdj).sum(1, keepdim=True) / in_vdj.sum(1, keepdim=True).clamp_min(1.)
        out["indel_count"] = (reading.inserted * in_vdj).sum(1, keepdim=True)
        if self.anchors is not None:                  # the V end and J start, counted along the bank from the anchors
            refine_boundaries(out, tokens, mask, end_mask, self.bank)
        if "d" in self.genes and not self.training:   # the D call: the scan's times the junction's local alignment
            local = local_d_log_posterior(out, tokens, mask, self.bank, self.counts["d"])
            out["d_log_posterior"] = torch.log_softmax(out["d_log_posterior"] + local, -1)
            out["d_allele"] = torch.exp(out["d_log_posterior"] - out["d_log_posterior"].amax(-1, keepdim=True))
            refine_d_boundaries(out, tokens, mask, end_mask, self.bank)     # the D's edges on the called germline
        if self.anchors is not None and not self.training:     # the V call: the scan's times its banded alignment
            rescore_v(out, tokens, self.bank)

        if chip_batch:
            out["chip_emissions"], out["chip_transitions"] = parsed.emissions, parsed.transitions
            out["chip_log_z"], out["chip_edit_logits"] = parsed.log_z, reading.edit_logits
            out.update({f"chip_coord_logits_{gene}": logits for gene, logits in reading.coord_logits.items()})
        return out
