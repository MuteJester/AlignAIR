"""The chip's objective: every unit trained on its own GenAIRR truth, each loss divided by its chance level so all start
near 1, summed with fixed weights (``LOSS_WEIGHTS``, all 1; no learned uncertainty weights, which quietly turn down
the hard tasks).

- ``orientation``: cross entropy over the 4 orientations, / log 4.
- ``parse``: the parser's minus log probability of the paths the read cannot tell apart from the true one
  (``region_allowed``), per boundary: over (the true path's region changes + 1) x log(read length), about the
  boundaries' chance level. A read has a handful of boundaries and hundreds of easy interior bases, so a per-base
  scale would dilute a misplaced boundary a hundredfold.
- ``v_anchor``, ``j_anchor``: cross entropy over the read's positions and the "not in read" slot, / log(L + 1).
- ``shm``: unweighted cross entropy over germline / substitution / insertion on the V, D and J bases (deletion-tagged
  bases left out), / log 3: unweighted, so ``m`` stays a calibrated probability, as the scans' costs need.
- ``coord``: cross entropy over each gene's frame columns on the bases its alignment maps, / log(width), averaged
  over the genes.
- ``scan_{gene}``: minus the log posterior mass on the true allele set over the bank rows the batch allows, / log(the
  number of alleles allowed); a read whose bank allows fewer than 2 alleles has nothing to choose and no term.
- ``d_presence``: cross entropy over absent / too short / callable, / log 3.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ..core.anchors import ANCHOR_GENES, ANCHOR_KEYS, ANCHOR_LOGITS
from ..core.d_presence import D_CLASSES, D_PRESENCE, D_PRESENCE_LOGITS
from ..core.edit_states import EDIT_INDEX
from ..core.regions import REGIONS
from ..core.orientation import NUM_ORIENTATIONS
from ..core.tokens import IGNORE_INDEX
from ..tasks.metrics import MetricReport, alignment_metrics
from ..tasks.supervision import supervised_rows

from .parser import GENE_STATE, ParserOutput, crf_nll
from .reader import READ_EDITS
from .scan import OUT

LOSS_WEIGHTS = {name: 1.0 for name in ("orientation", "parse", *ANCHOR_KEYS.values(), "shm", "coord", "scan_v",
                                       "scan_d", "scan_j", "d_presence")}


def shm_labels(state_labels: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """The product's per-base edit states (germline, substitution, insertion, deletion) as the reader's classes on V,
    D and J bases; ``IGNORE_INDEX`` for a deletion-tagged base, a base outside V, D and J, and a pad."""
    germline_encoded = torch.zeros_like(region, dtype=torch.bool)
    for state in GENE_STATE.values():
        germline_encoded |= region == state
    keep = germline_encoded & (state_labels >= 0) & (state_labels != EDIT_INDEX["deletion"])
    return state_labels.masked_fill(~keep, IGNORE_INDEX)


def scan_loss(log_posterior: torch.Tensor, true: torch.Tensor, supervise: torch.Tensor):
    """``(loss (B,), rows (B,) bool)``: per read, minus the log posterior mass on its true alleles that the bank
    allows, over log(alleles allowed); ``rows`` marks the reads with a term (supervised, at least 2 alleles allowed,
    a true allele among them)."""
    allowed = log_posterior > OUT / 2
    count = allowed.sum(-1)
    true = true & allowed
    rows = supervise & (count >= 2) & true.any(-1)
    mass = torch.logsumexp(log_posterior.masked_fill(~true, OUT), -1)
    return -mass / torch.log(count.clamp_min(2).to(log_posterior.dtype)), rows


def parse_loss(output: ParserOutput, allowed: torch.Tensor, region: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) the parser's NLL of the allowed paths per boundary: over (region changes of the true path + 1) x
    log(read length)."""
    changes = ((region[:, 1:] != region[:, :-1]) & mask[:, 1:]).sum(1)
    length = mask.sum(1).clamp_min(2).to(output.log_z.dtype)
    return crf_nll(output, allowed, mask) / ((changes + 1) * torch.log(length))


def _mean(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return values[rows].mean() if rows.any() else values.sum() * 0.


def _allowed_states(bits: torch.Tensor) -> torch.Tensor:
    """(B, L) bitmasks over ``REGIONS`` -> (B, L, S) bool."""
    return ((bits.unsqueeze(-1) >> torch.arange(len(REGIONS), device=bits.device)) & 1).bool()


class ChipObjective:
    """The chip's training objective (``tasks.TrainingTasks``): its loss terms, metrics and checkpoint score."""

    def __init__(self, config):
        self.config = config
        self.genes = config.genes

    def loss(self, outputs: dict, targets: dict):
        """The weighted total loss (``LOSS_WEIGHTS``) and its terms."""
        mask = outputs["start_mask"]
        parts = {}
        if "orientation_logits" in outputs and "orientation" in targets:
            parts["orientation"] = (F.cross_entropy(outputs["orientation_logits"], targets["orientation"])
                                    / math.log(NUM_ORIENTATIONS))

        parsed = ParserOutput(outputs["chip_emissions"], outputs["chip_transitions"], mask, outputs["chip_log_z"],
                              None, {})
        parts["parse"] = parse_loss(parsed, _allowed_states(targets["region_allowed"]), targets["region"], mask).mean()

        length = mask.shape[1]
        slots = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        for gene in ANCHOR_GENES:
            if ANCHOR_LOGITS[gene] in outputs:
                key = ANCHOR_KEYS[gene]
                logits = outputs[ANCHOR_LOGITS[gene]].masked_fill(~slots, OUT)
                ce = F.cross_entropy(logits, targets[key], reduction="none")
                parts[key] = _mean(ce, targets[f"{key}_supervise"].bool()) / math.log(length + 1)

        labels = shm_labels(targets["state_labels"], targets["region"])
        edit = outputs["chip_edit_logits"]
        parts["shm"] = (F.cross_entropy(edit.reshape(-1, len(READ_EDITS)), labels.reshape(-1),
                                        ignore_index=IGNORE_INDEX)
                        if (labels >= 0).any() else edit.sum() * 0.) / math.log(len(READ_EDITS))

        coord = []
        for gene in self.genes:
            logits, columns = outputs[f"chip_coord_logits_{gene}"], targets[f"coord_{gene}"]
            if (columns >= 0).any():
                coord.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), columns.reshape(-1),
                                             ignore_index=IGNORE_INDEX) / math.log(logits.shape[-1]))
        parts["coord"] = torch.stack(coord).mean() if coord else edit.sum() * 0.

        for gene in self.genes:
            loss, rows = scan_loss(outputs[f"{gene}_log_posterior"], targets[f"{gene}_allele"] > 0.5,
                                   supervised_rows(targets, gene, outputs[f"{gene}_log_posterior"]))
            parts[f"scan_{gene}"] = _mean(loss, rows)

        if D_PRESENCE_LOGITS in outputs and D_PRESENCE in targets:
            ce = F.cross_entropy(outputs[D_PRESENCE_LOGITS], targets[D_PRESENCE], reduction="none")
            parts[D_PRESENCE] = _mean(ce, targets[f"{D_PRESENCE}_supervise"].bool()) / math.log(len(D_CLASSES))
        total = sum(LOSS_WEIGHTS[name] * value for name, value in parts.items())
        return total, parts

    def metric_report(self, outputs: dict, targets: dict) -> MetricReport:
        """The batch's metrics, as numerators and counts that merge across batches."""
        report = alignment_metrics(outputs, targets, self.genes)
        mask = outputs["start_mask"]
        slots = {"start": mask, "end": outputs["end_mask"]}
        for gene in self.genes:
            rows = supervised_rows(targets, gene, outputs[f"{gene}_allele"])
            for side in ("start", "end"):
                decoded = outputs[f"{gene}_{side}_logits"].masked_fill(~slots[side], OUT).argmax(-1)
                report.add(f"parse_{gene}_{side}_exact", decoded == targets[f"{gene}_{side}"].reshape(-1).round(),
                           rows)
            log_posterior = outputs[f"{gene}_log_posterior"]
            best = log_posterior.argmax(-1, keepdim=True)
            report.add(f"scan_{gene}_top1", targets[f"{gene}_allele"].gather(1, best).squeeze(1) > 0.5, rows)
            report.add(f"bank_{gene}_size", (log_posterior > OUT / 2).sum(-1).float(), keep_empty=True)
            if f"chip_coord_logits_{gene}" in outputs:
                columns = targets[f"coord_{gene}"]
                report.add(f"coord_{gene}_top1", outputs[f"chip_coord_logits_{gene}"].argmax(-1) == columns,
                           columns >= 0)
        anchor_slots = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        for gene in ANCHOR_GENES:
            if ANCHOR_LOGITS[gene] in outputs:
                key = ANCHOR_KEYS[gene]
                decoded = outputs[ANCHOR_LOGITS[gene]].masked_fill(~anchor_slots, OUT).argmax(-1)
                report.add(f"anchor_{gene}_exact", decoded == targets[key], targets[f"{key}_supervise"].bool())
        if "chip_edit_logits" in outputs:
            labels = shm_labels(targets["state_labels"], targets["region"])
            probs = outputs["chip_edit_logits"].softmax(-1)
            onehot = F.one_hot(labels.clamp_min(0), len(READ_EDITS)).to(probs.dtype)
            report.add("shm_brier", ((probs - onehot) ** 2).sum(-1), labels >= 0)
        if D_PRESENCE_LOGITS in outputs and D_PRESENCE in targets:
            report.add("d_presence_acc", outputs[D_PRESENCE_LOGITS].argmax(-1) == targets[D_PRESENCE],
                       targets[f"{D_PRESENCE}_supervise"].bool())
        return report

    def metrics(self, outputs: dict, targets: dict) -> dict[str, float]:
        """The batch's metric means."""
        return self.metric_report(outputs, targets).means()

    def selection_score(self, metrics: dict[str, float]) -> float:
        """The score that selects the best checkpoint: the mean top-1 scan accuracy over the genes (NaN without)."""
        values = [metrics[f"scan_{gene}_top1"] for gene in self.genes if f"scan_{gene}_top1" in metrics]
        return sum(values) / len(values) if values and all(math.isfinite(v) for v in values) else math.nan
