"""Detector training loss.

Adapted from DETR's SetCriterion (Facebook DETR, Apache-2.0) and YOLOX's head loss (Megvii,
Apache-2.0), specialised to fixed typed V/D/J queries (no Hungarian matching — the assignment is
by gene). Per gene:
  - span      : L1 + generalized-IoU on the (start, end) interval (DETR box loss, 1-D)
  - objectness: BCE-with-logits on gene presence (YOLOX obj branch)
  - allele    : multi-positive InfoNCE over the candidate set (see matching.contrastive_match_loss)
  - trim      : L1 on the two germline trims
Span / trim / allele losses apply only where the gene is present; objectness is always supervised.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_decoder import GENES
from .matching import contrastive_match_loss

_EPS = 1e-7


def interval_giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU of 1-D intervals. pred/target (N, 2) = (start, end); order-agnostic. -> (N,)."""
    plo, phi = pred.min(dim=-1).values, pred.max(dim=-1).values
    tlo, thi = target.min(dim=-1).values, target.max(dim=-1).values
    inter = (torch.minimum(phi, thi) - torch.maximum(plo, tlo)).clamp(min=0)
    union = (phi - plo) + (thi - tlo) - inter
    enclosing = torch.maximum(phi, thi) - torch.minimum(plo, tlo)
    iou = inter / union.clamp(min=_EPS)
    return iou - (enclosing - union) / enclosing.clamp(min=_EPS)


class DetectorLoss(nn.Module):
    def __init__(self, w_span: float = 5.0, w_giou: float = 2.0, w_obj: float = 1.0,
                 w_allele: float = 1.0, w_trim: float = 1.0, w_retr: float = 1.0):
        super().__init__()
        self.w = dict(span=w_span, giou=w_giou, obj=w_obj, allele=w_allele, trim=w_trim,
                      retr=w_retr)

    def forward(self, out: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """out[gene]     = {'span','objectness','allele_scores','trim'} (model output).
        targets[gene] = {'span'(B,2),'present'(B,),'allele'(B,Kg) multi-hot,'trim'(B,2)}.
        -> (scalar total loss, {name: float} for logging)."""
        logs, total = {}, out[GENES[0]]["objectness"].new_zeros(())
        for g in GENES:
            o, t = out[g], targets[g]
            present = t["present"].bool()
            obj = F.binary_cross_entropy_with_logits(o["objectness"], present.float())
            allele = contrastive_match_loss(o["allele_scores"], t["allele"])   # zeros absent rows
            # retriever: rank the true allele high among the FULL reference (trains recall@k)
            retr = (contrastive_match_loss(o["retrieval_scores"], t["allele"])
                    if "retrieval_scores" in o else total.new_zeros(()))
            if present.any():
                ps, ts = o["span"][present], t["span"][present]
                span = F.l1_loss(ps, ts)
                giou = (1.0 - interval_giou(ps, ts)).mean()
                trim = F.l1_loss(o["trim"][present], t["trim"][present])
            else:
                span = giou = trim = total.new_zeros(())
            gene_loss = (self.w["span"] * span + self.w["giou"] * giou + self.w["obj"] * obj
                         + self.w["allele"] * allele + self.w["trim"] * trim
                         + self.w["retr"] * retr)
            total = total + gene_loss
            logs.update({f"{g}/span": float(span.detach()), f"{g}/giou": float(giou.detach()),
                         f"{g}/obj": float(obj.detach()), f"{g}/allele": float(allele.detach()),
                         f"{g}/trim": float(trim.detach()), f"{g}/retr": float(retr.detach())})
        logs["total"] = float(total.detach())
        return total, logs
