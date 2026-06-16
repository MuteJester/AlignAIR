"""Hierarchical multi-task loss with Kendall uncertainty weighting.

Faithful port of the legacy ``hierarchical_loss``, refactored into its own
nn.Module that owns the uncertainty-weighting parameters.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig
from ..nn.weighting import UncertaintyWeight
from .functional import (
    soft_targets, expectation_from_logits, soft_label_cross_entropy, interval_iou_loss,
)


def _bce_label_smoothing(target: torch.Tensor, prob: torch.Tensor,
                         smoothing: float = 0.1, eps: float = 1e-7) -> torch.Tensor:
    """Binary cross-entropy on probabilities with label smoothing (Keras-style)."""
    target = target * (1.0 - smoothing) + 0.5 * smoothing
    prob = prob.clamp(eps, 1.0 - eps)
    return -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob)).mean()


class AlignAIRLoss(nn.Module):
    def __init__(self, config: ModelConfig, sigma: float = 1.5):
        super().__init__()
        self.config = config
        self.L = config.max_seq_length
        self.has_d_gene = config.has_d_gene
        self.is_multi_chain = config.is_multi_chain
        self.sigma = sigma

        names = ["v_start", "v_end", "j_start", "j_end",
                 "v_classification", "j_classification",
                 "mutation", "indel", "productivity"]
        if self.has_d_gene:
            names += ["d_start", "d_end", "d_classification"]
        if self.is_multi_chain:
            names += ["chain_type"]
        self.weights = nn.ModuleDict({n: UncertaintyWeight() for n in names})

    @torch.no_grad()
    def apply_constraints(self) -> None:
        for w in self.weights.values():
            w.apply_constraints()

    def forward(self, y_true: Dict[str, torch.Tensor],
                y_pred: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        L = self.L

        # --- Segmentation soft-label CE, per boundary, uncertainty-weighted ---
        boundaries = ["v_start", "v_end", "j_start", "j_end"]
        if self.has_d_gene:
            boundaries += ["d_start", "d_end"]
        seg_losses = {}
        for b in boundaries:
            t = soft_targets(y_true[b], L, self.sigma)
            seg_losses[b] = soft_label_cross_entropy(t, y_pred[f"{b}_logits"])
        segmentation_loss = sum(seg_losses[b] * self.weights[b]() for b in boundaries)

        # --- Auxiliary segmentation: Huber length + IoU + hinge ---
        def exp_of(b):
            return expectation_from_logits(y_pred[f"{b}_logits"], L)

        genes = ["v", "j"] + (["d"] if self.has_d_gene else [])
        len_loss = torch.zeros((), device=segmentation_loss.device)
        iou_loss = torch.zeros((), device=segmentation_loss.device)
        hinge_loss = torch.zeros((), device=segmentation_loss.device)
        for g in genes:
            s_exp, e_exp = exp_of(f"{g}_start"), exp_of(f"{g}_end")
            len_pred = (e_exp - s_exp).squeeze(-1)
            len_true = (y_true[f"{g}_end"].float() - y_true[f"{g}_start"].float()).squeeze(-1)
            len_loss = len_loss + F.huber_loss(len_pred, len_true, delta=1.0)
            iou_loss = iou_loss + interval_iou_loss(
                s_exp, e_exp,
                y_true[f"{g}_start"].float().squeeze(-1),
                y_true[f"{g}_end"].float().squeeze(-1))
            hinge_loss = hinge_loss + torch.relu(1.0 - len_pred).mean()
        segmentation_loss = segmentation_loss + 0.1 * len_loss + 0.1 * iou_loss + 0.05 * hinge_loss

        # --- Classification BCE (label smoothing), uncertainty-weighted ---
        clf_v = _bce_label_smoothing(y_true["v_allele"], y_pred["v_allele"])
        clf_j = _bce_label_smoothing(y_true["j_allele"], y_pred["j_allele"])
        classification_loss = clf_v * self.weights["v_classification"]() \
            + clf_j * self.weights["j_classification"]()
        if self.has_d_gene:
            clf_d = _bce_label_smoothing(y_true["d_allele"], y_pred["d_allele"])
            classification_loss = classification_loss + clf_d * self.weights["d_classification"]()
            d_len = (y_pred["d_end"] - y_pred["d_start"]).squeeze(-1)
            short_d_prob = y_pred["d_allele"][:, -1]
            classification_loss = classification_loss + ((d_len < 5).float() * short_d_prob).mean()

        # --- Analysis losses ---
        mutation_loss = (y_true["mutation_rate"].float() - y_pred["mutation_rate"].float()).abs().mean()
        indel_loss = (y_true["indel_count"].float() - y_pred["indel_count"].float()).abs().mean()
        productive_loss = F.binary_cross_entropy(
            y_pred["productive"].clamp(1e-7, 1 - 1e-7), y_true["productive"].float())

        weighted_mutation = mutation_loss * self.weights["mutation"]()
        weighted_indel = indel_loss * self.weights["indel"]()
        weighted_productive = productive_loss * self.weights["productivity"]()

        total = (segmentation_loss + classification_loss
                 + weighted_mutation + weighted_indel + weighted_productive)

        components = {
            "segmentation_loss": segmentation_loss.detach(),
            "classification_loss": classification_loss.detach(),
            "mutation_rate_loss": weighted_mutation.detach(),
            "indel_count_loss": weighted_indel.detach(),
            "productive_loss": weighted_productive.detach(),
        }

        if self.is_multi_chain and "chain_type" in y_pred:
            # Categorical CE on softmax probabilities.
            ct_pred = y_pred["chain_type"].clamp(1e-7, 1.0)
            chain_type_loss = -(y_true["chain_type"] * torch.log(ct_pred)).sum(dim=-1).mean()
            weighted_ct = chain_type_loss * self.weights["chain_type"]()
            total = total + weighted_ct
            components["chain_type_loss"] = weighted_ct.detach()

        # Regularization penalties from the uncertainty weights.
        total = total + sum(w.regularization() for w in self.weights.values())

        components["total_loss"] = total.detach()
        return total, components
