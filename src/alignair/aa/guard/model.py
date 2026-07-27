"""Stage-4 — GuardModel: assemble encoder -> (immutable HMM fusion) -> heads (+ source-adversarial head via
GRL). Candidate variants C1–C4 progressively enable fusion, energy/OE, contrastive, and source-adversarial.
C0 is the HMM-only rule baseline (c0_rules) and has no learned parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from alignair.aa.guard.encoder import ApplicabilityEncoder
from alignair.aa.guard.heads import GuardHeads
from alignair.aa.guard.losses import grad_reverse

# Which objectives each learned candidate enables (masks/terms activated at train time).
CANDIDATE_CONFIG = {
    "C1": {"fusion": False, "energy": False, "contrast": False, "source": False},
    "C2": {"fusion": True, "energy": False, "contrast": False, "source": False},
    "C3": {"fusion": True, "energy": True, "contrast": False, "source": False},
    "C4": {"fusion": True, "energy": True, "contrast": True, "source": True},
}
LEARNED_CANDIDATES = ("C1", "C2", "C3", "C4")


class GuardModel(nn.Module):
    def __init__(self, cid: str, hmm_dim: int, *, width=128, depth=4, heads=4, n_sources=4):
        super().__init__()
        assert cid in CANDIDATE_CONFIG, cid
        self.cid = cid
        self.cfg = CANDIDATE_CONFIG[cid]
        self.encoder = ApplicabilityEncoder(width=width, depth=depth, heads=heads)
        seq_dim = 2 * width                                    # cls ⊕ mean
        fused_dim = seq_dim + (hmm_dim if self.cfg["fusion"] else 0)
        self.heads = GuardHeads(fused_dim)
        self.source_head = nn.Linear(seq_dim, n_sources) if self.cfg["source"] else None

    def forward(self, ids, mask, hmm_norm=None, grl_lambda: float = 1.0):
        cls, mean = self.encoder(ids, mask)
        rep = torch.cat([cls, mean], dim=-1)                  # sequence representation
        if self.cfg["fusion"]:
            assert hmm_norm is not None, "C2+ require normalized HMM features"
            fused = torch.cat([rep, hmm_norm.to(rep.dtype)], dim=-1)
        else:
            fused = rep
        out = self.heads(fused)
        out["energy"] = GuardHeads.energy(out["domain_type"])
        out["rep"] = rep
        if self.source_head is not None:
            out["source_logits"] = self.source_head(grad_reverse(rep, grl_lambda))   # GRL: reverse into rep
        else:
            out["source_logits"] = torch.zeros(rep.shape[0], 1, device=rep.device)
        return out
