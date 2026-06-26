"""Per-position state head: germline / SHM-mutation / sequencing-noise / insertion / deletion."""
import torch
import torch.nn as nn

STATES = ("germline", "substitution", "insertion", "deletion")
STATE_INDEX = {name: i for i, name in enumerate(STATES)}


class PerPositionStateHead(nn.Module):
    """(B, L, d) -> (B, L, len(STATES)) per-position state logits."""

    def __init__(self, d_model: int, n_states: int = len(STATES)):
        super().__init__()
        self.fc = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def state_reliability(state_logits: torch.Tensor, r_min: float = 0.25) -> torch.Tensor:
    """Per-position base-channel reliability r = r_min + (1-r_min)*(1 - P(substitution)).

    Used to condition the soft-DP raw-token emission: where the state head thinks a
    position is an SHM substitution, r -> r_min so the +1/-1 base match/mismatch is
    down-weighted (SHM shouldn't penalise the true allele). r_min keeps a diagnostic-SNP
    floor so true allele-distinguishing differences are never fully erased."""
    pi_sub = state_logits.softmax(dim=-1)[..., STATE_INDEX["substitution"]]
    return r_min + (1.0 - r_min) * (1.0 - pi_sub)


def state_counts(state_logits: torch.Tensor, mask: torch.Tensor) -> dict:
    """Per-sample counts of substitution / indel from argmax states (padding ignored)."""
    labels = state_logits.argmax(dim=-1)  # (B, L)
    valid = mask

    def count(name: str) -> torch.Tensor:
        return ((labels == STATE_INDEX[name]) & valid).sum(dim=-1)

    return {
        "substitution_count": count("substitution"),
        "indel_count": count("insertion") + count("deletion"),
    }
