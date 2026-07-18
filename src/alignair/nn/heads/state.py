"""Per-position state head: germline / SHM-mutation / sequencing-noise / insertion / deletion."""
import torch
import torch.nn as nn

STATES = ("germline", "substitution", "insertion", "deletion")
STATE_INDEX = {name: i for i, name in enumerate(STATES)}

# Per-position labels are dominated by "germline"; up-weight the rare, useful edit classes (the
# indels especially) so the weighted CE does not collapse to predicting germline everywhere.
STATE_CLASS_WEIGHTS = (1.0, 3.0, 8.0, 8.0)


class PerPositionStateHead(nn.Module):
    """(B, L, d) -> (B, L, len(STATES)) per-position state logits."""

    def __init__(self, d_model: int, n_states: int = len(STATES)):
        super().__init__()
        self.fc = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class PerPositionStateBranch(nn.Module):
    """Per-position edit-state predictor: ``emb (B, L, C)`` -> state logits ``(B, L, len(STATES))``.

    A stack of dilated, same-padded 1-D convs gives each position a wide receptive field — enough to
    flag a local germline deviation (substitution) or the alignment shift an indel produces — then a
    linear state head. It runs on the *canonicalized* embedding, so its output is in the forward frame
    and aligns position-for-position with the forward-frame ``state_labels`` target. Dilations
    ``(1,2,4,8,16)`` with kernel 3 give a receptive field of 63 positions."""

    def __init__(self, in_dim: int, hidden: int = 128, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        layers = []
        d = in_dim
        for dil in dilations:
            layers += [nn.Conv1d(d, hidden, kernel_size=3, padding=dil, dilation=dil), nn.GELU()]
            d = hidden
        self.convs = nn.Sequential(*layers)
        self.head = PerPositionStateHead(hidden)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:            # (B, L, C) -> (B, L, S)
        h = self.convs(emb.transpose(1, 2)).transpose(1, 2)
        return self.head(h)


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
