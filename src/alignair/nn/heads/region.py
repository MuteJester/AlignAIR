"""Per-position region tagging and exact-boundary decoding."""
import torch
import torch.nn as nn

REGIONS = ("pad", "pre", "V", "N1", "D", "N2", "J", "post")
REGION_INDEX = {name: i for i, name in enumerate(REGIONS)}


class RegionTagger(nn.Module):
    """Per-position region classifier: (B, L, d) -> (B, L, len(REGIONS)) logits."""

    def __init__(self, d_model: int, n_regions: int = len(REGIONS)):
        super().__init__()
        self.fc = nn.Linear(d_model, n_regions)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def decode_boundaries(region_logits: torch.Tensor, mask: torch.Tensor,
                      has_d: bool = True) -> list:
    """Argmax region labels -> per-gene [start, end) from the contiguous run.

    Returns a list (one dict per sample) with v/j(/d)_start/_end; -1 if absent."""
    labels = region_logits.argmax(dim=-1)  # (B, L)
    B = labels.shape[0]
    genes = ["V", "J"] + (["D"] if has_d else [])
    out = []
    for b in range(B):
        valid = mask[b]
        rec = {}
        for g in genes:
            gid = REGION_INDEX[g]
            pos = ((labels[b] == gid) & valid).nonzero(as_tuple=True)[0]
            key = g.lower()
            if pos.numel() == 0:
                rec[f"{key}_start"], rec[f"{key}_end"] = -1, -1
            else:
                rec[f"{key}_start"] = int(pos.min().item())
                rec[f"{key}_end"] = int(pos.max().item()) + 1
        out.append(rec)
    return out
