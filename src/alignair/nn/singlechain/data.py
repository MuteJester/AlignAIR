"""Bridge the GenAIRR gym (`gym_collate`) into SingleChainAlignAIR inputs + targets.

The model wants a fixed-length (576) sequence and per-gene boundary positions + allele multi-hots.
The gym collate already emits in-read boundaries, allele multi-hots (over the fixed reference), and
mutation/indel/productivity, so this just right-pads the read to `max_len` and reshapes the targets.
Boundaries are in the read frame; right-padding leaves them unchanged.

This is the FIXED-reference path: allele targets index the training reference order, matching the
model's `Dense(allele_count)` heads. (Dynamic-reference matching is a later patch.)
"""
import torch


def singlechain_inputs(collated: dict, max_len: int = 576, has_d: bool = True, device=None):
    """collated: one `gym_collate` output. -> (tokens (B, max_len) long, target dict)."""
    device = device or collated["tokens"].device
    tok = collated["tokens"]
    B, Lc = tok.shape
    n = min(Lc, max_len)
    tokens = torch.zeros(B, max_len, dtype=torch.long, device=device)
    tokens[:, :n] = tok[:, :n].to(device)

    tgt = {}
    for g in (["v", "j"] + (["d"] if has_d else [])):
        tgt[f"{g}_start"] = collated[f"{g}_start"].to(device).float().clamp(0, max_len - 1)
        tgt[f"{g}_end"] = collated[f"{g}_end"].to(device).float().clamp(0, max_len - 1)
        tgt[f"{g}_allele"] = collated[f"{g}_allele"].to(device).float()
    for k in ("mutation_rate", "indel_count", "productive"):
        tgt[k] = collated[k].to(device).float().reshape(B)
    return tokens, tgt
