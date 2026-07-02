"""Germline coordinate decoding.

``decode_germline_coords`` turns start/end germline-position posteriors (from the
seed-and-extend banded DP) into integer germline_start / germline_end coordinates,
used by the trainer, evaluator, and inference.
"""
import torch


def decode_germline_coords(start_logits: torch.Tensor, end_logits: torch.Tensor,
                           soft: bool = False):
    """Germline_start and germline_end (end-exclusive). soft=False: argmax
    (gs=argmax, ge=argmax+1). soft=True: rounded soft-argmax expected position over
    valid (finite) columns — sub-integer-stable, kills argmax-plateau jitter."""
    if not soft:
        gs = start_logits.argmax(dim=-1)
        ge = end_logits.argmax(dim=-1) + 1
        return gs, ge

    def _expected(logits):
        pos = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
        p = torch.softmax(logits.float(), dim=-1)        # NEG columns -> ~0 weight
        return (p * pos).sum(dim=-1)

    gs = _expected(start_logits).round().long()
    ge = (_expected(end_logits).round().long()) + 1
    return gs, ge
