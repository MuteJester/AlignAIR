"""One gene's reported boundaries straight from a model's raw outputs, as the prediction's clean stage decodes them."""
from collections.abc import Mapping

import torch

from alignair.predict.decode import boundary_slots, decode_boundaries


def decoded_boundaries(outputs: Mapping, gene: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One gene's reported ``(start, end)`` from a model's outputs (tensors or arrays), or None when the outputs
    carry no logits or masks to decode."""
    keys = (f"{gene}_start_logits", f"{gene}_end_logits")
    if not all(key in outputs for key in keys) or not {"start_mask", "end_mask"} <= outputs.keys():
        return None
    start_slots, end_slots = boundary_slots(outputs)
    return decode_boundaries(torch.as_tensor(outputs[keys[0]]), torch.as_tensor(outputs[keys[1]]),
                             start_slots, end_slots)
