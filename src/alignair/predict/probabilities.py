"""Selected categorical-head probabilities over one shared inference pass per batch."""
from __future__ import annotations

import numpy as np
import torch

from ..data.inference import SequenceInputAdapter
from ..data.tokenizer import AAProteinTokenizer, AA_TOKEN_DICT
from ..errors import InferenceError, InputError, NumericalError, error_context
from .forward import _output_keys, collect_prediction_outputs, output_numpy, prediction_batches


def softmax_probabilities(logits, *, device=None) -> np.ndarray:
    """Normalize categorical logits, preserving their device unless a placement is explicit."""
    with error_context(stage="class_probabilities"):
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] == 0
                or not logits.is_floating_point()):
            raise InferenceError("class logits must be a floating-point batch-by-class matrix")
        if not torch.isfinite(logits).all():
            raise NumericalError("non-finite class logits")
        values = logits.detach().to(device) if device is not None else logits.detach()
        values = values.float() if values.dtype == torch.bfloat16 else values
        return output_numpy(torch.softmax(values, dim=-1))


def class_probabilities(model, sequences, input_adapter, *, keys, batch_size=256, probability_device="cpu") -> dict:
    """Collect requested heads from a single forward per batch; softmax placement is explicit.

    The CPU default retains research-calibration arithmetic. Raw outputs remain available through
    prediction_batches/forward_batch when coordinates or other heads are also needed.
    """
    keys = _output_keys(keys)
    if keys is None:
        raise InputError("select categorical output keys explicitly")
    batches = prediction_batches(model, sequences, input_adapter, batch_size=batch_size, output_keys=keys)
    return collect_prediction_outputs(
        batches, transform=lambda logits: softmax_probabilities(logits, device=probability_device))


def protein_class_probabilities(model, seqs, cfg, device, keys, batch=256) -> dict:
    """Protein research policy: uppercase declared residues (including X/*), no truncation."""
    adapter = SequenceInputAdapter(AAProteinTokenizer(), cfg.max_seq_length, device,
                                    alphabet=frozenset(AA_TOKEN_DICT), vocab_size=getattr(cfg, "vocab_size", None))
    sequences = adapter.validate(seqs)
    return class_probabilities(model, sequences, adapter, keys=keys, batch_size=batch)


def class_probs(model, seqs, cfg, device, key, batch: int = 256) -> np.ndarray:
    """Compatibility single-head interface; multi-head callers use protein_class_probabilities."""
    return protein_class_probabilities(model, seqs, cfg, device, (key,), batch)[key]
