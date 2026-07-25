"""Predeclared V+J penultimate representation for Stage-2, extracted from the FROZEN model without
changing any prediction.

The representation is the input to each gene branch's ``cls_head`` — the pooled per-branch feature
``silu(cls_mid(cls_tower(emb*mask)))`` that the class head reads — captured via a forward pre-hook and
concatenated across the ``v`` and ``j`` branches (predeclared order). This is a LEARNED VH representation
(unlike the raw token+position embedding, which would encode composition/position only). Hooks observe;
they do not alter the model outputs, so predictions are byte-identical with or without extraction.
"""
from __future__ import annotations

import numpy as np
import torch

REPRESENTATION_VERSION = "vj_penultimate_v1"


def branch_features(predictor, sequences, branches=("v", "j")) -> np.ndarray:
    """Return ``[n, d]`` float features: the concatenated ``cls_head``-input of the named branches."""
    cfg = predictor.cfg
    L = cfg.max_seq_length
    toks = torch.zeros(len(sequences), L, dtype=torch.long)
    for i, s in enumerate(sequences):
        enc = predictor.tokenizer.encode(s)
        if len(enc) > L:
            raise ValueError(f"sequence length {len(enc)} exceeds max_seq_length {L}")
        toks[i, :len(enc)] = torch.from_numpy(enc)

    model = predictor.model
    model.eval()
    captured: dict = {}
    handles = [model.branches[b].cls_head.register_forward_pre_hook(
        lambda mod, inp, nm=b: captured.__setitem__(nm, inp[0].detach())) for b in branches]
    try:
        with torch.no_grad():
            model({"tokenized_sequence": toks.to(predictor.device)})
    finally:
        for h in handles:
            h.remove()
    return torch.cat([captured[b] for b in branches], dim=-1).cpu().numpy()
