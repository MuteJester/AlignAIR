"""Predeclared V+J penultimate representation for Stage-2, extracted from the FROZEN model without
changing any prediction.

The representation is the input to each gene branch's ``cls_head`` — the pooled per-branch feature
``silu(cls_mid(cls_tower(emb*mask)))`` that the class head reads — captured via a forward pre-hook and
concatenated across the ``v`` and ``j`` branches (predeclared order). This is a LEARNED VH representation
(unlike the raw token+position embedding, which would encode composition/position only). Hooks observe;
they do not alter the model outputs, so predictions are byte-identical with or without extraction.
"""
from __future__ import annotations

from alignair.data.inference import SequenceInputAdapter
from alignair.predict.features import component_features

REPRESENTATION_VERSION = "vj_penultimate_v1"


def branch_features(predictor, sequences, branches=("v", "j")):
    """Extract the predeclared branch-head inputs through the shared feature observer."""
    adapter = SequenceInputAdapter(predictor.tokenizer, predictor.cfg.max_seq_length, predictor.device,
                                    vocab_size=getattr(predictor.cfg, "vocab_size", None))
    return component_features(predictor.model, sequences, adapter,
                               components=tuple(f"{branch}.cls_head" for branch in branches))
