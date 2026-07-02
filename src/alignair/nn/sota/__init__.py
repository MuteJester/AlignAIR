"""SOTA open-vocabulary detection components adapted to 1-D DNA (clean re-implementations).

Concepts adapted from permissively-licensed sources (see ATTRIBUTION.md):
  - symmetric InfoNCE + learnable logit_scale ....... open_clip (MIT)
  - late-interaction / region-word MaxSim matching .. GLIP (MIT), ColBERT
  - decoupled detection head ........................ YOLOX (Apache-2.0)
  - object-query decoder ............................ DETR (Apache-2.0)
"""
from .matching import TokenMatch, maxsim_scores, contrastive_match_loss
from .fusion import BiAttentionBlock, ReferenceFusion
from .query_decoder import TypedVDJDecoder, GENES
from .span_head import SpanHead
from .detector import OpenVocabVDJDetector
from .loss import DetectorLoss, interval_giou
from .retrieval import retrieve_topk, maxsim_scores_batched, gather_candidates

__all__ = ["TokenMatch", "maxsim_scores", "contrastive_match_loss",
           "BiAttentionBlock", "ReferenceFusion",
           "TypedVDJDecoder", "GENES", "SpanHead",
           "OpenVocabVDJDetector", "DetectorLoss", "interval_giou",
           "retrieve_topk", "maxsim_scores_batched", "gather_candidates"]
