"""Germline alignment subpackage: base-match utilities, diagonal ops, the shared soft-DP
recurrence, and the seed-and-extend aligner (band head + banded exact DP)."""
from .base_match import base_match_channel, base_match_matrix
from .diagonal_ops import weighted_leading_diag, weighted_reverse_diag, banded_start_end
from .soft_dp import soft_dp_end_logits, NEG
from .germline_aligner import decode_germline_coords
from .band_head import BandHead, band_offset_loss, peak_evidence
from .banded_dp import band_mask_scores, SeedExtendAligner

__all__ = [
    "base_match_channel", "base_match_matrix",
    "weighted_leading_diag", "weighted_reverse_diag", "banded_start_end",
    "soft_dp_end_logits", "NEG",
    "decode_germline_coords",
    "BandHead", "band_offset_loss", "peak_evidence",
    "band_mask_scores", "SeedExtendAligner",
]
