"""Germline alignment subpackage: base-match utilities, diagonal ops, and the aligner
variants (soft-DP oracle, legacy diagonal, pointer baseline, the seed-and-extend band head
and banded DP)."""
from .base_match import base_match_channel, base_match_matrix
from .diagonal_ops import weighted_leading_diag, weighted_reverse_diag, banded_start_end
from .soft_dp import soft_dp_end_logits, SoftDPAligner, NEG
from .germline_aligner import GermlineAligner, decode_germline_coords
from .pointer import BandedPointerAligner
from .band_head import BandHead, band_offset_loss, peak_evidence
from .banded_dp import band_mask_scores, SeedExtendAligner

__all__ = [
    "base_match_channel", "base_match_matrix",
    "weighted_leading_diag", "weighted_reverse_diag", "banded_start_end",
    "soft_dp_end_logits", "SoftDPAligner", "NEG",
    "GermlineAligner", "decode_germline_coords",
    "BandedPointerAligner",
    "BandHead", "band_offset_loss", "peak_evidence",
    "band_mask_scores", "SeedExtendAligner",
]
