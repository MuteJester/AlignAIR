"""Stage-3 Task 3.4 — HmmFusion: normalize the raw HMM feature vector with stored train-fit stats and fuse
it with the sequence representation, while PRESERVING the raw HMM vector for provenance (never overwritten).

Provenance integrity is enforced HARD: fusion is bound to an exact ``feature_names`` ordering and the
``bank_checksum`` the stats were fit on. If either changes, ``fuse``/``normalize`` RAISE ``FusionContractError``
rather than silently feed features into the wrong positions.
"""
from __future__ import annotations

from alignair.errors import DataContractError

import numpy as np
import torch


class FusionContractError(DataContractError):
    """Raised when the incoming HMM feature ordering or bank checksum does not match what fusion was fit on."""
    code = "ALIGNAIR_FUSION_CONTRACT"


class HmmFusion:
    def __init__(self, feature_names, mean, std, bank_checksum: str):
        self.feature_names = tuple(feature_names)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.std_safe = np.where(self.std > 1e-8, self.std, 1.0)
        self.bank_checksum = bank_checksum
        if not (len(self.feature_names) == self.mean.shape[0] == self.std.shape[0]):
            raise ValueError("feature_names / mean / std length mismatch")

    def _check(self, feature_names, bank_checksum):
        if tuple(feature_names) != self.feature_names:
            raise FusionContractError(
                f"HMM feature ordering changed: fusion was fit on {self.feature_names}, got {tuple(feature_names)}")
        if bank_checksum != self.bank_checksum:
            raise FusionContractError(
                f"HMM bank checksum changed: fusion was fit on {self.bank_checksum}, got {bank_checksum}")

    def normalize(self, hmm_vec, *, feature_names, bank_checksum):
        """(hmm_vec - mean) / std using STORED stats; hard-fails on ordering/checksum mismatch."""
        self._check(feature_names, bank_checksum)
        v = np.asarray(hmm_vec, dtype=np.float64)
        return torch.tensor((v - self.mean) / self.std_safe, dtype=torch.float32)

    def fuse(self, seq_rep, hmm_vec, *, feature_names, bank_checksum):
        """Return (fused_tensor, raw_hmm_vec_unchanged). The RAW vector is returned alongside and never
        overwritten. Hard-fails if the feature ordering or bank checksum has changed."""
        raw = torch.as_tensor(np.asarray(hmm_vec, dtype=np.float64), dtype=torch.float32)   # untouched provenance
        norm = self.normalize(hmm_vec, feature_names=feature_names, bank_checksum=bank_checksum).to(seq_rep.dtype)
        if seq_rep.dim() == 2 and norm.dim() == 1:
            norm = norm.unsqueeze(0).expand(seq_rep.shape[0], -1)
        fused = torch.cat([seq_rep, norm.to(seq_rep.device)], dim=-1)
        return fused, raw


def fit_fusion(feature_names, vectors, bank_checksum) -> HmmFusion:
    """Fit normalization stats (mean/std) on a stack of raw HMM feature vectors (train/dev only)."""
    arr = np.asarray(vectors, dtype=np.float64)
    return HmmFusion(feature_names, arr.mean(axis=0), arr.std(axis=0), bank_checksum)
