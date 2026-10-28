"""Predeclared positive-only scorer candidates for Stage-2 (completeness and membership use the SAME
candidate family, fit on their own positive-only data).

Both score higher = more in-manifold, are fit on positives only (no negative ever touches fitting), and
serialize pickle-free (plain float arrays via ``np.savez``). Candidates:
  - ``shrinkage_mahalanobis`` : Ledoit-Wolf-shrunk Mahalanobis distance (essential — the V+J penultimate
    dim exceeds the fit count, so the sample covariance is singular);
  - ``knn`` : negative mean Euclidean distance to the k nearest fit points.
Selection between them (and any representation) happens on VALIDATION only (see the Stage-2 fit pipeline).
"""
from __future__ import annotations

import io

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf

_EPS = 1e-6


def _standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0) + _EPS
    return mu, sd


def _pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})
    return buf.getvalue()


def _unpack(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:      # allow_pickle=False -> pickle-free
        return {k: z[k] for k in z.files}


class ShrinkageMahalanobisScorer:
    name = "shrinkage_mahalanobis"

    def fit(self, X: np.ndarray):
        self.mu, self.sd = _standardize_fit(X)
        Z = (X - self.mu) / self.sd
        lw = LedoitWolf().fit(Z)
        self.mean = lw.location_.astype(np.float64)
        self.precision = lw.precision_.astype(np.float64)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd - self.mean
        m = np.einsum("ij,jk,ik->i", Z, self.precision, Z)
        return -np.sqrt(np.maximum(m, 0.0))                        # higher = more in-manifold

    def to_bytes(self) -> bytes:
        return _pack(mu=self.mu, sd=self.sd, mean=self.mean, precision=self.precision)

    @classmethod
    def from_bytes(cls, blob: bytes):
        d = _unpack(blob)
        s = cls()
        s.mu, s.sd, s.mean, s.precision = d["mu"], d["sd"], d["mean"], d["precision"]
        return s


class KNNScorer:
    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: np.ndarray):
        self.mu, self.sd = _standardize_fit(X)
        self.Z = ((X - self.mu) / self.sd).astype(np.float64)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Zq = (X - self.mu) / self.sd
        D = cdist(Zq, self.Z)
        k = min(self.k, self.Z.shape[0])
        return -np.sort(D, axis=1)[:, :k].mean(1)                 # higher = more in-manifold

    def to_bytes(self) -> bytes:
        return _pack(mu=self.mu, sd=self.sd, Z=self.Z, k=np.array([self.k]))

    @classmethod
    def from_bytes(cls, blob: bytes):
        d = _unpack(blob)
        s = cls(k=int(d["k"][0]))
        s.mu, s.sd, s.Z = d["mu"], d["sd"], d["Z"]
        return s


CANDIDATES = {"shrinkage_mahalanobis": ShrinkageMahalanobisScorer, "knn": KNNScorer}
