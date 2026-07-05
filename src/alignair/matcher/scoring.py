import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_NEG = -1e4


def masked_maxsim(read_emb, read_mask, cand_emb, cand_sal, cand_mask):
    """Salience-weighted MaxSim of each candidate against one read.

    read_emb [Ls,d], read_mask [Ls] bool; cand_emb [C,Lc,d], cand_sal [C,Lc],
    cand_mask [C,Lc] bool. Returns scores [C].
    """
    sim = cand_emb @ read_emb.t()                       # [C,Lc,Ls]
    sim = sim.masked_fill(~read_mask.view(1, 1, -1), _NEG)
    best = sim.max(dim=-1).values                       # [C,Lc]  best read match per cand pos
    best = best.masked_fill(~cand_mask, 0.0)
    w = cand_sal.masked_fill(~cand_mask, 0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # normalize salience per candidate
    return (w * best).sum(dim=-1)                        # [C]


def pick_strand(v_scores_fwd, v_scores_rc) -> int:
    """0 if forward's best V candidate beats revcomp's, else 1."""
    return 0 if v_scores_fwd.max() >= v_scores_rc.max() else 1


def _expand_candidate_tensor(x: torch.Tensor, batch_size: int, batched_ndim: int) -> torch.Tensor:
    """Accept candidate tensors as unbatched or batched, with explicit expected rank."""
    if x.dim() == batched_ndim:
        if x.shape[0] != batch_size:
            raise ValueError(f"batched candidate tensor has batch {x.shape[0]}, expected {batch_size}")
        return x
    if x.dim() == batched_ndim - 1:
        return x.unsqueeze(0).expand(batch_size, *x.shape)
    raise ValueError(
        f"candidate tensor rank must be {batched_ndim - 1} or {batched_ndim}, got {x.dim()}"
    )


def _band_positions(offsets: torch.Tensor, cand_len: int, read_len: int, radius: int):
    """Return read positions [B,K,Lc,W] for allele position i at offset+i+delta."""
    if offsets.dim() != 2:
        raise ValueError("offsets must have shape [B,K]")
    dev = offsets.device
    i = torch.arange(cand_len, device=dev).view(1, 1, cand_len, 1)
    d = torch.arange(-radius, radius + 1, device=dev).view(1, 1, 1, 2 * radius + 1)
    pos = offsets.to(torch.long).unsqueeze(-1).unsqueeze(-1) + i + d
    in_bounds = (pos >= 0) & (pos < read_len)
    return pos.clamp(0, max(read_len - 1, 0)), in_bounds


def _gather_read(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather read values at [B,K,Lc,W] positions.

    values is [B,L] or [B,L,D]. The result is [B,K,Lc,W] or [B,K,Lc,W,D].
    """
    B, L = values.shape[:2]
    K, Lc, W = positions.shape[1:]
    flat_pos = positions.reshape(B * K, Lc * W)
    if values.dim() == 2:
        flat = values.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
        return flat.gather(1, flat_pos).reshape(B, K, Lc, W)
    if values.dim() == 3:
        D = values.shape[-1]
        flat = values.unsqueeze(1).expand(B, K, L, D).reshape(B * K, L, D)
        got = flat.gather(1, flat_pos.unsqueeze(-1).expand(B * K, Lc * W, D))
        return got.reshape(B, K, Lc, W, D)
    raise ValueError("read values must have shape [B,L] or [B,L,D]")


def _base_match(cand_ids: torch.Tensor, read_ids: torch.Tensor) -> torch.Tensor:
    """Raw +1 match / -1 mismatch / 0 non-ACGT channel over token ids."""
    real = (cand_ids >= 1) & (cand_ids <= 4) & (read_ids >= 1) & (read_ids <= 4)
    return real.to(torch.float32) * (2.0 * (cand_ids == read_ids).to(torch.float32) - 1.0)


def banded_pair_scores(
    read_emb: torch.Tensor,
    read_mask: torch.Tensor,
    cand_emb: torch.Tensor,
    cand_mask: torch.Tensor,
    offsets: torch.Tensor,
    radius: int,
    *,
    read_ids: Optional[torch.Tensor] = None,
    cand_ids: Optional[torch.Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    base_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pair scores only in a seeded diagonal band.

    Shapes:
      read_emb/read_mask: [B,Ls,D], [B,Ls]
      cand_emb/cand_mask: [C,Lc,D]/[C,Lc] or [B,K,Lc,D]/[B,K,Lc]
      offsets: [B,K], where read position j ~= allele position i + offset

    Returns:
      pair [B,K,Lc,2r+1], valid [B,K,Lc,2r+1]

    This is the expensive part of the replacement scorer, and its complexity is
    O(B*K*Lc*(2r+1)*D), instead of MaxSim's O(B*K*Lc*Ls*D).
    """
    B, Ls, _ = read_emb.shape
    cand_emb = _expand_candidate_tensor(cand_emb, B, 4)
    cand_mask = _expand_candidate_tensor(cand_mask, B, 3)
    _, K, Lc, _ = cand_emb.shape
    if offsets.shape != (B, K):
        raise ValueError(f"offsets must have shape {(B, K)}, got {tuple(offsets.shape)}")

    pos, in_bounds = _band_positions(offsets, Lc, Ls, radius)
    read_band = _gather_read(read_emb, pos)                         # [B,K,Lc,W,D]
    read_valid = _gather_read(read_mask.to(torch.bool), pos) & in_bounds
    valid = cand_mask.to(torch.bool).unsqueeze(-1) & read_valid

    pair = (read_band * cand_emb.unsqueeze(-2)).sum(dim=-1)
    if logit_scale is not None:
        pair = pair * logit_scale.clamp(-2.0, 3.0).exp()

    if read_ids is not None and cand_ids is not None:
        cand_ids = _expand_candidate_tensor(cand_ids, B, 3)
        read_tok = _gather_read(read_ids, pos)
        base = _base_match(cand_ids.unsqueeze(-1), read_tok)
        if base_weight is None:
            pair = pair + base
        else:
            pair = pair + (0.25 + F.softplus(base_weight)) * base

    return pair.masked_fill(~valid, _NEG), valid


class BandedAlignmentScorer(nn.Module):
    """Seeded, reference-agnostic replacement for salience-weighted MaxSim.

    The seed supplies one diagonal offset per read/candidate. The scorer evaluates only a
    narrow band around that offset, adds a raw-base match channel so unseen alleles still
    get an exact SNP signal, then aggregates with either:

      * ``diag``: salience-weighted logsumexp over fixed diagonals.
      * ``softpath``: a compact band DP that allows the path to drift by one band column per
        allele position with a learned penalty, approximating small indels without building
        the full read x allele matrix.
    """

    def __init__(
        self,
        radius: int = 8,
        aggregation: str = "diag",
        temperature: float = 0.2,
        coverage_weight: float = 0.05,
    ):
        super().__init__()
        if aggregation not in {"diag", "softpath"}:
            raise ValueError("aggregation must be 'diag' or 'softpath'")
        self.radius = int(radius)
        self.aggregation = aggregation
        self.temperature = float(temperature)
        self.coverage_weight = float(coverage_weight)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.base_weight = nn.Parameter(torch.tensor(0.0))
        self._drift_penalty = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        read_emb: torch.Tensor,
        read_mask: torch.Tensor,
        cand_emb: torch.Tensor,
        cand_sal: torch.Tensor,
        cand_mask: torch.Tensor,
        offsets: torch.Tensor,
        *,
        read_ids: Optional[torch.Tensor] = None,
        cand_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return candidate scores [B,K]."""
        B = read_emb.shape[0]
        cand_sal = _expand_candidate_tensor(cand_sal, B, 3)
        cand_mask = _expand_candidate_tensor(cand_mask, B, 3)
        pair, valid = banded_pair_scores(
            read_emb,
            read_mask,
            cand_emb,
            cand_mask,
            offsets,
            self.radius,
            read_ids=read_ids,
            cand_ids=cand_ids,
            logit_scale=self.logit_scale,
            base_weight=self.base_weight,
        )
        if self.aggregation == "diag":
            return self._diag_score(pair, valid, cand_sal, cand_mask)
        return self._softpath_score(pair, valid, cand_sal, cand_mask)

    def _diag_score(self, pair, valid, cand_sal, cand_mask):
        w = cand_sal.masked_fill(~cand_mask, 0.0).unsqueeze(-1) * valid.to(pair.dtype)
        denom = w.sum(dim=2).clamp_min(1e-6)                    # [B,K,W]
        num = (pair.masked_fill(~valid, 0.0) * w).sum(dim=2)
        diag = num / denom
        total = cand_sal.masked_fill(~cand_mask, 0.0).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coverage = (denom / total).clamp_min(1e-4)
        diag = diag + self.coverage_weight * coverage.log()
        diag = diag.masked_fill(denom <= 1e-6, _NEG)
        t = max(self.temperature, 1e-4)
        return t * torch.logsumexp(diag / t, dim=-1)

    def _softpath_score(self, pair, valid, cand_sal, cand_mask):
        B, K, Lc, W = pair.shape
        pair = pair.reshape(B * K, Lc, W)
        valid = valid.reshape(B * K, Lc, W)
        sal = cand_sal.masked_fill(~cand_mask, 0.0).reshape(B * K, Lc)
        cmask = cand_mask.reshape(B * K, Lc)

        sal_sum = sal.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weight = (sal / sal_sum).unsqueeze(-1)
        emit = pair.masked_fill(~valid, 0.0) * weight
        gap = -F.softplus(self._drift_penalty)
        t = max(self.temperature, 1e-4)
        H = pair.new_zeros(B * K, W)
        for i in range(Lc):
            stay = H
            left = F.pad(H[:, :-1], (1, 0), value=_NEG) + gap
            right = F.pad(H[:, 1:], (0, 1), value=_NEG) + gap
            trans = t * torch.logsumexp(torch.stack((stay, left, right), dim=0) / t, dim=0)
            trans = trans - t * math.log(3.0)
            nxt = trans + emit[:, i]
            H = torch.where(cmask[:, i].unsqueeze(-1), nxt, H)

        score = t * torch.logsumexp(H / t, dim=-1) - t * math.log(float(W))
        covered = (valid & cmask.unsqueeze(-1)).any(dim=-1).to(pair.dtype)
        coverage = ((covered * sal).sum(dim=-1) / sal_sum.squeeze(-1)).clamp_min(1e-4)
        score = score + self.coverage_weight * coverage.log()
        return score.reshape(B, K)
