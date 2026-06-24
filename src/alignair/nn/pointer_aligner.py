"""BandedPointerAligner: a fully-parallel, reference-conditioned germline pointer
head that replaces the sequential soft-DP. Start/end logits are weighted leading /
reverse diagonals of the score matrix M, extracted in single CUDA launches via
as_strided (no S-length recurrence). See the design spec §4."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_match import base_match_channel

NEG = -1e4


def weighted_leading_diag(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[b,o] = (Σ_i w[b,i]·M[b,i,o+i]) / Σ_i w[b,i], for o+i < Lg. (B,Lg)."""
    B, S, Lg = M.shape
    Mp = F.pad(M, (0, S))                                  # (B,S,Lg+S)
    bs, ss, es = Mp.stride()
    diag = Mp.as_strided((B, S, Lg), (bs, ss + es, es))    # diag[b,i,o] = Mp[b,i,o+i]
    num = (w * diag).sum(dim=1)                            # (B,Lg)
    return num / w.sum(dim=1).clamp(min=1e-6)


def weighted_reverse_diag(M: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[b,o] = (Σ_i w[b,S-1-i]·M[b,S-1-i,o-i]) / Σ_i w[b,i], for 0<=o-i<Lg. (B,Lg).
    CRITICAL: w is flipped into the flipped-row frame (wf = flip(w)), else w[i] pairs with
    row S-1-i and the head trains to the wrong coordinate once w is non-uniform (spec §4.2 B1)."""
    B, S, Lg = M.shape
    Mf = torch.flip(M, (1, 2))                             # reverse read rows AND germ cols
    Mfp = F.pad(Mf, (0, S))
    bs, ss, es = Mfp.stride()
    led = Mfp.as_strided((B, S, Lg), (bs, ss + es, es))    # led[b,i,o] = Mf[b,i,o+i]
    wf = torch.flip(w, (1,))                               # weights into led's frame
    num = torch.flip((wf * led).sum(dim=1), (1,))          # (B,Lg)
    return num / w.sum(dim=1).clamp(min=1e-6)
