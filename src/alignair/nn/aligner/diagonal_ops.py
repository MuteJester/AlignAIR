"""Parallel diagonal score-extraction ops shared by the pointer aligner and the band head.

A germline start offset `o` places read position `i` against germline column `o+i`; the
score of that diagonal is a weighted sum along it. These extract all offsets' diagonal
scores in a single CUDA launch via as_strided (no S-length Python loop).
"""
import torch
import torch.nn.functional as F

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


def banded_start_end(M, w, gamma, G):
    """Combine leading/reverse diagonals over germline offsets Δ∈[-G,G] by
    logsumexp(diag_Δ + γ_Δ), so the coordinate tolerates an indel net-length shift.
    The Δ-shifted score matrix shifts germline columns with torch.roll, then NEG-masks
    the wrapped region. END uses the flip-w reverse diagonal per Δ (spec §4.4 B1)."""
    starts, ends = [], []
    for k, delta in enumerate(range(-G, G + 1)):
        Md = torch.roll(M, shifts=-delta, dims=2).clone()     # column shift by Δ
        if delta > 0:
            Md[:, :, -delta:] = NEG
        elif delta < 0:
            Md[:, :, :(-delta)] = NEG
        starts.append(weighted_leading_diag(Md, w) + gamma[k])
        ends.append(weighted_reverse_diag(Md, w) + gamma[k])
    start = torch.logsumexp(torch.stack(starts, 0), dim=0)
    end = torch.logsumexp(torch.stack(ends, 0), dim=0)
    return start, end
