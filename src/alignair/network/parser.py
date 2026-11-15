"""The parser: one ordered path of recombination regions per read.

A read is an ordered concatenation of regions, ``STATES`` = (pre, V, N1, D, N2, J, post); any region may be empty, so
a read without a D, a light chain or a fragment is the same path with regions skipped. The parser is a linear-chain
CRF over those states whose transitions only move forward (b >= a): per base, an emission score for each state (a 1x1
convolution of the trunk's features), and per base a score for each transition from the previous base's state (a 1x1
convolution to S x S, so a boundary's evidence sits where it is). It cannot put V after J or two Vs in one read.

Inference is exact. With ``M_i(a, b) = transitions_i(a, b) + emissions_i(b)`` (pads: the log-semiring identity, so
padding changes nothing), the prefix products ``M_1 ⊗ ... ⊗ M_i`` in the log semiring come from a Hillis-Steele scan
(log2 L batched steps rather than L sequential ones); ``α_i = emissions_0 ⊗ M_1 ⊗ ... ⊗ M_i``, ``β`` from the same scan
over the transposed, reversed sequence, ``log Z = logsumexp α_{n-1}``, the per-base marginals ``α + β - log Z`` and the
pairwise ``α_{i-1}(a) + M_i(a, b) + β_i(b) - log Z`` that place each boundary. Everything is differentiable, so a
unit that reads the marginals (the router, the anchors) trains the parser through them.

The scan multiplies full S x S matrices at every one of its log2 L levels: few, wide steps suit a GPU. A CPU runs the
L-step forward-backward instead (``α_i = α_{i-1} ⊗ M_i``, ``β_{i-1} = M_i ⊗ β_i``: vectors, not matrices; the two
recursions as one loop over a doubled batch, since a step's cost on a CPU is mostly its fixed overhead), which computes
the same parse; ``parallel`` chooses explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..core.regions import REGIONS


STATES = REGIONS
GENES = {"v": STATES.index("V"), "d": STATES.index("D"), "j": STATES.index("J")}
IMPOSSIBLE = -1e4          # the boundary logit of a slot no path uses (finite, so a softmax over the slots is defined)
NEG = -1e9                 # a forbidden score: finite, so a logsumexp over forbidden terms only keeps finite gradients


def expected_slot(logits: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The expected slot (B, 1) of the boundary distribution ``logits`` (B, L) over the allowed ``slots``."""
    probs = logits.masked_fill(~slots, IMPOSSIBLE).softmax(-1)
    return (probs * torch.arange(logits.shape[-1], device=logits.device)).sum(-1, keepdim=True)


@dataclass
class ParserOutput:
    emissions: torch.Tensor           # (B, L, S)
    transitions: torch.Tensor         # (B, L, S, S); transitions[:, i] enter base i from base i - 1
    mask: torch.Tensor                # (B, L) bool
    log_z: torch.Tensor               # (B,)
    marginals: torch.Tensor           # (B, L, S): P(base i is in state s); 0 on pads
    boundary_logits: dict             # {g}_start_logits / {g}_end_logits, (B, L) over the half-open slots


def _log_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(a.unsqueeze(-1) + b.unsqueeze(-3), dim=-2)


def _prefix_products(m: torch.Tensor) -> torch.Tensor:
    """(B, N, S, S) -> the inclusive prefix products ``m_0 ⊗ ... ⊗ m_k`` in the log semiring (Hillis-Steele)."""
    step = 1
    while step < m.shape[1]:
        m = torch.cat([m[:, :step], _log_matmul(m[:, :-step], m[:, step:])], 1)
        step *= 2
    return m


def _steps(emissions, transitions, mask):
    """``M_i`` for every base (index 0 unused), with forbidden transitions at -inf and pads as the identity."""
    states = emissions.shape[-1]
    forward = torch.ones(states, states, dtype=torch.bool, device=emissions.device).triu()
    m = (transitions + emissions.unsqueeze(-2)).masked_fill(~forward, NEG)
    identity = torch.full((states, states), NEG, device=emissions.device).fill_diagonal_(0.)
    return torch.where(mask[:, :, None, None], m, identity)


def _parallel(m, parallel):
    return m.is_cuda if parallel is None else parallel


def _alpha_sequential(emissions, m):
    alpha = [emissions[:, 0]]
    for i in range(1, m.shape[1]):
        alpha.append(torch.logsumexp(alpha[-1].unsqueeze(-1) + m[:, i], dim=-2))
    return torch.stack(alpha, 1)


def _forward_backward_sequential(emissions, m):
    """``(α, β)`` in one L-step loop: β's recursion ``β_{i-1}(a) = ⊕_b M_i(a, b) ⊗ β_i(b)`` is α's on the transposed
    steps taken from the end, so the two run as one batch of 2B vectors (half the steps, each as cheap)."""
    batch = m.shape[0]
    steps = torch.cat([m[:, 1:], m[:, 1:].flip(1).transpose(-1, -2)])                     # (2B, L - 1, S, S)
    state = torch.cat([emissions[:, 0], m.new_zeros(batch, m.shape[-1])])
    both = [state]
    for step in steps.unbind(1):
        state = torch.logsumexp(state.unsqueeze(-1) + step, dim=-2)
        both.append(state)
    both = torch.stack(both, 1)
    return both[:batch], both[batch:].flip(1)


def _alpha(emissions, m, parallel=None):
    if not _parallel(m, parallel):
        return _alpha_sequential(emissions, m)
    alpha0 = emissions[:, 0]
    if m.shape[1] == 1:
        return alpha0.unsqueeze(1)
    prefix = _prefix_products(m[:, 1:])
    rest = torch.logsumexp(alpha0[:, None, :, None] + prefix, dim=2)
    return torch.cat([alpha0.unsqueeze(1), rest], 1)


def _beta(m):
    batch, length, states, _ = m.shape
    last = m.new_zeros(batch, 1, states)
    if length == 1:
        return last
    suffix = _prefix_products(m[:, 1:].flip(1).transpose(-1, -2))
    return torch.cat([torch.logsumexp(suffix, dim=-2).flip(1), last], 1)


def log_partition(emissions, transitions, mask, parallel: bool | None = None) -> torch.Tensor:
    """``log Z`` (B,) of the paths through the read's bases (``mask``)."""
    return torch.logsumexp(_alpha(emissions, _steps(emissions, transitions, mask), parallel)[:, -1], -1)


def crf(emissions: torch.Tensor, transitions: torch.Tensor, mask: torch.Tensor,
        parallel: bool | None = None) -> ParserOutput:
    """The parse; ``parallel``: the log-depth scan (True) or the L-step forward-backward (False); by default the
    scan on a GPU and the forward-backward elsewhere."""
    m = _steps(emissions, transitions, mask)
    if _parallel(m, parallel):
        alpha, beta = _alpha(emissions, m, parallel=True), _beta(m)
    else:
        alpha, beta = _forward_backward_sequential(emissions, m)
    log_z = torch.logsumexp(alpha[:, -1], -1)
    log_marginals = alpha + beta - log_z[:, None, None]
    marginals = log_marginals.exp() * mask.unsqueeze(-1)
    pairs = alpha[:, :-1, :, None] + m[:, 1:] + beta[:, 1:, None, :] - log_z[:, None, None, None]   # (B, L-1, S, S)
    lengths = mask.sum(1)
    last = log_marginals.gather(1, (lengths - 1).clamp_min(0)[:, None, None].expand(-1, 1, emissions.shape[-1]))[:, 0]
    boundary_logits = {}
    batch, length = mask.shape
    for gene, state in GENES.items():
        start = torch.full((batch, length), NEG, device=emissions.device)
        start[:, 0] = log_marginals[:, 0, state]
        start[:, 1:] = torch.logsumexp(pairs[:, :, :state, state], -1)
        end = torch.full((batch, length), NEG, device=emissions.device)
        end[:, 1:] = torch.logsumexp(pairs[:, :, state, state + 1:], -1)
        at_end = lengths.clamp_max(length - 1)                  # a read reaching the window's last slot ends there
        end = end.scatter(1, at_end[:, None], torch.logaddexp(end.gather(1, at_end[:, None]), last[:, state, None]))
        boundary_logits[f"{gene}_start_logits"] = start.clamp_min(IMPOSSIBLE)
        boundary_logits[f"{gene}_end_logits"] = end.clamp_min(IMPOSSIBLE)
    return ParserOutput(emissions, transitions, mask, log_z, marginals, boundary_logits)


def crf_nll(output: ParserOutput, allowed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,) ``log Z - log Z_allowed``: minus the log probability of the paths that visit only allowed states
    (``allowed`` (B, L, S) bool), i.e. of the regions the read cannot tell apart from the true ones."""
    restricted = output.emissions.masked_fill(mask.unsqueeze(-1) & ~allowed, NEG)
    return output.log_z - log_partition(restricted, output.transitions, mask)


class Parser(nn.Module):
    """``(features (B, L, C), mask (B, L)) -> ParserOutput``."""

    def __init__(self, channels: int):
        super().__init__()
        self.emit = nn.Conv1d(channels, len(STATES), 1)
        self.trans = nn.Conv1d(channels, len(STATES) ** 2, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> ParserOutput:
        batch, length, _ = features.shape
        x = features.transpose(1, 2)
        emissions = self.emit(x).transpose(1, 2)
        transitions = self.trans(x).transpose(1, 2).reshape(batch, length, len(STATES), len(STATES))
        return crf(emissions, transitions, mask)
