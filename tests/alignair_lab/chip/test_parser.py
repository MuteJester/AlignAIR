"""The chip's parser: one ordered path of recombination regions per read, as a linear-chain CRF whose transitions only
move forward (pre, V, N1, D, N2, J, post), computed by a log-semiring scan."""
import itertools

import pytest
import torch

from alignair_lab.architectures.chip.parser import STATES, Parser, crf, crf_nll

pytestmark = pytest.mark.component

S = len(STATES)
V, N1, J = STATES.index("V"), STATES.index("N1"), STATES.index("J")


def _random(batch, length, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, length, S, generator=generator),
            torch.randn(batch, length, S, S, generator=generator))


def _brute_log_z(emissions, transitions, n):
    scores = []
    for path in itertools.product(range(S), repeat=n):
        if any(b < a for a, b in zip(path, path[1:])):
            continue
        score = emissions[0, path[0]]
        for i in range(1, n):
            score = score + transitions[i, path[i - 1], path[i]] + emissions[i, path[i]]
        scores.append(score)
    return torch.logsumexp(torch.stack(scores), 0)


def test_log_z_matches_brute_force_enumeration():
    emissions, transitions = _random(2, 5)
    out = crf(emissions, transitions, torch.ones(2, 5, dtype=torch.bool))
    for row in range(2):
        assert torch.allclose(out.log_z[row], _brute_log_z(emissions[row], transitions[row], 5), atol=1e-4)


def test_marginals_sum_to_one_on_read_positions_and_zero_on_pads():
    emissions, transitions = _random(2, 9)
    mask = torch.arange(9) < torch.tensor([[9], [6]])
    out = crf(emissions, transitions, mask)
    assert torch.allclose(out.marginals.sum(-1), mask.float(), atol=1e-5)


def test_padding_does_not_change_log_z_or_marginals():
    emissions, transitions = _random(1, 9)
    short = crf(emissions[:, :6], transitions[:, :6], torch.ones(1, 6, dtype=torch.bool))
    padded = crf(emissions, transitions, torch.arange(9).unsqueeze(0) < 6)
    assert torch.allclose(short.log_z, padded.log_z, atol=1e-5)
    assert torch.allclose(short.marginals, padded.marginals[:, :6], atol=1e-5)


def test_allowed_nll_is_zero_when_every_path_is_allowed_and_the_path_nll_when_one_is():
    emissions, transitions = _random(1, 5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    out = crf(emissions, transitions, mask)
    assert torch.allclose(crf_nll(out, torch.ones(1, 5, S, dtype=torch.bool), mask), torch.zeros(1), atol=1e-5)
    path = [0, V, V, N1, J]
    only = torch.zeros(1, 5, S, dtype=torch.bool)
    only[0, torch.arange(5), torch.tensor(path)] = True
    score = emissions[0, 0, path[0]] + sum(transitions[0, i, path[i - 1], path[i]] + emissions[0, i, path[i]]
                                            for i in range(1, 5))
    assert torch.allclose(crf_nll(out, only, mask), out.log_z - score, atol=1e-4)


def test_boundary_logits_peak_at_a_confident_path():
    emissions = torch.full((1, 12, S), -10.)
    for region, (start, end) in ((V, (0, 4)), (N1, (4, 6)), (J, (6, 9))):
        emissions[0, start:end, region] = 10.
    out = crf(emissions, torch.zeros(1, 12, S, S), torch.arange(12).unsqueeze(0) < 9)
    logits = out.boundary_logits
    assert logits["v_start_logits"].shape == (1, 12)
    assert logits["v_start_logits"].argmax().item() == 0 and logits["v_end_logits"].argmax().item() == 4
    assert logits["j_start_logits"].argmax().item() == 6 and logits["j_end_logits"].argmax().item() == 9
    assert torch.isfinite(logits["d_start_logits"]).all()


def test_gradients_flow_to_emissions_and_transitions():
    parser = Parser(channels=8)
    features = torch.randn(2, 10, 8)
    mask = torch.arange(10) < torch.tensor([[10], [7]])
    out = parser(features, mask)
    (out.log_z.sum() + out.marginals[..., V].sum()).backward()
    assert parser.emit.weight.grad is not None and parser.emit.weight.grad.abs().sum() > 0
    assert parser.trans.weight.grad is not None and parser.trans.weight.grad.abs().sum() > 0
