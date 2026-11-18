"""The chip's V re-score: the read's V span aligned with gaps to each of its top V rows, in a band around the diagonal
the conserved Cys fixes, so an indel does not throw the allele call."""
import math

import pytest
import torch

from alignair.core.tokens import TOKEN_DICT

from alignair.network.refine.v_rescore import GAP_COST, banded_scores

pytestmark = pytest.mark.component

V1 = "CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTC"
V2 = V1[:20] + "A" + V1[21:40] + "T" + V1[41:]          # two substitutions from V1
ANCHOR = 70                                              # the row base the (toy) conserved codon starts at
M = 0.05
MATCH, MISMATCH = math.log(4 * (1 - M)), math.log(4 * M / 3)


def _tokens(sequence):
    return torch.tensor([[TOKEN_DICT[base] for base in sequence]])


def _score(read, start, read_anchor):
    rows = torch.cat([_tokens(V1), _tokens(V2)])[None]                               # (1, K=2, row length)
    return banded_scores(_tokens(read), torch.tensor([start]), torch.tensor([len(read)]),
                         torch.tensor([read_anchor]), rows, torch.tensor([[ANCHOR, ANCHOR]]), torch.tensor([[M]]))


def test_a_clean_read_scores_every_base_as_a_match_of_its_allele():
    read = V1[10:]
    scores = _score(read, 0, ANCHOR - 10)
    assert scores[0, 0].item() == pytest.approx(len(read) * MATCH, rel=1e-5)
    assert scores[0, 0] > scores[0, 1]


def test_a_deletion_costs_one_gap_and_keeps_the_allele():
    read = V1[10:35] + V1[36:]                            # germline base 35 deleted, upstream of the anchor
    scores = _score(read, 0, ANCHOR - 11)
    assert scores[0, 0].item() == pytest.approx(len(read) * MATCH - GAP_COST, rel=1e-4)
    assert scores[0, 0] > scores[0, 1]


def test_an_insertion_costs_one_gap_and_keeps_the_allele():
    read = V1[10:35] + "G" + V1[35:]                      # one base inserted upstream of the anchor
    scores = _score(read, 0, ANCHOR - 9)
    assert scores[0, 0].item() == pytest.approx((len(read) - 1) * MATCH - GAP_COST, rel=1e-4)
    assert scores[0, 0] > scores[0, 1]


def test_bases_outside_the_span_do_not_count():
    read = "TTTTT" + V1[10:]
    assert _score(read, 5, ANCHOR - 5)[0, 0].item() == pytest.approx((len(read) - 5) * MATCH, rel=1e-5)


def _reference_band_dp(pair, active):
    """The band's recursion as written first: ``best`` (B, K, W) over the band's diagonals, base by base."""
    from torch.nn import functional as F
    width = pair.shape[-1]
    ramp = GAP_COST * torch.arange(width, dtype=pair.dtype)
    best = torch.zeros(*pair.shape[:2], width, dtype=pair.dtype)
    for t in range(pair.shape[2]):
        deleted = torch.cummax(best + ramp, -1).values - ramp
        inserted = F.pad(best[..., 1:], (0, 1), value=-1e9) - GAP_COST
        best = torch.where(active[:, :, t], torch.maximum(deleted + pair[:, :, t], inserted), best)
    return best.amax(-1)


def test_the_band_recursion_is_the_base_by_base_definition():
    """Pair scores from matches to the outside penalty and bases outside the span (inactive steps) included: the
    recursion's scores are the definition's."""
    from alignair.network.refine.ratios import OUTSIDE
    from alignair.network.refine.v_rescore import band_dp
    torch.manual_seed(0)
    pair = torch.randn(3, 2, 40, 17) * 2
    pair[0, :, :5] = OUTSIDE
    active = torch.rand(3, 1, 40, 1) < 0.8
    assert torch.allclose(band_dp(pair, active), _reference_band_dp(pair, active), atol=1e-4)


def _random_band_inputs(seed, batch=24, candidates=5, length=120, row_length=110):
    """Inputs shaped like the chip's call: right-padded reads with N bases, zero-padded rows of varying lengths, spans
    (some empty, as an unusable read's 0..0), anchors that put the band past either end of a row, and rates within
    ``RATE``."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, 5, (batch, length), generator=g)
    tokens[torch.rand(batch, length, generator=g) < 0.03] = 5
    for b, n in enumerate(torch.randint(length // 2, length + 1, (batch,), generator=g).tolist()):
        tokens[b, n:] = 0
    rows = torch.randint(1, 5, (batch, candidates, row_length), generator=g)
    for b in range(batch):
        for k, n in enumerate(torch.randint(row_length // 2, row_length + 1, (candidates,), generator=g).tolist()):
            rows[b, k, n:] = 0
    starts = torch.randint(0, length // 3, (batch,), generator=g)
    ends = torch.minimum(starts + torch.randint(0, length, (batch,), generator=g), torch.tensor(length))
    starts[:3], ends[:3] = 0, 0
    anchors = torch.randint(0, length, (batch,), generator=g)
    row_anchors = torch.randint(-1, row_length, (batch, candidates), generator=g)
    m = (0.01 + 0.24 * torch.rand(batch, 1, generator=g)).float()
    return tokens, starts, ends, anchors, rows, row_anchors, m


@pytest.mark.parametrize("seed", range(6))
def test_the_compiled_band_gives_the_torch_band_scores_bit_for_bit(seed):
    pytest.importorskip("alignair.network.refine._vband_cy")
    from alignair.network.refine import v_rescore as vrescore
    inputs = _random_band_inputs(seed)
    assert vrescore.VBAND_BACKEND == "cython"
    assert torch.equal(vrescore.banded_scores(*inputs), vrescore._torch_banded_scores(*inputs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_compiled_band_takes_and_returns_the_reads_device_with_the_torch_band_scores():
    pytest.importorskip("alignair.network.refine._vband_cy")
    from alignair.network.refine import v_rescore as vrescore
    inputs = [x.cuda() for x in _random_band_inputs(0)]
    scores = vrescore.banded_scores(*inputs)
    assert scores.device == inputs[0].device
    assert torch.equal(scores, vrescore._torch_banded_scores(*inputs))


def test_the_v_rescore_keeps_a_small_locus_calls_in_it():
    """A read's locus holds 3 V alleles (``refine.locus`` put the others at ``OUT``), so the re-score's top 5 takes 2
    alleles of another locus; those match the read exactly, the locus's own alleles carry two substitutions. The call
    stays in the locus and the other locus's alleles keep a score of 0."""
    from types import SimpleNamespace

    from alignair.core.anchors import ANCHOR_LOGITS
    from alignair.network.scan import OUT
    from alignair.network.refine.v_rescore import rescore_v
    read = V1[10:]
    length = len(read) + 1
    rows = torch.cat([_tokens(V2)] * 3 + [_tokens(V1)] * 3)                      # 3 in the locus, 3 outside
    bank = SimpleNamespace(anchor_base_v=torch.full((6,), ANCHOR), sequence_v=rows)

    def peak(at):
        logits = torch.full((1, length), -10.)
        logits[0, at] = 10.
        return logits

    log_posterior = torch.log_softmax(torch.tensor([[0., -0.5, -1., OUT, OUT, OUT]]), -1)
    out = {"v_log_posterior": log_posterior, ANCHOR_LOGITS["v"]: peak(ANCHOR - 10), "v_start_logits": peak(0),
           "v_end_logits": peak(len(read)), "mutation_rate": torch.tensor([[M]])}
    tokens = torch.cat([_tokens(read), torch.zeros(1, 1, dtype=torch.long)], 1)
    rescore_v(out, tokens, bank)
    assert not torch.equal(out["v_log_posterior"], log_posterior)                # the re-score ran on this read
    assert int(out["v_log_posterior"][0].argmax()) < 3 and (out["v_allele"][0, 3:] == 0).all()
