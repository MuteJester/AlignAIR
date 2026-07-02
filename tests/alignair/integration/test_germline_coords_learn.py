import numpy as np
import torch
from alignair.data.tokenizer import pad_tokenize
from alignair.nn.encoder.shared import SharedNucleotideEncoder
from alignair.nn.aligner.banded_dp import SeedExtendAligner
from alignair.nn.aligner.germline_aligner import decode_germline_coords

BASES = "ACGT"


def _rand(rng, n):
    return "".join(BASES[i] for i in rng.integers(0, 4, size=n))


def _mutate(rng, s, p=0.05):
    return "".join(BASES[rng.integers(0, 4)] if rng.random() < p else c for c in s)


def _make_batch(rng, B, glen=80):
    germs, starts, ends, obs = [], [], [], []
    for _ in range(B):
        g = _rand(rng, glen)
        gs = int(rng.integers(0, 15))
        ge = glen - int(rng.integers(0, 15))   # ge in [65, 80], always > gs
        germs.append(g)
        starts.append(gs)
        ends.append(ge)
        obs.append(_mutate(rng, g[gs:ge], 0.05))
    gt, gm = pad_tokenize(germs)
    ot, om = pad_tokenize(obs)
    return gt, gm, ot, om, torch.tensor(starts), torch.tensor(ends)


def test_seed_extend_aligner_learns_exact_trims():
    # keeper path: one shared encoder + the seed-and-extend banded DP (full band here, so the
    # exact soft-DP with the base-match channel). The encoder + aligner learn exact germline trims.
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    enc = SharedNucleotideEncoder(d_model=64, n_layers=1, nhead=4, max_len=256)
    aligner = SeedExtendAligner(d_model=64)
    opt = torch.optim.Adam(list(enc.parameters()) + list(aligner.parameters()), lr=2e-3)
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(250):
        gt, gm, ot, om, gs, ge = _make_batch(rng, 16)
        G = enc.forward_positions(gt, gm, token_type=SharedNucleotideEncoder.GERMLINE)
        S = enc.forward_positions(ot, om, token_type=SharedNucleotideEncoder.READ)
        center = torch.zeros(gt.shape[0], dtype=torch.long)
        w = gt.shape[1]                                    # full band == full soft-DP
        sl, el = aligner(S, om, G, gm, center, w, seg_tok=ot, germ_tok=gt)
        loss = ce(sl, gs) + ce(el, ge - 1)                 # end target is last aligned position
        opt.zero_grad()
        loss.backward()
        opt.step()

    enc.eval()
    aligner.eval()
    with torch.no_grad():
        gt, gm, ot, om, gs, ge = _make_batch(rng, 64)
        G = enc.forward_positions(gt, gm, token_type=SharedNucleotideEncoder.GERMLINE)
        S = enc.forward_positions(ot, om, token_type=SharedNucleotideEncoder.READ)
        center = torch.zeros(gt.shape[0], dtype=torch.long)
        sl, el = aligner(S, om, G, gm, center, gt.shape[1], seg_tok=ot, germ_tok=gt)
        pgs, pge = decode_germline_coords(sl, el)
    start_dev = (pgs - gs).abs().float().mean().item()
    end_dev = (pge - ge).abs().float().mean().item()
    assert start_dev <= 1.0, f"germline start deviation too high: {start_dev}"
    assert end_dev <= 1.0, f"germline end deviation too high: {end_dev}"
