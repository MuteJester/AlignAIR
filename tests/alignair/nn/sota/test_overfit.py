"""End-to-end sanity: the assembled detector + loss can overfit a fixed batch.

If the architecture is wired correctly, gradient descent on a single fixed (read, candidates,
targets) batch should drive the loss down and make the model call the right alleles and spans.
This is the cheapest signal that encoder->fusion->queries->heads->loss all connect and learn.
"""
import torch

from alignair.nn.sota.detector import OpenVocabVDJDetector
from alignair.nn.sota.loss import DetectorLoss
from alignair.nn.sota.query_decoder import GENES


def test_detector_overfits_a_fixed_batch():
    torch.manual_seed(0)
    B, L, Sc = 4, 40, 30
    read = torch.randint(1, 5, (B, L))
    read_mask = torch.ones(B, L, dtype=torch.bool)

    cands, targets = {}, {}
    ks = dict(V=8, D=5, J=4)
    for g in GENES:
        k = ks[g]
        cands[g] = {"tokens": torch.randint(1, 5, (k, Sc)),
                    "mask": torch.ones(k, Sc, dtype=torch.bool)}
        true_allele = torch.arange(B) % k                       # each read -> a distinct allele
        allele = torch.zeros(B, k)
        allele[torch.arange(B), true_allele] = 1.0
        targets[g] = {"span": torch.rand(B, 2).sort(dim=-1).values,
                      "present": torch.ones(B), "allele": allele,
                      "trim": torch.rand(B, 2), "_true": true_allele}

    model = OpenVocabVDJDetector(d_model=32, nhead=4, encoder_layers=2,
                                 fusion_layers=1, decoder_layers=2)
    crit = DetectorLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    model.train()
    first = None
    for step in range(250):
        opt.zero_grad()
        out = model(read, read_mask, cands)
        loss, _ = crit(out, targets)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
    final = float(loss.detach())

    assert final < 0.4 * first, f"loss did not drop enough: {first:.3f} -> {final:.3f}"

    model.eval()
    with torch.no_grad():
        out = model(read, read_mask, cands)
    for g in GENES:
        picked = out[g]["allele_scores"].argmax(dim=-1)
        assert (picked == targets[g]["_true"]).all(), f"{g}: {picked} vs {targets[g]['_true']}"
