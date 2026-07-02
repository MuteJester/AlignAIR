"""Gym -> detector bridge: candidate bank + target reshaping, wired through model + loss."""
import torch

from alignair.nn.sota.data import CandidateBank, detector_inputs
from alignair.nn.sota.detector import OpenVocabVDJDetector
from alignair.nn.sota.loss import DetectorLoss
from alignair.nn.sota.query_decoder import GENES


def test_candidate_bank_shapes_match_reference(reference):
    bank = CandidateBank(reference)
    for G in GENES:
        k = len(reference.gene(G).names)
        assert bank.tokens[G].shape[0] == k and bank.sizes[G] == k
        assert bank.lengths[G].shape == (k,)
        assert bank.lengths[G].max() <= bank.tokens[G].shape[1]


def test_targets_are_normalized_and_shaped(reference, collated):
    bank = CandidateBank(reference)
    read_tokens, read_mask, cands, targets = detector_inputs(collated, bank)
    B = read_tokens.shape[0]
    for G in GENES:
        assert targets[G]["span"].shape == (B, 2)
        assert (targets[G]["span"] >= 0).all() and (targets[G]["span"] <= 1).all()
        assert targets[G]["trim"].shape == (B, 2)
        assert (targets[G]["trim"] >= 0).all() and (targets[G]["trim"] <= 1).all()
        assert targets[G]["allele"].shape == (B, bank.sizes[G])
        assert targets[G]["present"].shape == (B,)
        # span normalized by per-sample read length (16): start=2 -> 2/16 for row 0
        assert torch.allclose(targets[G]["span"][0, 0], torch.tensor(2 / 16), atol=1e-4)
        assert torch.equal(cands[G]["force_include"], collated[f"{G.lower()}_primary_idx"])


def test_bridge_runs_through_model_and_loss(reference, collated):
    bank = CandidateBank(reference)
    model = OpenVocabVDJDetector(d_model=32, nhead=4, encoder_layers=2,
                                 fusion_layers=1, decoder_layers=2)
    read_tokens, read_mask, cands, targets = detector_inputs(collated, bank)
    out = model(read_tokens, read_mask, cands, top_k=2)      # force retrieval on V (K=4 > 2)
    loss, logs = DetectorLoss()(out, targets)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
