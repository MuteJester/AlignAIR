"""End-to-end assembly: encoder -> GLIP fusion -> DETR queries -> YOLOX/CLIP heads."""
import torch

from alignair.nn.sota.detector import OpenVocabVDJDetector
from alignair.nn.sota.query_decoder import GENES


def _toy_batch(B=3, L=40, kv=8, kd=5, kj=4, Sc=30):
    torch.manual_seed(0)
    read = torch.randint(1, 5, (B, L))
    read_mask = torch.ones(B, L, dtype=torch.bool)
    read_mask[:, 30:] = False
    cands = {}
    for g, k in zip(GENES, (kv, kd, kj)):
        tok = torch.randint(1, 5, (k, Sc))
        m = torch.ones(k, Sc, dtype=torch.bool)
        m[:, 20:] = False
        cands[g] = {"tokens": tok, "mask": m}
    return read, read_mask, cands


def _detector():
    return OpenVocabVDJDetector(d_model=32, nhead=4, encoder_layers=2,
                                fusion_layers=1, decoder_layers=2)


def test_detector_output_shapes():
    read, read_mask, cands = _toy_batch()
    out = _detector()(read, read_mask, cands)
    assert set(out) == set(GENES)
    B = read.shape[0]
    for g in GENES:
        assert out[g]["span"].shape == (B, 2)
        assert out[g]["objectness"].shape == (B,)
        assert out[g]["trim"].shape == (B, 2)
        assert out[g]["allele_scores"].shape == (B, cands[g]["tokens"].shape[0])
        assert (out[g]["span"] >= 0).all() and (out[g]["span"] <= 1).all()
        assert torch.isfinite(out[g]["allele_scores"]).all()


def test_dynamic_genotype_mask_forbids_disabled_alleles():
    """A candidate_mask (the caller's reference restriction) drives disabled alleles to -inf,
    so they can never be called — the dynamic-genotype mechanism."""
    read, read_mask, cands = _toy_batch()
    cm = torch.ones(cands["V"]["tokens"].shape[0], dtype=torch.bool)
    cm[2] = False                                   # allele 2 not in this caller's reference
    cands["V"]["candidate_mask"] = cm
    out = _detector()(read, read_mask, cands)
    assert torch.isinf(out["V"]["allele_scores"][:, 2]).all()
    assert torch.isfinite(out["V"]["allele_scores"][:, [0, 1, 3]]).all()


def test_topk_retrieval_scatters_scores_back_to_full_reference():
    """With top_k set and a large reference, only the shortlist is discriminated; scores are
    scattered back to full (B, Kg) with non-retrieved alleles at -inf. The forced-in positive
    must be finite so the contrastive loss can see it."""
    torch.manual_seed(0)
    B, L = 3, 40
    read = torch.randint(1, 5, (B, L))
    read_mask = torch.ones(B, L, dtype=torch.bool)
    cands = {}
    for g, k in zip(GENES, (60, 40, 30)):                 # large references
        cands[g] = {"tokens": torch.randint(1, 5, (k, 30)),
                    "mask": torch.ones(k, 30, dtype=torch.bool)}
    cands["V"]["force_include"] = torch.tensor([55, 3, 40])  # positives unlikely in cosine top-8
    out = _detector()(read, read_mask, cands, top_k=8)
    for g in GENES:
        Kg = cands[g]["tokens"].shape[0]
        assert out[g]["allele_scores"].shape == (B, Kg)
        finite = torch.isfinite(out[g]["allele_scores"]).sum(dim=-1)
        assert (finite <= 8).all()                        # at most top_k alleles scored
    for b in range(B):                                    # forced positive is always scored
        assert torch.isfinite(out["V"]["allele_scores"][b, cands["V"]["force_include"][b]])


def test_detector_gradients_flow_everywhere():
    read, read_mask, cands = _toy_batch()
    det = _detector()
    out = det(read, read_mask, cands)
    loss = sum(out[g]["span"].sum() + out[g]["objectness"].sum()
               + out[g]["trim"].sum() + out[g]["allele_scores"].clamp(-10, 10).sum()
               for g in GENES)
    loss.backward()
    # every trainable param gets a gradient EXCEPT the encoder's pooling projection, which the
    # detector never uses (it calls forward_positions, not the pooled forward).
    missing = [n for n, p in det.named_parameters()
               if p.requires_grad and p.grad is None]
    assert missing == ["encoder.proj.weight", "encoder.proj.bias"], missing
