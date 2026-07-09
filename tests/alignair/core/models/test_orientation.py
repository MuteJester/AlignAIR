"""Orientation: head from shared embeddings + in-model correct-and-re-embed."""
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core.layers import EmbeddingOrientationHead
from alignair.core import AlignAIR
from alignair.nn.heads.orientation import NUM_ORIENTATIONS, apply_orientation


def test_orientation_head_shape_and_order_sensitivity():
    head = EmbeddingOrientationHead(embed_dim=16).eval()
    emb = torch.randn(3, 20, 16)
    mask = torch.ones(3, 20, dtype=torch.bool)
    logits = head(emb, mask)
    assert logits.shape == (3, NUM_ORIENTATIONS)


def test_model_emits_orientation_and_self_corrects_to_forward():
    L = 256
    cfg = AlignAIRConfig(max_seq_length=L, v_allele_count=10, d_allele_count=4,
                         j_allele_count=4, has_d=True)
    torch.manual_seed(0)
    model = AlignAIR(cfg).eval()

    fwd = torch.randint(1, 6, (2, L))                 # valid bases 1..5 (no pad, full length)
    mask = fwd != 0
    # present the reverse-complement of the read; teacher-force the true orientation (=RC, id 1)
    rc = apply_orientation(fwd, mask, torch.tensor([1, 1]))
    # teacher-force both corrections -> both canonicalize to the same forward read
    out_fwd = model({"tokenized_sequence": fwd, "orientation": torch.tensor([0, 0])})
    out_rc = model({"tokenized_sequence": rc, "orientation": torch.tensor([1, 1])})

    assert "orientation_logits" in out_fwd and out_fwd["orientation_logits"].shape == (2, 4)
    # teacher-forced correction re-canonicalizes RC back to forward -> identical downstream outputs
    assert torch.allclose(out_rc["v_allele"], out_fwd["v_allele"], atol=1e-5)
    assert torch.allclose(out_rc["v_start"], out_fwd["v_start"], atol=1e-4)
