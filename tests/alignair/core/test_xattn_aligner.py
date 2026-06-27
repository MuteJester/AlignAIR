import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.reference.reference_set import ReferenceSet
from alignair.core.xattn_aligner import XAttnAligner
from alignair.data.tokenizer import pad_tokenize


def _model_and_ref():
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=4, dim_feedforward=64)
    model = XAttnAligner(cfg).eval()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGK_OGRDB)   # V/J only -> small
    ref_emb = model.encode_reference(rs)
    return model, rs, ref_emb


def test_forward_produces_four_heads_with_shapes():
    model, rs, ref_emb = _model_and_ref()
    reads = ["ACGTACGT" * 12, "TTGCAACGTACG" * 8]
    tok, msk = pad_tokenize(reads)
    with torch.no_grad():
        out = model(tok, msk, ref_emb, topk=8)
    B = len(reads)
    assert out["orientation_logits"].shape[0] == B
    assert out["region_logits"].shape[0] == B
    for g in ("V", "J"):
        m = out["match"][g]
        assert m["allele_logits"].shape[0] == B and m["allele_logits"].shape[1] == 8
        assert m["best_global_idx"].shape == (B,)
        assert m["germ_start"].shape == (B,) and m["germ_end"].shape == (B,)
        assert int(m["best_global_idx"].max()) < len(rs.gene(g).names)


def test_genotype_mask_restricts_candidate_pool():
    model, rs, ref_emb = _model_and_ref()
    reads = ["ACGTACGT" * 12]
    tok, msk = pad_tokenize(reads)
    K = len(rs.gene("V").names)
    allowed = torch.zeros(K, dtype=torch.bool); allowed[:3] = True       # only first 3 V alleles
    with torch.no_grad():
        out = model(tok, msk, ref_emb, candidate_masks={"V": allowed}, topk=8)
    assert int(out["match"]["V"]["best_global_idx"][0]) in (0, 1, 2)     # call stays in genotype


def test_seed_admission_widens_pool_and_mechanism_admits_true_allele():
    # The seed path must (1) be wired into forward — pool widens by seed_m — and (2) admit the true
    # allele from a read's own k-mers. (2) is asserted on the full read directly because the tiny
    # UNTRAINED segmentation produces an empty V segment, so an e2e assertion would test
    # segmentation, not the seed mechanism. SeedPrefilter effectiveness is unit-tested separately.
    model, rs, ref_emb = _model_and_ref()
    vg = rs.gene("V")
    true_idx = 5
    read = vg.sequences[true_idx]
    tok, msk = pad_tokenize([read])
    with torch.no_grad():
        base = model(tok, msk, ref_emb, topk=4)                              # retrieval only
        seeded = model(tok, msk, ref_emb, topk=4, seed_m=8, reference_set=rs)  # + seed union
    assert base["match"]["V"]["pool_idx"].shape[1] == 4
    assert seeded["match"]["V"]["pool_idx"].shape[1] == 12                   # seed slots wired in
    assert true_idx in model._seed.candidates(read, "V", 8)                  # mechanism admits truth
