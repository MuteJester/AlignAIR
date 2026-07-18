"""The unified AlignAIR model: GeneSpec-driven assembly, GeneBranch/MetaHead blocks, and the
dataconfig-driven single-vs-multi behavior (1 dataconfig -> single, N -> multi)."""
import torch

import GenAIRR.data as gd

from alignair.core.config import AlignAIRConfig, GeneSpec
from alignair.core.model import AlignAIR
from alignair.core.gene_branch import GeneBranch, MetaHead


def test_gene_specs_are_canonical_and_data_driven():
    heavy = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4,
                           d_allele_count=4, has_d=True)
    assert [s.name for s in heavy.gene_specs] == ["v", "d", "j"]      # canonical order, D present
    d_spec = next(s for s in heavy.gene_specs if s.name == "d")
    assert d_spec.short_d_penalty and d_spec.cls_kernels == (3, 3, 2, 2, 5)
    light = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4, has_d=False)
    assert [s.name for s in light.gene_specs] == ["v", "j"]           # no D


def test_gene_branch_segment_and_classify_shapes():
    cfg = AlignAIRConfig(max_seq_length=256, v_allele_count=8, j_allele_count=4, has_d=False)
    spec = GeneSpec("v", allele_count=8)
    branch = GeneBranch(spec, cfg).eval()
    emb = torch.randn(3, cfg.max_seq_length, cfg.embed_dim)
    s_log, e_log, s_exp, e_exp = branch.segment(emb)
    assert s_log.shape == (3, cfg.max_seq_length) and s_exp.shape == (3, 1)
    allele = branch.classify(emb, s_exp, e_exp)
    assert allele.shape == (3, 8) and (allele >= 0).all() and (allele <= 1).all()


def test_meta_head_optional_mid_and_clamp():
    with_mid = MetaHead(16, 1, mid_dim=8, out_act=torch.relu, clamp=(0.0, 1.0))
    assert with_mid.mid is not None
    with_mid.head.weight.data.fill_(5.0); with_mid.clamp_()
    assert with_mid.head.weight.max().item() <= 1.0
    no_mid = MetaHead(16, 1, out_act=torch.sigmoid)                   # productivity-style
    assert no_mid.mid is None
    assert no_mid(torch.randn(2, 16)).shape == (2, 1)


def test_from_dataconfigs_single_is_single_chain():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=256)
    assert cfg.has_d and cfg.num_chain_types == 1                     # one chain type -> single
    assert cfg.v_allele_count == 198 and cfg.j_allele_count == 7 and cfg.d_allele_count == 33
    out = AlignAIR(cfg).eval()({"tokenized_sequence": torch.randint(1, 6, (2, 256))})
    assert "chain_type_logits" not in out                            # single-chain: no locus head
    assert out["v_allele"].shape == (2, 198) and "d_allele" in out


def test_from_dataconfigs_multiple_is_multi_chain():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGK_OGRDB, max_seq_length=256)
    assert cfg.num_chain_types == 2                                   # two chain types -> multi
    assert cfg.has_d                                                  # union: IGH has D
    assert cfg.v_allele_count == len(
        set(a.name for a in gd.HUMAN_IGH_OGRDB.allele_list("v"))
        | set(a.name for a in gd.HUMAN_IGK_OGRDB.allele_list("v")))   # union of V alleles
    out = AlignAIR(cfg).eval()({"tokenized_sequence": torch.randint(1, 6, (2, 256))})
    assert out["chain_type_logits"].shape == (2, 2)                  # multi-chain: locus head present
