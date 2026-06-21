import pytest
import torch
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.inference.dnalignair_infer import predict_reads


def test_predict_reads_returns_valid_predictions():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 8, "ACGTACGTNNACGT" * 6]
    preds = predict_reads(model, rs, reads, batch_size=2)
    assert len(preds) == len(reads)
    vnames, jnames, dnames = (set(rs.gene(g).names) for g in ("V", "J", "D"))
    for p in preds:
        assert p["v_call"] in vnames and p["j_call"] in jnames and p["d_call"] in dnames
        for g in ("v", "d", "j"):
            assert isinstance(p[f"{g}_sequence_start"], int)
            assert 0 <= p[f"{g}_germline_start"] and p[f"{g}_germline_end"] >= 0


def _genotype(rs, n_v=4, n_d=3, n_j=2):
    return {
        "v": rs.gene("V").names[:n_v],
        "d": rs.gene("D").names[:n_d],
        "j": rs.gene("J").names[:n_j],
    }


def test_genotype_restricts_every_call():
    # Property 1: a dynamic genotype subset must restrict EVERY call to its alleles,
    # on both the stage-1 argmax and the learned reranker (which must never see a
    # disallowed candidate — top-k is capped to the allowed count).
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        aligner="softdp"))
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 8, "ACGTACGTNNACGT" * 6, "GATTACA" * 12]
    gt = _genotype(rs, n_v=4, n_d=3, n_j=2)  # n_j=2 < default top-k → exercises the cap
    allowed = {g.upper(): set(v) for g, v in gt.items()}
    for rerank in ("none", "learned"):
        preds = predict_reads(model, rs, reads, batch_size=2, genotype=gt, rerank=rerank)
        for p in preds:
            assert p["v_call"] in allowed["V"]
            assert p["d_call"] in allowed["D"]
            assert p["j_call"] in allowed["J"]
            for g in ("v", "d", "j"):
                assert set(p[f"{g}_topk"]) <= allowed[g.upper()]       # top-k stays inside
                if rerank == "learned":
                    assert set(p[f"{g}_call_set"]) <= allowed[g.upper()]  # equivalence set too


def test_genotype_with_novel_alleles_is_callable():
    # Property 1: novel alleles (never trained on) supplied as the reference are valid
    # call targets — the model conditions on whatever germlines it is handed.
    torch.manual_seed(0)
    igh = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    v0 = igh.gene("V").sequences[0]
    genes = {
        "V": {"IGHV1-2*02": v0, "NOVEL-V*01": v0[:120]},  # one real + one novel
        "D": {"IGHD3-10*01": igh.gene("D").sequences[0]},
        "J": {"IGHJ6*02": igh.gene("J").sequences[0]},
    }
    rs = ReferenceSet.from_genotype(genes)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        aligner="softdp"))
    preds = predict_reads(model, rs, ["ACGT" * 40, v0], batch_size=2, rerank="learned")
    for p in preds:
        assert p["v_call"] in {"IGHV1-2*02", "NOVEL-V*01"}  # novel is a legal output


def test_learned_rerank_emits_calibration_fields():
    # the multi-label set path emits benchmark-friendly keys + per-candidate scores
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        aligner="softdp"))
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 8]
    preds = predict_reads(model, rs, reads, batch_size=2, rerank="learned", emit_scores=True)
    for p in preds:
        for g in ("v", "d", "j"):
            assert p[f"{g}_calls"] == p[f"{g}_call_set"]             # adapter alias
            assert 0.0 <= p[f"{g}_set_confidence"] <= 1.0 + 1e-6     # posterior mass in set
            assert p[f"{g}_call"] in p[f"{g}_call_set"]              # top-1 is in its own set
            assert all(isinstance(nm, str) and isinstance(sc, float)
                       for nm, sc in p[f"{g}_scores"])               # (name, raw_score) pairs


def test_calibration_widens_set_via_temperature():
    # a high temperature flattens the posterior -> the LR-band keeps more candidates
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        aligner="softdp"))
    reads = ["ACGTACGTACGT" * 10, "TTGCAACGTACG" * 8]
    base = predict_reads(model, rs, reads, batch_size=2, rerank="learned", set_epsilon=0.5)
    hot = {"V": {"temperature": 5.0, "epsilon": 0.5}}
    warm = predict_reads(model, rs, reads, batch_size=2, rerank="learned",
                         set_epsilon=0.5, calibration=hot)
    base_sz = sum(len(p["v_call_set"]) for p in base)
    warm_sz = sum(len(p["v_call_set"]) for p in warm)
    assert warm_sz >= base_sz                                       # softer T -> not-smaller set


def test_resolve_hierarchy_levels():
    from alignair.inference.dnalignair_infer import resolve_hierarchy
    # singleton -> allele
    assert resolve_hierarchy(["IGHV3-23*01"], "IGHV3-23*01") == ("IGHV3-23*01", "allele")
    # small set, one gene -> gene level
    assert resolve_hierarchy(["IGHV3-23*01", "IGHV3-23*04"], "IGHV3-23*01") == ("IGHV3-23", "gene")
    # many alleles but one gene -> gene level
    big = [f"IGHV3-23*{i:02d}" for i in range(1, 8)]
    assert resolve_hierarchy(big, big[0]) == ("IGHV3-23", "gene")
    # spans genes but one family -> family level
    assert resolve_hierarchy(["IGHV3-23*01", "IGHV3-30*02"], "IGHV3-23*01") == ("IGHV3", "family")
    # spans families -> abstain
    r, lvl = resolve_hierarchy(["IGHV3-23*01", "IGHV1-2*02"], "IGHV3-23*01")
    assert r is None and lvl == "none"
    # empty set falls back to top1
    assert resolve_hierarchy([], "IGHV3-23*01") == ("IGHV3-23*01", "allele")


def test_predict_reads_emits_hierarchical_fields():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        aligner="softdp"))
    preds = predict_reads(model, rs, ["ACGT" * 30, "TTGCAACGTACG" * 6], rerank="learned")
    for p in preds:
        for g in ("v", "d", "j"):
            assert p[f"{g}_call_level"] in ("allele", "gene", "family", "none")


def test_contaminant_gate_flag_only_and_off_by_default():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64,
                                        aligner="softdp"))
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 6]
    # off by default: a contaminant_score is emitted, but no is_contaminant flag without tau
    p0 = predict_reads(model, rs, reads, rerank="learned")
    assert "is_contaminant" not in p0[0] and p0[0]["contaminant_score"] is not None
    # very high tau -> everything below it -> all flagged; calls RETAINED (flag-only)
    phi = predict_reads(model, rs, reads, rerank="learned", contaminant_tau=1e9)
    assert all(p["is_contaminant"] for p in phi) and all(p["v_call"] for p in phi)
    # very low tau -> nothing flagged
    plo = predict_reads(model, rs, reads, rerank="learned", contaminant_tau=-1e9)
    assert not any(p["is_contaminant"] for p in plo)
    # tau is read from the calibration dict's 'contaminant' entry
    pcal = predict_reads(model, rs, reads, rerank="learned",
                         calibration={"contaminant": {"tau": 1e9}})
    assert all(p["is_contaminant"] for p in pcal)


def test_genotype_empty_gene_raises():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    with pytest.raises(ValueError):
        predict_reads(model, rs, ["ACGT" * 40], genotype={"v": ["does-not-exist*01"]})
