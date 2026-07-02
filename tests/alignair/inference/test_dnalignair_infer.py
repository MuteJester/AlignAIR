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


def test_seed_extend_learned_reader_runs():
    # seed_extend has no germline_encoder: the learned reader must read segment reps off the
    # backbone and place a band head center for the exact banded DP (previously crashed here).
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    assert getattr(model, "germline_encoder", None) is None
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 8, "ACGTACGTNNACGT" * 6]
    # v_reader="learned" forces the DP reader on V too (not the parasail fallback)
    preds = predict_reads(model, rs, reads, batch_size=2, rerank="learned", v_reader="learned")
    assert len(preds) == len(reads)
    vnames = set(rs.gene("V").names)
    for p in preds:
        assert p["v_call"] in vnames
        assert isinstance(p["v_call_set"], list) and p["v_call_set"]   # equivalence set emitted
        assert set(p["v_call_set"]) <= vnames


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
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
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
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    preds = predict_reads(model, rs, ["ACGT" * 40, v0], batch_size=2, rerank="learned")
    for p in preds:
        assert p["v_call"] in {"IGHV1-2*02", "NOVEL-V*01"}  # novel is a legal output


def test_learned_rerank_emits_calibration_fields():
    # the multi-label set path emits benchmark-friendly keys + per-candidate scores
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    reads = ["ACGT" * 40, "TTGCAACGTACG" * 8]
    preds = predict_reads(model, rs, reads, batch_size=2, rerank="learned", emit_scores=True)
    for p in preds:
        for g in ("v", "d", "j"):
            assert p[f"{g}_calls"] == p[f"{g}_call_set"]             # adapter alias
            assert 0.0 <= p[f"{g}_set_confidence"] <= 1.0 + 1e-6     # posterior mass in set
            assert p[f"{g}_call"] in p[f"{g}_call_set"]              # top-1 is in its own set
            assert all(isinstance(nm, str) and isinstance(sc, float)
                       for nm, sc in p[f"{g}_scores"])               # (name, raw_score) pairs


def test_parasail_v_reader_keeps_call_in_ordered_set():
    # the classical parasail V reader: V call resolved by raw SW; v_calls[0] == v_call (ordered set);
    # D/J still come from the learned soft-DP path.
    pytest.importorskip("parasail")
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    reads = ["ACGTACGTACGT" * 12, "TTGCAACGTACG" * 10]
    preds = predict_reads(model, rs, reads, batch_size=2, rerank="learned",
                          v_reader="parasail", emit_scores=True)
    vnames = set(rs.gene("V").names)
    for p in preds:
        assert p["v_call"] in vnames
        assert p["v_call"] in p["v_call_set"]
        assert p["v_call_set"][0] == p["v_call"]                 # set ordered: best first
        assert 0.0 <= p["v_set_confidence"] <= 1.0 + 1e-6
        assert p["j_call"] in set(rs.gene("J").names)            # D/J unaffected


def test_calibration_widens_set_via_temperature():
    # a high temperature flattens the posterior -> the LR-band keeps more candidates
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
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
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    preds = predict_reads(model, rs, ["ACGT" * 30, "TTGCAACGTACG" * 6], rerank="learned")
    for p in preds:
        for g in ("v", "d", "j"):
            assert p[f"{g}_call_level"] in ("allele", "gene", "family", "none")


def test_contaminant_gate_flag_only_and_off_by_default():
    torch.manual_seed(0)
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
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


def test_swap_reference_allele_callable_via_seed_path_when_retrieval_misses():
    # The swap-robustness guarantee on the FULL IGH reference: when neural retrieval omits the
    # true allele from its top-k (the failure mode for divergent/novel germlines), the non-learned
    # k-mer seed prefilter must still admit it so WFA picks it and aligns full-length. This is the
    # mechanism that makes "swap the genotype/germline on a trained model" robust; it is asserted at
    # the caller level because an UNTRAINED model's segmentation can't yield a usable segment e2e.
    from alignair.align import SeedPrefilter, get_aligner
    from alignair.inference.wfa_caller import call_segment
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    sp, al = SeedPrefilter(rs, k=11), get_aligner()
    vg = rs.gene("V")
    true_idx = 7
    seg = vg.sequences[true_idx]                                 # a read that IS the true germline
    wrong_topk = [i for i in range(20) if i != true_idx][:16]    # retrieval MISSES the true allele
    call = call_segment(seg, "V", wrong_topk, rs, sp, al)
    assert true_idx not in wrong_topk
    assert true_idx in call.pool_idx                             # admitted by the seed prefilter
    assert call.best_idx == true_idx                             # WFA picks it from the union pool
    assert call.germ_start == 0 and call.germ_end == len(seg)    # aligned full-length
