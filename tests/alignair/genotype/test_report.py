"""Genotype task 5: explainable report (json + text)."""
from alignair.genotype.report import build_report, format_report
from alignair.genotype.zygosity import GeneCall


def _gc():
    return GeneCall(
        gene="v",
        alleles=[{"name": "IGHV1-2*01", "support": 0.9, "evidence": True,
                  "count": 1200, "usage_fraction": 0.42, "mean_confidence": 0.94}],
        zygosity="homozygous", deletion_candidate=False,
        reasons=["IGHV1-2*01: included (residual 0.900, diagnostic SNP covered)",
                 "IGHV1-2*05: excluded (residual 0.010 < 0.05, no covered diagnostic SNP)"],
        novel=[{"call": "novel", "near": "IGHV3-23*01", "positions": [165],
                "sequence": "ACGT", "source_mask": ["ref", "observed"]}],
    )


def test_build_report_carries_counts_reasons_and_novel_mask():
    r = build_report([_gc()], meta={"n_reads": 5000})
    g = r["genes"]["v"]
    assert g["zygosity"] == "homozygous"
    assert g["alleles"][0]["count"] == 1200 and g["alleles"][0]["usage_fraction"] == 0.42
    nv = g["novel_candidates"][0]
    assert nv["positions"] == [165] and nv["source_mask"] == ["ref", "observed"]
    assert any("excluded" in x for x in g["reasons"])
    assert r["meta"]["n_reads"] == 5000


def test_format_report_renders_gene_sections_and_reasons():
    txt = format_report(build_report([_gc()]))
    assert "V" in txt and "homozygous" in txt
    assert "excluded" in txt and "novel" in txt.lower()
    assert "IGHV1-2*01" in txt
