from alignair.compare import compare_airr, read_airr, format_report_md


def _write(path, rows):
    cols = ["sequence_id", "v_call", "v_call_set", "d_call", "j_call", "junction", "productive"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def test_compare_agreement_and_set_rescue(tmp_path):
    a = tmp_path / "alignair.tsv"
    b = tmp_path / "igblast.tsv"
    _write(a, [
        {"sequence_id": "r1", "v_call": "IGHV1-1*01", "v_call_set": "IGHV1-1*01,IGHV1-1*02",
         "d_call": "IGHD1-1*01", "j_call": "IGHJ1*01", "junction": "TGTGCG", "productive": "T"},
        {"sequence_id": "r2", "v_call": "IGHV2-1*01", "v_call_set": "IGHV2-1*01",
         "d_call": "IGHD2-1*01", "j_call": "IGHJ1*01", "junction": "TGTAAA", "productive": "T"},
        {"sequence_id": "r3", "v_call": "IGHV3-1*01", "v_call_set": "IGHV3-1*01",
         "d_call": "IGHD3-1*01", "j_call": "IGHJ2*01", "junction": "TGTCCC", "productive": "F"},
    ])
    _write(b, [
        {"sequence_id": "r1", "v_call": "IGHV1-1*02", "d_call": "IGHD1-1*01", "j_call": "IGHJ1*01",
         "junction": "TGTGCG", "productive": "T"},   # V differs but in a's set; D/J/junction agree
        {"sequence_id": "r2", "v_call": "IGHV2-1*01", "d_call": "IGHD2-1*01", "j_call": "IGHJ1*01",
         "junction": "TGTAAA", "productive": "T"},   # full agree
        {"sequence_id": "r3", "v_call": "IGHV9-9*01", "d_call": "IGHD3-1*01", "j_call": "IGHJ2*01",
         "junction": "TGTGGG", "productive": "F"},   # V differs, NOT in set; junction differs
    ])
    rep = compare_airr(read_airr(str(a)), read_airr(str(b)), "AlignAIR", "IgBLAST")
    assert rep["coverage"]["matched"] == 3
    v = rep["genes"]["v"]
    assert v["allele_agreement"] == round(1 / 3, 4)        # only r2 agrees on allele
    assert v["gene_agreement"] == round(2 / 3, 4)          # r1,r2 same gene; r3 differs
    assert v["set_rescue_rate"] == 0.5                     # of 2 disagreements, r1 rescued by set
    assert rep["genes"]["d"]["allele_agreement"] == 1.0    # D all agree
    assert rep["junction_nt_agreement"] == round(2 / 3, 4)  # r1,r2 match; r3 differs
    assert rep["productive_agreement"] == 1.0
    assert "AIRR agreement" in format_report_md(rep)       # renders without error
