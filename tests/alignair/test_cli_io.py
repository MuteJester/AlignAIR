import csv

import pytest

from alignair.io.sequence_reader import read_sequences, validate
from alignair.io.airr import write_airr, COLUMNS


def test_validate_iupac_and_drop():
    assert validate("acgtACGT") == "ACGTACGT"
    assert validate("ACGTRYKM") == "ACGTNNNN"          # IUPAC ambiguity -> N
    assert validate("XXXXXXXXXX") is None              # >20% junk -> dropped
    assert validate("") is None


def test_read_fasta(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(">a desc\nACGT\nACGT\n>b\nTTTT\n")
    ids, seqs, info = read_sequences(str(p))
    assert ids == ["a", "b"] and seqs == ["ACGTACGT", "TTTT"]
    assert info["format"] == "fasta" and info["n_dropped"] == 0


def test_read_fastq(tmp_path):
    p = tmp_path / "x.fastq"
    p.write_text("@r1\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTGGGG\n+\nIIIIIIII\n")
    ids, seqs, info = read_sequences(str(p))
    assert ids == ["r1", "r2"] and seqs == ["ACGTACGT", "TTTTGGGG"]
    assert info["format"] == "fastq"


def test_read_csv_with_sequence_column(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("sequence_id,sequence\ns1,ACGTACGT\ns2,GGGGCCCC\n")
    ids, seqs, _ = read_sequences(str(p))
    assert ids == ["s1", "s2"] and seqs == ["ACGTACGT", "GGGGCCCC"]


def test_read_txt_one_per_line(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("ACGTACGT\nTTTTGGGG\n")
    ids, seqs, info = read_sequences(str(p))
    assert seqs == ["ACGTACGT", "TTTTGGGG"] and info["format"] == "txt"


def test_write_airr_columns_coords_and_sets(tmp_path):
    preds = [{
        "v_call": "IGHV1-2*02", "d_call": "IGHD3-10*01", "j_call": "IGHJ6*02",
        "productive": True,
        "v_sequence_start": 0, "v_sequence_end": 295, "v_germline_start": 0, "v_germline_end": 295,
        "d_sequence_start": 300, "d_sequence_end": 312, "d_germline_start": 2, "d_germline_end": 14,
        "j_sequence_start": 320, "j_sequence_end": 360, "j_germline_start": 1, "j_germline_end": 41,
        "v_call_set": ["IGHV1-2*02", "IGHV1-2*04"], "v_call_level": "gene", "v_set_confidence": 0.91,
    }]
    out = tmp_path / "r.tsv"
    write_airr(str(out), ["read1"], ["ACGT" * 90], preds, locus="IGH")
    rows = list(csv.DictReader(open(out), delimiter="\t"))
    assert list(rows[0].keys()) == COLUMNS
    r = rows[0]
    assert r["sequence_id"] == "read1" and r["locus"] == "IGH" and r["v_call"] == "IGHV1-2*02"
    assert r["v_sequence_start"] == "1"                 # 0-based -> 1-based AIRR
    assert r["v_sequence_end"] == "295"
    assert r["v_call_set"] == "IGHV1-2*02,IGHV1-2*04" and r["v_call_level"] == "gene"
    assert r["v_set_confidence"] == "0.9100"


def test_cli_predict_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    from alignair.core.dnalignair import DNAlignAIR
    from alignair import cli
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64, aligner="softdp")
    model = DNAlignAIR(cfg)
    ck = tmp_path / "m.pt"; torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, ck)
    fa = tmp_path / "reads.fasta"
    fa.write_text(">read1\n" + "ACGT" * 40 + "\n>junk\nXXXXXXXXXX\n")
    out = tmp_path / "out.tsv"
    cli.main(["predict", str(fa), "-o", str(out), "--model", str(ck), "--device", "cpu", "--batch", "4"])
    rows = list(csv.DictReader(open(out), delimiter="\t"))
    assert len(rows) == 1 and rows[0]["sequence_id"] == "read1"   # junk dropped
    assert rows[0]["v_call"] and rows[0]["locus"] == "IGH"
