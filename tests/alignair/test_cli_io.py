import csv

import pytest

from alignair.io.sequence_reader import read_sequences, validate
from alignair.io.airr import write_airr, COLUMNS, _cigar
from alignair import cli


def _pred(**over):
    p = {"orientation_id": 0, "productive": True, "junction": "TGTGCGAAA", "junction_aa": "CAK",
         "junction_length": 9}
    for g in ("v", "d", "j"):
        p[f"{g}_call"] = f"IGH{g.upper()}1-1*01"
        p[f"{g}_call_set"] = [f"IGH{g.upper()}1-1*01"]
        p[f"{g}_call_level"] = "allele"
        p[f"{g}_set_confidence"] = 0.9
    p.update({"v_sequence_start": 0, "v_sequence_end": 290, "v_germline_start": 0, "v_germline_end": 290,
              "d_sequence_start": 295, "d_sequence_end": 305, "d_germline_start": 2, "d_germline_end": 12,
              "j_sequence_start": 310, "j_sequence_end": 360, "j_germline_start": 5, "j_germline_end": 55})
    p.update(over)
    return p


def test_reference_validate_and_convert(tmp_path):
    y = tmp_path / "ref.yaml"
    y.write_text("v:\n  IGHV1-2*01: ACGTACGT\nd:\n  IGHD1-1*01: GGGG\nj:\n  IGHJ1*01: TTTT\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["reference", "validate", str(y)])
    assert e.value.code == 0                              # clean reference -> OK
    fa = tmp_path / "ref.fasta"
    cli.main(["reference", "convert", str(y), "-o", str(fa)])
    txt = fa.read_text()
    assert ">IGHV1-2*01" in txt and "ACGTACGT" in txt


def test_reference_validate_flags_empty_sequence(tmp_path):
    y = tmp_path / "bad.yaml"
    y.write_text("v:\n  IGHV1-2*01: ''\nj:\n  IGHJ1*01: TTTT\n")   # empty V sequence
    with pytest.raises(SystemExit) as e:
        cli.main(["reference", "validate", str(y)])
    assert e.value.code != 0


def test_iter_sequences_chunks_and_counts(tmp_path):
    from alignair.io.sequence_reader import iter_sequences
    p = tmp_path / "reads.fasta"
    p.write_text("".join(f">r{i}\nACGTACGT\n" for i in range(10)) + ">bad\nXXXXXXXX\n")
    chunks = list(iter_sequences(str(p), chunk_size=3))
    sizes = [len(ids) for ids, _, _ in chunks]
    assert sum(sizes) == 10 and max(sizes) <= 3              # 10 valid reads, chunked by 3
    assert sum(d for _, _, d in chunks) == 1                 # the all-X read is dropped
    # ids and seqs stay aligned and valid
    all_ids = [i for ids, _, _ in chunks for i in ids]
    assert all_ids[0] == "r0" and len(all_ids) == 10


def test_airr_writer_incremental_matches_oneshot(tmp_path):
    from alignair.io.airr import AirrWriter
    seq = "ACGT" * 90
    preds = [_pred(), _pred(), _pred()]
    one = tmp_path / "one.tsv"
    write_airr(str(one), ["a", "b", "c"], [seq] * 3, preds, locus="IGH")
    inc = tmp_path / "inc.tsv"
    w = AirrWriter(str(inc), "IGH")
    w.write(["a"], [seq], preds[:1])            # two separate chunks
    w.write(["b", "c"], [seq, seq], preds[1:])
    w.close()
    assert one.read_text() == inc.read_text()   # streaming == one-shot, byte-identical


def test_read_sequences_custom_columns(tmp_path):
    p = tmp_path / "in.csv"
    p.write_text("name,dna\na,ACGTACGT\nb,TTTTGGGG\n")
    ids, seqs, info = read_sequences(str(p), seq_column="dna", id_column="name")
    assert ids == ["a", "b"] and seqs == ["ACGTACGT", "TTTTGGGG"]
    with pytest.raises(ValueError, match="sequence column 'missing'"):
        read_sequences(str(p), seq_column="missing")


def test_read_sequences_stdin(monkeypatch):
    import alignair.io.sequence_reader as sr
    monkeypatch.setattr(sr, "_STDIN_CACHE", None)
    monkeypatch.setattr(sr, "_slurp_stdin", lambda: ">r1\nACGTACGT\n>r2\nTTTTGGGG\n")
    ids, seqs, info = sr.read_sequences("-")
    assert ids == ["r1", "r2"] and info["format"] == "fasta"


def test_write_airr_to_stdout(capsys):
    write_airr("-", ["r1"], ["ACGT" * 90], [_pred()], locus="IGH")
    out = capsys.readouterr().out
    assert out.startswith("sequence_id\t") and "IGHV1-1*01" in out


def test_realign_gapped_alignment():
    pytest.importorskip("parasail")
    from alignair.io.alignment import realign
    from alignair.reference.reference_set import ReferenceSet
    vg = "ACGTACGTACGTACGTACGT"          # 20bp V germline
    jg = "TTTTGGGGTTTTGGGG"              # 16bp J germline
    rs = ReferenceSet.from_genotype({"v": {"V1*01": vg}, "j": {"J1*01": jg}})
    seq = vg + "CCCCC" + jg              # V + 5bp non-templated N + J (exact)
    p = {"v_call": "V1*01", "v_sequence_start": 0, "v_sequence_end": 20,
         "j_call": "J1*01", "j_sequence_start": 25, "j_sequence_end": 41}
    out = realign(seq, p, rs)
    assert len(out["sequence_alignment"]) == len(out["germline_alignment"])   # aligned pair
    assert out["v_identity"] == 1.0 and out["j_identity"] == 1.0              # exact match
    assert out["germline_alignment"].count("N") == 5                          # the N region
    assert out["v_germline_start"] == 0 and out["v_germline_end"] == 20
    assert "M" in out["v_cigar"]


def test_realign_recovers_offset_and_indel():
    pytest.importorskip("parasail")
    from alignair.io.alignment import realign
    from alignair.reference.reference_set import ReferenceSet
    vg = "AAAACCCCGGGGTTTTACGT"
    rs = ReferenceSet.from_genotype({"v": {"V1*01": vg}, "j": {"J1*01": "TTTTGGGG"}})
    # read V segment == germline[4:] (a 4bp 5' germline offset)
    seq = vg[4:] + "TTTTGGGG"
    p = {"v_call": "V1*01", "v_sequence_start": 0, "v_sequence_end": len(vg) - 4,
         "j_call": "J1*01", "j_sequence_start": len(vg) - 4, "j_sequence_end": len(vg) - 4 + 8}
    out = realign(seq, p, rs)
    assert out["v_germline_start"] == 4          # alignment recovers the 5' offset
    assert out["v_cigar"].startswith("4N")       # encoded as a germline skip


def test_cigar():
    assert _cigar(377, 0, 290, 0, 290) == "290M87S"        # V at read start, no germline skip
    assert _cigar(377, 310, 360, 5, 55) == "310S5N50M17S"  # J: clip + germline skip + match + tail
    assert _cigar(377, None, None, None, None) == ""        # absent segment
    # germline span (296) longer than read span (287) -> a 9-base deletion op
    assert _cigar(377, 0, 287, 0, 296) == "287M9D90S"
    # read span longer than germline span -> insertion op
    assert _cigar(100, 0, 50, 0, 45) == "45M5I50S"


def test_write_airr_is_airr_c_valid(tmp_path):
    airrlib = pytest.importorskip("airr")
    seq = "ACGT" * 90                                  # len 360
    out = tmp_path / "rearr.tsv"
    write_airr(str(out), ["r1", "r2"], [seq, seq], [_pred(), _pred()], locus="IGH")
    assert airrlib.validate_rearrangement(str(out)) is True
    rows = list(airrlib.read_rearrangement(str(out)))   # Change-O/Immcantation-compatible reader
    assert len(rows) == 2 and rows[0]["v_call"] == "IGHV1-1*01"


def test_predict_missing_input_clean_error(tmp_path):
    # a known-bad model path errors before reading sequences (covered elsewhere); here just
    # confirm validate-airr rejects a non-AIRR file cleanly
    bad = tmp_path / "notairr.tsv"
    bad.write_text("a\tb\n1\t2\n")
    with pytest.raises(SystemExit) as e:
        cli.main(["validate-airr", str(bad)])
    assert e.value.code != 0


def test_doctor_runs_and_reports_ok():
    # doctor exits 0 when core deps (torch, GenAIRR) are importable in the test env
    with pytest.raises(SystemExit) as e:
        cli.main(["doctor"])
    assert e.value.code == 0


def test_predict_missing_model_errors_cleanly():
    with pytest.raises(SystemExit) as e:
        cli.main(["predict", "x.fasta", "-o", "out.tsv", "--model", "/no/such/model.pt"])
    assert e.value.code != 0


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


def test_canonicalize_sequence_recovers_forward():
    from alignair.inference.dnalignair_infer import canonicalize_sequence
    fwd = "ACGTACGTTTGCAAACGT"
    comp = str.maketrans("ACGTN", "TGCAN")
    rc = fwd.translate(comp)[::-1]
    assert canonicalize_sequence(fwd, 0) == fwd
    assert canonicalize_sequence(rc, 1) == fwd                      # RC input -> forward
    assert canonicalize_sequence(canonicalize_sequence(fwd, 1), 1) == fwd   # involution
    assert canonicalize_sequence(fwd, 2) == fwd.translate(comp)
    assert canonicalize_sequence(fwd, 3) == fwd[::-1]


def test_canonicalize_matches_model_token_transform():
    torch = pytest.importorskip("torch")
    from alignair.data.tokenizer import pad_tokenize
    from alignair.nn.orientation import apply_orientation
    from alignair.inference.dnalignair_infer import canonicalize_sequence
    seq = "ACGTACGTNNACGTTGCA"
    tok, msk = pad_tokenize([seq])
    inv = {1: "A", 2: "T", 3: "G", 4: "C", 5: "N"}
    for oid in (0, 1, 2, 3):
        canon = apply_orientation(tok, msk, torch.full((1,), oid))
        model_str = "".join(inv[int(t)] for t, m in zip(canon[0], msk[0]) if int(m) == 1)
        assert canonicalize_sequence(seq, oid) == model_str        # string == token transform


def test_airr_coords_match_emitted_sequence_for_rev_comp(tmp_path):
    # regression for the canonical-frame bug: coordinates must match the emitted (canonical)
    # `sequence` regardless of input orientation.
    from alignair.inference.dnalignair_infer import canonicalize_sequence
    fwd = "AAAACCCCGGGGTTTT" + "ACGT" * 10
    rc = fwd.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    pred = {"v_call": "X*01", "v_sequence_start": 4, "v_sequence_end": 12, "orientation_id": 1}
    canon = canonicalize_sequence(rc, 1)
    out = tmp_path / "rc.tsv"
    write_airr(str(out), ["rc"], [canon], [pred])
    row = next(csv.DictReader(open(out), delimiter="\t"))
    assert row["rev_comp"] == "T" and row["sequence"] == fwd
    s, e = int(row["v_sequence_start"]) - 1, int(row["v_sequence_end"])
    assert row["sequence"][s:e] == fwd[4:12]                       # coords point to the right region


def test_write_airr_columns_coords_and_sets(tmp_path):
    preds = [{
        "v_call": "IGHV1-2*02", "d_call": "IGHD3-10*01", "j_call": "IGHJ6*02",
        "productive": True, "orientation_id": 0,
        "v_sequence_start": 0, "v_sequence_end": 295, "v_germline_start": 0, "v_germline_end": 295,
        "d_sequence_start": 300, "d_sequence_end": 312, "d_germline_start": 2, "d_germline_end": 14,
        "j_sequence_start": 320, "j_sequence_end": 360, "j_germline_start": 1, "j_germline_end": 41,
        "v_call_set": ["IGHV1-2*02", "IGHV1-2*04"], "v_call_level": "gene", "v_set_confidence": 0.91,
        "is_contaminant": True,
    }]
    out = tmp_path / "r.tsv"
    write_airr(str(out), ["read1"], ["ACGT" * 90], preds, locus="IGH")
    rows = list(csv.DictReader(open(out), delimiter="\t"))
    assert list(rows[0].keys()) == COLUMNS
    r = rows[0]
    assert r["sequence_id"] == "read1" and r["locus"] == "IGH" and r["v_call"] == "IGHV1-2*02"
    assert r["rev_comp"] == "F"                         # orientation_id 0 -> forward
    assert r["is_contaminant"] == "T"                   # out-of-scope flag
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


def test_cli_bundle_and_predict_equivalence(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("GenAIRR")
    from alignair.config.dnalignair_config import DNAlignAIRConfig
    from alignair.core.dnalignair import DNAlignAIR
    from alignair import cli
    torch.manual_seed(0)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64, aligner="softdp")
    ck = tmp_path / "m.pt"
    torch.save({"model": DNAlignAIR(cfg).state_dict(), "config": cfg.to_dict()}, ck)
    fa = tmp_path / "r.fasta"; fa.write_text(">a\n" + "ACGT" * 50 + "\n>b\n" + "TTGCAACGTACG" * 6 + "\n")

    # package into a bundle, then predict from BOTH raw ckpt and the bundle
    bdir = tmp_path / "bundle"
    cli.main(["bundle", "--model", str(ck), "-o", str(bdir), "--dataconfig", "HUMAN_IGH_OGRDB"])
    out_raw, out_bundle = tmp_path / "raw.tsv", tmp_path / "bundle.tsv"
    cli.main(["predict", str(fa), "-o", str(out_raw), "--model", str(ck), "--device", "cpu"])
    cli.main(["predict", str(fa), "-o", str(out_bundle), "--model", str(bdir), "--device", "cpu"])
    assert open(out_raw).read() == open(out_bundle).read()        # identical predictions
