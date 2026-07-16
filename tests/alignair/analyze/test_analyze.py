"""`alignair analyze` — repertoire + QC + AIRR validation of a predict TSV."""
from alignair.analyze import analyze_file, analyze_rows, format_text, validate_airr
from alignair.io.airr import write_airr


def _records():
    base = dict(v_germline_start=0, v_germline_end=30, j_sequence_start=40, j_sequence_end=60,
                d_call="IGHD3-3*01", junction_aa="CAK")
    return [
        dict(base, v_call="IGHV1-2*01", j_call="IGHJ4*02", productive=True, vj_in_frame=True,
             stop_codon=False, junction="TGTGCGAAA", junction_length=9, v_sequence_start=0,
             v_sequence_end=30, v_cigar="30M", j_cigar="20M", v_set_confidence=0.90, v_identity=0.98,
             orientation_id=0),
        dict(base, v_call="IGHV1-2*01", j_call="IGHJ6*03", productive=True, vj_in_frame=True,
             stop_codon=False, junction="TGTGCGAGAGGG", junction_length=12, v_sequence_start=0,
             v_sequence_end=30, v_cigar="15M1I14M", j_cigar="20M", v_set_confidence=0.30,
             v_identity=0.80, orientation_id=1),                        # reoriented + indel + low conf
        dict(base, v_call="IGHV3-23*01", j_call="IGHJ4*02", productive=False, vj_in_frame=False,
             stop_codon=True, junction="", junction_length=0, v_sequence_start=0, v_sequence_end=30,
             v_cigar="30M", j_cigar="20M", orientation_id=0),           # nonproductive, no junction
    ]


def _write_tsv(tmp_path):
    recs = _records()
    ids = [f"r{i}" for i in range(len(recs))]
    seqs = ["ACGTACGTAC" * 6] * len(recs)
    out = str(tmp_path / "airr.tsv")
    write_airr(out, ids, seqs, recs)
    return out


def test_repertoire_section(tmp_path):
    rep = analyze_file(_write_tsv(tmp_path))["repertoire"]
    assert rep["n_reads"] == 3
    assert rep["productive"]["n"] == 2 and abs(rep["productive"]["pct"] - 66.7) < 0.2
    assert rep["stop_codon"]["n"] == 1
    assert dict(rep["gene_usage"]["v"])["IGHV1-2*01"] == 2          # top V allele counted
    assert rep["unique_junctions_aa"] == 1                          # only "CAK" (empty excluded)
    assert rep["cdr3_length"]["max"] == 12 and rep["cdr3_length"]["n"] == 2   # zero-length excluded


def test_qc_section(tmp_path):
    qc = analyze_file(_write_tsv(tmp_path))["qc"]
    assert qc["orientation"] == {"forward": 2, "reoriented": 1}
    assert qc["indel_flagged_reads"] == 1                          # the 1I cigar
    assert qc["low_confidence_reads"]["n"] == 1                    # v_set_confidence 0.30 < 0.5
    assert qc["field_completeness_pct"]["v_call"] == 100.0
    assert abs(qc["v_set_confidence"]["mean"] - 0.6) < 0.01        # mean of 0.90, 0.30


def test_validation_section_flags_length_mismatch(tmp_path):
    rows = analyze_rows_from_records()
    val = validate_airr(rows, columns=list(rows[0].keys()))
    assert val["junction_length_violations"] == 1                  # doctored row below
    assert not val["valid"]


def analyze_rows_from_records():
    # a row whose junction_length disagrees with len(junction) must be flagged
    return [{"sequence_id": "r1", "sequence": "ACGT", "v_call": "IGHV1-2*01", "j_call": "IGHJ4*02",
             "junction": "TGTGCG", "junction_length": "99", "productive": "T", "vj_in_frame": "T",
             "stop_codon": "F", "v_cigar": "30M", "j_cigar": "20M",
             "v_sequence_start": "1", "v_sequence_end": "30"}]


def test_format_text_has_all_sections(tmp_path):
    text = format_text(analyze_file(_write_tsv(tmp_path)))
    assert "REPERTOIRE" in text and "QC" in text and "VALIDATION" in text
    assert "productive" in text.lower()


def test_analyze_rows_direct():
    rep = analyze_rows(_records())["repertoire"]
    assert rep["n_reads"] == 3


def test_cli_analyze_writes_report(tmp_path):
    from alignair.cli.main import build_parser
    tsv = _write_tsv(tmp_path)
    out = str(tmp_path / "report.txt")
    args = build_parser().parse_args(["analyze", tsv, "--out", out])
    assert args.func(args) == 0
    text = open(out).read()
    assert "REPERTOIRE" in text and "VALIDATION" in text
    # json format
    out2 = str(tmp_path / "report.json")
    args = build_parser().parse_args(["analyze", tsv, "--out", out2, "--format", "json"])
    assert args.func(args) == 0
    import json
    assert json.load(open(out2))["repertoire"]["n_reads"] == 3
