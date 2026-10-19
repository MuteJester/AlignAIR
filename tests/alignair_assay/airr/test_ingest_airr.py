import builtins

from alignair_assay.airr import reader
from alignair_assay.airr.reader import load_airr_tsv

SID = "human_igh_ogrdb_broad:amplicon_janchor:000000"
SEQ = "ACGT" * 30


def _write(path, header, rows):
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_alignair_row(tmp_path):
    header = ["sequence_id", "sequence", "rev_comp", "productive", "v_call", "v_call_set",
              "v_sequence_start", "v_sequence_end", "v_germline_start", "v_germline_end", "v_cigar"]
    row = [SID, SEQ, "F", "", "IGHVF1-G3*02", "IGHVF1-G3*02,IGHVF1-G3*04,IGHVF1-G3*06",
           "1", "44", "256", "300", "26M1D18M"]
    preds, stats = load_airr_tsv(_write(tmp_path / "a.tsv", header, [row]))
    p = preds[SID]
    v = p.segments["v"]
    assert v.call == "IGHVF1-G3*02"
    assert v.call_set == ("IGHVF1-G3*02", "IGHVF1-G3*04", "IGHVF1-G3*06")
    assert (v.seq_start, v.seq_end, v.germ_start, v.germ_end) == (0, 44, 255, 300)
    assert v.cigar == "26M1D18M"
    assert p.rev_comp is False
    assert p.productive is None
    assert p.junction_start is None
    assert p.invalid == ()
    assert stats == {"rows": 1, "duplicates": 0, "invalid_rows": 0}


def test_igblast_row(tmp_path):
    header = ["sequence_id", "sequence", "rev_comp", "productive", "v_call",
              "v_sequence_start", "v_sequence_end", "v_germline_start", "v_germline_end",
              "v_cigar", "j_cigar", "j_sequence_start", "j_sequence_end",
              "j_germline_start", "j_germline_end"]
    row = [SID, SEQ, "F", "T", "IGHVF1-G2*01,IGHVF1-G2*02", "1", "45", "256", "301",
           "255N26M1D19M77S", "60S1N52M", "61", "112", "2", "53"]
    preds, _ = load_airr_tsv(_write(tmp_path / "i.tsv", header, [row]))
    p = preds[SID]
    v = p.segments["v"]
    assert v.germ_start == 255
    assert v.cigar == "26M1D19M"
    assert v.call == "IGHVF1-G2*01"
    assert v.call_set == ("IGHVF1-G2*01", "IGHVF1-G2*02")
    assert p.productive is True
    assert p.segments["j"].cigar == "52M"
    assert p.segments["j"].call is None


def test_malformed_row(tmp_path):
    header = ["sequence_id", "sequence", "v_call", "v_sequence_start", "v_sequence_end",
              "v_germline_start", "v_germline_end", "v_cigar", "j_sequence_start",
              "j_sequence_end", "j_cigar"]
    row = [SID, SEQ, "IGHV1*01", "None", "9999", "10", "20", "", "5", "20", "3M"]
    preds, stats = load_airr_tsv(_write(tmp_path / "m.tsv", header, [row]))
    p = preds[SID]
    v = p.segments["v"]
    assert v.seq_start is None
    assert v.seq_end is None
    assert v.cigar is None
    assert v.germ_start == 9
    # J has read coordinates but no germline coordinates, so its CIGAR cannot be validated and is dropped.
    assert p.segments["j"].cigar is None
    assert "j_cigar: dropped, coordinates unavailable to validate '3M'" in p.invalid
    assert p.invalid
    assert stats["invalid_rows"] == 1


def test_cigar_disagreeing_with_coordinates_is_dropped(tmp_path):
    header = ["sequence_id", "sequence", "v_call", "v_sequence_start", "v_sequence_end", "v_germline_start",
              "v_germline_end", "v_cigar"]
    # 44 read and 45 germline bases: 26M1D18M fits, 40M does not.
    good = [SID, SEQ, "IGHV1*01", "1", "44", "256", "300", "26M1D18M"]
    bad = ["other", SEQ, "IGHV1*01", "1", "44", "256", "300", "40M"]
    preds, stats = load_airr_tsv(_write(tmp_path / "c.tsv", header, [good, bad]))
    assert preds[SID].segments["v"].cigar == "26M1D18M" and preds[SID].invalid == ()
    v = preds["other"].segments["v"]
    assert v.cigar is None and (v.seq_start, v.seq_end, v.germ_start, v.germ_end) == (0, 44, 255, 300)
    assert preds["other"].invalid == ("v_cigar: '40M' disagrees with coordinates",)
    assert stats["invalid_rows"] == 1


def test_integral_float_coordinates(tmp_path):
    header = ["sequence_id", "sequence", "v_sequence_start", "v_sequence_end", "junction_start", "junction_end"]
    row = [SID, SEQ, "12.0", "40.0", "11.5", "17"]
    p = load_airr_tsv(_write(tmp_path / "f.tsv", header, [row]))[0][SID]
    assert (p.segments["v"].seq_start, p.segments["v"].seq_end) == (11, 40)
    assert p.junction_start is None and p.invalid == ("junction_start: unparseable coordinate '11.5'",)


def test_tsv_read_as_utf8(tmp_path, monkeypatch):
    path = _write(tmp_path / "u.tsv", ["sequence_id", "sequence", "note"], [[SID, SEQ, "naïve β cell"]])
    encodings = []

    def spy(file, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return builtins.open(file, *args, **kwargs)

    monkeypatch.setattr(reader, "open", spy, raising=False)
    preds, _ = load_airr_tsv(path)
    assert encodings == ["utf-8"] and preds[SID].sequence == SEQ


def test_junction_coordinates(tmp_path):
    header = ["sequence_id", "sequence", "junction_start", "junction_end", "cdr3_start", "cdr3_end"]
    rows = [
        ["reversed", SEQ, "20", "10", "", ""],  # end before start
        ["cdr3", SEQ, "", "", "14", "20"],  # junction = CDR3 plus the anchor codons
        ["both", SEQ, "11", "23", "50", "60"],  # reported junction coordinates win
        ["edge", SEQ, "", "", "2", "119"],  # the derived junction would leave the read
    ]
    preds, _ = load_airr_tsv(_write(tmp_path / "j.tsv", header, rows))
    p = preds["reversed"]
    assert (p.junction_start, p.junction_end) == (None, None)
    assert p.invalid == ("junction: end before start",)
    assert (preds["cdr3"].junction_start, preds["cdr3"].junction_end) == (10, 23)
    assert (preds["both"].junction_start, preds["both"].junction_end) == (10, 23)
    assert (preds["edge"].junction_start, preds["edge"].junction_end) == (None, None)
    assert preds["cdr3"].invalid == preds["both"].invalid == preds["edge"].invalid == ()


def test_duplicates(tmp_path):
    header = ["sequence_id", "sequence", "v_call"]
    rows = [[SID, SEQ, "A*01"], [SID, SEQ, "B*01"]]
    preds, stats = load_airr_tsv(_write(tmp_path / "d.tsv", header, rows))
    assert preds[SID].segments["v"].call == "A*01"
    assert stats["duplicates"] == 1
    assert stats["rows"] == 2
    assert len(preds) == 1


def test_junction_and_bool_parsing(tmp_path):
    header = ["sequence_id", "sequence", "junction", "junction_aa", "junction_start",
              "junction_end", "vj_in_frame", "stop_codon", "rev_comp"]
    row = [SID, SEQ, "tgtgcc", "CA", "11", "17", "TRUE", "0", "nan"]
    preds, _ = load_airr_tsv(_write(tmp_path / "j.tsv", header, [row]))
    p = preds[SID]
    assert p.junction_nt == "TGTGCC"
    assert (p.junction_start, p.junction_end) == (10, 17)
    assert p.vj_in_frame is True
    assert p.stop_codon is False
    assert p.rev_comp is None
