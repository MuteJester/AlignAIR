
import pytest
import builtins

from alignair_assay.airr import reader
from alignair_assay.airr.reader import load_airr_tsv

pytestmark = pytest.mark.component

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


def test_rows_already_read_parse_like_the_file(tmp_path, monkeypatch):
    header = ["sequence_id", "sequence", "rev_comp", "v_call", "v_sequence_start", "v_sequence_end"]
    rows = [[SID, SEQ, "T", "A*01", "1", "44"], ["", SEQ, "F", "", "", ""], [SID, SEQ, "F", "B*01", "", ""],
            ["other", SEQ, "x", "C*01", "1", "999"]]
    path = _write(tmp_path / "r.tsv", header, rows)
    in_memory = [dict(zip(header, row)) for row in rows]
    in_memory[2]["note"] = None  # a cell that is not text is ignored, as the file reader ignores a short row's None
    assert reader.rows_to_predictions(in_memory) == load_airr_tsv(path)
    preds, stats = reader.rows_to_predictions(in_memory)
    assert stats == {"rows": 4, "duplicates": 1, "invalid_rows": 2}
    assert preds[SID].segments["v"].call == "A*01" and preds["other"].invalid

    # The file reader is that function over the TSV's rows: one parser for both sources.
    seen = []
    monkeypatch.setattr(reader, "rows_to_predictions", lambda rows: seen.append(list(rows)) or ({}, {}))
    load_airr_tsv(path)
    assert seen == [[dict(zip(header, row)) for row in rows]]


V_HEADER = ["sequence_id", "sequence", "v_call", "v_sequence_start", "v_sequence_end", "v_germline_start",
            "v_germline_end", "v_cigar"]


def test_the_v_germline_span_of_an_imgt_gapped_row_comes_from_its_cigar(tmp_path):
    """AlignAIR's table (and Change-O's MakeDb's) counts the V germline positions in the IMGT-gapped germline, from
    its first position: v_germline_start is 1 and v_germline_end counts the gaps. The AIRR CIGAR counts ungapped
    germline bases, its N the germline skipped before the V, so the V germline span is read from it: here germline
    255..300 for a read V of 44 bases with one deleted germline base."""
    rows = [
        # AlignAIR: the whole query, S for the read after V; v_germline_end the gapped end
        ["alignair", SEQ, "IGHV1*01", "1", "44", "1", "330", "255N26M1D18M76S"],
        # a partial AlignAIR row: no gapped alignment, so no V germline positions; the CIGAR still places the V
        ["partial", SEQ, "IGHV1*01", "1", "44", "", "", "255N26M1D18M76S"],
        # AlignAIR, the read's V starting at the germline's first base: no N
        ["from_first", SEQ, "IGHV1*01", "1", "44", "1", "70", "26M1D18M76S"],
    ]
    preds, stats = load_airr_tsv(_write(tmp_path / "g.tsv", V_HEADER, rows))
    assert stats["invalid_rows"] == 0
    spans = {sid: (p.segments["v"].germ_start, p.segments["v"].germ_end, p.segments["v"].cigar)
             for sid, p in preds.items()}
    assert spans == {"alignair": (255, 300, "26M1D18M"), "partial": (255, 300, "26M1D18M"),
                     "from_first": (0, 45, "26M1D18M")}


def test_a_makedb_row_reads_its_v_germline_span_from_its_cigar(tmp_path):
    """MakeDb (igblast) writes the V germline positions IMGT-gapped from position 1 and a CIGAR with = and X, a leading
    S and N but no trailing S. With a leading N, or a leading S and no N (the V at the germline's first base), the
    CIGAR places the V. A V at both the read's and the germline's first base has neither: that CIGAR cannot be told
    from an AlignAIR 3 core CIGAR (below), so it is dropped with its message."""
    rows = [
        ["offset", SEQ, "IGHV1*01", "1", "44", "1", "330", "255N4=1X21=1D18="],
        ["flank", SEQ, "IGHV1*01", "11", "54", "1", "70", "10S26=1D17=1X"],
        ["at_first_bases", SEQ, "IGHV1*01", "1", "44", "1", "70", "26=1D17=1X"],
    ]
    preds, stats = load_airr_tsv(_write(tmp_path / "m.tsv", V_HEADER, rows))
    assert [(preds[s].segments["v"].germ_start, preds[s].segments["v"].germ_end) for s in ("offset", "flank")] \
        == [(255, 300), (0, 45)]
    assert preds["offset"].segments["v"].cigar == "4=1X21=1D18="
    assert preds["at_first_bases"].segments["v"].cigar is None
    assert preds["at_first_bases"].invalid == ("v_cigar: '26=1D17=1X' disagrees with coordinates",)
    assert stats["invalid_rows"] == 1


def test_an_alignair_3_row_keeps_its_positions_and_drops_its_v_cigar(tmp_path):
    """An AlignAIR 3 row (a benchmark run's ``human_igh_ogrdb_broad:moderate_full:000000``): a core CIGAR, no S or N,
    with the V germline positions IMGT-gapped from position 1. Its CIGAR does not say where the V starts in the
    germline and the positions do not count ungapped bases (320, not the CIGAR's 290), so the CIGAR is dropped and
    flagged, as before."""
    sequence = ("ACGT" * 91)[:362]
    row = ["human_igh_ogrdb_broad:moderate_full:000000", sequence, "IGHVF10-G40*07", "1", "290", "1", "320",
           "1D27M1I262M"]
    preds, stats = load_airr_tsv(_write(tmp_path / "v3.tsv", V_HEADER, [row]))
    p = preds["human_igh_ogrdb_broad:moderate_full:000000"]
    v = p.segments["v"]
    assert (v.seq_start, v.seq_end, v.germ_start, v.germ_end, v.cigar) == (0, 290, 0, 320, None)
    assert p.invalid == ("v_cigar: '1D27M1I262M' disagrees with coordinates",) and stats["invalid_rows"] == 1


def test_the_assay_writer_round_trips_a_v_spanning_the_whole_read(tmp_path):
    """The assay's own table: a core CIGAR over the whole read with ungapped germline positions (10..110). The
    positions agree with the CIGAR, so they place the V: a core CIGAR over the whole read is not read as an AIRR CIGAR
    without an N (germline from 0)."""
    from alignair_assay.airr.writer import write_airr_tsv
    from alignair_assay.core.prediction import PredSegment, Prediction
    segment = PredSegment("IGHV1*01", ("IGHV1*01",), 0, 100, 10, 110, "100M")
    written = Prediction(sequence_id="r", sequence="ACGT" * 25, segments={"v": segment, "d": PredSegment(),
                                                                          "j": PredSegment()})
    path = tmp_path / "w.tsv"
    write_airr_tsv([written], path)
    preds, stats = load_airr_tsv(path)
    assert preds["r"].segments["v"] == segment and stats["invalid_rows"] == 0
