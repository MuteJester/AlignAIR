"""The AIRR table follows the AIRR Rearrangement schema, and Change-O's frames where the schema leaves room, so
Change-O's ``CreateGermlines.py`` reads it as written.

The schema: positions are 1-based and closed; ``*_sequence_start``/``_end`` and the FWR/CDR ``_start``/``_end`` are
positions in ``sequence`` (in its reverse complement for a ``rev_comp`` row); ``*_alignment_start``/``_end`` are
positions in ``sequence_alignment`` and ``germline_alignment``; a CIGAR spans the whole query (leading S, the
germline's skipped 5' bases as N, the alignment, trailing S) and counts ungapped germline bases. Where it leaves room,
the table does what Change-O's ``MakeDb.py igblast`` writes: the V germline positions count in the IMGT-gapped V
germline, from its first position, as ``sequence_alignment`` does; the untemplated bases are N in
``germline_alignment``; a row without a D call holds every base between V and J in ``np1``. FWR4 is IMGT's FR4, from
the J Trp or Phe codon (the junction's last), as IgBLAST and MakeDb write it.

``conformance_errors`` checks one written row against the germlines its calls name; the test runs it over the test
chip's rows of the example reads and the golden reads (a reverse-complemented read, a cropped read, amplicons, a V
indel), with the D presence head forced to absent and to short on some of them.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "reads.fasta"
REGIONS = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")
_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _ops(cigar: str) -> list[tuple[int, str]]:
    return [(int(n), op) for n, op in re.findall(r"(\d+)([A-Z=])", cigar)]


def _int(value: str) -> int | None:
    return int(value) if value not in ("", None) else None


def _core(ops):
    """The CIGAR without its leading S/N and trailing S/N: (leading S, leading N, core ops, trailing S)."""
    ops = list(ops)
    lead_s = ops.pop(0)[0] if ops and ops[0][1] == "S" else 0
    lead_n = ops.pop(0)[0] if ops and ops[0][1] == "N" else 0
    tail_n = ops.pop()[0] if ops and ops[-1][1] == "N" else 0
    tail_s = ops.pop()[0] if ops and ops[-1][1] == "S" else 0
    return lead_s, lead_n, ops, tail_s, tail_n


def _germline_on_read(germline: str, start: int, core) -> str:
    """The germline from 0-based ``start`` laid on the read along ``core``: a germline base under each M, ``-`` under
    each I, and the bases under D (which the read lacks) left out."""
    out, g = [], start
    for n, op in core:
        if op in "M=X":
            out.append(germline[g:g + n])
            g += n
        elif op == "I":
            out.append("-" * n)
        else:
            g += n
    return "".join(out)


def _inserted(start: int, core) -> set[int]:
    """The 0-based read positions of the bases ``core`` (from read position ``start``) inserts."""
    out, r = set(), start
    for n, op in core:
        if op == "I":
            out.update(range(r, r + n))
        if op in "M=XI":
            r += n
    return out


def changeo_germline(row: dict, germlines) -> str:
    """The germline Change-O's CreateGermlines builds for ``row`` (``changeo.Gene.buildGermline``, full): the gapped V
    from ``v_germline_start`` for ``v_germline_end - v_germline_start + 1`` columns, N for ``np1_length``, the D from
    ``d_germline_start`` (none without a D call), N for ``np2_length``, the J from ``j_germline_start``."""
    def part(ref, gene):
        start, end = _int(row[f"{gene}_germline_start"]), _int(row[f"{gene}_germline_end"])
        if start is None or end is None:
            return ""
        return ref[start - 1:end]
    v = part(germlines.v_gapped[row["v_call"].split(",")[0]], "v")
    d = part(germlines.d[row["d_call"].split(",")[0]], "d") if row["d_call"] else ""
    j = part(germlines.j[row["j_call"].split(",")[0]], "j")
    return v + "N" * int(row["np1_length"] or 0) + d + "N" * int(row["np2_length"] or 0) + j


def _germline_errors(row: dict, g: str, cigar: str, lead_n: int, core, germlines) -> list[str]:
    """The germline positions against the CIGAR and the germline alignment: the V's count in the IMGT-gapped V germline
    from its first position, and name ``v_germline_alignment``; the D's and J's count ungapped bases, as the CIGAR's
    N and M/D do, and name the germline laid along the CIGAR in ``{g}_germline_alignment``."""
    gs, ge = _int(row[f"{g}_germline_start"]), _int(row[f"{g}_germline_end"])
    call = row[f"{g}_call"].split(",")[0]
    germ_span = lead_n + sum(n for n, op in core if op in "MD=X")
    if g == "v":
        if gs is None:
            return ["a complete row without V germline positions"] if row["airr_assembly_status"] == "complete" else []
        gapped = germlines.v_gapped[call]
        if gs != 1 or not ge or ge > len(gapped):
            return [f"v_germline_start/end {gs}-{ge}: not the gapped V from its first position"]
        if len(gapped[:ge].replace(".", "")) != germ_span:
            return [f"v_germline_end {ge} (gapped) is not where v_cigar {cigar} ends in the germline"]
        if row["v_germline_alignment"] and gapped[gs - 1:ge] != row["v_germline_alignment"]:
            return ["the gapped V v_germline_start/end name is not v_germline_alignment"]
        return []
    if gs is None or ge is None or lead_n != gs - 1 or germ_span != ge:
        return [f"{g}_cigar {cigar} disagrees with {g}_germline_start/end {gs}-{ge}"]
    if call and row[f"{g}_germline_alignment"]:
        ref = germlines.d[call] if g == "d" else germlines.j[call]
        if _germline_on_read(ref, gs - 1, core) != row[f"{g}_germline_alignment"]:
            return [f"{g}_germline_alignment is not the germline {g}_germline_start/end name"]
    return []


def _segment_errors(row: dict, g: str, germlines, cores: dict) -> list[str]:
    """One gene's read positions and CIGAR: the CIGAR spans the whole query (S, N, the alignment, S) and agrees with
    the positions; its core (no S or N) goes to ``cores``."""
    cigar, length = row[f"{g}_cigar"], len(row["sequence"])
    ss, se = _int(row[f"{g}_sequence_start"]), _int(row[f"{g}_sequence_end"])
    if not cigar:
        return [f"{g}: coordinates without a CIGAR"] if ss is not None else []
    if ss is None or se is None or not 1 <= ss <= se <= length:
        return [f"{g}_sequence_start/end {ss}-{se} not in 1..{length}"]
    ops = _ops(cigar)
    if "".join(f"{n}{op}" for n, op in ops) != cigar or any(op in "HP" for _, op in ops):
        return [f"{g}_cigar {cigar!r} is not an AIRR CIGAR"]
    lead_s, lead_n, core, tail_s, tail_n = _core(ops)
    cores[g] = core
    errors = []
    query = sum(n for n, op in ops if op in "MIS=X")
    if query != length:
        errors.append(f"{g}_cigar {cigar} covers {query} query bases, sequence has {length}")
    if lead_s != ss - 1 or tail_s != length - se:
        errors.append(f"{g}_cigar {cigar} soft clips {lead_s}/{tail_s}, coordinates {ss}-{se} of {length}")
    if sum(n for n, op in core if op in "MI=X") != se - ss + 1:
        errors.append(f"{g}_cigar {cigar} aligns another read span than {ss}-{se}")
    if any(op in "SN" for _, op in core) or tail_n:
        errors.append(f"{g}_cigar {cigar} has S or N inside or a trailing N")
    return errors + _germline_errors(row, g, cigar, lead_n, core, germlines)


def _alignment_errors(row: dict) -> list[str]:
    """Each gene's alignment positions name its part of the sequence and germline alignments."""
    errors, seq_aln, germ_aln = [], row["sequence_alignment"], row["germline_alignment"]
    for g in "vdj":
        s, e = _int(row[f"{g}_alignment_start"]), _int(row[f"{g}_alignment_end"])
        if s is None:
            continue
        if e is None or not 1 <= s <= e <= len(seq_aln):
            errors.append(f"{g}_alignment_start/end {s}-{e} outside sequence_alignment ({len(seq_aln)})")
        elif seq_aln[s - 1:e] != row[f"{g}_sequence_alignment"]:
            errors.append(f"sequence_alignment[{g}_alignment_start-1:{g}_alignment_end] != {g}_sequence_alignment")
        elif row[f"{g}_germline_alignment"] and germ_aln[s - 1:e] != row[f"{g}_germline_alignment"]:
            errors.append(f"germline_alignment[{g}_alignment_start-1:{g}_alignment_end] != {g}_germline_alignment")
    return errors


def _region_errors(row: dict, read: str, inserted: set[int]) -> list[str]:
    """Each FWR/CDR's positions name its read bases in ``sequence``. The FWR1-FWR3 and CDR1-CDR2 bases are the gapped
    alignment's, which leaves a V insertion (a read base in ``inserted``) out; CDR3 and FWR4 are the read's own."""
    errors = []
    for region in REGIONS:
        s, e = _int(row[f"{region}_start"]), _int(row[f"{region}_end"])
        read_bases = re.sub(r"[.\-]", "", row[region]).upper()
        if s is None or e is None:
            if read_bases:
                errors.append(f"{region} holds read bases but no position")
        elif not 1 <= s <= e <= len(read):
            errors.append(f"{region}_start/end {s}-{e} not in 1..{len(read)}")
        else:
            left_out = inserted if region not in ("cdr3", "fwr4") else set()
            if "".join(read[p] for p in range(s - 1, e) if p not in left_out).upper() != read_bases:
                errors.append(f"sequence[{region}_start-1:{region}_end] is not {region} without its gaps")
    return errors


def _junction_errors(row: dict, read: str, cores: dict, germlines) -> list[str]:
    """A complete row's NP regions are the bases between its called genes (between V and J without a D call), N in
    the germline alignment, and, where its D and J align without an indel, CreateGermlines builds its germline
    alignment base for base."""
    errors, germ_aln = [], row["germline_alignment"]
    v_end, j_start = _int(row["v_sequence_end"]), _int(row["j_sequence_start"])
    np1, np2 = row["np1"], row["np2"]
    if row["d_call"]:
        d_start, d_end = _int(row["d_sequence_start"]), _int(row["d_sequence_end"])
        expected = (read[v_end:d_start - 1], read[d_end:j_start - 1])
        np_columns = ((_int(row["v_alignment_end"]), _int(row["d_alignment_start"]) - 1),
                      (_int(row["d_alignment_end"]), _int(row["j_alignment_start"]) - 1))
    else:
        expected = (read[v_end:j_start - 1], "")
        np_columns = ((_int(row["v_alignment_end"]), _int(row["j_alignment_start"]) - 1),)
    if (np1, np2) != expected:
        errors.append(f"np1/np2 {np1!r}/{np2!r} are not the bases between the called genes {expected}")
    if [row["np1_length"], row["np2_length"]] != [str(len(np1)), str(len(np2))]:
        errors.append("np1_length/np2_length are not the lengths of np1/np2")
    if any(set(germ_aln[a:b]) - {"N"} for a, b in np_columns):
        errors.append("germline_alignment holds bases in an untemplated region")
    indel = any(op in "ID" for g in "dj" for _, op in cores.get(g, ()))
    if not indel and changeo_germline(row, germlines) != germ_aln:
        errors.append("CreateGermlines would build another germline than germline_alignment")
    return errors


def _fwr4_errors(row: dict) -> list[str]:
    """FWR4 is IMGT's FR4: it begins with the junction's last codon (the J Trp or Phe, IMGT position 118), the codon
    after the CDR3."""
    junction, fwr4 = row["junction"], row["fwr4"]
    if not junction or not fwr4:
        return []
    errors = []
    if fwr4[:3] != junction[-3:]:
        errors.append(f"fwr4 {fwr4[:6]}... does not begin with the junction's last codon {junction[-3:]}")
    if row["cdr3_end"] and row["fwr4_start"] and int(row["fwr4_start"]) != int(row["cdr3_end"]) + 1:
        errors.append(f"fwr4_start {row['fwr4_start']} is not the position after cdr3_end {row['cdr3_end']}")
    return errors


def conformance_errors(row: dict, germlines) -> list[str]:
    """What in ``row`` (a written AIRR row, as text) breaks the schema's coordinates and CIGARs or the frames chosen
    where it leaves room; empty when nothing does. ``germlines`` holds the reference's gapped V and its D and J
    (``predict.airr.alignment.GermlineMaps``)."""
    read = row["sequence"].translate(_RC)[::-1] if row["rev_comp"] == "T" else row["sequence"]
    cores: dict = {}
    errors = [error for g in "vdj" for error in _segment_errors(row, g, germlines, cores)]
    inserted = _inserted(_int(row["v_sequence_start"]) - 1, cores["v"]) if "v" in cores else set()
    errors += _alignment_errors(row) + _region_errors(row, read, inserted) + _fwr4_errors(row)
    if row["airr_assembly_status"] == "complete":
        errors += _junction_errors(row, read, cores, germlines)
    return errors


def _read_fasta(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith(">")]


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    """The test chip's AIRR table of the example reads and the golden reads, plus the golden reads again with the D
    presence head forced to absent and to short: ``(rows, germline maps, path)``."""
    from tests.alignair.predict.golden_rows import FIXTURE, golden_model
    from tests.alignair.predict.test_d_gate import DPresenceStub, _encoded

    from alignair import Aligner
    from alignair.io.airr import write_airr
    from alignair.predict.airr.alignment import GermlineMaps
    model, reference = golden_model()
    golden = json.loads(FIXTURE.read_text())["reads"]
    reads = _read_fasta(EXAMPLES) + golden
    records = Aligner.from_model(model, reference, device="cpu").predict(reads, batch_size=8).records
    for klass in (0, 1):
        stub = DPresenceStub(model, {_encoded(read): klass for read in golden[:5]})
        records += Aligner.from_model(stub, reference, device="cpu").predict(golden[:5], batch_size=8).records
        reads += golden[:5]
    path = tmp_path_factory.mktemp("airr") / "table.tsv"
    write_airr(str(path), [f"r{i}" for i in range(len(reads))], reads, records, locus="IGH")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows, GermlineMaps.of(reference), path


def test_every_row_follows_the_airr_coordinates_and_cigars(table):
    rows, germlines, _ = table
    failures = {row["sequence_id"]: errors for row in rows if (errors := conformance_errors(row, germlines))}
    assert failures == {}


def test_the_rows_cover_the_cases_the_frames_differ_in(table):
    """Reverse-complemented and cropped reads, V starting past the germline's first base (an N in v_cigar), rows
    without a D call (absent and short), and complete rows CreateGermlines can rebuild."""
    rows, _, _ = table
    complete = [row for row in rows if row["airr_assembly_status"] == "complete"]
    assert any(row["rev_comp"] == "T" for row in complete)
    assert any(row["length_cropped"] == "T" for row in complete)
    assert any(re.match(r"^(\d+S)?\d+N", row["v_cigar"]) for row in complete)
    assert any(not row["d_call"] and row["d_sequence_start"] for row in complete)       # short: its span kept
    assert any(not row["d_call"] and not row["d_sequence_start"] for row in complete)   # absent
    assert len(complete) >= 30


def test_the_table_passes_validate_airr(table):
    from alignair.io.airr_validate import validate_airr_file
    _, _, path = table
    result = validate_airr_file(str(path))
    assert result["missing_columns"] == [] and result["errors"] == []
