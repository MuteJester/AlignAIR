"""IgBLAST on every locus: the J reading-frame file (``-auxiliary_data``) is written from the reference's own J
anchors, the receptor type (``-ig_seqtype``) follows the reference's loci, and a locus without D still gets the D
database ``igblastn`` requires (it searches none on a VJ chain).

The comparison with the release's own human file runs only when ``ALIGNAIR_ASSAY_IGBLAST_DIR`` names an unpacked
IgBLAST release.
"""
import os
from pathlib import Path

import pytest

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import Reference
from alignair_assay.runners.igblast import IgBlastRunner
from alignair_assay.runners.reference import igblast_auxiliary_lines, receptor_type

pytestmark = pytest.mark.component
IGBLAST_DIR = os.environ.get("ALIGNAIR_ASSAY_IGBLAST_DIR")
IGHJ1 = "GCTGAATACTTCCAGCACTGGGGCCAGGGCACCCTGGTCACCGTCTCCTCAG"   # IGHJ1*01: its Trp codon starts at 18


def _reference(v: dict, j: dict, anchors: dict, d: dict | None = None) -> Reference:
    alleles = {"v": v, "d": d, "j": j} if d else {"v": v, "j": j}
    return Reference(alleles=alleles, anchors=anchors)


def _fake_release(directory: Path) -> Path:
    """A stand-in IgBLAST release whose ``igblastn`` prints a version and whose ``makeblastdb`` succeeds."""
    (directory / "bin").mkdir(parents=True)
    (directory / "optional_file").mkdir()
    (directory / "optional_file" / "human_gl.aux").write_text("aux\n")
    for name, body in {"igblastn": 'echo "igblastn: 1.22.0"\n', "makeblastdb": "exit 0\n"}.items():
        path = directory / "bin" / name
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)
    return directory


IGK = _reference({"IGKV1-39*01": "GACATCCAGATGACC"}, {"IGKJ5-CLVV*01": "A" * 38}, {"IGKJ5-CLVV*01": 10})
TRB = _reference({"TRBV2*01": "GAACCTGAAGTC"}, {"TRBJ1-1*01": "A" * 48}, {"TRBJ1-1*01": 16},
                 d={"TRBD1*01": "GGGACAGGGGGC"})


def test_auxiliary_line_is_frame_chain_cdr3_stop_and_bases_past_the_last_codon():
    reference = _reference({"IGHV1-2*02": "CAGGTG"}, {"IGHJ1*01": IGHJ1}, {"IGHJ1*01": 18})
    # IgBLAST's own human file has this line for IGHJ1*01: frame 0, JH, CDR3 ends at 17, 1 base past the last codon.
    assert igblast_auxiliary_lines(reference) == ["IGHJ1*01\t0\tJH\t17\t1"]


def test_auxiliary_chain_type_follows_each_j_locus_and_an_unanchored_j_is_left_out():
    reference = _reference({"IGKV1-39*01": "GAC"},
                           {"IGKJ5-CLVV*01": "A" * 38, "TRAJ31*01": "A" * 57, "TRBJ2-2P*01": "A" * 50},
                           {"IGKJ5-CLVV*01": 10, "TRAJ31*01": 23})
    assert igblast_auxiliary_lines(reference) == ["IGKJ5-CLVV*01\t1\tJK\t9\t1", "TRAJ31*01\t2\tJA\t22\t1"]


def test_auxiliary_refuses_a_j_of_no_known_locus():
    reference = _reference({"IGKV1-39*01": "GAC"}, {"XJ1*01": "A" * 30}, {"XJ1*01": 9})
    with pytest.raises(ValueError, match="XJ1\\*01"):
        igblast_auxiliary_lines(reference)


def test_receptor_type_follows_the_reference_loci():
    assert receptor_type(IGK) == "Ig"
    assert receptor_type(TRB) == "TCR"
    mixed = _reference({"IGKV1-39*01": "GAC", "TRBV2*01": "GAA"}, {"IGKJ5-CLVV*01": "A" * 38}, {})
    with pytest.raises(ValueError, match="both"):
        receptor_type(mixed)


def test_a_locus_without_d_gets_a_placeholder_d_database_that_is_not_an_allele(tmp_path):
    runner = IgBlastRunner(_fake_release(tmp_path / "release"), threads=1)
    provenance = runner.prepare(RunLayout(tmp_path / "run"), IGK)
    database = provenance["database"]
    directory = Path(database["directory"])
    assert [command[-1] for command in database["commands"]] == [str(directory / x) for x in "VDJ"]
    assert database["alleles"] == {"v": 1, "j": 1}
    assert (directory / "germline.aux").read_text() == "IGKJ5-CLVV*01\t1\tJK\t9\t1\n"
    assert {"germline.aux", "placeholder_d.fasta"} <= set(database["files"])


def test_run_searches_with_the_reference_auxiliary_file_and_receptor_type(tmp_path):
    runner = IgBlastRunner(_fake_release(tmp_path / "release"), threads=1)
    database = Path(runner.prepare(RunLayout(tmp_path / "run"), TRB)["database"]["directory"])
    reads = tmp_path / "reads.fasta"
    reads.write_text(">r1\nGAACCTGAAGTC\n")
    result = runner.run(reads, tmp_path / "out.tsv", suite="S0", arm="shared", log_dir=tmp_path / "logs")
    argv = result.commands[-1]
    assert argv[argv.index("-auxiliary_data") + 1] == str(database / "germline.aux")
    assert argv[argv.index("-ig_seqtype") + 1] == "TCR"
    assert argv[argv.index("-germline_db_D") + 1] == str(database / "D")
    assert (database / "germline_d.fasta").is_file() and not (database / "placeholder_d.fasta").exists()


@pytest.mark.skipif(not IGBLAST_DIR, reason="set ALIGNAIR_ASSAY_IGBLAST_DIR to an unpacked IgBLAST release")
def test_igh_frames_and_cdr3_ends_are_the_release_file_s(genairr):
    release = {}
    for line in (Path(IGBLAST_DIR) / "optional_file" / "human_gl.aux").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4 and not fields[0].startswith("#"):
            release[fields[0]] = fields[1:4]
    lines = igblast_auxiliary_lines(Reference.from_dataconfig("HUMAN_IGH_OGRDB"))
    assert len(lines) == 7
    assert all(line.split("\t")[1:4] == release[line.split("\t")[0]] for line in lines)
