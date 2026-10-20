"""Real donors' genotypes (``real.genotype``): the VDJbase loader, the OGRDB IUIS map and the mapping by sequence.

The offline tests build their VDJbase CSV, OGRDB JSON, Supplementary Data 3 sheet and reference in the test; the CSV
header is the real one, copied from a pilot sample's file. The test of the eight pilot donors' real files runs only
when ``ALIGNAIR_ASSAY_DATA`` names the data root (``.../PRJNA555323``).
"""
from __future__ import annotations

import csv
import json
import os
import random
import zipfile
from pathlib import Path

import pytest

from alignair_assay.core.reference import Reference
from alignair_assay.real.genotype import (
    IuisMap,
    RealAllele,
    RealGenotype,
    Sd3Allele,
    filter_d,
    load_vdjbase_csv,
    merge_identical,
    sd3_novel,
    sd3_only,
    upto,
)

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")

# The header of a VDJbase genomic per-sample CSV, copied from P25_I58_S1.csv (study P25).
HEADER = (
    "project", "subject", "sample_name", "haplotype", "contig", "gene", "genotyper_allele", "vdjbase_allele",
    "closest_imgt_allele", "sense", "imgt_allele_match", "closest_imgt_seq", "ref_extension", "notes", "exon_reads",
    "exon_min_score", "gene_sequence", "gene_sequence_CIGAR", "L-PART1", "L-PART1_CIGAR", "V-INTRON",
    "V-INTRON_CIGAR", "L-PART2", "L-PART2_CIGAR", "V-REGION", "V-REGION_CIGAR", "V-REGION-GAPPED",
    "V-REGION-GAPPED_CIGAR", "V-HEPTAMER", "V-HEPTAMER_CIGAR", "V-SPACER", "V-SPACER_CIGAR", "V-NONAMER",
    "V-NONAMER_CIGAR", "V-UTR", "V-UTR_CIGAR", "D-5_NONAMER", "D-5_NONAMER_CIGAR", "D-5_SPACER", "D-5_SPACER_CIGAR",
    "D-5_HEPTAMER", "D-5_HEPTAMER_CIGAR", "D-REGION", "D-REGION_CIGAR", "D-3_HEPTAMER", "D-3_HEPTAMER_CIGAR",
    "D-3_SPACER", "D-3_SPACER_CIGAR", "D-3_NONAMER", "D-3_NONAMER_CIGAR", "J-NONAMER", "J-NONAMER_CIGAR",
    "J-SPACER", "J-SPACER_CIGAR", "J-HEPTAMER", "J-HEPTAMER_CIGAR", "J-REGION", "J-REGION_CIGAR", "C-REGION",
    "C-REGION_CIGAR", "C-EXON_1", "C-EXON_1_CIGAR", "REGION_start", "REGION_end", "Total_Positions",
    "Average_Coverage", "Mismatched_Positions", "Matched_Positions", "Position_Mismatches", "Position_Matches",
    "Percent_Accuracy", "Positions_With_At_Least_10x_Coverage", "Fully_Spanning_Reads",
    "Fully_Spanning_Reads_100%_Match",
)
DONOR = "SC-23"
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _random(n: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=n))


def _snp(sequence: str, position: int) -> str:
    base = "A" if sequence[position] != "A" else "C"
    return sequence[:position] + base + sequence[position + 1:]


def _gap(sequence: str) -> str:
    """An IMGT-style gapped form: gap columns after the 27th and the 69th base."""
    return sequence[:27] + "..." + sequence[27:69] + "......" + sequence[69:]


def _rc(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


V_A, V_B, V_U, V_C = (_random(296, seed) for seed in (1, 2, 3, 4))
D_44, D_33, D_X = _random(16, 5), _random(31, 6), _random(19, 7)
J_4 = _random(48, 8)
# The shared reference names V alleles by OGRDB F/G labels, which carry no IUIS name: mapping is by sequence only.
REFERENCE = Reference(
    alleles={"v": {"IGHVF1-G1*01": V_A, "IGHVF1-G2*01": V_B, "IGHVF9-G9*01": V_U},
             "d": {"IGHD4-4*01": D_44, "IGHD4-11*01": D_44, "IGHD3-3*01": D_33},
             "j": {"IGHJ4*02": J_4}},
    anchors={})
OGRDB = {"GermlineSet": [{
    "germline_set_id": "OGRDB:G00060", "germline_set_name": "IGH_VDJ", "release_version": 8,
    "release_date": "2024-01-11", "germline_set_ref": "OGRDB:IGH_VDJ.8", "locus": "IGH",
    "allele_descriptions": [
        {"label": "IGHV1-2*02", "coding_sequence": V_A, "sequence_type": "V", "functional": True},
        {"label": "IGHV1-69D*01", "coding_sequence": V_B, "sequence_type": "V", "functional": True},
        {"label": "IGHV1-69*01", "coding_sequence": V_B, "sequence_type": "V", "functional": True},
        # One base longer than the shared IGHVF9-G9*01: another sequence, so that allele stays unmapped.
        {"label": "IGHV3-13*06", "coding_sequence": V_U + "A", "sequence_type": "V", "functional": True},
        {"label": "IGHD4-4*01", "coding_sequence": D_44, "sequence_type": "D", "functional": True},
        {"label": "IGHD4-11*01", "coding_sequence": D_44, "sequence_type": "D", "functional": True},
        {"label": "IGHD3-3*01", "coding_sequence": D_33, "sequence_type": "D", "functional": True},
        {"label": "IGHJ4*02", "coding_sequence": J_4, "sequence_type": "J", "functional": True},
    ]}]}
# Six rows, as VDJbase writes them: one per haplotype per gene, haplotype 0 = unphased.
ROWS = [
    {"haplotype": "1", "gene": "IGHV1-2", "vdjbase_allele": "IGHV1-2*02", "imgt_allele_match": "100",
     "V-REGION": V_A, "V-REGION-GAPPED": _gap(V_A)},
    {"haplotype": "2", "gene": "IGHV1-2", "vdjbase_allele": "IGHV1-2*02_a41c", "imgt_allele_match": "99.66",
     "V-REGION": _snp(V_A, 40), "V-REGION-GAPPED": _gap(_snp(V_A, 40))},
    {"haplotype": "0", "gene": "IGHV1-69D", "vdjbase_allele": "IGHV1-69D*01", "imgt_allele_match": "100",
     "V-REGION": V_B, "V-REGION-GAPPED": _gap(V_B)},
    {"haplotype": "1", "gene": "IGHV3-62", "vdjbase_allele": "IGHV3-62*03", "imgt_allele_match": "100",
     "notes": "Stop codon in V-REGION before 2nd cysteine", "V-REGION": V_C, "V-REGION-GAPPED": _gap(V_C)},
    {"haplotype": "0", "gene": "IGHD4-11", "vdjbase_allele": "IGHD4-11*01", "imgt_allele_match": "100",
     "D-REGION": D_44},
    {"haplotype": "1", "gene": "IGHD3-3", "vdjbase_allele": "IGHD3-3*01_g8-_c18a", "imgt_allele_match": "56.25",
     "notes": "3 Heptamer does not match reference", "D-REGION": D_X},
]


def _write_csv(path: Path, rows, *, subject: str = DONOR, header=HEADER) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({"project": "PMID37479682", "subject": subject, "sample_name": subject, "sense": "-",
                             **{k: v for k, v in row.items() if k in header}})
    return path


@pytest.fixture()
def iuis(tmp_path) -> IuisMap:
    path = tmp_path / "ogrdb.json"
    path.write_text(json.dumps(OGRDB), encoding="utf-8")
    return IuisMap.from_ogrdb(path, REFERENCE)


@pytest.fixture()
def genotype(tmp_path, iuis) -> RealGenotype:
    return load_vdjbase_csv(_write_csv(tmp_path / "P25_I58_S1.csv", ROWS), DONOR, REFERENCE, iuis)


def _allele(genotype: RealGenotype, allele_id: str) -> RealAllele:
    (found,) = [allele for allele in genotype.alleles if allele.allele_id == allele_id]
    return found


# --- mapping by sequence ----------------------------------------------------------------------------------------

def test_shared_names_by_sequence(genotype, tmp_path, iuis):
    exact = _allele(genotype, "IGHV1-2*02")
    assert exact.shared == ("IGHVF1-G1*01",) and exact.iuis == ("IGHV1-2*02",)
    assert exact.segment == "v" and exact.gene == "IGHV1-2" and exact.sequence == V_A and exact.gapped == _gap(V_A)
    variant = _allele(genotype, "IGHV1-2*02_a41c")
    assert variant.shared == () and variant.iuis == ()
    assert _allele(genotype, "IGHD4-11*01").shared == ("IGHD4-11*01", "IGHD4-4*01")
    assert _allele(genotype, "IGHD4-11*01").gapped is None
    # A name is never matched: an allele named like a shared allele, with another sequence, maps to nothing.
    named = load_vdjbase_csv(_write_csv(tmp_path / "named.csv", [
        {"haplotype": "1", "gene": "IGHD3-3", "vdjbase_allele": "IGHD3-3*01", "imgt_allele_match": "100",
         "D-REGION": _snp(D_33, 3)}]), DONOR, REFERENCE, iuis)
    assert named.alleles[0].shared == () and named.alleles[0].iuis == ()


def test_iuis_map_paralogs(iuis, genotype):
    assert iuis.labels("IGHVF1-G2*01") == ("IGHV1-69*01", "IGHV1-69D*01")
    assert iuis.genes("IGHVF1-G2*01") == ("IGHV1-69", "IGHV1-69D")
    assert iuis.labels("IGHVF1-G1*01") == ("IGHV1-2*02",) and iuis.genes("IGHVF1-G1*01") == ("IGHV1-2",)
    assert iuis.labels("IGHD4-4*01") == ("IGHD4-11*01", "IGHD4-4*01")
    assert iuis.shared_for_sequence(V_B) == ("IGHVF1-G2*01",) and iuis.shared_for_sequence(V_B.lower()) == (
        "IGHVF1-G2*01",)
    assert iuis.shared_for_sequence(D_44) == ("IGHD4-11*01", "IGHD4-4*01")
    assert iuis.shared_for_sequence(_snp(V_B, 9)) == ()
    assert iuis.labels_for_sequence(V_U + "A") == ("IGHV3-13*06",)
    # One base shorter than OGRDB's IGHV3-13*06: a different sequence, never a near match.
    assert iuis.unmapped() == ("IGHVF9-G9*01",) and iuis.labels("IGHVF9-G9*01") == ()
    with pytest.raises(KeyError):
        iuis.labels("IGHV1-69*01")
    paralog = _allele(genotype, "IGHV1-69D*01")
    assert paralog.shared == ("IGHVF1-G2*01",) and paralog.iuis == ("IGHV1-69*01", "IGHV1-69D*01")
    assert iuis.source["release_version"] == 8 and iuis.source["germline_set_id"] == "OGRDB:G00060"
    assert iuis.source["reference_digest"] == REFERENCE.digest() and len(iuis.source["sha256"]) == 64


def test_absent_genes(genotype):
    # IGHV1-2 carries an allele absent from the shared reference; IGHV3-62's only allele is absent; the low-identity
    # D call was dropped, so it makes no gene absent.
    assert genotype.absent_genes() == {"IGHV1-2", "IGHV3-62"}
    assert genotype.genes("v") == {"IGHV1-2", "IGHV1-69D", "IGHV3-62"}
    assert genotype.genes("d") == {"IGHD4-11"} and genotype.genes("j") == set()
    assert [a.allele_id for a in genotype.carried("IGHV1-2")] == ["IGHV1-2*02", "IGHV1-2*02_a41c"]
    assert genotype.carried("IGHV4-34") == ()
    assert genotype.captured is None and genotype.donor == DONOR


def test_functional_follows_the_v_region_notes(genotype):
    assert _allele(genotype, "IGHV3-62*03").functional is False
    assert _allele(genotype, "IGHV1-2*02").functional is True
    assert _allele(genotype, "IGHD4-11*01").functional is True


def test_unphased_is_not_homozygous(genotype):
    (unphased,) = genotype.carried("IGHV1-69D")
    assert unphased.haplotype == 0
    assert [a.haplotype for a in genotype.carried("IGHV1-2")] == [1, 2]
    assert _allele(genotype, "IGHD4-11*01").haplotype == 0


def test_the_loader_drops_noisy_d_calls_and_records_them(genotype):
    assert all(a.allele_id != "IGHD3-3*01_g8-_c18a" for a in genotype.alleles)
    assert genotype.source["d_dropped"] == [
        ["IGHD3-3*01_g8-_c18a", "gap-indel token g8-; 56.25% identity to the closest IMGT allele, under 90%"]]
    assert genotype.source["donor"] == DONOR and genotype.source["rows"] == 6
    assert genotype.source["ogrdb"]["release_version"] == 8 and len(genotype.source["sha256"]) == 64


def test_the_loader_refuses_inconsistent_inputs(tmp_path, iuis):
    path = _write_csv(tmp_path / "other.csv", ROWS[:1], subject="W-97")
    with pytest.raises(ValueError, match="W-97.*not SC-23|subject W-97"):
        load_vdjbase_csv(path, DONOR, REFERENCE, iuis)
    other = Reference(alleles={**REFERENCE.alleles, "j": {"IGHJ6*02": _random(60, 9)}}, anchors={})
    with pytest.raises(ValueError, match="another reference"):
        load_vdjbase_csv(_write_csv(tmp_path / "a.csv", ROWS), DONOR, other, iuis)
    short = [name for name in HEADER if name != "V-REGION-GAPPED"]
    with pytest.raises(ValueError, match="V-REGION-GAPPED"):
        load_vdjbase_csv(_write_csv(tmp_path / "b.csv", ROWS, header=short), DONOR, REFERENCE, iuis)
    with pytest.raises(ValueError, match="no V-REGION"):
        load_vdjbase_csv(_write_csv(tmp_path / "c.csv", [{**ROWS[0], "V-REGION": ""}]), DONOR, REFERENCE, iuis)
    with pytest.raises(ValueError, match="haplotype"):
        load_vdjbase_csv(_write_csv(tmp_path / "d.csv", [{**ROWS[0], "haplotype": "3"}]), DONOR, REFERENCE, iuis)


def test_a_gapped_sequence_that_is_not_the_allele_is_left_out(tmp_path, iuis):
    # VDJbase marks a deleted base with "-" in the gapped form; both "." and "-" are gaps.
    deleted = V_A[:10] + V_A[11:]
    rows = [{**ROWS[0], "vdjbase_allele": "IGHV1-2*02_g11-", "V-REGION": deleted,
             "V-REGION-GAPPED": _gap(V_A[:10] + "-" + V_A[11:])},
            {**ROWS[1], "V-REGION-GAPPED": _gap(V_A)}]
    genotype = load_vdjbase_csv(_write_csv(tmp_path / "g.csv", rows), DONOR, REFERENCE, iuis)
    assert _allele(genotype, "IGHV1-2*02_g11-").gapped == _gap(V_A[:10] + "-" + V_A[11:])
    assert _allele(genotype, "IGHV1-2*02_a41c").gapped is None
    assert genotype.source["gapped_mismatch"] == ["IGHV1-2*02_a41c"]


def test_identical_rows_are_one_allele(tmp_path, iuis):
    # Two contigs on one haplotype can carry the same allele: one record.
    rows = [{**ROWS[4], "contig": "c=igh:1-6370_h=0"}, {**ROWS[4], "contig": "c=igh:9-9999_h=0"}]
    genotype = load_vdjbase_csv(_write_csv(tmp_path / "dup.csv", rows), DONOR, REFERENCE, iuis)
    assert len(genotype.alleles) == 1 and genotype.source["duplicate_rows"] == 1


# --- the D-call filter ------------------------------------------------------------------------------------------

def _d(allele_id: str, identity: float | None, segment: str = "d") -> RealAllele:
    return RealAllele(allele_id=allele_id, gene=allele_id.split("*")[0], segment=segment, sequence=D_33, gapped=None,
                      haplotype=1, functional=True, shared=(), iuis=(), imgt_identity=identity)


def test_filter_d_drops_gap_indel_and_low_identity():
    alleles = [
        _d("IGHD4-23*01_g8-_g9-_t10-_c18a", 95.0),          # single-base deletions
        _d("IGHD4-23*01_8---a11_c18a", 95.0),                # a block with gaps
        _d("IGHD2-8*01_c12g_i15ac", 95.0),                   # an insertion
        _d("IGHD2-8*01_c12g_a15g_17accagc22_a29g", 70.97),   # a same-length block: substitutions, low identity
        _d("IGHD5-24*01_a3g_g6t", 90.0),                     # exactly 90%: kept
        _d("IGHD1-1*01", None),                              # no identity recorded: kept
        _d("IGHD3-22*01_a14t_c28-", 83.33),                  # both reasons
        _d("IGHV1-24*01_g22-", 50.0, segment="v"),           # not a D call: never filtered here
    ]
    kept, dropped = filter_d(alleles)
    assert [a.allele_id for a in kept] == ["IGHD5-24*01_a3g_g6t", "IGHD1-1*01", "IGHV1-24*01_g22-"]
    assert dropped == [
        ("IGHD4-23*01_g8-_g9-_t10-_c18a", "gap-indel tokens g8-, g9-, t10-"),
        ("IGHD4-23*01_8---a11_c18a", "gap-indel token 8---a11"),
        ("IGHD2-8*01_c12g_i15ac", "gap-indel token i15ac"),
        ("IGHD2-8*01_c12g_a15g_17accagc22_a29g", "70.97% identity to the closest IMGT allele, under 90%"),
        ("IGHD3-22*01_a14t_c28-", "gap-indel token c28-; 83.33% identity to the closest IMGT allele, under 90%"),
    ]


# --- identical alleles and IMGT position 318 ----------------------------------------------------------------------

def test_merge_identical_groups_d4_4_and_d4_11():
    pair = Reference(alleles={"d": {"IGHD4-4*01": D_44, "IGHD4-11*01": D_44}}, anchors={})
    both = frozenset({"IGHD4-4*01", "IGHD4-11*01"})
    assert merge_identical(pair) == {"IGHD4-4*01": both, "IGHD4-11*01": both}
    merged = merge_identical(REFERENCE)
    assert merged["IGHD4-11*01"] == both and merged["IGHD3-3*01"] == frozenset({"IGHD3-3*01"})
    assert set(merged) == set(REFERENCE.sequences())


def test_upto_318():
    gapped = _gap(_random(311, 10))
    assert len(gapped) == 320
    at_319 = gapped[:318] + ("A" if gapped[318] != "A" else "C") + gapped[319:]
    assert upto(gapped) == upto(at_319) == gapped[:318].replace(".", "")
    at_318 = gapped[:317] + ("A" if gapped[317] != "A" else "C") + gapped[318:]
    assert upto(gapped) != upto(at_318)
    assert upto("ac..g-t", 5) == "ACG" and upto(gapped, 400) == gapped.replace(".", "")
    with pytest.raises(ValueError):
        upto(gapped, 0)


# --- the paper's novel alleles (Supplementary Data 3) -----------------------------------------------------------

SD3_HEADER = ["Allele name", "closest IMGT allele", "IMGT percent identity", "Number of samples with allele sequence",
              "Number of HiFi reads with exact matches to allele sequence", "BLAT confirmed", "IGV confirmed",
              "Sample count of allele and RS sequence", "Gene", "Allele sequence", "RS sequence", "Nonamer", "Spacer",
              "Heptamer"]
NOVEL_V = _snp(V_B, 100)
SD3_ROWS = [
    ["gene=IGHV1-2_allele=Novel-141_function=.", "Homsap IGHV1-2*02 F", "99.65", "1", "17", "TRUE", "Yes", "1",
     "IGHV1-2", _rc(_snp(V_A, 40)), "GGTTTCTGA", "GGTTTCTGA", "CACCCTCAGG", "CACTGTG"],
    ["gene=IGHV1-69D_allele=Novel-503_function=.", "Homsap IGHV1-69*01 F, or Homsap IGHV1-69D*01 F", "99.6", "2",
     "30", "TRUE", "Yes", "2", "IGHV1-69D", _rc(NOVEL_V), "", "", "", ""],
    ["", "", ""],
    ["gene=IGHD4-23_allele=Novel-792_function=.", "NA", "NA", "1", "23", "TRUE", "Yes", "1", "IGHD4-23", _rc(D_X),
     "", "", "", ""],
]
SD3_EXPECTED = [
    Sd3Allele("IGHV1-2*02", _snp(V_A, 40), "IGHV1-2", ("IGHV1-2*02",)),
    Sd3Allele("IGHV1-69*01", NOVEL_V, "IGHV1-69D", ("IGHV1-69*01", "IGHV1-69D*01")),
    Sd3Allele("", D_X, "IGHD4-23", ()),
]


def _write_sd3_tsv(path: Path) -> Path:
    path.write_text("\n".join("\t".join(row) for row in [SD3_HEADER, *SD3_ROWS]) + "\n", encoding="utf-8")
    return path


def _column(index: int) -> str:
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]


def _write_xlsx(path: Path, sheets: dict[str, list[list[str]]]) -> Path:
    """A minimal workbook as Excel writes one: each sheet's strings in ``sharedStrings.xml``."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package = "http://schemas.openxmlformats.org/package/2006/relationships"
    strings: list[str] = []
    with zipfile.ZipFile(path, "w") as archive:
        entries, relations = [], []
        for number, (name, rows) in enumerate(sheets.items(), start=1):
            entries.append(f'<sheet name="{name}" sheetId="{number}" r:id="rId{number}"/>')
            relations.append(f'<Relationship Id="rId{number}" Type="{rel}/worksheet" '
                             f'Target="worksheets/sheet{number}.xml"/>')
            cells = []
            for r, row in enumerate(rows, start=1):
                values = []
                for c, value in enumerate(row):
                    if value:
                        strings.append(value)
                        values.append(f'<c r="{_column(c)}{r}" t="s"><v>{len(strings) - 1}</v></c>')
                cells.append(f'<row r="{r}">{"".join(values)}</row>')
            archive.writestr(f"xl/worksheets/sheet{number}.xml",
                             f'<worksheet xmlns="{main}"><sheetData>{"".join(cells)}</sheetData></worksheet>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>{"".join(entries)}'
                                            f'</sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{package}">{"".join(relations)}'
                                                       f'</Relationships>')
        archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{main}">'
                                                 + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    return path


def test_sd3_reverse_complement_corrected(tmp_path):
    novel = sd3_novel(_write_sd3_tsv(tmp_path / "sd3.tsv"))
    assert novel == SD3_EXPECTED
    closest, sequence = novel[1][0], novel[1][1]
    assert (closest, sequence) == ("IGHV1-69*01", NOVEL_V) and novel[1].gene == "IGHV1-69D"


def test_sd3_reads_the_supplementary_workbook(tmp_path):
    path = _write_xlsx(tmp_path / "suppl.xlsx", {"Table index": [["Table", "Title"]],
                                                 "Supplementary Data 3": [SD3_HEADER, *SD3_ROWS]})
    assert sd3_novel(path) == SD3_EXPECTED
    without = _write_xlsx(tmp_path / "other.xlsx", {"Table index": [["Table", "Title"]]})
    with pytest.raises(ValueError, match="Supplementary Data 3"):
        sd3_novel(without)


def test_sd3_only_touches_the_gene_and_every_closest_gene(genotype):
    # The IGHV1-2 novel allele is carried by the donor (found); the others are in no donor's VDJbase alleles.
    assert sd3_only([genotype], SD3_EXPECTED) == {"IGHV1-69D", "IGHV1-69", "IGHD4-23"}
    assert sd3_only([], SD3_EXPECTED[:1]) == {"IGHV1-2"}
    assert sd3_only([genotype], []) == set()


# --- the pilot donors' real files -------------------------------------------------------------------------------

PILOT = {"SC-23": "P25_I58_S1", "SC-4": "P25_I61_S1", "SC-11": "P25_I48_S1", "SC-6": "P25_I62_S1",
         "W-97": "P25_I151_S1", "W-89": "P25_I145_S1", "W-133": "P25_I93_S1", "W-131": "P25_I91_S1"}
# V genes per donor in VDJbase's genotype API (the recon), which lists only alleles with no V-REGION defect note.
API_V_GENES = {"SC-23": 47, "SC-4": 49, "SC-11": 50, "SC-6": 49, "W-97": 39, "W-89": 53, "W-133": 40, "W-131": 48}
# Every V gene in the per-sample CSV, functional or not.
CSV_V_GENES = {"SC-23": 51, "SC-4": 53, "SC-11": 54, "SC-6": 52, "W-97": 42, "W-89": 57, "W-133": 44, "W-131": 52}


@pytest.mark.skipif(not DATA, reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root")
def test_the_pilot_donors_vdjbase_genotypes(reference):
    root = Path(DATA)
    iuis = IuisMap.from_ogrdb(root / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)
    assert iuis.unmapped() == ("IGHVF10-G33*01", "IGHVF10-G42*01")
    assert sum(len(iuis.labels(name)) > 1 for name in reference.sequences()) == 11
    assert iuis.labels("IGHVF6-G21*06") == ("IGHV1-69*01", "IGHV1-69D*01")
    merged = merge_identical(reference)
    assert sorted(sorted(group) for group in set(merged.values()) if len(group) > 1) == [
        ["IGHD4-11*01", "IGHD4-4*01"], ["IGHD5-18*01", "IGHD5-5*01"]]

    genotypes = {donor: load_vdjbase_csv(root / "genotypes/vdjbase_genomic_samples" / f"{sample}.csv", donor,
                                         reference, iuis) for donor, sample in PILOT.items()}
    functional = {donor: len({a.gene for a in g.alleles if a.segment == "v" and a.functional})
                  for donor, g in genotypes.items()}
    assert functional == API_V_GENES
    assert {donor: len(g.genes("v")) for donor, g in genotypes.items()} == CSV_V_GENES
    # No D or J truth for these two donors: their genotype arm must keep the shared D and J alleles.
    for donor in ("SC-23", "W-97"):
        assert genotypes[donor].genes("d") == set() and genotypes[donor].genes("j") == set()
    # The recon's exact matches against the shared reference, per distinct (donor, sequence): V 460 of 517 and
    # J 6 of 6 (its D count, 90 of 112, is before the D filter).
    v = {(d, a.sequence): bool(a.shared) for d, g in genotypes.items() for a in g.alleles if a.segment == "v"}
    assert (len(v), sum(v.values())) == (517, 460)
    j = {(d, a.sequence): bool(a.shared) for d, g in genotypes.items() for a in g.alleles if a.segment == "j"}
    assert (len(j), sum(j.values())) == (6, 6)
    for g in genotypes.values():
        assert all(a.gapped is not None for a in g.alleles if a.segment == "v")
        assert g.source["gapped_mismatch"] == []

    novel = sd3_novel(root / "genotypes/rodriguez2023_suppl/41467_2023_40070_MOESM4_ESM.xlsx")
    assert len(novel) == 137 and sum(item.gene.startswith("IGHV") for item in novel) == 132
    assert all(set(item.sequence) <= set("ACGT") for item in novel)
    touched = sd3_only(genotypes.values(), novel)
    assert touched and all(gene.startswith(("IGHV", "IGHD")) for gene in touched)
