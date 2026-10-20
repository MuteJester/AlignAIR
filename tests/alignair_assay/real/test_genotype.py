"""Real donors' genotypes (``real.genotype``): the VDJbase loader, the OGRDB IUIS map, the mapping by sequence and
the genotype arm's alleles.

The offline tests build their VDJbase CSV, OGRDB JSON, Supplementary Data 3 sheet and reference in the test; the CSV
header is the real one, copied from a pilot sample's file. The test of the eight pilot donors' real files runs only
when ``ALIGNAIR_ASSAY_DATA`` names the data root (``.../PRJNA555323``); their genotype arms also need
``ALIGNAIR_ASSAY_DOCKER=1`` (the captured genes come from the contig BAMs, read in the samtools container).
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from alignair_assay.core.reference import Reference
from alignair_assay.real.genotype import (
    IuisMap,
    RealAllele,
    RealGenotype,
    Sd3Allele,
    arm_alleles,
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
        # One base longer than the shared IGHVF9-G9*01: another sequence, and with no IMGT-gapped form to place
        # position 318 on either, that allele stays unmapped.
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


def _ogrdb_v(label: str, coding: str, aligned: str | None = None) -> dict:
    """An OGRDB V allele description, with its IMGT delineation when ``aligned`` is given."""
    description = {"label": label, "coding_sequence": coding, "sequence_type": "V", "functional": True}
    if aligned is not None:
        description["v_gene_delineations"] = [{"delineation_scheme": "IMGT", "aligned_sequence": aligned}]
    return description


def test_an_unlabelled_shared_allele_takes_the_iuis_genes_it_equals_through_318(tmp_path):
    # 311-nt V alleles gapped to 320 columns: IMGT position 318 is ungapped base 309
    long_13, long_28, long_15, long_7 = (_random(311, seed) for seed in (11, 12, 13, 14))
    assert len(_gap(long_13)) == 320 and len(upto(_gap(long_13))) == 309
    # one base short of OGRDB's allele, as IGHVF10-G33*01 is of IGHV3-13*06: a 3' prefix ending past 318
    prefix = long_13[:-1]
    # the last base differs: equal through 318 (the shared allele's own gapped form places it)
    beyond = long_28[:-1] + ("A" if long_28[-1] != "A" else "C")
    short = long_15[:250]  # a prefix ending before 318: the bases past it are unknown
    early = _snp(long_7, 100)  # differs at IMGT ~110
    reference = Reference(alleles={"v": {"IGHVF10-G33*01": prefix, "IGHVF3-G9*09": beyond, "IGHVF9-G1*01": short,
                                         "IGHVF7-G1*01": early, "IGHVF1-G1*01": V_A}},
                          anchors={}, gapped_v={"IGHVF3-G9*09": _gap(beyond), "IGHVF7-G1*01": _gap(early)})
    ogrdb = {"GermlineSet": [{**OGRDB["GermlineSet"][0], "allele_descriptions": [
        _ogrdb_v("IGHV3-13*06", long_13, _gap(long_13)), _ogrdb_v("IGHV4-28*01", long_28, _gap(long_28)),
        _ogrdb_v("IGHV3-15*01", long_15, _gap(long_15)), _ogrdb_v("IGHV3-7*01", long_7, _gap(long_7)),
        _ogrdb_v("IGHV1-2*02", V_A)]}]}
    path = tmp_path / "ogrdb.json"
    path.write_text(json.dumps(ogrdb), encoding="utf-8")
    iuis = IuisMap.from_ogrdb(path, reference)
    assert iuis.labels("IGHVF10-G33*01") == () and iuis.inferred("IGHVF10-G33*01") == ("IGHV3-13*06",)
    assert iuis.genes("IGHVF10-G33*01") == ("IGHV3-13",)
    assert iuis.inferred("IGHVF3-G9*09") == ("IGHV4-28*01",) and iuis.genes("IGHVF3-G9*09") == ("IGHV4-28",)
    # a labelled allele keeps its exact labels only
    assert iuis.labels("IGHVF1-G1*01") == ("IGHV1-2*02",) and iuis.inferred("IGHVF1-G1*01") == ()
    assert iuis.unmapped() == ("IGHVF7-G1*01", "IGHVF9-G1*01")
    assert iuis.genes("IGHVF9-G1*01") == () and iuis.genes("IGHVF7-G1*01") == ()
    assert iuis.source["inferred"] == {"IGHVF10-G33*01": ["IGHV3-13*06"], "IGHVF3-G9*09": ["IGHV4-28*01"]}
    with pytest.raises(KeyError):
        iuis.inferred("IGHV3-13*06")


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


def test_the_dropped_d_calls_stay_on_the_genotype(genotype):
    # a gene whose only calls were dropped has its truth dropped: it is not a gene the donor lacks
    (dropped,) = genotype.dropped
    assert dropped.allele_id == "IGHD3-3*01_g8-_c18a" and dropped.gene == "IGHD3-3" and dropped.segment == "d"
    assert genotype.dropped_genes() == {"IGHD3-3"} and "IGHD3-3" not in genotype.genes("d")
    # a gene with a kept call as well is carried, not dropped
    kept = RealAllele(allele_id="IGHD3-3*01", gene="IGHD3-3", segment="d", sequence=D_33, gapped=None, haplotype=2,
                      functional=True, shared=("IGHD3-3*01",), iuis=("IGHD3-3*01",))
    both = RealGenotype(donor=DONOR, alleles=(*genotype.alleles, kept), dropped=genotype.dropped)
    assert both.dropped_genes() == set() and "IGHD3-3" in both.genes("d")
    assert RealGenotype(donor=DONOR, alleles=genotype.alleles).dropped_genes() == set()


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


def test_sd3_only_falls_back_to_the_closest_allele_without_closest_all():
    # ``closest_all`` defaults to (): a record built with only ``closest`` still touches the closest allele's gene.
    assert sd3_only([], [Sd3Allele("IGHV1-69*01", NOVEL_V, "IGHV1-69D")]) == {"IGHV1-69D", "IGHV1-69"}
    assert sd3_only([], [Sd3Allele("", D_X, "IGHD4-23")]) == {"IGHD4-23"}


# --- the genotype arm's alleles -----------------------------------------------------------------------------------

# A shared reference for the genotype arm: shared name -> (sequence, OGRDB labels with that sequence).
ARM_V = {
    "IGHVF1-G1*01": (_random(296, 21), ("IGHV1-2*02",)),
    "IGHVF1-G1*02": (_random(296, 22), ("IGHV1-2*04",)),
    "IGHVF6-G21*06": (V_B, ("IGHV1-69*01", "IGHV1-69D*01")),     # an identical-sequence paralog pair
    "IGHVF6-G21*07": (_random(296, 23), ("IGHV1-69*06", "IGHV1-69D*06")),
    "IGHVF6-G21*02": (_random(296, 24), ("IGHV1-69*02",)),
    "IGHVF3-G23*01": (_random(296, 25), ("IGHV3-23*01", "IGHV3-23D*01")),
    "IGHVF3-G23*04": (_random(296, 26), ("IGHV3-23*04",)),
    "IGHVF3-G23*02": (_random(296, 27), ("IGHV3-23D*02",)),
    "IGHVF5-G5*01": (_random(296, 28), ("IGHV3-30*18", "IGHV3-30-5*01")),
    "IGHVF5-G5*02": (_random(296, 29), ("IGHV3-30-5*02",)),
    "IGHVF5-G5*03": (_random(296, 30), ("IGHV3-30-5*03",)),
    "IGHVF3-G62*03": (_random(296, 31), ("IGHV3-62*03",)),
    "IGHVF3-G11*01": (_random(296, 32), ("IGHV3-11*01",)),
    "IGHVF4-G34*01": (_random(296, 33), ("IGHV4-34*01",)),
    "IGHVF9-G9*01": (V_U, ()),                                   # no label: no IUIS gene
}
ARM_REFERENCE = Reference(
    alleles={"v": {name: sequence for name, (sequence, _) in ARM_V.items()},
             "d": {"IGHD4-4*01": D_44, "IGHD4-11*01": D_44, "IGHD3-3*01": D_33},
             "j": {"IGHJ4*02": J_4, "IGHJ6*02": _random(63, 34)}},
    anchors={})


@pytest.fixture()
def arm_iuis(tmp_path) -> IuisMap:
    descriptions = [{"label": label, "coding_sequence": sequence, "sequence_type": "V", "functional": True}
                    for sequence, labels in ARM_V.values() for label in labels]
    descriptions += [{"label": name, "coding_sequence": sequence, "sequence_type": letter.upper(), "functional": True}
                     for letter in ("d", "j") for name, sequence in ARM_REFERENCE.alleles[letter].items()]
    path = tmp_path / "arm_ogrdb.json"
    path.write_text(json.dumps({"GermlineSet": [{**OGRDB["GermlineSet"][0], "allele_descriptions": descriptions}]}),
                    encoding="utf-8")
    return IuisMap.from_ogrdb(path, ARM_REFERENCE)


def _carried(iuis: IuisMap, allele_id: str, sequence: str, *, segment: str = "v", functional: bool = True,
             haplotype: int = 1) -> RealAllele:
    """A donor allele as the loader maps it: its shared and IUIS names by exact sequence."""
    return RealAllele(allele_id=allele_id, gene=allele_id.split("*")[0], segment=segment, sequence=sequence,
                      gapped=None, haplotype=haplotype, functional=functional,
                      shared=iuis.shared_for_sequence(sequence), iuis=iuis.labels_for_sequence(sequence))


def _arm_donor(iuis: IuisMap, *, d_truth: bool = False) -> RealGenotype:
    """A donor with every V case the arm rule tells apart; ``d_truth`` adds a D allele."""
    seq = {name: sequence for name, (sequence, _) in ARM_V.items()}
    alleles = [
        _carried(iuis, "IGHV1-2*02", seq["IGHVF1-G1*01"]),
        _carried(iuis, "IGHV1-2*02_a41c", _snp(seq["IGHVF1-G1*01"], 40), haplotype=2),  # absent: a shared one too
        _carried(iuis, "IGHV1-69D*01", V_B, haplotype=0),
        _carried(iuis, "IGHV3-23*01_t100a", _snp(seq["IGHVF3-G23*01"], 99)),            # its gene's only allele
        _carried(iuis, "IGHV3-30-5*02", seq["IGHVF5-G5*02"]),
        _carried(iuis, "IGHV3-62*03", seq["IGHVF3-G62*03"], functional=False),
    ]
    if d_truth:
        alleles.append(_carried(iuis, "IGHD4-11*01", D_44, segment="d", haplotype=0))
    # IGHV3-11 is captured and not carried (a deletion); IGHV3-30 and IGHV4-34 are not captured.
    captured = frozenset({"IGHV1-2", "IGHV1-69", "IGHV1-69D", "IGHV3-23", "IGHV3-23D", "IGHV3-30-5", "IGHV3-62",
                          "IGHV3-11", *(("IGHD4-11",) if d_truth else ())})
    return RealGenotype(donor="W-97", alleles=tuple(alleles), captured=captured)


def test_arm_alleles_keeps_shared_for_missing_segments(arm_iuis):
    # No D or J truth (W-97, SC-23), or D truth for some genes: D and J keep every shared allele either way.
    for d_truth in (False, True):
        kept = set(arm_alleles(_arm_donor(arm_iuis, d_truth=d_truth), ARM_REFERENCE, arm_iuis))
        assert {name for name in kept if ARM_REFERENCE.gene_letter(name) == "d"} == set(ARM_REFERENCE.alleles["d"])
        assert {name for name in kept if ARM_REFERENCE.gene_letter(name) == "j"} == set(ARM_REFERENCE.alleles["j"])


def test_arm_alleles_restricts_captured_v_only(arm_iuis):
    kept = arm_alleles(_arm_donor(arm_iuis), ARM_REFERENCE, arm_iuis)
    assert kept == sorted(kept) and len(kept) == len(set(kept))
    v = {name for name in kept if ARM_REFERENCE.gene_letter(name) == "v"}
    assert v == {
        "IGHVF1-G1*01",     # a donor allele's shared name: IGHV1-2 is restricted to it, its absent allele aside
        "IGHVF6-G21*06",    # the donor's IGHV1-69D*01, the shared allele it shares with IGHV1-69*01
        "IGHVF5-G5*02",     # the donor's IGHV3-30-5*02
        "IGHVF3-G62*03",    # the donor's nonfunctional IGHV3-62*03: the genome carries it
        "IGHVF3-G23*01",    # IGHV3-23 is carried only with an allele absent from the shared reference
        "IGHVF3-G23*04",    # ... so all its shared alleles stay
        "IGHVF5-G5*01",     # one of its genes, IGHV3-30, is not captured
        "IGHVF4-G34*01",    # IGHV4-34 is not captured
        "IGHVF9-G9*01",     # no IUIS gene: no truth names its gene
    }
    # Dropped: another allele of a carried gene (IGHVF1-G1*02, IGHVF5-G5*03), an allele of a captured gene the
    # donor lacks (IGHV3-11, IGHV1-69*02, IGHV3-23D*02), and a paralog pair allele whose genes are both captured
    # and neither carried only with absent alleles (IGHVF6-G21*07).
    assert set(ARM_REFERENCE.alleles["v"]) - v == {"IGHVF1-G1*02", "IGHVF5-G5*03", "IGHVF3-G11*01",
                                                   "IGHVF6-G21*02", "IGHVF3-G23*02", "IGHVF6-G21*07"}


def test_arm_alleles_keeps_a_paralog_allele_by_any_of_its_genes(arm_iuis):
    donor = _arm_donor(arm_iuis)
    # IGHVF6-G21*07 (IGHV1-69*06 = IGHV1-69D*06) is kept once either gene is not captured ...
    for gene in ("IGHV1-69", "IGHV1-69D"):
        uncaptured = replace(donor, captured=donor.captured - {gene})
        assert "IGHVF6-G21*07" in arm_alleles(uncaptured, ARM_REFERENCE, arm_iuis)
    # ... and IGHVF3-G23*01 (IGHV3-23*01 = IGHV3-23D*01) by IGHV3-23 alone, carried only with an absent allele,
    # while IGHV3-23D is captured and not carried.
    assert "IGHVF3-G23*01" in arm_alleles(donor, ARM_REFERENCE, arm_iuis)
    # Once IGHV3-23 carries a shared allele, its other alleles go, paralog pair allele included.
    documented = replace(donor, alleles=(*donor.alleles, _carried(arm_iuis, "IGHV3-23*04", ARM_V["IGHVF3-G23*04"][0])))
    kept = arm_alleles(documented, ARM_REFERENCE, arm_iuis)
    assert "IGHVF3-G23*04" in kept and "IGHVF3-G23*01" not in kept


def test_arm_alleles_never_empty_letter(arm_iuis):
    # A donor with few V genes and no D/J truth (W-97, W-133): every gene letter keeps alleles.
    few = RealGenotype(donor="W-133", alleles=(_carried(arm_iuis, "IGHV1-2*02", ARM_V["IGHVF1-G1*01"][0]),),
                       captured=frozenset({"IGHV1-2"}))
    kept = arm_alleles(few, ARM_REFERENCE, arm_iuis)
    assert {ARM_REFERENCE.gene_letter(name) for name in kept} == {"v", "d", "j"}
    assert "IGHVF1-G1*01" in kept and "IGHVF1-G1*02" not in kept
    # A restriction that would leave a letter with no allele is refused, never passed on.
    small = Reference(alleles={"v": {"IGHVF1-G1*01": ARM_V["IGHVF1-G1*01"][0]}, "d": {"IGHD3-3*01": D_33},
                               "j": {"IGHJ4*02": J_4}}, anchors={})
    path = Path(arm_iuis.source["path"])
    small_iuis = IuisMap.from_ogrdb(path, small)
    lacking = RealGenotype(donor="W-97", alleles=(_carried(small_iuis, "IGHV7-81*01", _random(296, 35)),),
                           captured=frozenset({"IGHV1-2", "IGHV7-81"}))
    with pytest.raises(ValueError, match="no v allele"):
        arm_alleles(lacking, small, small_iuis)


def test_arm_alleles_refuses_inconsistent_inputs(arm_iuis, iuis):
    donor = _arm_donor(arm_iuis)
    with pytest.raises(ValueError, match="captured"):
        arm_alleles(replace(donor, captured=None), ARM_REFERENCE, arm_iuis)
    with pytest.raises(ValueError, match="another reference"):
        arm_alleles(donor, ARM_REFERENCE, iuis)
    # A donor allele mapped against another reference names a shared allele this one does not hold.
    foreign = replace(donor.alleles[0], shared=("IGHVF7-G7*01",))
    with pytest.raises(ValueError, match=re.escape("IGHVF7-G7*01")):
        arm_alleles(replace(donor, alleles=(foreign, *donor.alleles[1:])), ARM_REFERENCE, arm_iuis)


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
    # each of the two unlabelled V alleles is one base short of an OGRDB allele and equal to it through IMGT 318
    assert iuis.labels("IGHVF10-G33*01") == iuis.labels("IGHVF10-G42*01") == ()
    assert iuis.inferred("IGHVF10-G33*01") == ("IGHV3-13*06",) and iuis.genes("IGHVF10-G33*01") == ("IGHV3-13",)
    assert iuis.inferred("IGHVF10-G42*01") == ("IGHV3-43D*05",) and iuis.genes("IGHVF10-G42*01") == ("IGHV3-43D",)
    assert iuis.unmapped() == ()
    assert sum(bool(iuis.inferred(name)) for name in reference.sequences()) == 2
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


# The genotype arm's V alleles per pilot donor (kept of the shared reference's 198), on HUMAN_IGH_OGRDB. W-97 and
# W-133 keep the most: their contigs capture the fewest V genes (45 and 44), so more genes keep all their alleles.
ARM_V_KEPT = {"SC-23": 111, "SC-4": 102, "SC-11": 110, "SC-6": 95, "W-97": 114, "W-89": 92, "W-133": 117, "W-131": 95}


@pytest.mark.skipif(not DATA or os.environ.get("ALIGNAIR_ASSAY_DOCKER") != "1",
                    reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root and ALIGNAIR_ASSAY_DOCKER=1")
def test_the_pilot_donors_genotype_arms(reference):
    from alignair_assay.real.locus import donor_capture

    root = Path(DATA)
    samples = root / "genotypes/vdjbase_genomic_samples"
    iuis = IuisMap.from_ogrdb(root / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)
    kept_v = {}
    for donor, sample in PILOT.items():
        genotype = donor_capture(load_vdjbase_csv(samples / f"{sample}.csv", donor, reference, iuis),
                                 samples / f"{sample}.bam", root / "genotypes/igenotyper/gene_coords.bed")
        kept = arm_alleles(genotype, reference, iuis)
        by_letter = {letter: {name for name in kept if reference.gene_letter(name) == letter} for letter in "vdj"}
        # D and J are never restricted; V keeps every donor allele's shared name and is restricted
        assert by_letter["d"] == set(reference.alleles["d"]) and by_letter["j"] == set(reference.alleles["j"])
        donor_v = {name for allele in genotype.alleles if allele.segment == "v" for name in allele.shared}
        assert donor_v <= by_letter["v"]
        assert 0 < len(by_letter["v"]) < len(reference.alleles["v"])
        kept_v[donor] = len(by_letter["v"])
    assert kept_v == ARM_V_KEPT
