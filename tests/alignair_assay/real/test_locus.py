"""Which genes each donor's genomic contigs capture (``real.locus``).

The offline tests use synthetic SAM text, a synthetic four-gene BED and genotypes built in the test; Docker is
never started (``subprocess.run`` answers as Docker would). The gated tests:

- ``ALIGNAIR_ASSAY_DATA`` (the PRJNA555323 data root): the pinned IGenotyper ``gene_coords.bed`` and its gene names
  against OGRDB's IUIS genes;
- with ``ALIGNAIR_ASSAY_DOCKER=1`` too: the eight pilot donors' contig BAMs read in the pinned samtools container,
  SC-23 in detail.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from alignair_assay.core import container
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import gene_name
from alignair_assay.real import locus
from alignair_assay.real.genotype import IuisMap, RealAllele, RealGenotype, load_vdjbase_csv
from alignair_assay.real.locus import (
    SAMTOOLS_IMAGE,
    captured_genes,
    contig_spans,
    donor_capture,
    parse_sam_spans,
    read_gene_coords,
    samtools_command,
    with_capture,
)

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
needs_data = pytest.mark.skipif(not DATA, reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root")
needs_docker = pytest.mark.skipif(not DOCKER, reason="set ALIGNAIR_ASSAY_DOCKER=1 to run the samtools container")


def _sam(*records: tuple) -> str:
    """SAM text without a header: ``(qname, flag, rname, pos, cigar, rg)`` per record, the rest filler."""
    return "".join(f"{qname}\t{flag}\t{rname}\t{pos}\t254\t{cigar}\t*\t0\t0\tACGT\t*\tnp:i:1\tRG:Z:{rg}\n"
                   for qname, flag, rname, pos, cigar, rg in records)


def _contig(haplotype, start: int = 100, end: int = 900, index: int = 0) -> str:
    """A contig name as IGenotyper writes it: assembled region, haplotype, index, contig count."""
    return f"c=igh:{start}-{end}_h={haplotype}_i={index}_t=2_/0/0_0"


# Four records IGenotyper keeps, then the three kinds it skips (``skip_read``). The RG tags disagree with the names'
# haplotypes on purpose: the haplotype is the name's h=, never the RG.
SAM = _sam(
    (_contig(0), 0, "igh", 101, "5S100=2D50=1I48=5S", 1),
    (_contig(1), 0, "igh", 1001, "10M5N10X1D", 0),
    (_contig(2), 16, "igh", 2001, "300M", 2),
    (_contig(1, index=1), 0, "igh", 5001, "4H20=3P10=", 2),
    (_contig(1, index=2), 256, "igh", 7001, "500M", 0),    # secondary
    (_contig(2, index=1), 2048, "igh", 8001, "500M", 0),   # supplementary
    (_contig(0, index=1), 4, "*", 0, "*", 0),              # unmapped
)


def test_parse_sam_spans():
    assert parse_sam_spans("@SQ\tSN:igh\tLN:1193129\n" + SAM) == [
        (0, 100, 300), (1, 1000, 1026), (2, 2000, 2300), (1, 5000, 5030)]


def test_parse_sam_spans_refuses_a_contig_without_a_haplotype():
    for name in ("c=igh:100-900_i=0_t=2_/0/0_0", "c=igh:100-900_h=3_i=0", "contig_7", "c=igh:1-2_h=_i=0"):
        with pytest.raises(ValueError, match=r"line 1 .*h=<0\|1\|2>"):
            parse_sam_spans(_sam((name, 0, "igh", 1, "10M", 0)))
    # a skipped record is not read, so its name does not matter
    assert parse_sam_spans(_sam(("contig_7", 256, "igh", 1, "10M", 0))) == []


def test_parse_sam_spans_refuses_what_it_cannot_place():
    refused = {
        (_contig(0), 0, "chr14", 1, "10M", 0): "reference chr14, not igh",
        (_contig(0), 0, "igh", 1, "*", 0): "no CIGAR",
        (_contig(0), 0, "igh", 1, "10Q", 0): "CIGAR",
        (_contig(0), 0, "igh", 1, "10S", 0): "spans no reference base",
        (_contig(0), 0, "igh", 0, "10M", 0): "position",
        (_contig(0), "x", "igh", 1, "10M", 0): "flag",
    }
    for record, problem in refused.items():
        with pytest.raises(ValueError, match=problem):
            parse_sam_spans(_sam(record))
    with pytest.raises(ValueError, match="line 1 has 3 fields"):
        parse_sam_spans("a\tb\tc\n")


BED = ("chr14\t21622284\t21622567\tTRAV1-1\r\n"
       "igh\t1000\t1300\tIGHV1-2\r\n"
       "igh\t2050\t2080\tIGHD3-3\r\n"
       "igh\t3000\t3050\tIGHJ4\r\n"
       "igh\t4000\t4300\tIGHV3-23\r\n")
COORDS = {"IGHV1-2": (1000, 1300), "IGHD3-3": (2050, 2080), "IGHJ4": (3000, 3050), "IGHV3-23": (4000, 4300)}


def test_read_gene_coords(tmp_path):
    path = tmp_path / "gene_coords.bed"
    path.write_bytes(BED.encode())
    assert read_gene_coords(path) == COORDS


def test_read_gene_coords_refuses_a_malformed_bed(tmp_path):
    path = tmp_path / "gene_coords.bed"
    for text, problem in {
        BED + "igh\t5000\t5300\tIGHV1-2\n": "IGHV1-2 is given twice",
        "igh\t10\t5\tIGHV1-2\n": "is not a non-empty interval",
        "igh\tten\t50\tIGHV1-2\n": "is not a non-empty interval",
        "igh\t10\t50\n": "not a BED line",
        "igh\t10\t50\tIGHV1-2*01\n": "not an IGH gene name",
        "chr14\t10\t50\tTRAV1\n": "no igh gene",
    }.items():
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=problem):
            read_gene_coords(path)


def test_gene_captured_by_unphased_contig():
    assert captured_genes([(0, 900, 1400)], COORDS) == {"IGHV1-2"}
    assert captured_genes([(0, 1000, 1300)], COORDS) == {"IGHV1-2"}  # exactly the gene's interval
    assert captured_genes([(0, 0, 5000)], COORDS) == set(COORDS)


def test_gene_needs_both_haplotypes_when_phased():
    assert captured_genes([(1, 900, 1400)], COORDS) == frozenset()
    assert captured_genes([(2, 900, 1400)], COORDS) == frozenset()
    assert captured_genes([(1, 900, 1400), (2, 950, 2100)], COORDS) == {"IGHV1-2"}
    # IGHD3-3 lies inside the haplotype-2 span only
    assert captured_genes([(1, 900, 2060), (2, 950, 2100)], COORDS) == {"IGHV1-2"}


def test_gene_partially_spanned_is_not_captured():
    assert captured_genes([(0, 1001, 1400)], COORDS) == frozenset()
    assert captured_genes([(0, 900, 1299)], COORDS) == frozenset()
    # two contigs that cover the gene only together do not capture it
    assert captured_genes([(0, 900, 1150), (0, 1150, 1400)], COORDS) == frozenset()
    assert captured_genes([(1, 900, 1400), (2, 1009, 1400)], COORDS) == frozenset()
    with pytest.raises(ValueError, match="haplotype 3"):
        captured_genes([(3, 900, 1400)], COORDS)


def _allele(gene: str, haplotype: int, allele: str = "01") -> RealAllele:
    segment = gene[3].lower()
    return RealAllele(allele_id=f"{gene}*{allele}", gene=gene, segment=segment, sequence="ACGT" * 4,
                      gapped="ACGT" * 4 if segment == "v" else None, haplotype=haplotype, functional=True,
                      shared=(), iuis=())


GENOTYPE = RealGenotype(donor="SC-23", alleles=(_allele("IGHV1-2", 1), _allele("IGHV1-2", 2, "02"),
                                                 _allele("IGHV3-25", 1), _allele("IGHJ4", 0)),
                        source={"kind": "vdjbase_genomic_csv", "sample": "P25_I58_S1"})


def test_with_capture_adds_every_carried_gene():
    """A gene the donor carries is captured whatever the spans say: IGHV3-25 is not in the BED, and IGHJ4 is
    spanned on one haplotype only. IGHV3-23 is spanned but not carried: a candidate deletion."""
    genotype = with_capture(GENOTYPE, frozenset({"IGHV1-2", "IGHV3-23"}))
    assert genotype.captured == {"IGHV1-2", "IGHV3-23", "IGHV3-25", "IGHJ4"}
    assert genotype.alleles == GENOTYPE.alleles and GENOTYPE.captured is None
    assert genotype.source["sample"] == "P25_I58_S1" and "capture" not in GENOTYPE.source
    assert genotype.source["capture"] == {"spanned": ["IGHV1-2", "IGHV3-23"],
                                          "carried_not_spanned": ["IGHJ4", "IGHV3-25"],
                                          "spanned_not_carried": ["IGHV3-23"], "captured": 4}
    recorded = with_capture(GENOTYPE, {"IGHV1-2"}, {"command": ["docker", "run"], "spanned": "ignored"})
    assert recorded.source["capture"]["command"] == ["docker", "run"]
    assert recorded.source["capture"]["spanned"] == ["IGHV1-2"]


def test_the_samtools_image_is_pinned_by_digest():
    name, _, digest = SAMTOOLS_IMAGE.partition("@")
    assert name.startswith("quay.io/biocontainers/samtools:") and len(digest) == 71
    assert container.pinned_repo_digest(SAMTOOLS_IMAGE) == f"quay.io/biocontainers/samtools@{digest}"


def test_samtools_command_reads_only_the_bams_directory_offline_as_the_user(tmp_path):
    bam = tmp_path / "P25_I58_S1.bam"
    argv = samtools_command(bam)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--name") + 1] == f"alignair-locus-P25_I58_S1-{os.getpid()}"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    assert [argv[i + 1] for i, part in enumerate(argv) if part == "-v"] == [f"{tmp_path}:/data:ro"]
    assert argv[argv.index(SAMTOOLS_IMAGE) + 1:] == ["samtools", "view", "/data/P25_I58_S1.bam"]


class _Docker:
    """``subprocess.run`` as Docker answers it: the pinned image is local, and ``samtools view`` prints ``sam``."""

    def __init__(self, sam: str = SAM, *, digest: str | None = None):
        self.sam, self.calls = sam, []
        self.repo_digest = container.pinned_repo_digest(SAMTOOLS_IMAGE) if digest is None else digest

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            return _Completed(json.dumps([{"Id": "sha256:" + "1" * 64, "RepoDigests": [self.repo_digest]}]))
        if argv[:2] == ["docker", "run"]:
            return _Completed(self.sam)
        return _Completed()


class _Completed:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_contig_spans_runs_the_pinned_container(tmp_path, monkeypatch):
    docker = _Docker()
    monkeypatch.setattr(container.subprocess, "run", docker)
    bam = tmp_path / "P25_I58_S1.bam"
    assert contig_spans(bam) == parse_sam_spans(SAM)
    assert docker.calls == [["docker", "image", "inspect", SAMTOOLS_IMAGE], samtools_command(bam)]


def test_contig_spans_refuses_another_image_and_kills_a_timed_out_view(tmp_path, monkeypatch):
    other = "quay.io/biocontainers/samtools@sha256:" + "f" * 64
    monkeypatch.setattr(container.subprocess, "run", _Docker(digest=other))
    with pytest.raises(container.ContainerError, match="pinned digest"):
        contig_spans(tmp_path / "x.bam")
    docker = _Docker()

    def run(argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            docker.calls.append(argv)
            raise container.subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
        return docker(argv, **kwargs)

    monkeypatch.setattr(container.subprocess, "run", run)
    with pytest.raises(container.ContainerError, match="was killed"):
        contig_spans(tmp_path / "x.bam", timeout=3)
    assert docker.calls[-1] == ["docker", "kill", f"alignair-locus-x-{os.getpid()}"]


def test_donor_capture_records_how_it_was_computed(tmp_path, monkeypatch):
    bam, bed = tmp_path / "P25_I58_S1.bam", tmp_path / "gene_coords.bed"
    bam.write_bytes(b"BAM synthetic")
    bed.write_bytes(BED.encode())
    sam = _sam((_contig(0), 0, "igh", 901, "500M", 0), (_contig(1), 0, "igh", 3901, "500M", 1))
    monkeypatch.setattr(container.subprocess, "run", _Docker(sam))
    genotype = donor_capture(GENOTYPE, bam, bed)
    assert genotype.captured == {"IGHV1-2", "IGHV3-25", "IGHJ4"}
    record = genotype.source["capture"]
    assert record["command"] == samtools_command(bam)
    assert record["image"]["image"] == SAMTOOLS_IMAGE
    assert record["image"]["repo_digests"] == [container.pinned_repo_digest(SAMTOOLS_IMAGE)]
    assert record["bam"] == {"path": str(bam), "sha256": sha256_file(bam)}
    assert record["gene_coords"] == {"path": str(bed), "sha256": sha256_file(bed), "genes": 4}
    assert record["spans"] == {"h0": 1, "h1": 1, "h2": 0}
    assert record["spanned"] == ["IGHV1-2"] and record["carried_not_spanned"] == ["IGHJ4", "IGHV3-25"]
    assert record["code_sha256"] == sha256_file(locus.__file__)
    # the fingerprint follows the inputs, the image and the code, and nothing else
    again = donor_capture(replace(GENOTYPE, source={}), bam, bed).source["capture"]["fingerprint"]
    assert again == record["fingerprint"]
    bed.write_bytes(BED.encode() + b"igh\t6000\t6300\tIGHV1-3\n")
    assert donor_capture(GENOTYPE, bam, bed).source["capture"]["fingerprint"] != record["fingerprint"]


def test_donor_capture_refuses_another_donors_bam_and_names_the_bam_of_a_bad_record(tmp_path, monkeypatch):
    bed = tmp_path / "gene_coords.bed"
    bed.write_bytes(BED.encode())
    other = tmp_path / "P25_I61_S1.bam"
    other.write_bytes(b"BAM synthetic")
    docker = _Docker(_sam(("contig_7", 0, "igh", 901, "500M", 0)))
    monkeypatch.setattr(container.subprocess, "run", docker)
    with pytest.raises(ValueError, match="not the contig BAM of SC-23's VDJbase sample P25_I58_S1"):
        donor_capture(GENOTYPE, other, bed)
    assert docker.calls == []  # refused before any container
    bam = tmp_path / "P25_I58_S1.bam"
    bam.write_bytes(b"BAM synthetic")
    with pytest.raises(ValueError, match=r"P25_I58_S1\.bam \(SC-23\): SAM line 1 \(contig_7\).*h=<0\|1\|2>"):
        donor_capture(GENOTYPE, bam, bed)


# --- the pilot donors (gated) --------------------------------------------------------------------------------------

GENE_COORDS = "genotypes/igenotyper/gene_coords.bed"
SAMPLES = "genotypes/vdjbase_genomic_samples"
PILOT = {"SC-23": "P25_I58_S1", "SC-11": "P25_I48_S1", "W-97": "P25_I151_S1", "W-133": "P25_I93_S1",
         "SC-4": "P25_I61_S1", "SC-6": "P25_I62_S1", "W-89": "P25_I145_S1", "W-131": "P25_I91_S1"}
# Per donor: genes the span rule captures (V, D, J), carried V genes it does not, and spanned V genes the donor
# does not carry (candidate deletions or assembly gaps).
SPANNED = {"SC-23": (48, 0, 0), "SC-11": (49, 19, 0), "W-97": (42, 0, 0), "W-133": (40, 3, 1),
           "SC-4": (49, 13, 0), "SC-6": (46, 20, 0), "W-89": (52, 14, 5), "W-131": (48, 23, 0)}
NO_COORDS = {"IGHV3-25", "IGHV3-62", "IGHV8-51-1"}


@pytest.fixture(scope="module")
def iuis(reference):
    return IuisMap.from_ogrdb(Path(DATA) / "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json", reference)


@needs_data
def test_the_pinned_gene_coords_name_iuis_genes(iuis):
    coords = read_gene_coords(Path(DATA) / GENE_COORDS)
    assert len(coords) == 92
    assert {kind: sum(gene.startswith(kind) for gene in coords) for kind in ("IGHV", "IGHD", "IGHJ")} == {
        "IGHV": 59, "IGHD": 27, "IGHJ": 6}
    ogrdb_genes = {gene_name(label) for labels in iuis.labels_by_sequence.values() for label in labels}
    # the BED's names are IUIS gene names: all OGRDB's but IGHV1-38-4, which OGRDB rev 8 lacks and VDJbase uses
    assert set(coords) - ogrdb_genes == {"IGHV1-38-4"}
    assert ogrdb_genes - set(coords) >= NO_COORDS


@needs_data
@needs_docker
def test_sc23_contigs_capture_every_v_gene_in_its_csv(reference, iuis):
    root = Path(DATA)
    genotype = load_vdjbase_csv(root / SAMPLES / "P25_I58_S1.csv", "SC-23", reference, iuis)
    captured = donor_capture(genotype, root / SAMPLES / "P25_I58_S1.bam", root / GENE_COORDS)
    assert genotype.genes("v") <= captured.captured
    record = captured.source["capture"]
    assert record["spans"] == {"h0": 15, "h1": 49, "h2": 39}  # the 103 primary alignments of 674 records
    # IGHV3-15 is carried on haplotype 1; the haplotype-2 contig starts 9 bases inside it
    assert set(record["carried_not_spanned"]) == NO_COORDS | {"IGHV3-15"}
    assert record["spanned_not_carried"] == ["IGHV4-30-2"]
    assert len(record["spanned"]) == 48 and len(captured.captured) == 52


@needs_data
@needs_docker
def test_the_pilot_donors_capture(reference, iuis):
    root = Path(DATA)
    found = {}
    for donor, sample in PILOT.items():
        genotype = load_vdjbase_csv(root / SAMPLES / f"{sample}.csv", donor, reference, iuis)
        captured = donor_capture(genotype, root / SAMPLES / f"{sample}.bam", root / GENE_COORDS)
        spanned = captured.source["capture"]["spanned"]
        found[donor] = tuple(sum(gene.startswith(kind) for gene in spanned) for kind in ("IGHV", "IGHD", "IGHJ"))
        assert captured.captured >= {allele.gene for allele in genotype.alleles}
        assert NO_COORDS & genotype.genes("v") <= set(captured.source["capture"]["carried_not_spanned"])
    assert found == SPANNED
