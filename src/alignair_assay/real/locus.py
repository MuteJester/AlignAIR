"""Which IGH genes each donor's genomic contigs capture: the part of the locus its genotype truth covers.

R1 must tell apart a call to a gene the donor lacks from a call to a gene its long-read sequencing did not resolve.
In the first case the gene's region is captured and the donor carries no allele of it, so the call is
``unsupported_gene``. In the second the call is excluded. The donor's IGenotyper contigs, aligned to IGenotyper's
custom ``igh`` reference (1,193,129 bp), are in VDJbase's per-sample contig BAM (study P25, CC0). The gene
intervals on that reference are in IGenotyper's ``data/gene_coords.bed``, pinned by ``real.fetch``.

**The rule.** A gene is captured when the donor carries an allele of it (any segment, any haplotype: the allele was
read off a contig, so a carried gene is captured by definition), or when its interval lies wholly inside one contig
of haplotype 0 (unphased), or inside a contig of haplotype 1 and a contig of haplotype 2. Two contigs of one
haplotype that cover a gene only together do not capture it, and neither does a partial overlap.

**Resolved.** A gene is resolved when both of the donor's haplotypes are known there: a haplotype-0 contig spans it
or carries an allele of it, or each of haplotypes 1 and 2 spans it or carries an allele of it. A carried gene can be
captured but not resolved: an allele read off a haplotype-1 contig while no haplotype-2 contig spans the gene (SC-23's
IGHV3-15, whose haplotype-2 contig starts 9 bases inside it). Its second allele is unknown, so R1 excludes an
allele-level unsupported call there (``real.metrics``).

- **The haplotype is the contig name's ``h=``, not the alignment's read group.** IGenotyper names each contig after
  the phased block and haplotype it was assembled from (``c=igh:<start>-<end>_h=<0|1|2>_i=<n>_t=<n>_/0/0_0``; 0
  is an unphased block, assembled from both haplotypes' reads). The ``RG`` tag means something else:
  ``phasing/reads.py:phase_read`` sets it per alignment from the phased SNVs that alignment covers, so ``RG:Z:0``
  marks an alignment over no phased SNV, or with mixed alleles, not an unphased contig. IGenotyper's own allele
  extraction (``common/helper.py:extract_sequence``) takes the haplotype from ``h=``, and the VDJbase CSVs'
  ``haplotype`` equals the contig's ``h=`` on all 840 rows of the eight pilot donors, while the RG differs from it
  on 60 of them (38 on W-89, 30 of those haplotype-1 contigs tagged RG 0). A contig name without an ``h=`` of 0, 1
  or 2 is refused; nothing is guessed.
- **Only the alignments IGenotyper uses.** A record is skipped as ``common/helper.py:skip_read`` skips it:
  unmapped, secondary or supplementary (``SKIP_FLAGS``). There is no MAPQ filter. BLASR reports up to ten
  placements per contig, so most records are secondary (571 of SC-23's 674): partial, mostly MAPQ 0 hits on
  paralogous regions. Counting them would add 8 to 17 genes per pilot donor that its contigs do not span.
- **Coordinates.** The BED is 0-based and half-open. A SAM record's span is ``[POS - 1, POS - 1 + L)``, where L is
  its CIGAR's reference-consuming length (``M``, ``D``, ``N``, ``=``, ``X``).
- **Names.** The BED's ``igh`` rows name 92 genes (59 IGHV, 27 IGHD, 6 IGHJ) by their IUIS gene names, the names
  ``RealGenotype`` uses (VDJbase's ``gene``). All but IGHV1-38-4 are among OGRDB rev 8's IUIS genes, and VDJbase
  uses IGHV1-38-4 too, so the names are used as they are. The BED has no IGHV3-62 or IGHV8-51-1, which every pilot
  donor carries, no IGHV3-25, which all but SC-4 carry, and none of OGRDB's IGHV3-19, IGHV3-47, IGHV3-69-1,
  IGHV3-NL1 or IGHV4-NL1. A carried gene is captured all the same, by the first clause of the rule; one the donor
  does not carry is uncaptured.
- **Deletions.** A CIGAR ``D`` consumes reference, so a contig that carries a haplotype's deletion as a long ``D``
  spans the genes it deletes (W-89: a 9.4 kb ``D`` over IGHD1-7 to IGHD3-3). A deletion that BLASR does not align
  across leaves its genes uncaptured: the rule cannot tell it from an assembly gap.

The IGenotyper sources cited are at the commit ``real.fetch`` pins its BED to (``40a00015``).

``contig_spans`` reads the BAM with ``samtools view`` in the pinned container (``SAMTOOLS_IMAGE``), run as the
current user with no network and no capabilities, unable to gain privileges, on the BAM's directory alone mounted
read-only, and named so that a run that times out is killed (``core.container``). Only its SAM text is parsed
(``parse_sam_spans``). ``donor_capture`` puts it all together and records in the genotype's
``source["capture"]`` how the set was computed (the image and its digests, the whole command, the SHA-256s of the
inputs, the genotype's CSV among them, and of this module) and a fingerprint of those inputs, so a stored capture
is reused only while it is current.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from alignair_assay.core.container import Mount, check_image, container_name, hardened_run, run_container
from alignair_assay.core.provenance import sha256_file
from alignair_assay.real.genotype import RealGenotype

# biocontainers samtools 1.23.1, pushed to quay.io 2026-03-18 (image sha256:0ea56912f089...).
SAMTOOLS_IMAGE = ("quay.io/biocontainers/samtools:1.23.1--ha83d96e_0"
                  "@sha256:23cda33a3a42125872766df9aaf1d2db67cdb8c85314b793465188435af31ba6")
# The sequence name of IGenotyper's custom IGH reference, in the BED and in the BAM.
IGH_REFERENCE = "igh"
# 0 for a contig assembled from an unphased block, 1 and 2 for the two haplotypes of a phased one.
HAPLOTYPES = (0, 1, 2)
# Unmapped (0x4), secondary (0x100) and supplementary (0x800): what IGenotyper's ``skip_read`` skips.
SKIP_FLAGS = 0x4 | 0x100 | 0x800
VIEW_TIMEOUT = 600.0  # seconds ``samtools view`` may take on one BAM
BAM_MOUNT = "/data"
SAM_FIELDS = 11  # the mandatory fields of a SAM record

_CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")
_REFERENCE_OPS = frozenset("MDN=X")
_HAPLOTYPE = re.compile(r"(?:^|_)h=([^_]*)")
_GENE = re.compile(r"IGH[VDJ][0-9A-Z][0-9A-Za-z-]*")
_CODE_SHA256 = sha256_file(__file__)


def read_gene_coords(bed_path, *, reference: str = IGH_REFERENCE) -> dict[str, tuple[int, int]]:
    """IUIS gene -> its ``(start, end)`` interval (0-based, half-open) on ``reference``, from the BED rows on that
    sequence; the rows on other sequences (IGenotyper's file also lists TR and IGK/IGL genes on hg19) are ignored.
    ValueError, naming the line, for a line with fewer than four fields, an interval that is not two integers with
    start < end, a name that is not an IGH gene, a gene given twice, or a file with no gene on ``reference``."""
    path = Path(bed_path)
    coords: dict[str, tuple[int, int]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.rstrip("\r").split("\t")
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        if len(fields) < 4:
            raise ValueError(f"{path}: line {number} is not a BED line with a name (chrom, start, end, name)")
        if fields[0] != reference:
            continue
        name = fields[3].strip()
        try:
            start, end = int(fields[1]), int(fields[2])
        except ValueError:
            start, end = 0, 0
        if not 0 <= start < end:
            raise ValueError(f"{path}: line {number} ({name}): {fields[1]!r}-{fields[2]!r} is not a non-empty "
                             "interval")
        if not _GENE.fullmatch(name):
            raise ValueError(f"{path}: line {number}: {name!r} is not an IGH gene name")
        if name in coords:
            raise ValueError(f"{path}: line {number}: {name} is given twice")
        coords[name] = (start, end)
    if not coords:
        raise ValueError(f"{path} has no {reference} gene")
    return coords


def parse_sam_spans(text: str, *, reference: str = IGH_REFERENCE) -> list[tuple[int, int, int]]:
    """``(haplotype, start, end)`` of each contig alignment in SAM text ``text`` (``samtools view`` output; header
    lines are skipped) that IGenotyper uses, in order: the haplotype is the contig name's ``h=``, and the span is
    0-based and half-open on ``reference`` (the module docstring). Records with a ``SKIP_FLAGS`` flag are skipped
    unread. ValueError, naming the line and contig, for a line that is not a SAM record, a kept record whose name
    has no ``h=`` of 0, 1 or 2, a record on another reference, or one without a valid position or CIGAR."""
    spans = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("@"):
            continue
        fields = line.split("\t", SAM_FIELDS)
        if len(fields) < SAM_FIELDS:
            raise ValueError(f"SAM line {number} has {len(fields)} fields, not a record's {SAM_FIELDS}")
        qname, flag_text, rname, pos_text, _, cigar = fields[:6]
        where = f"SAM line {number} ({qname})"
        try:
            flag = int(flag_text)
        except ValueError:
            raise ValueError(f"{where}: flag {flag_text!r} is not a number") from None
        if flag & SKIP_FLAGS:
            continue
        if rname != reference:
            raise ValueError(f"{where}: reference {rname}, not {reference}")
        found = _HAPLOTYPE.search(qname)
        if found is None or found[1] not in ("0", "1", "2"):
            raise ValueError(f"{where}: the contig name has no h=<0|1|2> haplotype field")
        try:
            position = int(pos_text)
        except ValueError:
            position = 0
        if position < 1:
            raise ValueError(f"{where}: position {pos_text!r} is not a 1-based position")
        if cigar == "*":
            raise ValueError(f"{where}: a mapped record with no CIGAR")
        operations = _CIGAR.findall(cigar)
        if "".join(count + op for count, op in operations) != cigar:
            raise ValueError(f"{where}: {cigar!r} is not a CIGAR")
        length = sum(int(count) for count, op in operations if op in _REFERENCE_OPS)
        if length == 0:
            raise ValueError(f"{where}: CIGAR {cigar} spans no reference base")
        spans.append((int(found[1]), position - 1, position - 1 + length))
    return spans


def samtools_command(bam_path) -> list[str]:
    """The container run of ``samtools view`` on ``bam_path``: the pinned image, as the current user, with no
    network and no capabilities, on the BAM's directory alone mounted read-only at ``BAM_MOUNT``, named after the
    BAM and this process (``core.container``)."""
    bam = Path(bam_path).absolute()
    return hardened_run(SAMTOOLS_IMAGE, ["samtools", "view", f"{BAM_MOUNT}/{bam.name}"],
                        name=container_name("locus", bam.stem), mounts=[Mount(bam.parent, BAM_MOUNT)])


def contig_spans(bam_path, *, timeout: float | None = None) -> list[tuple[int, int, int]]:
    """``(haplotype, start, end)`` of the contig alignments in contig BAM ``bam_path`` that IGenotyper uses
    (``parse_sam_spans`` of its ``samtools view`` in the pinned container). ContainerError when Docker is missing,
    the image is not the pinned one, or the view fails or takes longer than ``timeout`` seconds (``VIEW_TIMEOUT``),
    its container then killed; ValueError for a record ``parse_sam_spans`` refuses."""
    text, _, _ = _view(bam_path, timeout)
    return parse_sam_spans(text)


def _view(bam_path, timeout: float | None) -> tuple[str, list[str], dict]:
    """The BAM's SAM text, the command that wrote it and the image found (``core.container.check_image``)."""
    bam = Path(bam_path).absolute()
    image = check_image(SAMTOOLS_IMAGE)
    argv = samtools_command(bam)
    name = argv[argv.index("--name") + 1]
    completed = run_container(argv, name=name, timeout=timeout or VIEW_TIMEOUT, what=f"samtools view of {bam}")
    return completed.stdout, argv, image


def spanned_by_haplotype(spans: Iterable[tuple[int, int, int]],
                         coords: dict[str, tuple[int, int]]) -> dict[int, frozenset[str]]:
    """Haplotype (0, 1 and 2) -> the genes of ``coords`` whose interval lies wholly inside one span of that
    haplotype. ValueError for a haplotype other than 0, 1 or 2."""
    by_haplotype: dict[int, list[tuple[int, int]]] = {haplotype: [] for haplotype in HAPLOTYPES}
    for haplotype, start, end in spans:
        if haplotype not in by_haplotype:
            raise ValueError(f"haplotype {haplotype} is not 0 (unphased), 1 or 2")
        by_haplotype[haplotype].append((start, end))
    return {haplotype: frozenset(gene for gene, (start, end) in coords.items()
                                 if any(first <= start and end <= last for first, last in intervals))
            for haplotype, intervals in by_haplotype.items()}


def captured_genes(spans: Iterable[tuple[int, int, int]], coords: dict[str, tuple[int, int]]) -> frozenset[str]:
    """The genes of ``coords`` whose interval lies wholly inside one span of haplotype 0, or inside a span of
    haplotype 1 and a span of haplotype 2 (the rule's span clauses; ``with_capture`` adds the carried genes).
    ValueError for a haplotype other than 0, 1 or 2."""
    return _span_clauses(spanned_by_haplotype(spans, coords))


def _span_clauses(haplotypes: dict[int, frozenset[str]]) -> frozenset[str]:
    return haplotypes[0] | (haplotypes[1] & haplotypes[2])


def with_capture(genotype: RealGenotype, captured: Iterable[str], record: dict | None = None, *,
                 haplotypes: Mapping[int, Iterable[str]] | None = None) -> RealGenotype:
    """``genotype`` with ``captured`` set to the genes ``captured`` (the span clauses, ``captured_genes``) and every
    gene the donor carries an allele of, and ``resolved`` set to the genes known on both haplotypes (the module
    docstring). ``haplotypes`` is ``spanned_by_haplotype`` of the same spans; without it, a gene is resolved by the
    span clauses or the alleles' haplotypes alone. ``source["capture"]`` holds ``record`` (``donor_capture``'s
    provenance), then ``spanned``, ``carried_not_spanned``, ``spanned_not_carried`` (the candidate deletions or
    assembly gaps) and ``half_resolved`` (carried genes not resolved), sorted gene lists, and the ``captured`` and
    ``resolved`` counts. ValueError for a haplotype other than 0, 1 or 2, or for ``haplotypes`` whose span clauses
    are not ``captured``."""
    spanned = frozenset(captured)
    carried = {allele.gene for allele in genotype.alleles}
    known = {haplotype: {allele.gene for allele in genotype.alleles if allele.haplotype == haplotype}
             for haplotype in HAPLOTYPES}
    if haplotypes is not None:
        unknown = set(haplotypes) - set(HAPLOTYPES)
        if unknown:
            raise ValueError(f"haplotype {sorted(unknown)[0]} is not 0 (unphased), 1 or 2")
        by_haplotype = {haplotype: frozenset(haplotypes.get(haplotype, ())) for haplotype in HAPLOTYPES}
        if _span_clauses(by_haplotype) != spanned:
            raise ValueError("the spans per haplotype do not give the captured genes")
        for haplotype in HAPLOTYPES:
            known[haplotype] |= by_haplotype[haplotype]
    resolved = frozenset(spanned | known[0] | (known[1] & known[2]))
    summary = {"spanned": sorted(spanned), "carried_not_spanned": sorted(carried - spanned),
               "spanned_not_carried": sorted(spanned - carried), "half_resolved": sorted(carried - resolved),
               "captured": len(spanned | carried), "resolved": len(resolved)}
    return replace(genotype, captured=spanned | carried, resolved=resolved,
                   source={**genotype.source, "capture": {**(record or {}), **summary}})


def donor_capture(genotype: RealGenotype, bam_path, bed_path, *, timeout: float | None = None) -> RealGenotype:
    """``with_capture`` of ``genotype`` from its donor's contig BAM ``bam_path`` and the gene coordinates
    ``bed_path``, with the record of how it was computed: ``kind``, ``image`` (``check_image``), the whole
    ``command``, ``bam`` and ``gene_coords`` (path and SHA-256; the BED's gene count), ``genotype`` (the path and
    SHA-256 of the genotype's CSV, ``source["path"]`` and ``source["sha256"]``), ``spans`` per haplotype (``h0``,
    ``h1``, ``h2``), ``code_sha256`` (this module) and ``fingerprint`` (SHA-256 over the BAM's, the BED's and the
    genotype CSV's SHA-256s, the pinned image and the code's SHA-256). ValueError for a BAM that is not the
    genotype's VDJbase sample (``source["sample"]``, when it has one) or a record ``parse_sam_spans`` refuses (the
    error names the BAM); ContainerError as ``contig_spans``."""
    bam, bed = Path(bam_path), Path(bed_path)
    sample = genotype.source.get("sample")
    if sample and bam.stem != sample:
        raise ValueError(f"{bam} is not the contig BAM of {genotype.donor}'s VDJbase sample {sample}")
    coords = read_gene_coords(bed)
    bam_sha256, bed_sha256 = sha256_file(bam), sha256_file(bed)
    text, argv, image = _view(bam, timeout)
    try:
        spans = parse_sam_spans(text)
    except ValueError as error:
        raise ValueError(f"{bam} ({genotype.donor}): {error}") from None
    # the carried genes are captured too (``with_capture``), so the genotype's CSV is an input like the BAM and BED
    genotype_csv = {"path": genotype.source.get("path"), "sha256": genotype.source.get("sha256")}
    fingerprint = hashlib.sha256(json.dumps({"bam": bam_sha256, "gene_coords": bed_sha256, "image": SAMTOOLS_IMAGE,
                                             "genotype": genotype_csv["sha256"], "code": _CODE_SHA256},
                                            sort_keys=True).encode()).hexdigest()
    record = {"kind": "alignair_assay.real.locus.capture", "image": image, "command": argv,
              "bam": {"path": str(bam), "sha256": bam_sha256},
              "gene_coords": {"path": str(bed), "sha256": bed_sha256, "genes": len(coords)},
              "genotype": genotype_csv,
              "spans": {f"h{haplotype}": sum(span[0] == haplotype for span in spans) for haplotype in HAPLOTYPES},
              "code_sha256": _CODE_SHA256, "fingerprint": fingerprint}
    haplotypes = spanned_by_haplotype(spans, coords)
    return with_capture(genotype, _span_clauses(haplotypes), record, haplotypes=haplotypes)
