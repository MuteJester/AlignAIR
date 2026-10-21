"""Which IGH genes each donor's genomic contigs capture: the part of the locus its genotype truth covers.

R1 must tell apart a call to a gene the donor lacks from a call to a gene its long-read sequencing did not resolve.
In the first case the gene's region is captured, the donor carries no allele of it and its contigs show it deleted,
so the call is ``unsupported_gene``. In the second, or where the contigs hold a gene its genotype does not call, the
call is excluded. The donor's IGenotyper contigs, aligned to IGenotyper's
custom ``igh`` reference (1,193,129 bp), are in VDJbase's per-sample contig BAM (study P25, CC0). The gene
intervals on that reference are in IGenotyper's ``data/gene_coords.bed``, pinned by ``real.fetch``.

**The rule.** A gene is captured when the donor carries an allele of it (any segment, any haplotype: the allele was
read off a contig, so a carried gene is captured by definition), or when its interval lies wholly inside one contig
of haplotype 0 (unphased), or inside a contig of haplotype 1 and a contig of haplotype 2. Two contigs of one
haplotype that cover a gene only together do not capture it, and neither does a partial overlap.

**What a contig holds over a gene** (``span_evidence``). For every gene and every alignment that spans its interval,
the contig's bases over the interval (``interval_bases``: the bases aligned to it, ``M``, ``=`` and ``X``, with any
insertion inside it, and the count of its reference bases deleted, ``D`` and ``N``) say one of three things:

- ``deleted``: deletions cover at least half of the interval (``DELETED_SHARE``), or fewer than half of its bases are
  aligned;
- ``carried``: the bases exactly contain, or are exactly contained in, a carried allele of the gene, on either strand;
- ``present``: anything else, a record without its SEQ (``*``) among them, conservatively. For a gene the donor does
  not carry, the report names what it holds by its best OGRDB label (ungapped, best offset, either strand), with that
  label's identity and coverage: "an allele of the gene" or "another gene's sequence" at an identity of at least
  ``LABEL_IDENTITY`` (98%) over at least ``LABEL_COVERAGE`` (95%) of the label's sequence, else "no allele closely".
  The label never decides a class.

**Spanned and not called.** VDJbase's genotype can miss a gene the contigs hold (W-97's IGHV3-15 is 302 of 302
identical to IGHV3-15*01 on an unphased contig, and its IGHV4-30-2 interval holds IGHV4-28*01, a copy-number region),
so a gene spanned and not carried is not taken for one the donor lacks. Only the alignments of the clauses that
capture it count: every haplotype-0 alignment that spans it, and those of haplotypes 1 and 2 when both do. It is
``uncalled`` (``RealGenotype.uncalled``; R1 excludes every call to it) when any of them is ``present``, and ``absent``
(a call to it is ``unsupported_gene``) only when every one is ``deleted``.

**Resolved.** A carried gene is resolved when both of the donor's haplotypes are known there. A haplotype with a call
of the gene is known. One without is known only when it has alignments spanning the gene and each is ``deleted`` or
``carried`` there; one not spanned, or spanned and holding another sequence, is unknown. Haplotype 0 is unphased, not
homozygous: an unphased call, or haplotype-0 alignments that are each ``carried`` or ``deleted`` (a deleted one leaves
its haplotype known, hemizygous there), resolve the gene, while a ``present`` haplotype-0 alignment (it holds another
sequence than the carried alleles) leaves it unresolved. A carried gene that is not resolved is ``half_resolved``: an
allele on one of its haplotypes is unknown, its region not spanned (SC-23's IGHV3-15, whose haplotype-2 contig starts
9 bases inside it) or spanned but not called (W-133's IGHV3-30-5, whose unphased contigs hold sequences VDJbase did
not call), so R1 excludes every call to it (``real.metrics``). A gene ``absent`` on every capturing alignment is
resolved too.

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
(``parse_sam_alignments``; ``parse_sam_spans`` is its spans). ``donor_capture`` puts it all together and records in
the genotype's ``source["capture"]`` how the set was computed (the image and its digests, the whole command, the
SHA-256s of the inputs, the genotype's CSV and the IUIS map's OGRDB file among them, and of this module), a
fingerprint of those inputs, so a stored capture is reused only while it is current, and the evidence: each
``uncalled`` gene's capturing alignments (haplotype, contig, bases aligned and deleted, state, label, identity,
coverage and what it holds), the ``absent`` genes, and each half-resolved gene's ``unknown_haplotypes`` with why.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from alignair_assay.core.container import Mount, check_image, container_name, hardened_run, run_container
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.reference import gene_name
from alignair_assay.real.genotype import IuisMap, RealGenotype

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

# What an alignment holds over a gene's interval (``span_evidence``).
DELETED, CARRIED, PRESENT = "deleted", "carried", "present"
# An interval this share of which is deleted, or less than this share aligned, is deleted.
DELETED_SHARE = 0.5
# A label names what a contig holds (for the report only) at this identity over this share of its sequence.
LABEL_IDENTITY = 0.98
LABEL_COVERAGE = 0.95
# How far past full overlap a label is tried, each way.
LABEL_OVERHANG = 10

_CIGAR = re.compile(r"(\d+)([MIDNSHP=X])")
_REFERENCE_OPS = frozenset("MDN=X")
_QUERY_OPS = frozenset("MIS=X")
_ALIGNED_OPS = frozenset("M=X")
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
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


class ContigAlignment(NamedTuple):
    """One contig alignment IGenotyper uses (``parse_sam_alignments``)."""

    haplotype: int  # the contig name's h=: 0 (unphased), 1 or 2
    start: int  # its span on the reference, 0-based and half-open
    end: int
    cigar_ops: tuple[tuple[int, str], ...]  # (count, operation), in order
    seq: str  # SAM's SEQ, on the reference's forward strand; "*" when not stored
    contig: str = ""  # the contig's name


class IntervalBases(NamedTuple):
    """An alignment over a reference interval (``interval_bases``)."""

    bases: str | None  # the contig's bases aligned to it, inserted ones inside it included, uppercase; None for "*"
    aligned: int  # the interval's reference bases aligned (M, =, X)
    deleted: int  # the interval's reference bases deleted (D, N)


def parse_sam_alignments(text: str, *, reference: str = IGH_REFERENCE) -> list[ContigAlignment]:
    """Each contig alignment in SAM text ``text`` (``samtools view`` output; header lines are skipped) that IGenotyper
    uses, in order: the haplotype is the contig name's ``h=``, and the span is 0-based and half-open on ``reference``
    (the module docstring). Records with a ``SKIP_FLAGS`` flag are skipped unread. ValueError, naming the line and
    contig, for a line that is not a SAM record, a kept record whose name has no ``h=`` of 0, 1 or 2, a record on
    another reference, one without a valid position or CIGAR, or a stored SEQ whose length is not the bases its CIGAR
    reads (``M``, ``I``, ``S``, ``=``, ``X``)."""
    alignments = []
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
        operations = tuple((int(count), op) for count, op in _CIGAR.findall(cigar))
        if "".join(f"{count}{op}" for count, op in operations) != cigar:
            raise ValueError(f"{where}: {cigar!r} is not a CIGAR")
        length = sum(count for count, op in operations if op in _REFERENCE_OPS)
        if length == 0:
            raise ValueError(f"{where}: CIGAR {cigar} spans no reference base")
        seq = fields[9].split("\t", 1)[0]
        read = sum(count for count, op in operations if op in _QUERY_OPS)
        if seq != "*" and len(seq) != read:
            raise ValueError(f"{where}: a SEQ of {len(seq)} bases, where CIGAR {cigar} reads {read}")
        alignments.append(ContigAlignment(haplotype=int(found[1]), start=position - 1, end=position - 1 + length,
                                          cigar_ops=operations, seq=seq, contig=qname))
    return alignments


def parse_sam_spans(text: str, *, reference: str = IGH_REFERENCE) -> list[tuple[int, int, int]]:
    """``(haplotype, start, end)`` of each alignment of ``parse_sam_alignments(text)``, in order, with its
    refusals."""
    return [(aln.haplotype, aln.start, aln.end) for aln in parse_sam_alignments(text, reference=reference)]


def interval_bases(alignment: ContigAlignment, start: int, end: int) -> IntervalBases:
    """What ``alignment`` holds over the reference interval ``[start, end)``, which it must span: the contig's bases
    aligned to it (``M``, ``=``, ``X``) and inserted inside it (an ``I`` between two of its bases), uppercase, or None
    when the record stores no SEQ; how many of its bases are aligned; how many are deleted (``D``, ``N``).
    ValueError for an interval the alignment does not span."""
    if not alignment.start <= start < end <= alignment.end:
        raise ValueError(f"[{start}, {end}) is not inside the alignment's span [{alignment.start}, {alignment.end})")
    stored = alignment.seq != "*"
    pieces, aligned, deleted = [], 0, 0
    position, read = alignment.start, 0
    for count, op in alignment.cigar_ops:
        if position >= end:
            break
        if op in _ALIGNED_OPS:
            low, high = max(position, start), min(position + count, end)
            if high > low:
                aligned += high - low
                if stored:
                    pieces.append(alignment.seq[read + low - position:read + high - position])
            position += count
            read += count
        elif op in "DN":
            deleted += max(0, min(position + count, end) - max(position, start))
            position += count
        elif op == "I":
            if start < position < end and stored:  # between two reference bases of the interval
                pieces.append(alignment.seq[read:read + count])
            read += count
        elif op == "S":
            read += count
    return IntervalBases(bases="".join(pieces).upper() if stored else None, aligned=aligned, deleted=deleted)


class SpanEvidence(NamedTuple):
    """What one alignment spanning a gene's interval holds there (``span_evidence``)."""

    gene: str
    haplotype: int
    contig: str
    aligned: int  # the interval's bases aligned
    deleted: int  # the interval's bases deleted
    state: str  # DELETED, CARRIED or PRESENT
    # for a present alignment on a gene the donor does not carry, with its bases: the best OGRDB label, its identity
    # over their overlap and the share of its sequence the overlap covers; None otherwise
    label: str | None = None
    identity: float | None = None
    coverage: float | None = None

    @property
    def holds(self) -> str:
        """What the alignment holds, in words, for the report: ``deleted``, ``a carried allele``, ``bases not
        stored`` (a present record without its SEQ, or one not labelled), ``an allele of the gene`` or ``another
        gene's sequence`` (a label close enough), or ``no allele closely``."""
        if self.state == DELETED:
            return "deleted"
        if self.state == CARRIED:
            return "a carried allele"
        if self.label is None:
            return "bases not stored"
        if self.identity < LABEL_IDENTITY or self.coverage < LABEL_COVERAGE:
            return "no allele closely"
        return "an allele of the gene" if gene_name(self.label) == self.gene else "another gene's sequence"

    def to_json(self) -> dict:
        """The evidence as JSON-ready data, with ``holds``."""
        return {"haplotype": self.haplotype, "contig": self.contig, "aligned": self.aligned, "deleted": self.deleted,
                "state": self.state, "label": self.label,
                "identity": None if self.identity is None else round(self.identity, 4),
                "coverage": None if self.coverage is None else round(self.coverage, 4), "holds": self.holds}


def span_evidence(alignments: Iterable[ContigAlignment], coords: Mapping[str, tuple[int, int]],
                  genotype: RealGenotype, labels: Mapping[str, str] | None = None) -> list[SpanEvidence]:
    """One ``SpanEvidence`` per gene of ``coords`` and alignment of ``alignments`` whose span holds the gene's whole
    interval, gene by gene in ``coords``' order: what the alignment holds there (the module docstring), against the
    donor's carried alleles of the gene (``genotype``) and, for a present alignment on a gene the donor does not
    carry, its best label among ``labels`` (OGRDB label -> sequence) of the gene's segment (``best_label``)."""
    alignments = list(alignments)
    carried: dict[str, list[str]] = defaultdict(list)
    for allele in genotype.alleles:
        carried[allele.gene].append(allele.sequence.upper())
    evidence = []
    for gene, (start, end) in coords.items():
        for alignment in alignments:
            if not alignment.start <= start < end <= alignment.end:
                continue
            found = interval_bases(alignment, start, end)
            label = identity = coverage = None
            if found.deleted >= DELETED_SHARE * (end - start) or found.aligned < DELETED_SHARE * (end - start):
                state = DELETED
            elif found.bases is not None and _holds_carried(found.bases, carried.get(gene, ())):
                state = CARRIED
            else:
                state = PRESENT
                if gene not in carried and labels and found.bases:
                    label, identity, coverage = best_label(found.bases, labels, segment=gene[3:4])
            evidence.append(SpanEvidence(gene=gene, haplotype=alignment.haplotype, contig=alignment.contig,
                                         aligned=found.aligned, deleted=found.deleted, state=state, label=label,
                                         identity=identity, coverage=coverage))
    return evidence


def _holds_carried(bases: str, alleles: Iterable[str]) -> bool:
    """Whether ``bases`` exactly contain, or are exactly contained in, one of ``alleles``, on either strand."""
    strands = (bases, bases.translate(_COMPLEMENT)[::-1])
    return any(allele in strand or strand in allele for allele in alleles for strand in strands if strand)


def best_label(bases: str, labels: Mapping[str, str], *,
               segment: str | None = None) -> tuple[str | None, float | None, float | None]:
    """The label of ``labels`` (label -> sequence; with ``segment``, only the IGH labels of that segment letter, V, D
    or J) whose sequence best matches ``bases``, ungapped, on either strand: at its best offset (full overlap, or up
    to ``LABEL_OVERHANG`` past it either way), the most matches over the label's length, then the first name. Returns
    the label, its identity over the overlap and the share of its sequence the overlap covers; ``(None, None, None)``
    when no label is tried."""
    best: tuple[float, str, float, float] | None = None
    forward = bases.upper()
    strands = (forward, forward.translate(_COMPLEMENT)[::-1])
    for label in sorted(labels):
        if segment and label[3:4].upper() != segment.upper():
            continue
        allele = labels[label].upper()
        size = len(allele)
        for strand in strands:
            low = max(min(0, len(strand) - size) - LABEL_OVERHANG, 1 - size)
            high = min(max(0, len(strand) - size) + LABEL_OVERHANG, len(strand) - 1)
            for offset in range(low, high + 1):
                first, last = max(0, -offset), min(size, len(strand) - offset)
                overlap = last - first
                if overlap <= 0:
                    continue
                matches = sum(a == b for a, b in zip(allele[first:last], strand[offset + first:offset + last]))
                if best is None or matches / size > best[0]:
                    best = (matches / size, label, matches / overlap, overlap / size)
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


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


def with_capture(genotype: RealGenotype, evidence: Iterable[SpanEvidence], record: dict | None = None) -> RealGenotype:
    """``genotype`` with its capture from ``evidence`` (``span_evidence`` of its contigs; the module docstring):
    ``captured``, the genes of the span clauses and every gene the donor carries an allele of; ``uncalled``, the
    spanned genes not carried that a capturing alignment holds; and ``resolved``, the carried genes known on both
    haplotypes and the ``absent`` ones. ``source["capture"]`` holds ``record`` (``donor_capture``'s provenance), then
    ``spanned``, ``carried_not_spanned``, ``spanned_not_carried`` (the uncalled and absent genes together),
    ``uncalled`` (gene -> its capturing alignments' evidence, ``SpanEvidence.to_json``), ``absent``,
    ``half_resolved`` (the carried genes not resolved) and ``unknown_haplotypes`` (each half-resolved gene's unknown
    haplotypes, with why), sorted, and the ``captured`` and ``resolved`` counts. ValueError for a haplotype other than
    0, 1 or 2."""
    by_gene: dict[str, dict[int, list[SpanEvidence]]] = defaultdict(lambda: {haplotype: [] for haplotype in HAPLOTYPES})
    for entry in evidence:
        if entry.haplotype not in HAPLOTYPES:
            raise ValueError(f"haplotype {entry.haplotype} is not 0 (unphased), 1 or 2")
        by_gene[entry.gene][entry.haplotype].append(entry)
    spanned = _span_clauses({haplotype: frozenset(gene for gene, spans in by_gene.items() if spans[haplotype])
                             for haplotype in HAPLOTYPES})
    carried = {allele.gene for allele in genotype.alleles}
    called: dict[str, set[int]] = defaultdict(set)
    for allele in genotype.alleles:
        called[allele.gene].add(allele.haplotype)
    uncalled, absent = {}, []
    for gene in sorted(spanned - carried):
        clause = _capturing(by_gene[gene])
        if all(entry.state == DELETED for entry in clause):
            absent.append(gene)
        else:
            uncalled[gene] = [entry.to_json() for entry in clause]
    resolved, unknown = set(absent), {}
    for gene in sorted(carried):
        why = _unknown_haplotypes(by_gene.get(gene) or {haplotype: [] for haplotype in HAPLOTYPES}, called[gene])
        if why:
            unknown[gene] = why
        else:
            resolved.add(gene)
    summary = {"spanned": sorted(spanned), "carried_not_spanned": sorted(carried - spanned),
               "spanned_not_carried": sorted(spanned - carried), "uncalled": uncalled, "absent": absent,
               "half_resolved": sorted(carried - resolved), "unknown_haplotypes": unknown,
               "captured": len(spanned | carried), "resolved": len(resolved)}
    return replace(genotype, captured=frozenset(spanned | carried), resolved=frozenset(resolved),
                   uncalled=frozenset(uncalled), source={**genotype.source, "capture": {**(record or {}), **summary}})


def _capturing(spans: Mapping[int, list[SpanEvidence]]) -> list[SpanEvidence]:
    """The alignments of the clauses that capture a gene: every haplotype-0 one, and those of haplotypes 1 and 2
    when both span it."""
    clause = list(spans[0])
    if spans[1] and spans[2]:
        clause += [*spans[1], *spans[2]]
    return clause


def _unknown_haplotypes(spans: Mapping[int, list[SpanEvidence]], called: set[int]) -> list[dict]:
    """A carried gene's unknown haplotypes, each ``{"haplotype", "why"}``; [] when the gene is resolved (the module
    docstring)."""
    # only a present alignment unresolves the gene: a deleted one leaves its haplotype known (hemizygous there)
    differing = [entry for entry in spans[0] if entry.state == PRESENT]
    if differing:
        return [{"haplotype": 0, "why": f"an unphased contig ({differing[0].contig}) is {differing[0].state} there, "
                                        "not a carried allele"}]
    if 0 in called or spans[0]:
        return []
    unknown = []
    for haplotype in (1, 2):
        if haplotype in called:
            continue
        other = [entry for entry in spans[haplotype] if entry.state not in (DELETED, CARRIED)]
        if not spans[haplotype]:
            unknown.append({"haplotype": haplotype, "why": "not spanned"})
        elif other:
            unknown.append({"haplotype": haplotype, "why": f"spanned, not called: a contig ({other[0].contig}) holds "
                                                           "another sequence than the carried alleles"})
    return unknown


def donor_capture(genotype: RealGenotype, bam_path, bed_path, iuis: IuisMap, *,
                  timeout: float | None = None) -> RealGenotype:
    """``with_capture`` of ``genotype`` from the ``span_evidence`` of its donor's contig BAM ``bam_path`` over the gene
    coordinates ``bed_path``, what the contigs hold named by the OGRDB labels of the IUIS map ``iuis``, with the
    record of how it was computed: ``kind``, ``image`` (``check_image``), the whole ``command``, ``bam`` and
    ``gene_coords`` (path and SHA-256; the BED's gene count), ``genotype`` (the path and SHA-256 of the genotype's
    CSV, ``source["path"]`` and ``source["sha256"]``), ``ogrdb_sha256`` (the IUIS map's OGRDB file), ``spans`` per
    haplotype (``h0``, ``h1``, ``h2``), ``code_sha256`` (this module) and ``fingerprint`` (SHA-256 over the BAM's,
    the BED's, the genotype CSV's and the OGRDB file's SHA-256s, the pinned image and the code's SHA-256).
    ValueError for a BAM that is not the genotype's VDJbase sample (``source["sample"]``, when it has one) or a
    record ``parse_sam_alignments`` refuses (the error names the BAM); ContainerError as ``contig_spans``."""
    bam, bed = Path(bam_path), Path(bed_path)
    sample = genotype.source.get("sample")
    if sample and bam.stem != sample:
        raise ValueError(f"{bam} is not the contig BAM of {genotype.donor}'s VDJbase sample {sample}")
    coords = read_gene_coords(bed)
    bam_sha256, bed_sha256 = sha256_file(bam), sha256_file(bed)
    text, argv, image = _view(bam, timeout)
    try:
        alignments = parse_sam_alignments(text)
    except ValueError as error:
        raise ValueError(f"{bam} ({genotype.donor}): {error}") from None
    # the carried genes are captured too (``with_capture``), so the genotype's CSV is an input like the BAM and BED;
    # the OGRDB file names what the contigs hold
    genotype_csv = {"path": genotype.source.get("path"), "sha256": genotype.source.get("sha256")}
    ogrdb_sha256 = iuis.source.get("sha256")
    fingerprint = hashlib.sha256(json.dumps({"bam": bam_sha256, "gene_coords": bed_sha256, "image": SAMTOOLS_IMAGE,
                                             "genotype": genotype_csv["sha256"], "ogrdb": ogrdb_sha256,
                                             "code": _CODE_SHA256}, sort_keys=True).encode()).hexdigest()
    record = {"kind": "alignair_assay.real.locus.capture", "image": image, "command": argv,
              "bam": {"path": str(bam), "sha256": bam_sha256},
              "gene_coords": {"path": str(bed), "sha256": bed_sha256, "genes": len(coords)},
              "genotype": genotype_csv, "ogrdb_sha256": ogrdb_sha256,
              "spans": {f"h{haplotype}": sum(aln.haplotype == haplotype for aln in alignments)
                        for haplotype in HAPLOTYPES},
              "code_sha256": _CODE_SHA256, "fingerprint": fingerprint}
    labels = {label: sequence for sequence, names in iuis.labels_by_sequence.items() for label in names}
    return with_capture(genotype, span_evidence(alignments, coords, genotype, labels), record)
