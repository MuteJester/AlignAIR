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
    CARRIED,
    DELETED,
    PRESENT,
    SAMTOOLS_IMAGE,
    ContigAlignment,
    SpanEvidence,
    captured_genes,
    contig_spans,
    donor_capture,
    interval_bases,
    parse_sam_alignments,
    parse_sam_spans,
    read_gene_coords,
    samtools_command,
    span_evidence,
    spanned_by_haplotype,
    with_capture,
)

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
needs_data = pytest.mark.skipif(not DATA, reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root")
needs_docker = pytest.mark.skipif(not DOCKER, reason="set ALIGNAIR_ASSAY_DOCKER=1 to run the samtools container")


def _sam(*records: tuple) -> str:
    """SAM text without a header: ``(qname, flag, rname, pos, cigar, rg)`` per record, with no SEQ stored (``*``) and
    the rest filler; a seventh field, when given, is the SEQ."""
    return "".join(f"{qname}\t{flag}\t{rname}\t{pos}\t254\t{cigar}\t*\t0\t0\t{seq[0] if seq else '*'}\t*\tnp:i:1"
                   f"\tRG:Z:{rg}\n" for qname, flag, rname, pos, cigar, rg, *seq in records)


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
    # a stored SEQ must be as long as the bases its CIGAR reads (M, I, S, =, X)
    with pytest.raises(ValueError, match="SEQ of 9 bases"):
        parse_sam_alignments(_sam((_contig(0), 0, "igh", 1, "2S5M3I", 0, "ACGTACGTA")))


def test_parse_sam_alignments_keeps_the_bases_and_parse_sam_spans_projects_it():
    sam = _sam((_contig(0), 0, "igh", 101, "2S3=1D2X1I2=", 1, "GGACGTAACC"),
               (_contig(2, index=4), 16, "igh", 2001, "300M", 2),
               (_contig(1), 256, "igh", 7001, "10M", 0))
    found = parse_sam_alignments(sam)
    assert found == [
        ContigAlignment(haplotype=0, start=100, end=108, cigar_ops=((2, "S"), (3, "="), (1, "D"), (2, "X"), (1, "I"),
                                                                    (2, "=")),
                        seq="GGACGTAACC", contig=_contig(0)),
        ContigAlignment(haplotype=2, start=2000, end=2300, cigar_ops=((300, "M"),), seq="*",
                        contig=_contig(2, index=4))]
    assert parse_sam_spans(sam) == [(aln.haplotype, aln.start, aln.end) for aln in found] == [
        (0, 100, 108), (2, 2000, 2300)]
    assert parse_sam_spans(SAM) == [(aln.haplotype, aln.start, aln.end) for aln in parse_sam_alignments(SAM)]


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


def test_spanned_by_haplotype():
    spans = [(0, 2000, 2100), (1, 900, 1400), (1, 3900, 4400), (2, 950, 1350)]
    assert spanned_by_haplotype(spans, COORDS) == {0: {"IGHD3-3"}, 1: {"IGHV1-2", "IGHV3-23"}, 2: {"IGHV1-2"}}
    assert captured_genes(spans, COORDS) == {"IGHD3-3", "IGHV1-2"}
    assert spanned_by_haplotype([], COORDS) == {0: frozenset(), 1: frozenset(), 2: frozenset()}
    with pytest.raises(ValueError, match="haplotype 3"):
        spanned_by_haplotype([(3, 900, 1400)], COORDS)


def _alignment(start: int, cigar: str, seq: str, haplotype: int = 0) -> ContigAlignment:
    ops = tuple((int(count), op) for count, op in __import__("re").findall(r"(\d+)([MIDNSHP=X])", cigar))
    end = start + sum(count for count, op in ops if op in "MDN=X")
    return ContigAlignment(haplotype=haplotype, start=start, end=end, cigar_ops=ops, seq=seq,
                           contig=_contig(haplotype, start, end))


def test_interval_bases_reads_the_contig_over_an_interval():
    # reference 100..: 3 aligned (ACG), 2 deleted, 2 aligned (TA), an insertion of 2 (GG), 4 aligned (CCTT);
    # a soft clip of 2 first
    aln = _alignment(100, "2S3=2D2X2I4=", "NNACGTAGGCCTT")
    assert aln.end == 111
    # the whole alignment: every aligned base, the inserted ones inside it, and the 2 deleted reference bases
    assert interval_bases(aln, 100, 111) == ("ACGTAGGCCTT", 9, 2)
    # an interval past the insertion's flank keeps it; one ending at it leaves it out
    assert interval_bases(aln, 104, 109) == ("TAGGCC", 4, 1)
    assert interval_bases(aln, 100, 107) == ("ACGTA", 5, 2)
    assert interval_bases(aln, 107, 111) == ("CCTT", 4, 0)
    # deletions and skips (N) both count as deleted; bases are uppercased
    skip = _alignment(0, "2M3N2M", "acgt")
    assert interval_bases(skip, 0, 7) == ("ACGT", 4, 3)
    # hard clips and padding read no SEQ base
    assert interval_bases(_alignment(0, "3H2=1P2=", "ACGT"), 0, 4) == ("ACGT", 4, 0)
    # no SEQ stored: the counts only
    assert interval_bases(_alignment(0, "4M", "*"), 1, 3) == (None, 2, 0)
    with pytest.raises(ValueError, match="not inside"):
        interval_bases(aln, 99, 105)


# A synthetic igh reference and the alleles of its genes (``COORDS``): IGHV1-2*02 is the reference's own sequence over
# its interval, IGHV3-23 has two alleles, and IGHV4-28*01 is another gene's sequence, to be found in a contig over
# IGHV3-23's interval.
LOCUS = "".join(__import__("random").Random(7).choices("ACGT", k=5000))
V12 = LOCUS[1000:1300]
V323 = LOCUS[4000:4300]
V323_02 = V323[:150] + ("A" if V323[150] != "A" else "C") + V323[151:]
V428 = "".join(__import__("random").Random(8).choices("ACGT", k=300))
LABELS = {"IGHV1-2*02": V12, "IGHV3-23*01": V323, "IGHV3-23*02": V323_02, "IGHV4-28*01": V428,
          "IGHD3-3*01": LOCUS[2050:2080], "IGHJ4*02": LOCUS[3000:3050]}
_RC = str.maketrans("ACGT", "TGCA")


def _over(haplotype: int, start: int, end: int, *, put: dict[int, str] | None = None, index: int = 0,
          cigar: str | None = None, seq: str | None = None) -> tuple:
    """A SAM record of a contig of ``haplotype`` aligned to ``LOCUS[start:end]``, with ``put`` (reference start ->
    bases of the same length) written into it; ``cigar`` and ``seq`` replace the plain all-``M`` alignment."""
    bases = list(LOCUS[start:end])
    for at, text in (put or {}).items():
        bases[at - start:at - start + len(text)] = text
    return (_contig(haplotype, start, end, index), 0, "igh", start + 1, cigar or f"{end - start}M", 0,
            seq if seq is not None else "".join(bases))


def _genotype(*alleles: RealAllele) -> RealGenotype:
    return RealGenotype(donor="SC-23", alleles=alleles, source={"sample": "P25_I58_S1", "sha256": "c" * 64})


def _v(gene: str, haplotype: int, sequence: str, allele: str = "01") -> RealAllele:
    return RealAllele(allele_id=f"{gene}*{allele}", gene=gene, segment=gene[3].lower(), sequence=sequence,
                      gapped=sequence if gene[3] == "V" else None, haplotype=haplotype, functional=True, shared=(),
                      iuis=())


def _states(evidence) -> dict:
    return {(ev.gene, ev.haplotype, ev.contig.split("_i=")[1][0]): ev.state for ev in evidence}


GENOTYPE = RealGenotype(donor="SC-23", alleles=(_v("IGHV1-2", 1, V12), _v("IGHV1-2", 2, V12, "02"),
                                                 _v("IGHV3-25", 1, "ACGT" * 75), _v("IGHJ4", 0, LOCUS[3000:3050])),
                        source={"kind": "vdjbase_genomic_csv", "path": "/data/P25_I58_S1.csv", "sha256": "c" * 64,
                                "sample": "P25_I58_S1"})


def test_span_evidence_per_gene_and_spanning_alignment():
    """Each gene spanned by an alignment gets one record of what the alignment holds over its interval: deleted
    (half or more of it), the donor's allele of the gene (exactly, either strand, contained either way), or another
    sequence, labelled for the report by its best OGRDB label when the donor does not carry the gene."""
    genotype = _genotype(_v("IGHV3-23", 1, V323), _v("IGHV1-2", 0, V12))
    sam = _sam(
        _over(0, 900, 1400),                                            # IGHV1-2 as carried
        _over(1, 3900, 4400),                                           # IGHV3-23*01 as carried
        _over(2, 3900, 4400, put={4000: V323_02}),                      # another allele: not carried
        _over(0, 3900, 4400, index=1, put={4000: V428}),                # another gene's sequence
        _over(0, 3900, 4400, index=2, cigar="100M250D150M",
              seq=LOCUS[3900:4000] + LOCUS[4250:4400]),                 # 250 of 300 bases deleted
        _over(1, 3900, 4400, index=3, seq="*"),                         # no SEQ stored
        _over(2, 3990, 4310, index=1, put={4000: V323[::-1].translate(_RC)}),  # carried, reverse strand
        _over(1, 4050, 4400, index=4),                                  # does not span IGHV3-23
    )
    evidence = span_evidence(parse_sam_alignments(sam), COORDS, genotype, LABELS)
    assert all(isinstance(ev, SpanEvidence) for ev in evidence)
    assert _states(evidence) == {
        ("IGHV1-2", 0, "0"): CARRIED, ("IGHV3-23", 1, "0"): CARRIED, ("IGHV3-23", 2, "0"): PRESENT,
        ("IGHV3-23", 0, "1"): PRESENT, ("IGHV3-23", 0, "2"): DELETED, ("IGHV3-23", 1, "3"): PRESENT,
        ("IGHV3-23", 2, "1"): CARRIED}
    deleted = next(ev for ev in evidence if ev.state == DELETED)
    assert (deleted.aligned, deleted.deleted) == (50, 250)
    # the label is for the report and only where the donor lacks the gene (here it carries IGHV3-23)
    assert all(ev.label is None for ev in evidence)
    lacking = span_evidence(parse_sam_alignments(sam), COORDS, _genotype(_v("IGHV1-2", 0, V12)), LABELS)
    labelled = {(ev.haplotype, ev.contig.split("_i=")[1][0]): ev for ev in lacking if ev.gene == "IGHV3-23"}
    assert labelled[1, "0"].label == "IGHV3-23*01" and labelled[1, "0"].identity == labelled[1, "0"].coverage == 1.0
    assert labelled[1, "0"].holds == "an allele of the gene"
    assert labelled[2, "0"].label == "IGHV3-23*02" and labelled[2, "1"].label == "IGHV3-23*01"  # either strand
    assert labelled[0, "1"].label == "IGHV4-28*01" and labelled[0, "1"].holds == "another gene's sequence"
    assert labelled[0, "2"].state == DELETED and labelled[0, "2"].label is None
    assert labelled[1, "3"].state == PRESENT and labelled[1, "3"].label is None  # SEQ *: present, conservatively
    assert labelled[1, "3"].holds == "bases not stored"
    # a sequence like no label closely enough
    noisy = _sam(_over(0, 3900, 4400, put={4000 + k: "A" if V323[k] != "A" else "C" for k in range(0, 300, 20)}))
    (ev,) = [ev for ev in span_evidence(parse_sam_alignments(noisy), COORDS, _genotype(), LABELS)
             if ev.gene == "IGHV3-23"]
    assert ev.label == "IGHV3-23*01" and 0.9 < ev.identity < 0.98 and ev.holds == "no allele closely"


def test_a_spanned_gene_not_called_is_uncalled_unless_every_alignment_shows_it_deleted():
    """VDJbase's calls can miss a gene the contigs hold: a gene spanned and not carried is ``uncalled`` (its calls
    are excluded) when any alignment of the clause that captured it holds sequence there, and ``absent`` (a call to
    it is ``unsupported_gene``) only when every one is deleted. Its label never decides."""
    deleted = {"cigar": "100M250D150M", "seq": LOCUS[3900:4000] + LOCUS[4250:4400]}
    cases = {
        "present on h0": ([_over(0, 3900, 4400)], "uncalled"),
        "deleted on h0": ([_over(0, 3900, 4400, **deleted)], "absent"),
        "another gene's sequence": ([_over(0, 3900, 4400, put={4000: V428})], "uncalled"),
        "no SEQ stored": ([_over(0, 3900, 4400, seq="*")], "uncalled"),
        "h1 present, h2 deleted": ([_over(1, 3900, 4400), _over(2, 3900, 4400, **deleted)], "uncalled"),
        "h1 and h2 deleted": ([_over(1, 3900, 4400, **deleted), _over(2, 3900, 4400, **deleted)], "absent"),
        # only the capturing clause's alignments count: a lone h1 contig beside a deleted h0 one is not one
        "h0 deleted, a lone h1 present": ([_over(0, 3900, 4400, **deleted), _over(1, 3900, 4400)], "absent"),
        "two h0, one present": ([_over(0, 3900, 4400, **deleted), _over(0, 3900, 4400, index=1)], "uncalled"),
    }
    genotype = _genotype(_v("IGHV1-2", 0, V12))
    for case, (records, expected) in cases.items():
        evidence = span_evidence(parse_sam_alignments(_sam(*records)), COORDS, genotype, LABELS)
        captured = with_capture(genotype, evidence)
        assert "IGHV3-23" in captured.captured, case
        assert ("IGHV3-23" in captured.uncalled) == (expected == "uncalled"), case
        record = captured.source["capture"]
        assert record["absent"] == (["IGHV3-23"] if expected == "absent" else []), case
        assert record["spanned_not_carried"] == ["IGHV3-23"], case
        if expected == "uncalled":
            kept = record["uncalled"]["IGHV3-23"]
            assert {entry["contig"] for entry in kept} <= {ev.contig for ev in evidence}, case
            assert all({"haplotype", "contig", "aligned", "deleted", "state", "label", "identity", "holds"}
                       <= set(entry) for entry in kept), case
    # the IUIS map's labels only name what a contig holds: W-97's IGHV4-30-2 interval holds IGHV4-28*01 (a CNV)
    evidence = span_evidence(parse_sam_alignments(_sam(_over(0, 3900, 4400, put={4000: V428}))), COORDS, genotype,
                             LABELS)
    (entry,) = with_capture(genotype, evidence).source["capture"]["uncalled"]["IGHV3-23"]
    assert entry["label"] == "IGHV4-28*01" and entry["holds"] == "another gene's sequence"


def test_a_haplotype_without_a_call_is_known_only_where_its_contigs_show_the_gene_deleted_or_a_carried_allele():
    """h=0 is unphased, not homozygous: an unphased contig that holds another sequence than the carried allele makes
    the gene unresolved, and a haplotype without a call is known only when every one of its spanning alignments is
    deleted at the gene or holds a carried allele exactly. An unknown haplotype makes a carried gene half resolved."""
    deleted = {"cigar": "100M250D150M", "seq": LOCUS[3900:4000] + LOCUS[4250:4400]}
    on_h1 = _genotype(_v("IGHV3-23", 1, V323))
    cases = {
        "h2 holds the carried allele": (on_h1, [_over(2, 3900, 4400)], True),
        "h2 deleted": (on_h1, [_over(2, 3900, 4400, **deleted)], True),
        "h2 holds another allele": (on_h1, [_over(2, 3900, 4400, put={4000: V323_02})], False),
        "h2 not spanned": (on_h1, [], False),
        "h2 holds the allele and another": (on_h1, [_over(2, 3900, 4400), _over(2, 3900, 4400, index=1,
                                                                                  put={4000: V323_02})], False),
        "h2 not stored": (on_h1, [_over(2, 3900, 4400, seq="*")], False),
        "h0 holds the carried allele": (on_h1, [_over(0, 3900, 4400)], True),
        "h0 differs from it": (on_h1, [_over(0, 3900, 4400, put={4000: V323_02})], False),
        # a deleted alignment leaves its haplotype known: hemizygous there (only a present one unresolves the gene)
        "h0 deleted, h1 called": (on_h1, [_over(0, 3900, 4400, **deleted)], True),
        "h0 deleted and holding the carried allele": (on_h1, [_over(0, 3900, 4400, **deleted),
                                                              _over(0, 3900, 4400, index=1)], True),
        "h0 deleted and differing": (on_h1, [_over(0, 3900, 4400, **deleted),
                                             _over(0, 3900, 4400, index=1, put={4000: V323_02})], False),
        "h0 differs, h1 and h2 called": (_genotype(_v("IGHV3-23", 1, V323), _v("IGHV3-23", 2, V323)),
                                         [_over(0, 3900, 4400, put={4000: V428})], False),
        "an unphased call": (_genotype(_v("IGHV3-23", 0, V323)), [], True),
        "called on both": (_genotype(_v("IGHV3-23", 1, V323), _v("IGHV3-23", 2, V323_02, "02")),
                           [_over(1, 3900, 4400), _over(2, 3900, 4400, put={4000: V428})], True),
    }
    for case, (genotype, records, resolved) in cases.items():
        evidence = span_evidence(parse_sam_alignments(_sam(*records)), COORDS, genotype, LABELS)
        captured = with_capture(genotype, evidence)
        assert ("IGHV3-23" in captured.resolved) is resolved, case
        assert (captured.source["capture"]["half_resolved"] == ["IGHV3-23"]) is not resolved, case
        assert captured.uncalled == frozenset(), case
        if not resolved:
            unknown = captured.source["capture"]["unknown_haplotypes"]["IGHV3-23"]
            assert unknown and all(set(entry) == {"haplotype", "why"} for entry in unknown), case


def test_with_capture_adds_every_carried_gene():
    """A gene the donor carries is captured whatever the spans say: IGHV3-25 is not in the BED, and IGHJ4 is
    spanned on one haplotype only. IGHV3-23 is spanned (on h0, holding sequence) but not carried: uncalled."""
    sam = _sam(_over(0, 900, 1400), _over(0, 3900, 4400, index=1), _over(1, 2900, 3100))
    evidence = span_evidence(parse_sam_alignments(sam), COORDS, GENOTYPE, LABELS)
    genotype = with_capture(GENOTYPE, evidence)
    assert genotype.captured == {"IGHV1-2", "IGHV3-23", "IGHV3-25", "IGHJ4"}
    assert genotype.uncalled == {"IGHV3-23"}
    assert genotype.alleles == GENOTYPE.alleles and GENOTYPE.captured is None and GENOTYPE.uncalled is None
    assert genotype.source["sample"] == "P25_I58_S1" and "capture" not in GENOTYPE.source
    record = genotype.source["capture"]
    assert {key: record[key] for key in ("spanned", "carried_not_spanned", "spanned_not_carried", "absent",
                                         "half_resolved", "captured", "resolved")} == {
        "spanned": ["IGHV1-2", "IGHV3-23"], "carried_not_spanned": ["IGHJ4", "IGHV3-25"],
        "spanned_not_carried": ["IGHV3-23"], "absent": [], "half_resolved": ["IGHV3-25"], "captured": 4,
        "resolved": 2}
    assert list(record["uncalled"]) == ["IGHV3-23"]
    recorded = with_capture(GENOTYPE, evidence, {"command": ["docker", "run"], "spanned": "ignored"})
    assert recorded.source["capture"]["command"] == ["docker", "run"]
    assert recorded.source["capture"]["spanned"] == ["IGHV1-2", "IGHV3-23"]
    # the span clauses of the evidence are the span rule's (captured_genes)
    spans = parse_sam_spans(sam)
    assert set(record["spanned"]) == captured_genes(spans, COORDS)


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
    assert [argv[i + 1] for i, part in enumerate(argv) if part == "--mount"] == [
        f"type=bind,source={tmp_path},target=/data,readonly"]
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


def _labels_map(sha256: str = "f" * 64) -> IuisMap:
    """An IUIS map holding ``LABELS``: what ``donor_capture`` names a contig's sequence by."""
    return IuisMap(labels_by_shared={}, shared_by_sequence={},
                   labels_by_sequence={sequence: (label,) for label, sequence in LABELS.items()},
                   reference_digest="0" * 64, source={"sha256": sha256})


def test_donor_capture_records_how_it_was_computed(tmp_path, monkeypatch):
    bam, bed = tmp_path / "P25_I58_S1.bam", tmp_path / "gene_coords.bed"
    bam.write_bytes(b"BAM synthetic")
    bed.write_bytes(BED.encode())
    sam = _sam(_over(0, 900, 1400), _over(1, 3900, 4400))
    monkeypatch.setattr(container.subprocess, "run", _Docker(sam))
    iuis = _labels_map()
    genotype = donor_capture(GENOTYPE, bam, bed, iuis)
    assert genotype.captured == {"IGHV1-2", "IGHV3-25", "IGHJ4"}
    # IGHV3-23 lies inside the haplotype-1 contig only, and IGHV3-25 is carried on haplotype 1 only
    assert genotype.resolved == {"IGHV1-2", "IGHJ4"} and genotype.uncalled == frozenset()
    record = genotype.source["capture"]
    assert record["half_resolved"] == ["IGHV3-25"] and record["uncalled"] == {} and record["absent"] == []
    assert record["command"] == samtools_command(bam)
    assert record["image"]["image"] == SAMTOOLS_IMAGE
    assert record["image"]["repo_digests"] == [container.pinned_repo_digest(SAMTOOLS_IMAGE)]
    assert record["bam"] == {"path": str(bam), "sha256": sha256_file(bam)}
    assert record["gene_coords"] == {"path": str(bed), "sha256": sha256_file(bed), "genes": 4}
    assert record["spans"] == {"h0": 1, "h1": 1, "h2": 0}
    assert record["spanned"] == ["IGHV1-2"] and record["carried_not_spanned"] == ["IGHJ4", "IGHV3-25"]
    assert record["code_sha256"] == sha256_file(locus.__file__)
    assert record["genotype"] == {"path": "/data/P25_I58_S1.csv", "sha256": "c" * 64}
    assert record["ogrdb_sha256"] == "f" * 64
    # the fingerprint follows the inputs (the genotype CSV and the IUIS map's OGRDB file among them), the image and
    # the code, and nothing else
    again = donor_capture(replace(genotype, source={"sha256": "c" * 64}), bam, bed,
                          iuis).source["capture"]["fingerprint"]
    assert again == donor_capture(genotype, bam, bed, iuis).source["capture"]["fingerprint"]  # recorded: same
    assert again == record["fingerprint"]
    other_csv = replace(GENOTYPE, source={**GENOTYPE.source, "sha256": "e" * 64})
    assert donor_capture(other_csv, bam, bed, iuis).source["capture"]["fingerprint"] != record["fingerprint"]
    other_ogrdb = donor_capture(GENOTYPE, bam, bed, _labels_map("a" * 64)).source["capture"]["fingerprint"]
    assert other_ogrdb != record["fingerprint"]
    bed.write_bytes(BED.encode() + b"igh\t6000\t6300\tIGHV1-3\n")
    assert donor_capture(GENOTYPE, bam, bed, iuis).source["capture"]["fingerprint"] != record["fingerprint"]


def test_donor_capture_refuses_another_donors_bam_and_names_the_bam_of_a_bad_record(tmp_path, monkeypatch):
    bed = tmp_path / "gene_coords.bed"
    bed.write_bytes(BED.encode())
    other = tmp_path / "P25_I61_S1.bam"
    other.write_bytes(b"BAM synthetic")
    docker = _Docker(_sam(("contig_7", 0, "igh", 901, "500M", 0)))
    monkeypatch.setattr(container.subprocess, "run", docker)
    with pytest.raises(ValueError, match="not the contig BAM of SC-23's VDJbase sample P25_I58_S1"):
        donor_capture(GENOTYPE, other, bed, _labels_map())
    assert docker.calls == []  # refused before any container
    bam = tmp_path / "P25_I58_S1.bam"
    bam.write_bytes(b"BAM synthetic")
    with pytest.raises(ValueError, match=r"P25_I58_S1\.bam \(SC-23\): SAM line 1 \(contig_7\).*h=<0\|1\|2>"):
        donor_capture(GENOTYPE, bam, bed, _labels_map())


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
# Per donor: carried V genes with an unknown haplotype: not spanned, or spanned and holding a sequence the genotype
# does not call (W-133's IGHV3-30-5 among them).
HALF_RESOLVED_V = {"SC-23": ["IGHV3-15", "IGHV3-62", "IGHV4-28"], "SC-11": ["IGHV3-15", "IGHV3-25", "IGHV3-74"],
                   "W-97": ["IGHV3-62"], "W-133": ["IGHV3-21", "IGHV3-30", "IGHV3-30-5", "IGHV3-48", "IGHV3-7"],
                   "SC-4": ["IGHV3-74", "IGHV4-59", "IGHV4-61"],
                   "SC-6": ["IGHV3-30-5", "IGHV3-43D", "IGHV4-28", "IGHV4-31"],
                   "W-89": ["IGHV3-16", "IGHV3-23D", "IGHV3-30-5", "IGHV3-33", "IGHV4-4"],
                   "W-131": ["IGHV3-25", "IGHV3-35", "IGHV4-4"]}
# Per donor: the genes the contigs span and hold that the genotype does not call. None is absent (deleted on every
# capturing contig), so no call on these donors can be unsupported_gene.
UNCALLED = {"SC-23": ["IGHV4-30-2"], "SC-11": [], "W-97": ["IGHV3-11", "IGHV3-15", "IGHV4-30-2"], "W-133": [],
            "SC-4": ["IGHV4-31"], "SC-6": ["IGHD1-14", "IGHD1-26", "IGHD3-16", "IGHD3-22"], "W-89": ["IGHD2-8"],
            "W-131": ["IGHD1-7", "IGHD2-8", "IGHD3-3", "IGHD5-5"]}


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
    captured = donor_capture(genotype, root / SAMPLES / "P25_I58_S1.bam", root / GENE_COORDS, iuis)
    assert genotype.genes("v") <= captured.captured
    record = captured.source["capture"]
    assert record["spans"] == {"h0": 15, "h1": 49, "h2": 39}  # the 103 primary alignments of 674 records
    # IGHV3-15 is carried on haplotype 1; the haplotype-2 contig starts 9 bases inside it
    assert set(record["carried_not_spanned"]) == NO_COORDS | {"IGHV3-15"}
    assert record["spanned_not_carried"] == ["IGHV4-30-2"]
    assert len(record["spanned"]) == 48 and len(captured.captured) == 52
    # IGHV4-30-2 is not a gene SC-23 lacks: its unphased contig holds IGHV4-30-2*01 exactly
    assert captured.uncalled == {"IGHV4-30-2"} and record["absent"] == []
    (entry,) = record["uncalled"]["IGHV4-30-2"]
    assert (entry["haplotype"], entry["state"], entry["label"], entry["identity"]) == (0, "present", "IGHV4-30-2*01",
                                                                                       1.0)
    assert entry["holds"] == "an allele of the gene"
    # IGHV3-15's second allele is unknown, as are IGHV3-62's (carried on one haplotype, and not in the BED) and
    # IGHV4-28's (its haplotype-2 contig holds a sequence the genotype does not call)
    assert record["half_resolved"] == ["IGHV3-15", "IGHV3-62", "IGHV4-28"] and len(captured.resolved) == 48
    assert [entry["haplotype"] for entry in record["unknown_haplotypes"]["IGHV4-28"]] == [2]


@needs_data
@needs_docker
def test_the_pilot_donors_capture(reference, iuis):
    root = Path(DATA)
    coords = read_gene_coords(root / GENE_COORDS)
    found, half, uncalled = {}, {}, {}
    for donor, sample in PILOT.items():
        genotype = load_vdjbase_csv(root / SAMPLES / f"{sample}.csv", donor, reference, iuis)
        captured = donor_capture(genotype, root / SAMPLES / f"{sample}.bam", root / GENE_COORDS, iuis)
        record = captured.source["capture"]
        found[donor] = tuple(sum(gene.startswith(kind) for gene in record["spanned"]) for kind in ("IGHV", "IGHD",
                                                                                                  "IGHJ"))
        assert NO_COORDS & genotype.genes("v") <= set(record["carried_not_spanned"])
        # every call is read off a contig: where a contig of the call's haplotype spans its gene, one of them holds
        # the called allele exactly (the evidence agrees with VDJbase's own extraction)
        bam_sam, _, _ = locus._view(root / SAMPLES / f"{sample}.bam", None)
        evidence = span_evidence(parse_sam_alignments(bam_sam), coords, genotype)
        for allele in genotype.alleles:
            mine = [ev for ev in evidence if ev.gene == allele.gene and ev.haplotype == allele.haplotype]
            assert not mine or any(ev.state == CARRIED for ev in mine), (donor, allele.allele_id)
        # no gene is resolved against its contigs: a resolved carried gene has no alignment holding another
        # sequence on haplotype 0 or on a haplotype without a call of it
        for gene in captured.resolved & {allele.gene for allele in genotype.alleles}:
            called = {allele.haplotype for allele in genotype.carried(gene)}
            assert not [ev for ev in evidence if ev.gene == gene and ev.state == PRESENT
                        and (ev.haplotype == 0 or ev.haplotype not in called)], (donor, gene)
        assert record["absent"] == [] and set(record["spanned_not_carried"]) == captured.uncalled
        half[donor] = [gene for gene in record["half_resolved"] if gene.startswith("IGHV")]
        uncalled[donor] = sorted(captured.uncalled)
    assert found == SPANNED
    assert half == HALF_RESOLVED_V
    assert uncalled == UNCALLED
