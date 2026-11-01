"""Tool-neutral truth for real reads: the observed span, the known answer, the naive flag and the clonotype.

Truth here comes only from comparing each preprocessed read with the donor's own genomic V alleles
(``real.genotype``); no tool output is read. This layer is pure: it takes strings, dicts and dataclasses and reads
no file.

**Reads are forward.** pRESTO orients every preprocessed read V to C (``real.preprocess``), so no reverse
complement is searched. On the pilot reads, of those without a forward Cys104 anchor, the reverse complement has
one on 3 of SC-23's 10,733 reads and on none of W-97's 50,000.

**Observed span** (``observed_span``). The primer-masked spans of a read (``primer_masked``, ``[start, end)`` in the
read; a multiplex read's VH primer at its start and C primer at its end) carry no germline evidence: the assembly
can fill a masked base from the other mate, so those bases are made ``N`` whatever they hold (``mask_primers``).
The observed span runs from the first to the last base that is neither masked nor ``N``. ``N`` never matches a
base, so a masked or uncalled base inside the span blocks an exact match over it.

**IMGT numbering** (``cys104_end``). Codon 104, the second cysteine, is IMGT nt 310-312: columns 309-311 (0-based)
of an IMGT-gapped V-REGION, whose ``.`` and ``-`` columns are gaps. ``cys104_end`` is the ungapped index just past
column 311. On OGRDB rev 8's IGHV3-23*01 it is 288, OGRDB's own ``fwr3_end`` (FR3 ends at IMGT nt 312). It is None
unless the codon at 309-311 is ``TGT`` or ``TGC``: VDJbase gaps an allele with an insertion by adding a column,
so its columns past the insertion are not IMGT positions. On the eight pilot donors 27 of 677 V alleles have no
cysteine there, each with a V-REGION indel, a stop codon or VDJbase's "Second cysteine not found" note; none is
functional. Five of them are one call, ``IGHV4-38-2*02_g4c_t82g_a83g_i93agt_...`` (SC-4, SC-11, SC-6 and twice
W-131): IGHV4-39*01's sequence gapped with a spurious insertion, in donors that carry IGHV4-39*01 as well.

**Known answer** (``known_match``, ``known_answer``). Each donor V allele with a Cys104 gives its region through
codon 104 (``sequence[:cys104_end]``) and an anchor, the last ``ANCHOR_LENGTH`` (30) nt of it. For each exact
occurrence of an anchor inside the observed span, ending at read index ``q``, the allele's window is its region
from the position facing the first observed base to Cys104: ``region[max(0, cys104_end - (q - start)):]``. A
5'RACE read starts in the UTR and leader, so its window is the whole region; a multiplex read's starts after the
VH primer. The allele matches when the read holds its window exactly, ending at ``q``.

- Matches are grouped by window sequence and position. One group is the answer: alleles whose windows are the
  same sequence, which includes alleles that differ only before the first observed base (or past Cys104: the
  rearranged read cannot tell them apart, as with the 318 rule) and one allele on both haplotypes.
- Two groups (two distinct window sequences, such as one allele's full region and another's that starts three
  bases later, or one window at two positions) give None: the read does not tell them apart. An exact window
  shorter than ``min_window`` counts here too, so a short match can block an answer but never be one.
- The answer's window must be at least ``min_window`` (200) nt long.
- Identity is the sequence (``real.genotype``): a donor V allele without a Cys104 of its own takes the Cys104 of a
  donor allele with exactly its sequence, so the IGHV4-38-2 call above joins IGHV4-39*01's answers. One that
  still has none (no twin, or no gapped form) cannot be windowed; when the answer's window occurs anywhere in such
  an allele's V-REGION, that allele could be the source too, and the answer is None.
- ``known_shared`` is the union of the set's ``shared`` names; ``novel_known`` is True when it is empty (every
  allele in the set is absent from the shared reference). ``known_gene`` is the set's IUIS gene (``RealAllele.gene``,
  the locus where genomic sequencing found it); a set spanning several genes, such as identical paralogs
  (IGHV1-69*01 and IGHV1-69D*01), has them sorted and joined by ``|``.

**Reference-exact known answer** (``ref_known``, ``ref_known_gene``). The same rule, run against the shared
reference's V alleles instead of the donor's (each a ``RealAllele`` of its gapped form and IUIS genes; Cys104 from
``cys104_end`` with the identical-sequence rule): a read whose observed V is exactly one shared allele's window, at
least ``min_window`` (200) nt, ``N`` never a wildcard, identical windows one set and two distinct windows none. It
needs no donor genotype, so it covers the genes the donor's assembly misses (uncaptured or uncalled), and its truth
is the read itself: the metrics score it on every read with one, whatever its exclusions. ``ref_known`` is the set's
shared names; ``ref_known_gene`` its IUIS genes (``IuisMap.genes``, a name standing in for an unmapped allele),
sorted and joined by ``|``. ``ref_donor`` is the donor V alleles whose window, at the reference-exact window's
place, is that window (the rule above, so the read cannot tell them from the answer), and those that cannot be
windowed whose V-REGION holds it; empty when the donor has none. ``real_truths`` computes them given the IUIS map
(``iuis``); without one all three are None.

**Naive.** IgM (``NAIVE_ISOTYPE``; neither protocol has an IgD primer) and a known answer: the observed V up to
Cys104 is the allele without a mismatch. Mutations past Cys104 (CDR3, J) are not looked at.

**Clonotype** (``cys104_in_read``, ``junction_by_motif``). Every read, known or not, is placed by the same rule, so
mutated members of an expanded clone share an id:

1. Cys104 of a known answer is the end of its exact window (``KnownMatch.cys_end``).
2. Cys104 of any other read: the best Hamming match, at most ``MAX_ANCHOR_MISMATCHES`` (3), of any of the donor's
   anchors inside the observed span; the fewest mismatches win, then the leftmost position (which allele's anchor
   does not matter: only the position is used). ``N`` is a mismatch. The search is exact for this bound: an anchor
   split into four blocks keeps one block without a mismatch, and every place a block occurs is checked. On the
   pilot reads it puts every known answer's Cys104 at the end of its window too (the gated test checks it); it
   would not where another anchor occurs exactly further left in the read, which the window rules out.
3. Only a read without a donor anchor is then searched, by the same rule, against the anchors of the shared
   reference's V alleles: the 30 nt ending at each one's Cys104, from its IMGT-gapped form (``Reference.gapped``)
   and ``cys104_end``. This places the junction of a read from a gene the donor's genomic genotype lacks, and
   nothing else: never a known answer or ``known_gene``. A germline sequence used to find a codon is still
   tool-neutral; no aligner output is involved. A shared reference with V alleles none of which gives an anchor
   (one read from a FASTA file has no gapped form) is refused rather than let it switch this step off.
   ``RealTruth.anchor`` says which anchors placed Cys104 (``"donor"``, ``"shared"`` or None), whether or not a
   W118 follows.
4. W118: the first ``W118_MOTIF`` at or past the end of the Cys codon that ends an in-frame junction of
   ``MIN_JUNCTION`` to ``MAX_JUNCTION`` (15 to 150) nt. The motif is IMGT's J motif Trp118 Gly119 X Gly121
   (``TGG GG. ... GG.``; IGHJ2 has Arg where the other IGHJ genes have Gln), and the junction (IMGT: Cys104 to
   Trp118 inclusive) is the read from the Cys codon's first base to the end of the Trp codon. A motif out of frame
   with the Cys codon, or closer to it than 15 nt, is passed over for the next one; past 150 nt the search stops.
   Out-of-frame and truncated junctions would merge reads of different clones (on W-97, 1,029 junctions were out
   of frame and 46 under 15 nt); a nonproductive read without an in-frame motif is a clonotype of its own, which
   is conservative. ``N`` bases between the two (masked or uncalled) stay in it as ``N``, so a consensus ``N``
   puts a read in a cluster of its own.
5. The clonotype is ``f"{donor}/{junction}"``; a read without both anchors is ``f"{donor}/single:{read_id}"``.

**Clustering sensitivity** (``clonotype_sl15``, ``sl15_clusters``, ``junction_length_clusters``). Exact junctions are
the primary clonotype in P3a. A mutated member of an expanded clone whose junction differs from its siblings' by a
few bases is a clonotype of its own there, which can overstate significance where clonality is highest. Each truth
also records an SL15 clonotype: single linkage at ``SL15_SHARE`` (15%) nt Hamming or less (at most
``floor(0.15 * L)`` mismatches, ``N`` a mismatch against a base) among the donor's junctions of one length ``L``
(``single_linkage``), named after the component's first junction in sorted order; a read without a junction keeps
its single clonotype. The search is exact and has no cap on a group's size: a pair within ``k`` mismatches shares one
of ``k + 1`` blocks exactly, so only junctions sharing a block are compared (the dev donors' largest group, 4,261
distinct junctions, takes about a minute). SL15 is the candidate primary clustering for P3b.
``junction_length_clusters`` is a stress test, one cluster per donor and junction length.

**The observed V window** (``cys_end``, ``v_window_n``). Every truth records the read index just past its Cys104
(``cys_end``, from the rules above; None where none places it) and the offsets of the ``N`` bases inside
``[observed[0], cys_end)``, counted from the first observed base (``v_window_n``). The tie-aware metrics
(``real.metrics``) compare germline sequences over that window: what the read could tell apart.

``cys104_regions`` gives each donor V allele's region through Cys104, with the identical-sequence rule above (an
allele without a Cys104 of its own takes that of an allele with its sequence).

**V evidence** (``v_covered``). The bases of the observed span that an exact ``V_KMER`` (15) nt k-mer of any V allele
covers, the donor's or the shared reference's; overlapping k-mers count a base once, and ``N`` matches nothing, so a
masked or uncalled base breaks every k-mer over it. A read with none holds no V: a D-J rearrangement, primed from C
like any read but with no V upstream of its D, on which IgBLAST aligns a V of a dozen or so nt and a tool that must
name a V names one. ``real.metrics`` leaves such a read out of R1 for both tools (``excluded:no_v``). A V the read
holds keeps an exact 15-mer of its germline unless mutations or indels fall in every 15 nt of it, and a chance
15-mer in a read without a V only keeps the read in R1, as before the rule.

``clonotype_clusters`` gives ``stats.compare(..., clusters=...)`` its read id -> clonotype map.

**Truth health** (``truth_health``), a tool-neutral audit of the truth: the known-answer rule run against the shared
reference's V alleles instead of the donor's. A read whose V is exactly one shared allele's window (at least
``MIN_WINDOW`` nt) that no donor allele explains shows an allele the genotype lacks: the read has no known answer,
and no donor allele's window at that place is that window (``ref_donor`` is empty). Windows are compared, never
names: VDJbase calls as novel, with no shared name, alleles that equal a shared one over every read (SC-11's
``IGHV5-10-1*03_c319-_a320-`` differs from IGHV5-10-1*03 only past Cys104). Such reads are counted per gene (the
allele's IUIS genes), split by where the gene stands in the truth: ``carried`` (the donor carries another allele of
it: an allele VDJbase missed, or a haplotype it did not resolve), ``captured`` (spanned,
and deleted on every capturing contig), ``uncalled`` or ``not_captured``. A carried gene with ``TRUTH_GAP_FLAG``
(10) clonotypes of such reads or more is flagged, and whether all its alleles are unphased (``h0_only``) is
noted. ``truth_to_json`` and ``truth_from_json`` turn a truth into JSON-ready data and back, for the workflow's
truth file.

**On the pilot reads** (Task 4's preprocessed run): W-97 (multiplex, 50,000 sequences) has 5.7% known answers and
5.6% naive; Cys104 is placed on 92.8% of its reads (66.5% by the donor's anchors, 26.3% by the shared reference's)
and 88.0% have an in-frame junction of 15 to 150 nt (the first motif would give 89.5%, 1,075 of them out of frame
or under 15 nt; 320 of those take a later in-frame motif). SC-23 (5'RACE, 10,733) has 25.9% and 24.1%; 91.5%
(90.7% donor, 0.8% shared) and 88.9% (90.0%; 175, 56). W-97's genomic genotype holds no IGHV4 gene while about a
quarter of its reads are IGHV4: the shared anchors place them.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from alignair_assay.core.reference import Reference
from alignair_assay.real.genotype import IuisMap, RealAllele, RealGenotype

# The nt ending at Cys104 that place a V allele in a read.
ANCHOR_LENGTH = 30
# The most mismatches a read may have against an anchor and still be placed by it.
MAX_ANCHOR_MISMATCHES = 3
# The shortest window a known answer may rest on.
MIN_WINDOW = 200
# The k-mer length of a read's V evidence (``v_covered``): a read with no exact V_KMER-mer of any V allele has no V.
V_KMER = 15
# IMGT nt 310-312, codon 104, as 0-based columns of an IMGT-gapped V-REGION.
CYS104_COLUMNS = (309, 312)
CYS_CODONS = frozenset({"TGT", "TGC"})
# IMGT's J motif, Trp118 Gly119 X Gly121: TGG GG. ... GG. (IGHJ2 has Arg where the other IGHJ genes have Gln)
W118_MOTIF = re.compile(r"TGGGG....GG.")
# The shortest and longest junction (Cys104 through Trp118, in nt) a W118 motif may end.
MIN_JUNCTION = 15
MAX_JUNCTION = 150
# The isotype a naive read must be (``real.preprocess``'s C primer gene).
NAIVE_ISOTYPE = "IGHM"
# The SL15 clustering: junctions of one length within this share of nt mismatches are linked.
SL15_SHARE = 0.15
# A carried gene with at least this many clonotypes of reads that exactly match an allele the genotype lacks is
# flagged (``truth_health``).
TRUTH_GAP_FLAG = 10
# Where a gene of such reads stands in the truth, in order of precedence.
GAP_CLASSES = ("carried", "captured", "uncalled", "not_captured")

_GAPS = str.maketrans("", "", ".-")


@dataclass(frozen=True)
class RealTruth:
    """One preprocessed read's tool-neutral truth."""

    read_id: str
    donor: str
    isotype: str  # the C primer's gene: "IGHM", "IGHG", ...
    protocol: str  # "race" | "multiplex"
    # the read's ``[start, end)`` outside its masked primers and leading or trailing N
    observed: tuple[int, int]
    # the donor allele ids whose window the read holds exactly; None without a known answer
    known: frozenset[str] | None
    # the shared-reference alleles of ``known`` (empty for a novel answer); None without a known answer
    known_shared: frozenset[str] | None
    # the IUIS gene of ``known`` (several sorted and joined by "|"); None without a known answer
    known_gene: str | None
    # a known answer whose alleles are all absent from the shared reference
    novel_known: bool
    # IgM and a known answer
    naive: bool
    # f"{donor}/{junction}", or f"{donor}/single:{read_id}" without both anchors
    clonotype: str
    # where Cys104 was found, W118 or not: "donor" (a donor allele's anchor), "shared" (the shared reference's,
    # only without a donor anchor; it places the junction and nothing else) or None
    anchor: str | None
    # the read index just past Cys104; None when no anchor placed it
    cys_end: int | None = None
    # the offsets, from the first observed base, of the N bases inside [observed[0], cys_end), sorted
    v_window_n: tuple[int, ...] = ()
    # the SL15 clonotype: f"{donor}/sl15:{first junction of its component}", or the single clonotype; None unset
    clonotype_sl15: str | None = None
    # the shared alleles whose window the read holds exactly (the reference-exact known answer); None without one
    ref_known: frozenset[str] | None = None
    # their IUIS genes, sorted and joined by "|"; None without one
    ref_known_gene: str | None = None
    # the donor alleles whose window at the reference-exact window's place is that window, and those that cannot be
    # windowed whose V-REGION holds it; None without a reference-exact known answer
    ref_donor: frozenset[str] | None = None
    # the observed bases an exact V_KMER-mer of any V allele (the donor's or the shared reference's) covers: 0 is a
    # read without a V; None unset
    v_covered: int | None = None


class KnownMatch(NamedTuple):
    """A read's known answer: the alleles, the window they share, and the read index just past its Cys104."""

    alleles: tuple[RealAllele, ...]
    window: str
    cys_end: int


@dataclass(frozen=True)
class _Anchors:
    """Cys104 anchors, split for the Hamming search."""

    # a block of an anchor -> (anchor, offset of the block in it): every anchor split into max_mismatches + 1
    blocks: dict[str, tuple[tuple[str, int], ...]]
    max_mismatches: int


@dataclass(frozen=True)
class _Index:
    """A genotype's V alleles, ready to place reads."""

    # anchor -> (allele, region through Cys104) of each allele whose region ends with it
    windows: dict[str, tuple[tuple[RealAllele, str], ...]]
    # V alleles without a Cys104: never an answer, but they block one whose window they contain
    unanchored: tuple[RealAllele, ...]
    anchors: _Anchors


def cys104_end(gapped: str) -> int | None:
    """The ungapped index just past IMGT nt 312 (codon 104) of IMGT-gapped V-REGION ``gapped`` (``.`` and ``-``
    are gaps; any case); None unless columns 309-311 hold ``TGT`` or ``TGC`` (see the module docstring)."""
    start, end = CYS104_COLUMNS
    if len(gapped) < end or gapped[start:end].upper() not in CYS_CODONS:
        return None
    return len(gapped[:end].translate(_GAPS))


def mask_primers(read: str, primer_masked) -> str:
    """``read`` uppercased, with every base of the ``primer_masked`` spans (``[[start, end), ...]``, or that list
    as JSON) made ``N``. ValueError for a span that is not inside the read."""
    read = read.upper()
    for start, end in _spans(primer_masked, len(read)):
        read = read[:start] + "N" * (end - start) + read[end:]
    return read


def observed_span(read: str, primer_masked) -> tuple[int, int]:
    """The ``[start, end)`` of ``read`` outside its ``primer_masked`` spans (``mask_primers``) and its leading and
    trailing ``N``; ``(0, 0)`` when nothing is observed."""
    masked = mask_primers(read, primer_masked)
    start, end = len(masked) - len(masked.lstrip("N")), len(masked.rstrip("N"))
    return (start, end) if end > start else (0, 0)


def known_match(read: str, observed: tuple[int, int], alleles: Iterable[RealAllele], *,
                min_window: int = MIN_WINDOW) -> KnownMatch | None:
    """The known answer of the primer-masked read ``read`` (``mask_primers``) among the V alleles of ``alleles``
    (others are skipped), over its ``observed`` span (``observed_span``), as the module docstring defines it; None
    without one."""
    return _known_match(read.upper(), observed, _index(alleles), min_window)


def known_answer(read: str, observed: tuple[int, int], alleles: Iterable[RealAllele], *,
                 min_window: int = MIN_WINDOW) -> frozenset[str] | None:
    """The allele ids of ``known_match`` of the primer-masked read ``read``; None without a known answer."""
    match = known_match(read, observed, alleles, min_window=min_window)
    return None if match is None else frozenset(allele.allele_id for allele in match.alleles)


def cys104_in_read(read: str, observed: tuple[int, int], alleles: Iterable[RealAllele], *,
                   max_mismatches: int = MAX_ANCHOR_MISMATCHES) -> int | None:
    """The read index just past Cys104: the end of the best Hamming match (at most ``max_mismatches``; fewest,
    then leftmost) of any V allele's anchor inside the ``observed`` span; None without one."""
    return _anchor_end(read.upper(), observed, _index(alleles, max_mismatches).anchors)


def junction_by_motif(read: str, cys_end_in_read: int, *, min_length: int = MIN_JUNCTION,
                      max_length: int = MAX_JUNCTION) -> str | None:
    """The junction: from the Cys codon's first base (``cys_end_in_read - 3``) to the end of the Trp codon of the
    first ``W118_MOTIF`` at or past ``cys_end_in_read`` that makes it a whole number of codons, ``min_length`` to
    ``max_length`` nt long; None without one. ``N`` bases are kept. ValueError for a Cys end that does not leave a
    whole codon inside the read."""
    if not 3 <= cys_end_in_read <= len(read):
        raise ValueError(f"Cys104 end {cys_end_in_read} is not inside a read of {len(read)} nt")
    read, start = read.upper(), cys_end_in_read - 3
    found = W118_MOTIF.search(read, cys_end_in_read)
    while found is not None:
        length = found.start() + 3 - start
        if length > max_length:
            return None
        if length % 3 == 0 and length >= min_length:
            return read[start:found.start() + 3]
        found = W118_MOTIF.search(read, found.start() + 1)
    return None


def real_truths(reads: Mapping[str, str], meta: Iterable[Mapping], genotype: RealGenotype, reference: Reference, *,
                min_window: int = MIN_WINDOW, max_mismatches: int = MAX_ANCHOR_MISMATCHES,
                iuis: IuisMap | None = None) -> list[RealTruth]:
    """The truth of every read, in ``meta``'s order. ``reads`` maps a sequence id to its sequence (the FASTA);
    ``meta`` holds one row per read (the TSV: ``sequence_id``, ``isotype``, ``protocol`` and ``primer_masked``,
    as a list or as JSON); ``reference`` is the shared reference, whose V anchors place Cys104 for the clonotype
    of a read without a donor anchor, whose V alleles, named by the IUIS map ``iuis``, give the reference-exact
    known answer (None without ``iuis``), and whose V alleles, with the donor's, give the V evidence. ValueError,
    naming an id, for an id given twice, a row without a read, a read without a row, or a primer span outside its
    read; and for a genotype or an IUIS map mapped to another shared reference (the genotype's
    ``source["reference_digest"]``, where it has one)."""
    mapped_to = (genotype.source or {}).get("reference_digest")
    if mapped_to and mapped_to != reference.digest():
        raise ValueError(f"the genotype of {genotype.donor} was mapped to another shared reference (digest "
                         f"{mapped_to[:12]}, not {reference.digest()[:12]})")
    rows = list(meta)
    ids: set[str] = set()
    for row in rows:
        if row["sequence_id"] in ids:
            raise ValueError(f"sequence id {row['sequence_id']} is given twice in the metadata")
        ids.add(row["sequence_id"])
    missing = [read_id for read_id in ids if read_id not in reads]
    if missing:
        raise ValueError(f"{len(missing)} metadata rows have no read (e.g. {sorted(missing)[0]})")
    extra = [read_id for read_id in reads if read_id not in ids]
    if extra:
        raise ValueError(f"{len(extra)} reads have no metadata row (e.g. {sorted(extra)[0]})")
    if iuis is not None and iuis.reference_digest != reference.digest():
        raise ValueError(f"the IUIS map was built against another shared reference (digest "
                         f"{iuis.reference_digest[:12]}, not {reference.digest()[:12]})")
    index = _index(genotype.alleles, max_mismatches)
    shared_anchors = _shared_anchors(reference, max_mismatches)
    shared_index = None if iuis is None else _index(shared_alleles(reference, iuis), max_mismatches)
    v_kmers = _kmers([*(allele.sequence for allele in genotype.alleles if allele.segment == "v"),
                      *reference.alleles.get("v", {}).values()], V_KMER)
    donor = genotype.donor
    truths = []
    for row in rows:
        read_id = row["sequence_id"]
        try:
            read = mask_primers(reads[read_id], row.get("primer_masked"))
        except ValueError as error:
            raise ValueError(f"read {read_id}: {error}") from None
        observed = observed_span(read, [])
        match = _known_match(read, observed, index, min_window)
        # a known answer's Cys104 is the end of its exact window; any other read's, the best Hamming anchor
        anchor, cys_end = "donor", match.cys_end if match is not None else _anchor_end(read, observed, index.anchors)
        if cys_end is None:
            anchor, cys_end = "shared", _anchor_end(read, observed, shared_anchors)
        if cys_end is None:
            anchor = None
        junction = None if cys_end is None else junction_by_motif(read, cys_end)
        clonotype = f"{donor}/single:{read_id}" if junction is None else f"{donor}/{junction}"
        gaps = () if cys_end is None else tuple(offset for offset, base in enumerate(read[observed[0]:cys_end])
                                                if base == "N")
        known = shared = gene = None
        if match is not None:
            known = frozenset(allele.allele_id for allele in match.alleles)
            shared = frozenset(name for allele in match.alleles for name in allele.shared)
            gene = "|".join(sorted({allele.gene for allele in match.alleles}))
        ref = None if shared_index is None else _known_match(read, observed, shared_index, min_window)
        ref_donor = None if ref is None else _windowed_by(index, ref, observed[0])
        truths.append(RealTruth(read_id=read_id, donor=donor, isotype=row["isotype"], protocol=row["protocol"],
                                observed=observed, known=known, known_shared=shared, known_gene=gene,
                                novel_known=match is not None and not shared,
                                naive=match is not None and row["isotype"] == NAIVE_ISOTYPE, clonotype=clonotype,
                                anchor=anchor, cys_end=cys_end, v_window_n=gaps,
                                ref_known=None if ref is None else frozenset(a.allele_id for a in ref.alleles),
                                ref_known_gene=None if ref is None else "|".join(sorted(
                                    {gene for allele in ref.alleles for gene in allele.gene.split("|")})),
                                ref_donor=ref_donor, v_covered=_covered(read, observed, v_kmers, V_KMER)))
    roots = single_linkage((junction for truth in truths if (junction := _junction(truth))), SL15_SHARE)
    return [dataclasses.replace(truth, clonotype_sl15=_sl15(truth, roots)) for truth in truths]


def clonotype_clusters(truths: Iterable[RealTruth]) -> dict[str, str]:
    """Read id -> clonotype, the ``clusters`` of ``stats.compare``."""
    return {truth.read_id: truth.clonotype for truth in truths}


def shared_alleles(reference: Reference, iuis: IuisMap) -> list[RealAllele]:
    """The shared reference's V alleles as ``RealAllele`` records, for the known-answer rule: each named by its
    shared name, of its IUIS genes (``IuisMap.genes``, joined by ``|``; its name for an unmapped allele), with its
    gapped form (``Reference.gapped``)."""
    return [RealAllele(allele_id=name, gene="|".join(iuis.genes(name)) or name, segment="v",
                       sequence=sequence.upper(), gapped=reference.gapped(name), haplotype=0, functional=True,
                       shared=(name,), iuis=iuis.labels(name))
            for name, sequence in reference.alleles.get("v", {}).items()]


def truth_health(truths: Iterable[RealTruth], genotype: RealGenotype, iuis: IuisMap, *,
                 flag: int = TRUTH_GAP_FLAG) -> dict:
    """The truth audit of the module docstring over one donor's ``truths``, their reference-exact known answers
    (``real_truths(..., iuis=iuis)``) against the donor's captured ``genotype``: ``reads``; ``shared_exact`` (reads
    that are exactly one shared allele's window); ``not_carried`` (of those, the reads no donor allele explains: no
    known answer and an empty ``ref_donor``; windows compared, never names); ``by_class`` (``GAP_CLASSES`` ->
    reads); ``genes`` (one entry per gene: ``gene``, ``class``, ``reads``, ``clonotypes``, ``alleles`` (shared allele
    -> reads) and ``h0_only``; the most reads first);
    ``flagged`` (carried genes with at least ``flag`` clonotypes) and ``flag_clonotypes``. ``iuis`` is the map the
    truths were computed with."""
    carried = {allele.gene for allele in genotype.alleles}
    unphased = {gene for gene in carried if all(allele.haplotype == 0 for allele in genotype.carried(gene))}
    captured, uncalled = genotype.captured or frozenset(), genotype.uncalled or frozenset()
    found: dict[str, dict] = {}
    count = exact = 0
    for truth in truths:
        count += 1
        if truth.ref_known is None:
            continue
        exact += 1
        if truth.known is not None or truth.ref_donor:
            continue
        genes = truth.ref_known_gene.split("|")
        cls = ("carried" if set(genes) & carried else "uncalled" if set(genes) & uncalled
               else "captured" if set(genes) <= captured else "not_captured")
        entry = found.setdefault(truth.ref_known_gene, {"gene": truth.ref_known_gene, "class": cls, "reads": 0,
                                                        "clonotypes": set(), "alleles": defaultdict(int),
                                                        "h0_only": bool(set(genes) & unphased)})
        entry["reads"] += 1
        entry["clonotypes"].add(truth.clonotype)
        entry["alleles"]["|".join(sorted(truth.ref_known))] += 1
    genes = sorted(({**entry, "clonotypes": len(entry["clonotypes"]), "alleles": dict(sorted(entry["alleles"].items()))}
                    for entry in found.values()), key=lambda entry: (-entry["reads"], entry["gene"]))
    return {"reads": count, "shared_exact": exact, "not_carried": sum(entry["reads"] for entry in genes),
            "by_class": {cls: sum(entry["reads"] for entry in genes if entry["class"] == cls) for cls in GAP_CLASSES},
            "genes": genes, "flag_clonotypes": flag,
            "flagged": [entry["gene"] for entry in genes
                        if entry["class"] == "carried" and entry["clonotypes"] >= flag]}


def sl15_clusters(truths: Iterable[RealTruth]) -> dict[str, str]:
    """Read id -> SL15 clonotype (the module docstring): the stored ``clonotype_sl15``, else computed over
    ``truths`` (one donor's)."""
    truths = list(truths)
    if all(truth.clonotype_sl15 is not None for truth in truths):
        return {truth.read_id: truth.clonotype_sl15 for truth in truths}
    roots = single_linkage((junction for truth in truths if (junction := _junction(truth))), SL15_SHARE)
    return {truth.read_id: _sl15(truth, roots) for truth in truths}


def junction_length_clusters(truths: Iterable[RealTruth]) -> dict[str, str]:
    """Read id -> ``f"{donor}/length:{L}"`` for a read with a junction of L nt, else its single clonotype: the
    clustering stress test."""
    return {truth.read_id: truth.clonotype if (junction := _junction(truth)) is None
            else f"{truth.donor}/length:{len(junction)}" for truth in truths}


def single_linkage(junctions: Iterable[str], share: float = SL15_SHARE) -> dict[str, str]:
    """Each distinct junction -> the first junction, in sorted order, of its component when junctions of one length
    are linked at ``floor(share * L)`` mismatches or fewer (exact: a pair within ``k`` mismatches shares one of
    ``k + 1`` blocks, so only junctions sharing a block are compared)."""
    by_length: dict[int, list[str]] = defaultdict(list)
    for junction in sorted(set(junctions)):
        by_length[len(junction)].append(junction)
    roots: dict[str, str] = {}
    for length, group in by_length.items():
        limit = int(share * length)
        parent = list(range(len(group)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, junction in enumerate(group):
            for block, (start, end) in enumerate(_block_bounds(length, min(limit + 1, length))):
                buckets[block, junction[start:end]].append(index)
        for members in buckets.values():
            for first, i in enumerate(members):
                for j in members[first + 1:]:
                    root_i, root_j = find(i), find(j)
                    if root_i != root_j and _mismatches(group[i], group[j], 0, limit) <= limit:
                        parent[max(root_i, root_j)] = min(root_i, root_j)
        roots.update({junction: group[find(index)] for index, junction in enumerate(group)})
    return roots


def _junction(truth: RealTruth) -> str | None:
    """The read's junction, from its exact clonotype; None for a single clonotype."""
    tail = truth.clonotype.split("/", 1)[1]
    return None if tail.startswith("single:") else tail


def _sl15(truth: RealTruth, roots: Mapping[str, str]) -> str:
    junction = _junction(truth)
    return truth.clonotype if junction is None else f"{truth.donor}/sl15:{roots[junction]}"


def truth_to_json(truth: RealTruth) -> dict:
    """``truth`` as JSON-ready data, which ``truth_from_json`` reads back equal: the sets as sorted lists, the
    observed span and the window's N offsets as lists."""
    return {"read_id": truth.read_id, "donor": truth.donor, "isotype": truth.isotype, "protocol": truth.protocol,
            "observed": list(truth.observed), "known": None if truth.known is None else sorted(truth.known),
            "known_shared": None if truth.known_shared is None else sorted(truth.known_shared),
            "known_gene": truth.known_gene, "novel_known": truth.novel_known, "naive": truth.naive,
            "clonotype": truth.clonotype, "anchor": truth.anchor, "cys_end": truth.cys_end,
            "v_window_n": list(truth.v_window_n), "clonotype_sl15": truth.clonotype_sl15,
            "ref_known": None if truth.ref_known is None else sorted(truth.ref_known),
            "ref_known_gene": truth.ref_known_gene,
            "ref_donor": None if truth.ref_donor is None else sorted(truth.ref_donor), "v_covered": truth.v_covered}


def truth_from_json(record: Mapping) -> RealTruth:
    """The ``RealTruth`` of ``truth_to_json``'s data. ValueError for data that is not a truth's."""
    try:
        known, shared = record["known"], record["known_shared"]
        return RealTruth(read_id=record["read_id"], donor=record["donor"], isotype=record["isotype"],
                         protocol=record["protocol"], observed=tuple(record["observed"]),
                         known=None if known is None else frozenset(known),
                         known_shared=None if shared is None else frozenset(shared), known_gene=record["known_gene"],
                         novel_known=record["novel_known"], naive=record["naive"], clonotype=record["clonotype"],
                         anchor=record["anchor"], cys_end=record["cys_end"], v_window_n=tuple(record["v_window_n"]),
                         clonotype_sl15=record["clonotype_sl15"],
                         ref_known=None if record["ref_known"] is None else frozenset(record["ref_known"]),
                         ref_known_gene=record["ref_known_gene"],
                         ref_donor=None if record["ref_donor"] is None else frozenset(record["ref_donor"]),
                         v_covered=record["v_covered"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"not a read's truth (truth_to_json): {error!r}") from None


# --- placing reads -------------------------------------------------------------------------------------------------

def cys104_regions(alleles: Iterable[RealAllele]) -> tuple[list[tuple[RealAllele, str]], list[RealAllele]]:
    """``(regions, unplaced)``: each V allele of ``alleles`` (others are skipped) with its uppercase region through
    Cys104 (``sequence[:cys104_end]``), in order, and the V alleles without a Cys104. Identity is the sequence: an
    allele without a Cys104 of its own (no gapped form, or codon 104 off columns 309-311) takes that of an allele with
    exactly its sequence."""
    v = [allele for allele in alleles if allele.segment == "v"]
    own = {id(allele): cys104_end(allele.gapped) if allele.gapped else None for allele in v}
    by_sequence: dict[str, int] = {}
    for allele in v:
        if own[id(allele)] is not None:
            by_sequence.setdefault(allele.sequence.upper(), own[id(allele)])
    regions, unplaced = [], []
    for allele in v:
        end = own[id(allele)]
        end = by_sequence.get(allele.sequence.upper()) if end is None else end
        if end is None:
            unplaced.append(allele)
        else:
            regions.append((allele, allele.sequence[:end].upper()))
    return regions, unplaced


def _index(alleles: Iterable[RealAllele], max_mismatches: int = MAX_ANCHOR_MISMATCHES) -> _Index:
    regions, unanchored = cys104_regions(alleles)
    windows: dict[str, list[tuple[RealAllele, str]]] = defaultdict(list)
    for allele, region in regions:
        if len(region) < ANCHOR_LENGTH:
            unanchored.append(allele)
            continue
        windows[region[-ANCHOR_LENGTH:]].append((allele, region))
    return _Index(windows={anchor: tuple(entries) for anchor, entries in windows.items()},
                  unanchored=tuple(unanchored), anchors=_anchors(windows, max_mismatches))


def _shared_anchors(reference: Reference, max_mismatches: int) -> _Anchors:
    """The Cys104 anchors of the shared reference's V alleles that have a gapped form with a Cys104. ValueError for
    a reference with V alleles none of which gives one (a reference read from a FASTA file has no gapped form)."""
    anchors = set()
    v = reference.alleles.get("v", {})
    for name, sequence in v.items():
        gapped = reference.gapped(name)
        end = cys104_end(gapped) if gapped else None
        if end is not None and end >= ANCHOR_LENGTH:
            anchors.add(sequence[:end].upper()[-ANCHOR_LENGTH:])
    if v and not anchors:
        raise ValueError(f"none of the shared reference's {len(v)} V alleles has an IMGT-gapped form with a Cys104 "
                         "(a reference read from a FASTA file has no gapped form), so it cannot place the Cys104 of "
                         "a read without a donor anchor; use the reference built from the dataconfig")
    return _anchors(sorted(anchors), max_mismatches)


def _anchors(anchors: Iterable[str], max_mismatches: int) -> _Anchors:
    if max_mismatches < 0 or max_mismatches >= ANCHOR_LENGTH:
        raise ValueError(f"max_mismatches must be 0 to {ANCHOR_LENGTH - 1}, got {max_mismatches}")
    blocks: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for anchor in anchors:
        for start, end in _block_bounds(ANCHOR_LENGTH, max_mismatches + 1):
            blocks[anchor[start:end]].append((anchor, start))
    return _Anchors(blocks={block: tuple(entries) for block, entries in blocks.items()},
                    max_mismatches=max_mismatches)


def _block_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    """``parts`` contiguous ``[start, end)`` blocks covering ``length``, the first ones one longer when uneven."""
    size, extra = divmod(length, parts)
    bounds, start = [], 0
    for part in range(parts):
        end = start + size + (part < extra)
        bounds.append((start, end))
        start = end
    return bounds


def _known_match(read: str, observed: tuple[int, int], index: _Index, min_window: int) -> KnownMatch | None:
    start, end = observed
    found: dict[tuple[str, int], list[RealAllele]] = {}
    for anchor, entries in index.windows.items():
        at = read.find(anchor, start, end)
        while at != -1:
            cys_end = at + ANCHOR_LENGTH
            for allele, region in entries:
                window = _window(region, cys_end - start)
                if read.startswith(window, cys_end - len(window)):
                    found.setdefault((window, cys_end), []).append(allele)
            at = read.find(anchor, at + 1, end)
    if len(found) != 1:
        return None
    ((window, cys_end), alleles), = found.items()
    if len(window) < min_window or any(window in allele.sequence.upper() for allele in index.unanchored):
        return None
    return KnownMatch(alleles=tuple(alleles), window=window, cys_end=cys_end)


def _window(region: str, span: int) -> str:
    """An allele's window in a read: its ``region`` through Cys104 from the position facing the first observed base,
    ``span`` bases before the read's Cys104 end (the whole region when the read reaches past its start)."""
    return region[max(0, len(region) - span):]


def _windowed_by(index: _Index, match: KnownMatch, start: int) -> frozenset[str]:
    """The ids of ``index``'s alleles whose window at ``match``'s place (``start`` the first observed base) is
    ``match.window``, and of those that cannot be windowed whose V-REGION holds it (``_known_match``'s rule)."""
    span = match.cys_end - start
    same = {allele.allele_id for allele, region in index.windows.get(match.window[-ANCHOR_LENGTH:], ())
            if _window(region, span) == match.window}
    same |= {allele.allele_id for allele in index.unanchored if match.window in allele.sequence.upper()}
    return frozenset(same)


def _anchor_end(read: str, observed: tuple[int, int], anchors: _Anchors) -> int | None:
    start, end = observed
    limit = anchors.max_mismatches
    best: tuple[int, int] | None = None  # (mismatches, position)
    checked: set[tuple[str, int]] = set()
    for block, entries in anchors.blocks.items():
        at = read.find(block, start, end)
        while at != -1:
            for anchor, offset in entries:
                position = at - offset
                if position < start or position + ANCHOR_LENGTH > end or (anchor, position) in checked:
                    continue
                checked.add((anchor, position))
                mismatches = _mismatches(anchor, read, position, limit)
                if mismatches <= limit and (best is None or (mismatches, position) < best):
                    best = (mismatches, position)
            at = read.find(block, at + 1, end)
    return None if best is None else best[1] + ANCHOR_LENGTH


def _mismatches(anchor: str, read: str, position: int, limit: int) -> int:
    """The mismatches of ``anchor`` against ``read`` at ``position``, counted up to one past ``limit``."""
    count = 0
    for expected, base in zip(anchor, read[position:position + len(anchor)]):
        if expected != base:
            count += 1
            if count > limit:
                break
    return count


def _kmers(sequences: Iterable[str], k: int) -> frozenset[str]:
    """Every ``k``-mer of ``sequences``, uppercased."""
    return frozenset(seq[i:i + k] for seq in map(str.upper, sequences) for i in range(len(seq) - k + 1))


def _covered(read: str, observed: tuple[int, int], kmers: frozenset[str], k: int) -> int:
    """The bases of ``read``'s ``observed`` span that a ``k``-mer of ``kmers`` covers, each counted once."""
    covered, reach = 0, observed[0]
    for i in range(observed[0], observed[1] - k + 1):
        if read[i:i + k] in kmers:
            covered += i + k - max(i, reach)
            reach = i + k
    return covered


def _spans(primer_masked, length: int) -> list[tuple[int, int]]:
    """The ``[start, end)`` spans of ``primer_masked`` (a list, its JSON, or None / "" for none), each checked to
    lie inside a read of ``length`` nt."""
    if primer_masked is None or primer_masked == "":
        return []
    value = json.loads(primer_masked) if isinstance(primer_masked, str) else primer_masked
    if not isinstance(value, Sequence):
        raise ValueError(f"primer_masked {primer_masked!r} is not a list of [start, end) spans")
    spans = []
    for span in value:
        if (not isinstance(span, Sequence) or len(span) != 2
                or not all(isinstance(bound, int) and not isinstance(bound, bool) for bound in span)):
            raise ValueError(f"primer_masked {primer_masked!r} is not a list of [start, end) spans")
        start, end = span
        if not 0 <= start <= end <= length:
            raise ValueError(f"primer span [{start}, {end}) is not inside a read of {length} nt")
        spans.append((start, end))
    return spans
