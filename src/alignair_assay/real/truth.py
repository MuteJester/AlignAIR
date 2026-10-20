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

**Naive.** IgM (``NAIVE_ISOTYPE``; neither protocol has an IgD primer) and a known answer: the observed V up to
Cys104 is the allele without a mismatch. Mutations past Cys104 (CDR3, J) are not looked at.

**Clonotype** (``cys104_in_read``, ``junction_by_motif``). Every read, known or not, is placed by the same rule, so
mutated members of an expanded clone share an id:

1. Cys104: the best Hamming match, at most ``MAX_ANCHOR_MISMATCHES`` (3), of any of the donor's anchors inside the
   observed span; the fewest mismatches win, then the leftmost position (which allele's anchor does not matter:
   only the position is used). ``N`` is a mismatch. The search is exact for this bound: an anchor split into
   four blocks keeps one block without a mismatch, and every place a block occurs is checked.
2. W118: the first ``W118_MOTIF`` (``TGG GG. CA. GG.``, Trp118 Gly Gln Gly) at or past the end of the Cys codon.
   The junction (IMGT: Cys104 to Trp118 inclusive) is the read from the Cys codon's first base to the end of the
   Trp codon. ``N`` bases between the two (masked or uncalled) stay in it as ``N``, so a consensus ``N`` puts a
   read in a cluster of its own. IGHJ2's Trp118 is followed by Gly Arg Gly (``TGG GGC CGT GGC``), which the motif
   does not match: an IGHJ2 read has no junction and is a singleton.
3. The clonotype is ``f"{donor}/{junction}"``; a read without both anchors is ``f"{donor}/single:{read_id}"``.

A known answer's own Cys104 is the end of its exact window. The Hamming anchor puts it there too unless another
anchor occurs further left in the read with no mismatch; on the pilot reads the two agree for every known answer,
which the gated test checks. ``clonotype_clusters`` gives ``stats.compare(..., clusters=...)`` its read id ->
clonotype map.

**On the pilot reads** (Task 4's preprocessed run): W-97 (multiplex, 50,000 sequences) has 5.7% known answers,
5.6% naive, 66.5% with a Cys104 anchor and 60.5% with both anchors; SC-23 (5'RACE, 10,733) has 25.9%, 24.1%,
90.7% and 85.7%. W-97's genomic genotype holds no IGHV4 gene while about a quarter of its reads are IGHV4, so
those reads have no donor anchor. Of the reads with a Cys104 anchor but no W118 motif (W-97 2,975, SC-23 533),
1,729 and 382 have an IGHJ2-like ``TGG GG. ... GG.`` (Trp Gly Xaa Gly) instead.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from alignair_assay.real.genotype import RealAllele, RealGenotype

# The nt ending at Cys104 that place a V allele in a read.
ANCHOR_LENGTH = 30
# The most mismatches a read may have against an anchor and still be placed by it.
MAX_ANCHOR_MISMATCHES = 3
# The shortest window a known answer may rest on.
MIN_WINDOW = 200
# IMGT nt 310-312, codon 104, as 0-based columns of an IMGT-gapped V-REGION.
CYS104_COLUMNS = (309, 312)
CYS_CODONS = frozenset({"TGT", "TGC"})
# Trp118 Gly119 Gln120 Gly121 of an IGHJ: TGG GG. CA. GG.
W118_MOTIF = re.compile(r"TGGGG.CA.GG.")
# The isotype a naive read must be (``real.preprocess``'s C primer gene).
NAIVE_ISOTYPE = "IGHM"

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


class KnownMatch(NamedTuple):
    """A read's known answer: the alleles, the window they share, and the read index just past its Cys104."""

    alleles: tuple[RealAllele, ...]
    window: str
    cys_end: int


@dataclass(frozen=True)
class _Index:
    """A genotype's V alleles, ready to place reads."""

    # anchor -> (allele, region through Cys104) of each allele whose region ends with it
    windows: dict[str, tuple[tuple[RealAllele, str], ...]]
    # V alleles without a Cys104: never an answer, but they block one whose window they contain
    unanchored: tuple[RealAllele, ...]
    # a block of an anchor -> (anchor, offset of the block in it): every anchor split into max_mismatches + 1
    blocks: dict[str, tuple[tuple[str, int], ...]]
    max_mismatches: int


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
    """The known answer of ``read`` among the V alleles of ``alleles`` (others are skipped), over its ``observed``
    span (``observed_span``), as the module docstring defines it; None without one."""
    return _known_match(read.upper(), observed, _index(alleles), min_window)


def known_answer(read: str, observed: tuple[int, int], alleles: Iterable[RealAllele], *,
                 min_window: int = MIN_WINDOW) -> frozenset[str] | None:
    """The allele ids of ``known_match``; None without a known answer."""
    match = known_match(read, observed, alleles, min_window=min_window)
    return None if match is None else frozenset(allele.allele_id for allele in match.alleles)


def cys104_in_read(read: str, observed: tuple[int, int], alleles: Iterable[RealAllele], *,
                   max_mismatches: int = MAX_ANCHOR_MISMATCHES) -> int | None:
    """The read index just past Cys104: the end of the best Hamming match (at most ``max_mismatches``; fewest,
    then leftmost) of any V allele's anchor inside the ``observed`` span; None without one."""
    return _anchor_end(read.upper(), observed, _index(alleles, max_mismatches))


def junction_by_motif(read: str, cys_end_in_read: int) -> str | None:
    """The junction: from the Cys codon's first base (``cys_end_in_read - 3``) to the end of the Trp codon of the
    first ``W118_MOTIF`` at or past ``cys_end_in_read``; None without one. ``N`` bases are kept. ValueError for a
    Cys end that does not leave a whole codon inside the read."""
    if not 3 <= cys_end_in_read <= len(read):
        raise ValueError(f"Cys104 end {cys_end_in_read} is not inside a read of {len(read)} nt")
    found = W118_MOTIF.search(read.upper(), cys_end_in_read)
    return None if found is None else read[cys_end_in_read - 3:found.start() + 3].upper()


def real_truths(reads: Mapping[str, str], meta: Iterable[Mapping], genotype: RealGenotype, *,
                min_window: int = MIN_WINDOW, max_mismatches: int = MAX_ANCHOR_MISMATCHES) -> list[RealTruth]:
    """The truth of every read, in ``meta``'s order. ``reads`` maps a sequence id to its sequence (the FASTA);
    ``meta`` holds one row per read (the TSV: ``sequence_id``, ``isotype``, ``protocol`` and ``primer_masked``,
    as a list or as JSON). ValueError, naming an id, for an id given twice, a row without a read, a read without
    a row, or a primer span outside its read."""
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
    index = _index(genotype.alleles, max_mismatches)
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
        cys_end = _anchor_end(read, observed, index)
        junction = None if cys_end is None else junction_by_motif(read, cys_end)
        clonotype = f"{donor}/single:{read_id}" if junction is None else f"{donor}/{junction}"
        known = shared = gene = None
        if match is not None:
            known = frozenset(allele.allele_id for allele in match.alleles)
            shared = frozenset(name for allele in match.alleles for name in allele.shared)
            gene = "|".join(sorted({allele.gene for allele in match.alleles}))
        truths.append(RealTruth(read_id=read_id, donor=donor, isotype=row["isotype"], protocol=row["protocol"],
                                observed=observed, known=known, known_shared=shared, known_gene=gene,
                                novel_known=match is not None and not shared,
                                naive=match is not None and row["isotype"] == NAIVE_ISOTYPE, clonotype=clonotype))
    return truths


def clonotype_clusters(truths: Iterable[RealTruth]) -> dict[str, str]:
    """Read id -> clonotype, the ``clusters`` of ``stats.compare``."""
    return {truth.read_id: truth.clonotype for truth in truths}


# --- placing reads -------------------------------------------------------------------------------------------------

def _index(alleles: Iterable[RealAllele], max_mismatches: int = MAX_ANCHOR_MISMATCHES) -> _Index:
    if max_mismatches < 0 or max_mismatches >= ANCHOR_LENGTH:
        raise ValueError(f"max_mismatches must be 0 to {ANCHOR_LENGTH - 1}, got {max_mismatches}")
    v = [allele for allele in alleles if allele.segment == "v"]
    own = {id(allele): cys104_end(allele.gapped) if allele.gapped else None for allele in v}
    # identity is the sequence: an allele without a Cys104 of its own takes that of an allele with its sequence
    by_sequence: dict[str, int] = {}
    for allele in v:
        if own[id(allele)] is not None:
            by_sequence.setdefault(allele.sequence.upper(), own[id(allele)])
    windows: dict[str, list[tuple[RealAllele, str]]] = defaultdict(list)
    unanchored = []
    for allele in v:
        end = own[id(allele)]
        end = by_sequence.get(allele.sequence.upper()) if end is None else end
        if end is None or end < ANCHOR_LENGTH:
            unanchored.append(allele)
            continue
        region = allele.sequence[:end].upper()
        windows[region[-ANCHOR_LENGTH:]].append((allele, region))
    blocks: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for anchor in windows:
        for start, end in _block_bounds(ANCHOR_LENGTH, max_mismatches + 1):
            blocks[anchor[start:end]].append((anchor, start))
    return _Index(windows={anchor: tuple(entries) for anchor, entries in windows.items()},
                  unanchored=tuple(unanchored), blocks={block: tuple(entries) for block, entries in blocks.items()},
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
                length = min(len(region), cys_end - start)
                window = region[len(region) - length:]
                if read.startswith(window, cys_end - length):
                    found.setdefault((window, cys_end), []).append(allele)
            at = read.find(anchor, at + 1, end)
    if len(found) != 1:
        return None
    ((window, cys_end), alleles), = found.items()
    if len(window) < min_window or any(window in allele.sequence.upper() for allele in index.unanchored):
        return None
    return KnownMatch(alleles=tuple(alleles), window=window, cys_end=cys_end)


def _anchor_end(read: str, observed: tuple[int, int], index: _Index) -> int | None:
    start, end = observed
    limit = index.max_mismatches
    best: tuple[int, int] | None = None  # (mismatches, position)
    checked: set[tuple[str, int]] = set()
    for block, entries in index.blocks.items():
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
