"""Per-read observable truth: segments, junction, productivity and clone key from a simulated record.

Everything here is reachable by a perfect aligner: segments are projected onto the observed read, the
junction exists only when both conserved anchors are aligned bases of the read, and productivity is
defined only when the junction is.
"""
from __future__ import annotations

from collections.abc import Mapping

from alignair_assay.core.cigar import parse_cigar, walk
from alignair_assay.core.genotype import NovelAllele
from alignair_assay.core.reference import Reference, gene_name
from alignair_assay.core.seqops import revcomp, translate
from alignair_assay.truth.alleles import observable_alleles
from alignair_assay.truth.boundaries import boundary_intervals
from alignair_assay.truth.genairr import observe_forward, observe_inverted_d, record_calls
from alignair_assay.truth.model import JunctionTruth, ObservableTruth, SegmentTruth
from alignair_assay.truth.segment import ObservedSegment, check_invariants, observe_segment


def truth_from_genairr_record(record: dict, reference: Reference, *, case_id: str, stratum: str,
                              orientation: int = 0,
                              novel: Mapping[str, NovelAllele] | None = None) -> ObservableTruth:
    """Observable truth for a forward GenAIRR record (``rev_comp`` records are not supported yet).

    Each segment's primary allele is the generative one (``truth.genairr.record_calls``): ``truth_{g}_call``
    (records simulated with ``expose_provenance=True``) or, in GenAIRR's generating-reference encoding, ``{g}_call``.
    A record that does not name it falls back to the first entry of GenAIRR's post-SHM call list, and the allele
    truth says so (``AlleleTruth.primary_fallback``). That call list (``recalled_{g}_call`` in the
    generating-reference encoding, ``{g}_call`` before it) joins the scored set. In the encodings before it, a
    generative allele missing from the call list makes the segment re-called, decoded as ``truth.genairr``
    describes. A segment with no alignment column (an empty CIGAR) is not observed. A primary allele that is not in
    ``reference`` makes its segment unscorable.

    The scored allele set of a forward segment (V, J, forward D) also holds the as-near alleles
    (``as_near_alleles``): the primary's gene, covering the observed span, with no more mismatches against the read
    than the primary. A sequence-only aligner cannot tell them from the primary, and GenAIRR's re-call list,
    restricted to the donor's genotype in a cohort, does not always show them; relying on it would reward
    GenAIRR's conventions. An inverted D keeps GenAIRR's list only. Alleles of another gene that come as near are
    scored only when GenAIRR lists them.

    **Novel V alleles.** ``novel`` maps the novel alleles of the read's donor by name. A primary V allele found
    there (and not in ``reference``) is observed against the novel sequence, then re-based onto its base allele,
    which is what a perfect aligner restricted to ``reference`` reports: the same coordinates and CIGAR, the base's
    germline and mismatches counted against it (so every SNP shown is one), boundaries and the allele set built
    against the base, and the novel name replaced by the base in GenAIRR's call list. The as-near alleles are
    those of the base: GenAIRR's re-call compares the read with the novel sequence, so it cannot show them.
    ``ObservableTruth.v_novel_snps`` holds the read positions (canonical frame, ascending) aligned to the novel
    allele's SNPs inside the observed germline span ``[germ_start, germ_end)``. ValueError, naming the allele, when
    its base is not in ``reference`` or its sequence does not differ from the base's exactly at its SNPs.
    """
    if record.get("rev_comp"):
        raise ValueError(f"{case_id}: reverse-complemented GenAIRR records are not supported")
    read = record["sequence"].upper()
    warnings: list[str] = []

    observed: dict[str, tuple[ObservedSegment, str, list[str], bool, bool]] = {}
    unscorable: set[str] = set()
    v_novel_snps = None
    for g in "vdj":
        generative, calls, projecting = record_calls(record, g)
        primary = generative or (calls[0] if calls else None)
        if primary is None or not record.get(f"{g}_cigar"):
            # No call, or no alignment column: the read shows none of the segment. GenAIRR's assembled encoding
            # records a segment that end loss removed completely with an empty CIGAR and no germline coordinates.
            continue
        novel_allele = (novel or {}).get(primary) if g == "v" and not reference.has(primary) else None
        if novel_allele is None and not reference.has(primary):
            warnings.append(f"{g}: primary allele {primary} not in reference; segment dropped")
            unscorable.add(g)
            continue
        # Before the generating-reference encoding, GenAIRR projects the alignment onto its first call, which is the
        # generative allele whenever it is listed.
        recalled = projecting and bool(calls) and generative is not None and generative not in calls
        if novel_allele is not None:
            base_sequence = _base_sequence(novel_allele, reference)
            allele_seq = novel_allele.sequence
        else:
            allele_seq = reference.seq(primary)
        inverted = g == "d" and bool(record.get("d_inverted"))
        frame_seq = revcomp(allele_seq) if inverted else allele_seq
        problems: list[str] = []
        # The length of the re-called allele GenAIRR projected the alignment onto (see truth.genairr).
        projected = len(reference.seq(calls[0])) if recalled and reference.has(calls[0]) else None
        try:
            if not inverted:
                seg = observe_forward(read, allele_seq, primary, record, g, problems, recalled=recalled,
                                      projected_length=projected)
            elif recalled and not reference.has(calls[0]):
                seg = None
                problems.append(f"inverted D re-called as {calls[0]}, which is not in the reference; segment dropped")
            else:
                seg = observe_inverted_d(read, allele_seq, primary, record, problems, projected_length=projected)
        except (IndexError, TypeError, ValueError) as exc:
            seg = None
            problems.append(f"unusable simulator alignment ({exc}); segment dropped")
        # A missing segment with a problem was dropped; without one, the read simply does not show it.
        if seg is None and problems:
            unscorable.add(g)
        if seg is not None and novel_allele is not None:
            frame_seq = base_sequence
            seg, v_novel_snps = _rebased(read, seg, novel_allele, base_sequence)
            # GenAIRR calls the novel allele; on the reference, that call is its base.
            calls = list(dict.fromkeys(novel_allele.base if call == primary else call for call in calls))
        if seg is not None:
            problems += check_invariants(read, frame_seq, seg)
            observed[g] = (seg, frame_seq, calls, inverted, generative is None)
        warnings.extend(f"{g}: {p}" for p in problems)

    order = list(observed)
    segments: dict[str, SegmentTruth] = {}
    d_reference = None
    for k, g in enumerate(order):
        seg, frame_seq, calls, inverted, fallback = observed[g]
        left = observed[order[k - 1]][0].seq_end if k else 0
        right = observed[order[k + 1]][0].seq_start if k + 1 < len(order) else len(read)
        if left > seg.seq_start:
            warnings.append(f"{g}: overlaps the preceding segment ({seg.seq_start} < {left})")
        if inverted:
            d_reference = d_reference or _reverse_complemented(reference, "d")
        segments[g] = SegmentTruth(
            observed=seg,
            boundaries=boundary_intervals(read, frame_seq, seg, left, right),
            alleles=observable_alleles(seg, d_reference if inverted else reference, g, calls,
                                       primary_fallback=fallback,
                                       as_near=() if inverted else as_near_alleles(read, seg, reference, g)),
            observed_length=seg.germ_end - seg.germ_start,
            inverted=inverted,
        )

    junction = _junction(read, segments, reference, warnings)
    v_frameshift, j_frameshift = (_frameshift(segments[g].observed) if g in segments else None for g in "vj")
    vj_in_frame = stop_codon = productive = clone_key = None
    if junction is not None:
        vj_in_frame = junction.length % 3 == 0
        stop_codon = _stop_codon(read, junction.start, segments["v"].observed.seq_start,
                                 segments["j"].observed.seq_end)
        productive = vj_in_frame and not stop_codon and not (v_frameshift or j_frameshift)
        clone_key = (segments["v"].alleles.genes, segments["j"].alleles.genes, junction.length)

    return ObservableTruth(
        case_id=case_id, stratum=stratum, sequence=read, orientation=orientation, segments=segments,
        unscorable=frozenset(unscorable), junction=junction, vj_in_frame=vj_in_frame, stop_codon=stop_codon,
        productive=productive, clone_key=clone_key,
        v_mismatches=segments["v"].observed.n_mismatch if "v" in segments else None,
        v_frameshift=v_frameshift, j_frameshift=j_frameshift, excluded=None, warnings=tuple(warnings),
        v_novel_snps=v_novel_snps,
    )


def _base_sequence(novel: NovelAllele, reference: Reference) -> str:
    """The reference sequence of ``novel``'s base, checked against the novel sequence (ValueError otherwise)."""
    if not reference.has(novel.base):
        raise ValueError(f"novel allele {novel.name}: its base {novel.base} is not in the reference")
    base_sequence = reference.seq(novel.base)
    novel.check_base(base_sequence)
    return base_sequence


def _rebased(read: str, seg: ObservedSegment, novel: NovelAllele,
             base_sequence: str) -> tuple[ObservedSegment, tuple[int, ...]]:
    """``seg``, observed against ``novel``, re-based onto its base (same coordinates and CIGAR, the base's germline,
    mismatches recounted against ``base_sequence``), and the read positions aligned to the novel allele's SNPs."""
    rebased = observe_segment(read, base_sequence, novel.base, seg.seq_start, seg.germ_start, seg.cigar)
    if rebased is None:  # unreachable: ``seg`` already has an aligned column and no edge gap
        raise ValueError(f"novel allele {novel.name}: observed alignment {seg.cigar} has no aligned column")
    snps = set(novel.snps)
    positions = tuple(column.read_pos for column in walk(parse_cigar(seg.cigar), seg.seq_start, seg.germ_start)
                      if column.read_pos is not None and column.germ_pos in snps)
    return rebased, positions


def as_near_alleles(read: str, seg: ObservedSegment, reference: Reference, letter: str) -> list[str]:
    """The other alleles of the gene of ``seg.allele`` (the primary of a forward segment, or a re-based novel V's
    base), in name order, that cover the observed germline span (at least ``seg.germ_end`` long) and, compared
    position by position through ``seg``'s aligned columns, mismatch the read at no more columns than the primary
    (``seg.n_mismatch``; a read ``N`` never mismatches).

    A sequence-only aligner cannot tell these alleles from the primary on this read, so they are defensible calls.
    GenAIRR's re-call list does not always show them: it leaves out some ties, it is restricted to the donor's
    genotype in a cohort, and for a novel allele it compares the read with the novel sequence, not the base.
    """
    gene = gene_name(seg.allele)
    columns = [(column.read_pos, column.germ_pos)
               for column in walk(parse_cigar(seg.cigar), seg.seq_start, seg.germ_start)
               if column.read_pos is not None and column.germ_pos is not None]
    near = []
    for name, sequence in reference.alleles[letter].items():
        if name == seg.allele or gene_name(name) != gene or len(sequence) < seg.germ_end:
            continue
        if sum(read[r] != "N" and read[r] != sequence[p] for r, p in columns) <= seg.n_mismatch:
            near.append(name)
    return sorted(near)


def _reverse_complemented(reference: Reference, letter: str) -> Reference:
    return Reference(alleles={letter: {n: revcomp(s) for n, s in reference.alleles[letter].items()}},
                     anchors={})


def _anchor_read_pos(seg: ObservedSegment, germ_pos: int) -> int | None:
    """Read position aligned to ``germ_pos`` through an ``M`` column, else None."""
    for col in walk(parse_cigar(seg.cigar), seg.seq_start, seg.germ_start):
        if col.germ_pos == germ_pos:
            return col.read_pos
    return None


def _junction(read: str, segments: dict[str, SegmentTruth], reference: Reference,
              warnings: list[str]) -> JunctionTruth | None:
    if "v" not in segments or "j" not in segments:
        return None
    v, j = segments["v"].observed, segments["j"].observed
    a_v, a_j = reference.anchors.get(v.allele), reference.anchors.get(j.allele)
    if a_v is None or a_j is None:
        warnings.append(f"no anchor for {v.allele if a_v is None else j.allele}; junction undefined")
        return None
    start = _anchor_read_pos(v, a_v)
    last = _anchor_read_pos(j, a_j + 2)
    if start is None or last is None:
        return None
    end = last + 1
    if end <= start:
        warnings.append(f"J anchor ({last}) precedes V anchor ({start}); junction undefined")
        return None
    nt = read[start:end]
    length = end - start
    return JunctionTruth(start=start, end=end, nt=nt, aa=translate(nt) if length % 3 == 0 else None,
                         length=length)


def _frameshift(seg: ObservedSegment) -> bool:
    ops = parse_cigar(seg.cigar)
    return (sum(n for n, op in ops if op == "I") - sum(n for n, op in ops if op == "D")) % 3 != 0


def _stop_codon(read: str, junction_start: int, v_start: int, j_end: int) -> bool:
    """Stop in the junction's frame over full codons from the first V base to the last J base."""
    first = v_start + (junction_start - v_start) % 3
    n_codons = max((j_end - first) // 3, 0)
    return "*" in translate(read[first:first + 3 * n_codons])
