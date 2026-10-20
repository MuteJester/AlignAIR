"""R1 for real reads: each tool's top-1 V call against the donor's own genomic alleles, and the known answers.

This layer is pure: it takes predictions (``core.prediction``), truths (``real.truth``) and a donor's genotype with
its capture (``real.genotype``, ``real.locus``) and reads no file. It reads no tool output beyond the prediction it
scores, so both tools are classified by the same code against the same truth.

**The call.** The top-1 V call is the first allele a tool lists in ``v_call`` (``PredSegment.call``); a row without
one, or no row, is ``no_call``. Every preprocessed read is an IGH amplicon, so ``no_call`` counts as not supported.
A call must be an allele of the shared reference (ValueError otherwise); the genotype arm's tools run on a subset of
it, whose names are shared names, so both arms are classified against the whole shared reference. Identity is the
sequence (``real.genotype``): the IUIS labels of a shared allele are those with its exact sequence (``IuisMap``),
and alleles with the same sequence are one call (``merge_identical``).

**The call's genes** (``SupportIndex.genes``) are the IUIS genes of its labels, or for an allele without a label of
the labels it equals through IMGT 318 (``IuisMap.inferred``): ``IGHVF6-G21*06`` is IGHV1-69*01 and IGHV1-69D*01,
``IGHVF10-G33*01`` is inferred IGHV3-13*06.

**Exclusions** (``EXCLUSIONS``) are decided first, from the call's genes and tool-neutral truth alone (the donor's
genotype, and whether the read is a documented known answer), never from whether the call is right: a right call
and a wrong call to the same genes on the same read are excluded alike, so a tool cannot take its errors out of R1
by making them in an excluded gene. A call is excluded when any of its genes is, by the first reason that applies,
in this order:

- ``dropped_truth``: the gene's only calls were dropped by ``filter_d`` (D only): its truth is dropped, not absent.
- ``not_captured``: the gene's region is not captured (``RealGenotype.captured``, ``real.locus``): the call may be
  an allele the donor's long reads did not resolve.
- ``unmapped``: a guard for a call with no gene at all, labelled or inferred; none on HUMAN_IGH_OGRDB.
- ``novel_gene``: the gene has a donor allele absent from the shared reference, of any functionality
  (``RealGenotype.absent_genes``, per segment: a D gene's absent allele touches only D calls). Over-excluding is
  safe; scoring the gene is not, since a tool may call a read of the novel allele as the carried shared one. Not
  for a V call on a documented known answer (``is_documented``: a known answer that is not novel): that read's V
  is exactly a donor allele in the shared reference, so no novel allele hides behind the call, and a wrong call to
  the gene is ``unsupported_allele``. The flag is the read's, the same for both tools (``classify_v(pred, truth)``,
  ``read_calls(pred, truth)``).
- ``nonfunctional``: none of the gene's donor alleles is functional (each has a V-REGION defect note); such alleles
  are not truth (Mikelov et al. 2024).
- ``beyond_318``: a functional donor allele of the gene has no usable IMGT-gapped form (none, under 318 columns, or
  codon 104 off columns 309-311 as on VDJbase's insertion-gapped calls), so no call to the gene can be compared to
  it through 318. On the pilot donors none has. The shared reference must give every V allele a usable one
  (``SupportIndex`` refuses it otherwise; the dataconfig's does).
- ``half_resolved``: the gene is carried but not resolved (``RealGenotype.resolved``): one haplotype is known and
  the other's region is not spanned, so its unknown second allele could be any call.
- ``low_support``: the V gene has fewer than ``MIN_CLONOTYPES`` (10) clonotypes or ``MIN_NAIVE_CLONOTYPES`` (3)
  naive clonotypes among the known answers (Mikelov et al. 2024's thresholds, per gene; ``gene_support``).

Each tool's exclusions are counted per reason (``exclusion_counts``), so steering reads into excluded genes stays
visible.

**Support.** A call not excluded is ``supported`` when its sequence equals a functional donor allele's exactly or
through IMGT position 318, the two matches taken together: the IMGT-gapped forms' first 318 columns, ungapped
(``through_318``). Two V alleles that differ only past 318 cannot be told apart from rearranged reads (Peres et al.
2023, PIgLET), so the shared reference's ``IGHVF3-G9*04_A319G`` is supported for a donor carrying ``IGHVF3-G9*04``
(IGHV4-28*03 and *01), even where it is exactly a nonfunctional allele of the donor's. Matching by sequence means a
call is supported through any of its labels: ``IGHVF6-G21*06`` supports a donor that carries the sequence at
IGHV1-69D only. A nonfunctional donor allele supports no call. Otherwise the call is:

- ``unsupported_allele``: a gene of the call is carried, but not this allele.
- ``unsupported_gene``: no gene of the call is carried, and their regions are captured: a call to a gene the donor's
  long reads show it lacks.
- ``no_call``: no top-1 V call.

**Shared exclusions** (``shared_exclusions``). A read enters the paired comparison only when neither tool's call
is excluded; ``real_case_scores`` then leaves R1 out for both tools.

**Gene thresholds** (``gene_support``) are tool-neutral: per IUIS gene, the distinct clonotypes (``RealTruth
.clonotype``; a ``single:`` id is one clonotype) and naive clonotypes among the known-answer reads only. A known
answer of several genes (``IGHV3-23|IGHV3-23D``) counts toward each.

**Per-read scores** (``real_case_scores``), registered in ``metrics`` (INFORMATIONAL, ``_LOWER_IS_BETTER``):

- ``real.v.unsupported``: 1 for ``unsupported_allele``, ``unsupported_gene`` or ``no_call``, 0 for ``supported``;
  ``real.v.unsupported_allele``, ``real.v.unsupported_gene`` and ``real.v.no_call`` are its components. All four
  are absent for an excluded read.
- ``real.v.any_listed_supported`` (informational): some V allele the tool lists is supported. IgBLAST lists ties,
  and this shows whether a tie list hides a supported allele behind an unsupported top-1.
- ``real.v.known_top1`` and ``real.v.known_gene_top1``, on every known answer that is not novel, excluded or not:
  their truth is the known allele set, which is tool-neutral, and leaving them out would hide wrong calls into
  excluded genes. The top-1 is right when it, or an allele with its sequence, is a shared name of the set
  (``known_shared``), or equals one of its donor alleles through 318; its gene is right when one of its IUIS genes is
  one of the known answer's. No call is wrong.
- ``real.v.novel_known_call`` (informational): on a novel known answer, whether the call's IUIS gene is the known
  gene: where calls go when the true allele is not in the tool's reference.
- ``real.d.unsupported`` and ``real.j.unsupported`` (informational), where the donor has D or J truth: the same
  rule by exact sequence, with the same exclusions per segment (no 318 rule, no thresholds). Applicability is the
  read's: where the donor has the segment's truth and the read is not excluded for it, a missing D or J call is
  ``nan`` (not covered), never an absent key.

**The oracle** (``oracle_prediction``, ``oracle_self_check``): each known answer's own shared allele, as a tool
would call it, is supported or excluded by a rule of its genes (never ``novel_gene``: its reads are documented known
answers), and its ``known_top1`` is 1: the truth checked against itself (design §10.6).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import NamedTuple

from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.real.genotype import POSITION_318, IuisMap, RealAllele, RealGenotype, upto
from alignair_assay.real.truth import RealTruth, cys104_end

SUPPORTED = "supported"
UNSUPPORTED_ALLELE = "unsupported_allele"
UNSUPPORTED_GENE = "unsupported_gene"
NO_CALL = "no_call"
EXCLUDED = "excluded:"
# The exclusions, in the order a call's genes are checked: the first that applies to any of them is its class.
EXCLUSIONS = ("excluded:dropped_truth", "excluded:not_captured", "excluded:unmapped", "excluded:novel_gene",
              "excluded:nonfunctional", "excluded:beyond_318", "excluded:half_resolved", "excluded:low_support")
# The headline's components: a read with one of these classes is not supported.
UNSUPPORTED = (UNSUPPORTED_ALLELE, UNSUPPORTED_GENE, NO_CALL)
CLASSES = (SUPPORTED, *UNSUPPORTED, *EXCLUSIONS)
# The fewest clonotypes and naive clonotypes among the known answers a V gene needs to be scored.
MIN_CLONOTYPES = 10
MIN_NAIVE_CLONOTYPES = 3
SEGMENTS = ("v", "d", "j")
NAN = float("nan")  # a metric that applies to the read and that the tool did not cover
# What a gene of an unexcluded call is: carried by the donor, or captured and not carried.
_CARRIED, _ABSENT = "carried", "absent"


class ReadCalls(NamedTuple):
    """One tool's classes for one read (``SupportIndex.read_calls``)."""

    v: str  # the top-1 V call's class
    d: str | None  # the top-1 D call's class; None where the donor has no D truth
    j: str | None  # the top-1 J call's class; None where the donor has no J truth
    v_any_listed_supported: bool  # some V allele the tool lists is supported


def through_318(gapped: str | None) -> str | None:
    """The ungapped sequence of IMGT-gapped V ``gapped`` through IMGT position 318 (``real.genotype.upto``); None
    when it cannot be numbered there: no gapped form, fewer than 318 columns, or codon 104 off columns 309-311
    (``real.truth.cys104_end``)."""
    if gapped is None or len(gapped) < POSITION_318 or cys104_end(gapped) is None:
        return None
    return upto(gapped, POSITION_318)


class SupportIndex:
    """What one donor's calls are classified against, built once per donor: its genotype with the captured and
    resolved genes, the IUIS map and identical alleles of the shared reference the tools ran on, and the gene
    support from its known answers (``gene_support``). ValueError for a genotype without its capture
    (``real.locus.donor_capture``), an IUIS map or genotype mapped to another shared reference, or a shared
    reference with a V allele without a usable IMGT-gapped form (``through_318``; one read from a FASTA file has
    none), against which a call could not be compared through 318."""

    def __init__(self, genotype: RealGenotype, iuis: IuisMap, merged: Mapping[str, frozenset[str]],
                 reference: Reference, *, support: Mapping[str, tuple[int, int]]):
        if genotype.captured is None or genotype.resolved is None:
            raise ValueError(f"the genotype of {genotype.donor} has no capture (its captured and resolved genes): "
                             "compute it with real.locus.donor_capture")
        if iuis.reference_digest != reference.digest():
            raise ValueError(f"the IUIS map was built against another shared reference (digest "
                             f"{iuis.reference_digest[:12]}, not {reference.digest()[:12]})")
        mapped_to = (genotype.source or {}).get("reference_digest")
        if mapped_to and mapped_to != reference.digest():
            raise ValueError(f"the genotype of {genotype.donor} was mapped to another shared reference (digest "
                             f"{mapped_to[:12]}, not {reference.digest()[:12]})")
        self.genotype, self.iuis, self.merged, self.reference = genotype, iuis, merged, reference
        self.support = dict(support)
        self._by_sequence: dict[str, dict[str, list[RealAllele]]] = {s: defaultdict(list) for s in SEGMENTS}
        self._carried: dict[str, dict[str, list[RealAllele]]] = {s: defaultdict(list) for s in SEGMENTS}
        self._by_318: dict[str, list[RealAllele]] = defaultdict(list)
        self._by_id: dict[str, list[RealAllele]] = defaultdict(list)
        for allele in genotype.alleles:
            self._by_sequence[allele.segment][allele.sequence.upper()].append(allele)
            self._carried[allele.segment][allele.gene].append(allele)
            if allele.segment == "v":
                self._by_id[allele.allele_id].append(allele)
                through = through_318(allele.gapped)
                if through is not None:
                    self._by_318[through].append(allele)
        self._absent = {s: frozenset(a.gene for a in genotype.alleles if a.segment == s and not a.shared)
                        for s in SEGMENTS}
        self._dropped = frozenset(genotype.dropped_genes())
        self._truth = frozenset(allele.segment for allele in (*genotype.alleles, *genotype.dropped))
        self._shared_318 = {name: through_318(reference.gapped_v.get(name)) for name in reference.alleles.get("v", {})}
        unusable = sorted(name for name, through in self._shared_318.items() if through is None)
        if unusable:
            raise ValueError(f"{len(unusable)} V alleles of the shared reference have no usable IMGT-gapped form "
                             f"(e.g. {unusable[0]}): use the reference built from the dataconfig")
        self._classes: dict[tuple[str, str], str] = {}

    @property
    def donor(self) -> str:
        return self.genotype.donor

    def has_truth(self, segment: str) -> bool:
        """True when the donor's genotype has an allele of ``segment`` (kept or dropped)."""
        return segment in self._truth

    def genes(self, name: str) -> tuple[str, ...]:
        """The IUIS genes of shared allele ``name`` (``IuisMap.genes``; two for a paralog pair, none for an unmapped
        allele). ValueError for a name that is not in the shared reference."""
        try:
            return self.iuis.genes(name)
        except KeyError:
            raise ValueError(f"{name!r} is not an allele of the shared reference") from None

    def passes(self, gene: str) -> bool:
        """True when ``gene`` has at least ``MIN_CLONOTYPES`` clonotypes and ``MIN_NAIVE_CLONOTYPES`` naive ones
        among the known answers (a gene without a known answer has none)."""
        clonotypes, naive = self.support.get(gene, (0, 0))
        return clonotypes >= MIN_CLONOTYPES and naive >= MIN_NAIVE_CLONOTYPES

    def classify(self, name: str | None, segment: str = "v", *, documented: bool = False) -> str:
        """The class of a call of shared allele ``name`` for ``segment`` (the module docstring); ``no_call`` for
        None or ``""``. ``documented`` says the read is a documented known answer (``is_documented``), which lifts
        ``novel_gene`` from V calls. ValueError for a segment that is not V, D or J, or a name that is not one of
        the shared reference's alleles of it."""
        if segment not in SEGMENTS:
            raise ValueError(f"segment {segment!r} is not one of {', '.join(SEGMENTS)}")
        if not name:
            return NO_CALL
        key = (name, segment, documented)
        if key not in self._classes:
            self._classes[key] = self._classify(name, segment, documented)
        return self._classes[key]

    def classify_v(self, pred: Prediction | None, truth: RealTruth | None = None) -> str:
        """The class of ``pred``'s top-1 V call on the read of ``truth`` (``is_documented``; the same truth for every
        tool); ``no_call`` without a row, a V segment or a call."""
        return self.classify(_top1(pred, "v"), "v", documented=is_documented(truth))

    def read_calls(self, pred: Prediction | None, truth: RealTruth | None = None) -> ReadCalls:
        """``pred``'s classes on the read of ``truth`` (as ``classify_v``): its top-1 V, D and J calls (D and J only
        where the donor has their truth) and whether any V allele it lists, top-1 first, is supported."""
        segment = _segment(pred, "v")
        listed = dict.fromkeys(name for name in (segment.call, *segment.call_set) if name) if segment else {}
        known = is_documented(truth)
        return ReadCalls(v=self.classify_v(pred, truth),
                         d=self.classify(_top1(pred, "d"), "d") if self.has_truth("d") else None,
                         j=self.classify(_top1(pred, "j"), "j") if self.has_truth("j") else None,
                         v_any_listed_supported=any(self.classify(name, "v", documented=known) == SUPPORTED
                                                    for name in listed))

    def known_top1(self, name: str | None, truth: RealTruth) -> bool:
        """True when V call ``name`` is in ``truth``'s known set: it, or an allele with its sequence, is one of the
        set's shared names, or it equals one of the set's donor alleles through IMGT 318. False without a call or a
        known answer."""
        if not name or truth.known is None:
            return False
        if self.merged.get(name, frozenset({name})) & (truth.known_shared or frozenset()):
            return True
        through = self._shared_318.get(name)
        return through is not None and any(through_318(allele.gapped) == through
                                           for allele_id in truth.known for allele in self._by_id.get(allele_id, ()))

    def known_gene_top1(self, name: str | None, truth: RealTruth) -> bool:
        """True when one of V call ``name``'s IUIS genes is one of ``truth``'s known genes."""
        if not name or truth.known is None:
            return False
        return bool(set(self.genes(name)) & set(truth.known_gene.split("|")))

    def _classify(self, name: str, segment: str, documented: bool) -> str:
        table = self.reference.alleles.get(segment, {})
        if name not in table:
            raise ValueError(f"{name!r} is not a {segment.upper()} allele of the shared reference")
        genes = self.genes(name)
        if not genes:
            return "excluded:unmapped"  # a guard: no gene, labelled or inferred (none on HUMAN_IGH_OGRDB)
        # the exclusions are the genes' and the read's, never the call's: a right and a wrong call to the same genes
        # on the same read are excluded alike
        found = [self._gene_class(gene, segment, novel=not (documented and segment == "v")) for gene in genes]
        for reason in EXCLUSIONS:
            if reason in found:
                return reason
        if any(allele.functional for allele in self._matches(name, segment)):
            return SUPPORTED
        return UNSUPPORTED_ALLELE if _CARRIED in found else UNSUPPORTED_GENE

    def _matches(self, name: str, segment: str) -> list[RealAllele]:
        """The donor alleles shared allele ``name`` equals: exactly, and for V through IMGT 318 too (the union)."""
        matches = list(self._by_sequence[segment].get(self.reference.alleles[segment][name].upper(), ()))
        if segment == "v":
            matches += self._by_318.get(self._shared_318[name], ())
        return matches

    def _gene_class(self, gene: str, segment: str, *, novel: bool) -> str:
        """The exclusion of every call to ``gene`` (the module docstring; ``novel`` False lifts ``novel_gene``),
        else ``_CARRIED`` or ``_ABSENT`` (captured, not carried)."""
        if gene in self._dropped:
            return "excluded:dropped_truth"
        if gene not in self.genotype.captured:
            return "excluded:not_captured"
        alleles = self._carried[segment].get(gene)
        if not alleles:
            return _ABSENT
        if novel and gene in self._absent[segment]:
            return "excluded:novel_gene"
        functional = [allele for allele in alleles if allele.functional]
        if not functional:
            return "excluded:nonfunctional"
        if segment == "v" and any(through_318(allele.gapped) is None for allele in functional):
            return "excluded:beyond_318"
        if gene not in self.genotype.resolved:
            return "excluded:half_resolved"
        if segment == "v" and not self.passes(gene):
            return "excluded:low_support"
        return _CARRIED


def classify_v_call(pred: Prediction | None, genotype: RealGenotype, iuis: IuisMap,
                    merged: Mapping[str, frozenset[str]], *, reference: Reference,
                    support: Mapping[str, tuple[int, int]], truth: RealTruth | None = None) -> str:
    """The class of ``pred``'s top-1 V call against ``genotype`` on the read of ``truth``
    (``SupportIndex.classify_v``). Build a ``SupportIndex`` once to classify many reads."""
    return SupportIndex(genotype, iuis, merged, reference, support=support).classify_v(pred, truth)


def is_documented(truth: RealTruth | None) -> bool:
    """True for the truth of a documented known answer: a known answer whose alleles are in the shared reference
    (not ``novel_known``). Its V is exactly a documented donor allele, so no novel allele hides behind a call on it.
    False without a truth."""
    return truth is not None and truth.known is not None and not truth.novel_known


def gene_support(truths: Iterable[RealTruth]) -> dict[str, tuple[int, int]]:
    """IUIS gene -> (clonotypes, naive clonotypes) among the known-answer reads of ``truths``, one donor's: the
    distinct ``clonotype`` ids of the reads whose known answer is of the gene (each gene of a ``|``-joined
    ``known_gene``), and of those reads that are naive. Reads without a known answer never count. ValueError for
    the truths of several donors."""
    clonotypes: dict[str, set[str]] = defaultdict(set)
    naive: dict[str, set[str]] = defaultdict(set)
    donors = set()
    for truth in truths:
        donors.add(truth.donor)
        if truth.known is None:
            continue
        for gene in truth.known_gene.split("|"):
            clonotypes[gene].add(truth.clonotype)
            if truth.naive:
                naive[gene].add(truth.clonotype)
    if len(donors) > 1:
        raise ValueError(f"gene support is per donor: the truths are of donors {', '.join(sorted(donors))}")
    return {gene: (len(clonotypes[gene]), len(naive[gene])) for gene in sorted(clonotypes)}


def passing_genes(support: Mapping[str, tuple[int, int]]) -> set[str]:
    """The genes of ``support`` with at least ``MIN_CLONOTYPES`` clonotypes and ``MIN_NAIVE_CLONOTYPES`` naive
    ones."""
    return {gene for gene, (clonotypes, naive) in support.items()
            if clonotypes >= MIN_CLONOTYPES and naive >= MIN_NAIVE_CLONOTYPES}


def shared_exclusions(classes_a: Mapping[str, str | None], classes_b: Mapping[str, str | None]) -> set[str]:
    """The read ids excluded for both tools: those where either tool's class (read id -> class, for one segment;
    None where it has no truth) is an exclusion. ValueError when the two tools' classes are not of the same
    reads."""
    if classes_a.keys() != classes_b.keys():
        unpaired = sorted(classes_a.keys() ^ classes_b.keys())
        raise ValueError(f"the two tools were classified on different reads ({len(unpaired)} unpaired, e.g. read "
                         f"{unpaired[0]})")
    return {read_id for read_id, cls in classes_a.items() if _excluded(cls) or _excluded(classes_b[read_id])}


def exclusion_counts(classes: Mapping[str, str | None] | Iterable[str | None]) -> dict[str, int]:
    """One tool's exclusions: each reason of ``EXCLUSIONS``, in order, -> how many of ``classes`` (read id ->
    class, or the classes alone) it is. ValueError for an exclusion that is not one of them."""
    values = classes.values() if isinstance(classes, Mapping) else classes
    counts = Counter(cls for cls in values if _excluded(cls))
    unknown = set(counts) - set(EXCLUSIONS)
    if unknown:
        raise ValueError(f"{sorted(unknown)[0]} is not an exclusion")
    return {reason: counts.get(reason, 0) for reason in EXCLUSIONS}


def real_case_scores(pred: Prediction | None, truth: RealTruth, calls: str | ReadCalls, *,
                     excluded: bool | Collection[str], index: SupportIndex) -> CaseScores:
    """One tool's scores on one read (the module docstring). ``calls`` is the class of its top-1 V call, or its
    ``ReadCalls`` (which add the D, J and listed-V metrics); ``excluded`` is True when the read is out of R1 for both
    tools (``shared_exclusions``), or the segments it is out of (``{"v", "d"}``). ``pred`` None is a tool that
    emitted no row. ValueError for a prediction of another read, a truth of another donor than the index's, or a
    class that is an exclusion of a segment not excluded."""
    if pred is not None and pred.sequence_id != truth.read_id:
        raise ValueError(f"the prediction of read {pred.sequence_id} is scored against the truth of read "
                         f"{truth.read_id}")
    if truth.donor != index.donor:
        raise ValueError(f"read {truth.read_id} is of donor {truth.donor}, scored against {index.donor}'s genotype")
    skip = ({"v"} if excluded else set()) if isinstance(excluded, bool) else set(excluded)
    v = calls.v if isinstance(calls, ReadCalls) else calls
    scores: CaseScores = {}
    if "v" not in skip:
        _check(v, truth.read_id, "v")
        scores.update({"real.v.unsupported": float(v != SUPPORTED),
                       "real.v.unsupported_allele": float(v == UNSUPPORTED_ALLELE),
                       "real.v.unsupported_gene": float(v == UNSUPPORTED_GENE),
                       "real.v.no_call": float(v == NO_CALL)})
        if isinstance(calls, ReadCalls):
            scores["real.v.any_listed_supported"] = float(calls.v_any_listed_supported)
    if truth.known is not None:
        top = _top1(pred, "v")
        if truth.novel_known:
            scores["real.v.novel_known_call"] = float(index.known_gene_top1(top, truth))
        else:
            scores["real.v.known_top1"] = float(index.known_top1(top, truth))
            scores["real.v.known_gene_top1"] = float(index.known_gene_top1(top, truth))
    if isinstance(calls, ReadCalls):
        for segment, cls in (("d", calls.d), ("j", calls.j)):
            if cls is None or segment in skip:
                continue  # no truth for the segment, or the read is out for it: not applicable
            _check(cls, truth.read_id, segment)
            scores[f"real.{segment}.unsupported"] = NAN if cls == NO_CALL else float(cls != SUPPORTED)
    return scores


def oracle_prediction(truth: RealTruth) -> Prediction | None:
    """The oracle's row for a known answer that is not novel: its shared names, the first (sorted) as the top-1
    V call; None for any other read."""
    if truth.known is None or truth.novel_known:
        return None
    names = tuple(sorted(truth.known_shared))
    return Prediction(sequence_id=truth.read_id, segments={"v": PredSegment(call=names[0], call_set=names)})


def oracle_self_check(truths: Iterable[RealTruth], index: SupportIndex) -> dict:
    """The truth checked against itself (design §10.6): the oracle (``oracle_prediction``) on every known answer
    that is not novel. ``reads``; ``classes`` (class -> reads); ``excluded_genes`` (each exclusion -> the known
    genes of its reads); ``scored`` (reads not excluded); ``supported`` and ``unsupported`` (their shares of the
    scored reads, None without one); ``known_top1`` and ``known_gene_top1`` (shares of all ``reads``); and
    ``passed``: every scored read supported, and every read's known top-1 and gene right."""
    classes: Counter[str] = Counter()
    genes: dict[str, set[str]] = defaultdict(set)
    reads = scored = supported = top1 = gene_top1 = 0
    for truth in truths:
        pred = oracle_prediction(truth)
        if pred is None:
            continue
        reads += 1
        cls = index.classify_v(pred, truth)
        classes[cls] += 1
        if _excluded(cls):
            genes[cls].update(truth.known_gene.split("|"))
        else:
            scored += 1
            supported += cls == SUPPORTED
        scores = real_case_scores(pred, truth, cls, excluded=_excluded(cls), index=index)
        top1 += scores["real.v.known_top1"] == 1.0
        gene_top1 += scores["real.v.known_gene_top1"] == 1.0
    return {"reads": reads, "classes": dict(sorted(classes.items())),
            "excluded_genes": {cls: sorted(names) for cls, names in sorted(genes.items())},
            "scored": scored, "supported": supported / scored if scored else None,
            "unsupported": (scored - supported) / scored if scored else None,
            "known_top1": top1 / reads if reads else None, "known_gene_top1": gene_top1 / reads if reads else None,
            "passed": scored > 0 and supported == scored and top1 == reads and gene_top1 == reads}


def _segment(pred: Prediction | None, segment: str) -> PredSegment | None:
    return None if pred is None else pred.segments.get(segment)


def _top1(pred: Prediction | None, segment: str) -> str | None:
    found = _segment(pred, segment)
    return None if found is None else found.call


def _excluded(cls: str | None) -> bool:
    return cls is not None and cls.startswith(EXCLUDED)


def _check(cls: str, read_id: str, segment: str) -> None:
    """ValueError unless ``cls`` is a class R1 scores: an exclusion must have left the read out for both tools."""
    if _excluded(cls):
        raise ValueError(f"read {read_id}: its {segment.upper()} call is {cls}, so the read is out of R1 for both "
                         "tools (shared_exclusions); score it with excluded=True")
    if cls not in (SUPPORTED, *UNSUPPORTED):
        raise ValueError(f"read {read_id}: {cls!r} is not a class of a {segment.upper()} call")
