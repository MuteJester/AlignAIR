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
and a wrong call to the same genes are excluded alike; a call into an excluded gene still takes the read out of R1
for both tools (``shared_exclusions``), so a tool's errors into excluded genes leave R1 with their reads, which the
workflow counts per tool (``workflow.real_score.one_sided``). The genes are the call's own IUIS label genes, not
those of the donor alleles it matches: a call matches donor alleles only when it is right, so their genes would make
exclusion depend on it. This supersedes the earlier ruling (b), "genes of the matching donor alleles". A call is
excluded when any of its genes is, by the first reason that applies, in this order:

- ``dropped_truth``: the gene's only calls were dropped by ``filter_d`` (D only): its truth is dropped, not absent.
- ``not_captured``: the gene's region is not captured (``RealGenotype.captured``, ``real.locus``): the call may be
  an allele the donor's long reads did not resolve.
- ``uncalled``: the gene is spanned and not carried, but a capturing contig holds sequence there
  (``RealGenotype.uncalled``, ``real.locus``): VDJbase did not call it, which is no evidence the donor lacks it. For
  V, D and J alike. A tool's real error into such a gene leaves R1 too (SC-23: AlignAIR calls IGHV4-30-2 on reads
  that exactly match IGHV4-59 or IGHV4-NL1), the price of not counting a calling gap as the tool's error.
- ``unmapped``: a guard for a call with no gene at all, labelled or inferred; none on HUMAN_IGH_OGRDB.
- ``novel_gene``: the gene has a donor allele absent from the shared reference, of any functionality
  (``RealGenotype.absent_genes``, per segment: a D gene's absent allele touches only D calls). Over-excluding is
  safe; scoring the gene is not, since a tool may call a read of the novel allele as the carried shared one. Not
  for a V call on a documented known answer (``is_documented``: a known answer whose allele set is not entirely
  absent from the shared reference): that read's observed V is exactly a donor allele in the shared reference, so
  no novel allele hides behind the call, and a wrong call to the gene is ``unsupported_allele``. A set can mix a
  documented and an absent allele whose windows the read cannot tell apart (W-97 has 2 such reads, SC-23 10); it
  is documented. The flag is the read's, the same for both tools (``classify_v(pred, truth=truth)``,
  ``read_calls(pred, truth=truth)``; ``truth`` is required by keyword, so no tool is classified without it).
- ``nonfunctional``: the gene is carried only as nonfunctional alleles (each has a V-REGION defect note), so its
  truth is in doubt (Mikelov et al. 2024 leave such alleles out): every call to it is excluded, whether it equals
  one of them or not.
- ``beyond_318``: a functional donor allele of the gene has no usable IMGT-gapped form (none, under 318 columns, or
  codon 104 off columns 309-311 as on VDJbase's insertion-gapped calls), so no call to the gene can be compared to
  it through 318. On the pilot donors none has. The shared reference must give every V allele a usable one
  (``SupportIndex`` refuses it otherwise; the dataconfig's does).
- ``half_resolved``: the gene is carried but not resolved (``RealGenotype.resolved``): an allele on one haplotype is
  unknown, its region not spanned, or spanned but not called (a contig there holds a sequence the genotype does not
  call), so that allele could be any call.
- ``low_support``, in the gene-support sensitivity variant only (``SupportIndex(..., thresholds=True)``): the V
  gene has fewer than ``MIN_CLONOTYPES`` (10) clonotypes or ``MIN_NAIVE_CLONOTYPES`` (3) naive clonotypes among the
  known answers (Mikelov et al. 2024's thresholds, per gene; ``gene_support``). The R1 primary does not apply it
  (supersedes design §6): a gene the donor does not carry has no known answers, so the thresholds could only ever
  drop calls to carried genes, supported ones among them, while every ``unsupported_gene`` call stayed. On few
  reads that drives R1 to 1, and at full scale it still skews the denominator. The parent design applies these
  thresholds to the per-genotype secondary metric (P3b), not to the per-read primary.

Each tool's exclusions are counted per reason (``exclusion_counts``), so steering reads into excluded genes stays
visible.

**Support.** A call not excluded is ``supported`` when its sequence equals any donor allele's, functional or not,
exactly or through IMGT position 318, the two matches taken together: the IMGT-gapped forms' first 318 columns,
ungapped (``through_318``). R1 asks whether the donor's genome carries the call, and a nonfunctional allele is in the
genome. Two V alleles that differ only past 318 cannot be told apart from rearranged reads (Peres et al. 2023,
PIgLET), so the shared reference's ``IGHVF3-G9*04_A319G`` is supported for a donor carrying ``IGHVF3-G9*04``
(IGHV4-28*03 and *01). Matching by sequence means a call is supported through any of its labels: ``IGHVF6-G21*06``
supports a donor that carries the sequence at IGHV1-69D only. A donor allele without a usable IMGT-gapped form
(``through_318`` None) supports a call only by exact match. A functional one excludes its gene (``beyond_318``), so
this concerns nonfunctional alleles alone, and it changes no class on the pilot donors. Otherwise the call is:

- ``unsupported_allele``: a gene of the call is carried, but not this allele.
- ``unsupported_gene``: no gene of the call is carried, and every capturing contig shows it deleted (``absent``,
  ``real.locus``): a call to a gene the donor's long reads show it lacks.
- ``no_call``: no top-1 V call.

**Shared exclusions** (``shared_exclusions``). A read enters the paired comparison only when neither tool's call
is excluded; ``real_case_scores`` then leaves R1 out for both tools.

**Gene thresholds** (``gene_support``) are tool-neutral: per IUIS gene, the distinct clonotypes (``RealTruth
.clonotype``; a ``single:`` id is one clonotype) and naive clonotypes among the known-answer reads only. A known
answer of several genes (``IGHV3-23|IGHV3-23D``) counts toward each. They select the sensitivity variant "R1 with
Mikelov gene-support thresholds" (``thresholds=True``); the R1 primary is the index without them.

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
  one of the known answer's: a gene VDJbase found its alleles at, or an IUIS gene of their own sequence (an allele
  found at a paralog's locus may have a documented allele's sequence). No call is wrong.
- ``real.v.ref_known_top1`` and ``real.v.ref_known_gene_top1`` (compared, higher is better), on every read with a
  reference-exact known answer (``RealTruth.ref_known``: the read is exactly a shared allele's window, judged against
  the shared reference alone), excluded or not: the truth is the read itself, and it covers genes the donor's
  assembly misses. The top-1 is right when it, or an allele with its sequence (``merge_identical``), is in the set;
  its gene when one of its IUIS genes is one of the set's (``ref_known_gene``). No call is wrong.
- ``real.v.novel_known_call`` (informational): on a novel known answer, whether the call's IUIS gene is the known
  gene: where calls go when the true allele is not in the tool's reference.
- The tie-aware metrics (informational), which separate a read that cannot tell two alleles apart from a wrong call.
  A call's window over a read is the last ``L`` nt of its region through Cys104 (``real.truth.cys104_regions``; a
  shared allele's from its gapped form), ``L`` the read's ``cys_end`` less its first observed base: what the read
  observes of the V. Two windows are equal when they agree over their common length from Cys104 back (an allele
  shorter than ``L`` is compared over its own length), the read's ``N`` positions (``v_window_n``) left out. A read
  without a placed Cys104 is ``nan`` (not covered). ``SupportIndex`` caches each per call, window and Ns.

  - ``real.v.unsupported_observable`` (lower is better), on the reads R1 scores: 1 when the call is not supported
    and its window equals no donor allele's (any gene, any functionality; a missing call is 1), else 0.
  - ``real.v.known_top1_observable``, on every documented known answer (excluded or not, as ``known_top1``): 1 when
    the call's window equals the known alleles' window.
- ``real.d.unsupported`` and ``real.j.unsupported`` (informational), where the donor has D or J truth: the same
  rule by exact sequence, with the same exclusions per segment (no 318 rule, no thresholds). Applicability is the
  read's: where the donor has the segment's truth and the read is not excluded for it, a missing D or J call is
  ``nan`` (not covered), never an absent key.

**The oracle** (``oracle_prediction``, ``oracle_self_check``): each known answer's own shared allele, as a tool
would call it, is supported or excluded by a rule of its genes (never ``novel_gene``: its reads are documented known
answers), and its ``known_top1`` is 1: the truth checked against itself (design §10.6). It proves the truth and the
metric wiring on carried genes only, since every oracle call is a donor allele. Two decoys (``decoy_calls``) check
what it never calls: an allele of a carried, resolved gene that matches no donor allele must be
``unsupported_allele``, and an allele all of whose genes are uncalled must be ``excluded:uncalled``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import NamedTuple

from alignair_assay.core.prediction import PredSegment, Prediction
from alignair_assay.core.reference import Reference
from alignair_assay.metrics import CaseScores
from alignair_assay.real.genotype import POSITION_318, IuisMap, RealAllele, RealGenotype, upto
from alignair_assay.real.truth import RealTruth, cys104_end, cys104_regions

SUPPORTED = "supported"
UNSUPPORTED_ALLELE = "unsupported_allele"
UNSUPPORTED_GENE = "unsupported_gene"
NO_CALL = "no_call"
EXCLUDED = "excluded:"
# The exclusions, in the order a call's genes are checked: the first that applies to any of them is its class.
EXCLUSIONS = ("excluded:dropped_truth", "excluded:not_captured", "excluded:uncalled", "excluded:unmapped",
              "excluded:novel_gene", "excluded:nonfunctional", "excluded:beyond_318", "excluded:half_resolved",
              "excluded:low_support")
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
    support from its known answers (``gene_support``). ``thresholds`` selects the variant: False (the R1 primary)
    never excludes a call for low support; True applies the gene-support thresholds (``excluded:low_support``,
    the sensitivity variant), and needs ``support``. ValueError for a genotype without its capture
    (``real.locus.donor_capture``), an IUIS map or genotype mapped to another shared reference, a shared reference
    with a V allele without a usable IMGT-gapped form (``through_318``; one read from a FASTA file has none),
    against which a call could not be compared through 318, or ``thresholds`` without ``support``."""

    def __init__(self, genotype: RealGenotype, iuis: IuisMap, merged: Mapping[str, frozenset[str]],
                 reference: Reference, *, support: Mapping[str, tuple[int, int]] | None = None,
                 thresholds: bool = False):
        if thresholds and support is None:
            raise ValueError("the gene-support thresholds need the gene support of the donor's known answers "
                             "(gene_support)")
        if genotype.captured is None or genotype.resolved is None or genotype.uncalled is None:
            raise ValueError(f"the genotype of {genotype.donor} has no capture (its captured, resolved and uncalled "
                             "genes): compute it with real.locus.donor_capture")
        if iuis.reference_digest != reference.digest():
            raise ValueError(f"the IUIS map was built against another shared reference (digest "
                             f"{iuis.reference_digest[:12]}, not {reference.digest()[:12]})")
        mapped_to = (genotype.source or {}).get("reference_digest")
        if mapped_to and mapped_to != reference.digest():
            raise ValueError(f"the genotype of {genotype.donor} was mapped to another shared reference (digest "
                             f"{mapped_to[:12]}, not {reference.digest()[:12]})")
        self.genotype, self.iuis, self.merged, self.reference = genotype, iuis, merged, reference
        self.support = dict(support or {})
        self.thresholds = thresholds
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
            raise ValueError(f"{len(unusable)} V alleles of the shared reference have no usable IMGT-gapped form, so "
                             f"no call could be compared to them through IMGT 318: {', '.join(unusable)}; use the "
                             "reference built from the dataconfig")
        self._classes: dict[tuple[str, str, bool], str] = {}  # (name, segment, documented) -> class
        # the donor's V regions through Cys104, any gene or functionality, and each shared V allele's
        self._regions = cys104_regions(genotype.alleles)[0]
        self._shared_regions: dict[str, str | None] = {}
        self._observable: dict[tuple, bool] = {}  # (name, window length, Ns[, known]) -> equal

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

    def classify_v(self, pred: Prediction | None, *, truth: RealTruth | None) -> str:
        """The class of ``pred``'s top-1 V call on the read of ``truth`` (``is_documented``); ``no_call`` without a
        row, a V segment or a call. ``truth`` is required, by keyword, so every tool is classified with the same
        truth: None is a read without one, said explicitly."""
        return self.classify(_top1(pred, "v"), "v", documented=is_documented(truth))

    def read_calls(self, pred: Prediction | None, *, truth: RealTruth | None) -> ReadCalls:
        """``pred``'s classes on the read of ``truth`` (as ``classify_v``, ``truth`` required by keyword): its top-1
        V, D and J calls (D and J only where the donor has their truth) and whether any V allele it lists, top-1
        first, is supported."""
        segment = _segment(pred, "v")
        listed = dict.fromkeys(name for name in (segment.call, *segment.call_set) if name) if segment else {}
        known = is_documented(truth)
        return ReadCalls(v=self.classify_v(pred, truth=truth),
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

    def ref_known_top1(self, name: str | None, truth: RealTruth) -> bool:
        """True when V call ``name``, or an allele with its sequence, is in ``truth``'s reference-exact known answer
        (``ref_known``). False without a call or one."""
        if not name or truth.ref_known is None:
            return False
        return bool(self.merged.get(name, frozenset({name})) & truth.ref_known)

    def ref_known_gene_top1(self, name: str | None, truth: RealTruth) -> bool:
        """True when one of V call ``name``'s IUIS genes is one of ``truth``'s reference-exact known genes
        (``ref_known_gene``). False without a call or one."""
        if not name or truth.ref_known is None:
            return False
        return bool(set(self.genes(name)) & set(truth.ref_known_gene.split("|")))

    def known_gene_top1(self, name: str | None, truth: RealTruth) -> bool:
        """True when one of V call ``name``'s IUIS genes is one of ``truth``'s known genes: the genes VDJbase found
        its alleles at (``known_gene``) or the IUIS genes of their own sequence (those of ``known_shared``). VDJbase
        names an allele by its locus, and its sequence may be a paralog's documented allele (SC-11's
        ``IGHV3-30-3*03_g150a`` is IGHV3-30*01's sequence), which a call by sequence cannot tell apart."""
        if not name or truth.known is None:
            return False
        known = set(truth.known_gene.split("|"))
        known.update(gene for shared in truth.known_shared or () for gene in self.genes(shared))
        return bool(set(self.genes(name)) & known)

    def observably_supported(self, name: str | None, truth: RealTruth | None) -> bool | None:
        """Whether V call ``name``'s window over the read of ``truth`` equals some donor allele's window, of any gene
        or functionality (the module docstring): the read could not tell the call from an allele the donor carries.
        False without a call; None when the read has no placed Cys104."""
        if truth is None or truth.cys_end is None:
            return None
        if not name:
            return False
        length = truth.cys_end - truth.observed[0]
        key = (name, length, truth.v_window_n)
        if key not in self._observable:
            call = self._shared_region(name)
            self._observable[key] = call is not None and any(
                same_window(call, region, length, truth.v_window_n) for _, region in self._regions)
        return self._observable[key]

    def known_top1_observable(self, name: str | None, truth: RealTruth) -> bool:
        """Whether V call ``name``'s window over the read of ``truth`` equals its known alleles' window. False without
        a call, a known answer or a placed Cys104."""
        if not name or truth.known is None or truth.cys_end is None:
            return False
        length = truth.cys_end - truth.observed[0]
        key = (name, length, truth.v_window_n, truth.known)
        if key not in self._observable:
            call = self._shared_region(name)
            self._observable[key] = call is not None and any(
                same_window(call, region, length, truth.v_window_n)
                for allele, region in self._regions if allele.allele_id in truth.known)
        return self._observable[key]

    def _shared_region(self, name: str) -> str | None:
        """Shared V allele ``name``'s uppercase region through Cys104, from its gapped form; None without one."""
        if name not in self._shared_regions:
            gapped = self.reference.gapped_v.get(name)
            end = cys104_end(gapped) if gapped else None
            sequence = self.reference.alleles.get("v", {}).get(name)
            self._shared_regions[name] = None if end is None or sequence is None else sequence[:end].upper()
        return self._shared_regions[name]

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
        if self._matches(name, segment):  # any donor allele, functional or not: the donor's genome carries it
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
            return "excluded:uncalled" if gene in self.genotype.uncalled else _ABSENT
        if novel and gene in self._absent[segment]:
            return "excluded:novel_gene"
        functional = [allele for allele in alleles if allele.functional]
        if not functional:
            return "excluded:nonfunctional"
        if segment == "v" and any(through_318(allele.gapped) is None for allele in functional):
            return "excluded:beyond_318"
        if gene not in self.genotype.resolved:
            return "excluded:half_resolved"
        if self.thresholds and segment == "v" and not self.passes(gene):
            return "excluded:low_support"
        return _CARRIED


def same_window(a: str, b: str, length: int, gaps: Iterable[int] = ()) -> bool:
    """Whether regions ``a`` and ``b``, each ending at Cys104, agree over a read's window of ``length`` nt ending
    there: their last ``min(length, len(a), len(b))`` bases, leaving out the read's ``N`` offsets ``gaps`` (counted
    from the window's first base)."""
    size = min(length, len(a), len(b))
    skip = {length - 1 - offset for offset in gaps}  # positions from the 3' end; 0 is the last base
    if not skip:
        return a[len(a) - size:] == b[len(b) - size:]
    return all(a[-1 - k] == b[-1 - k] for k in range(size) if k not in skip)


def classify_v_call(pred: Prediction | None, genotype: RealGenotype, iuis: IuisMap,
                    merged: Mapping[str, frozenset[str]], *, reference: Reference,
                    support: Mapping[str, tuple[int, int]] | None = None, thresholds: bool = False,
                    truth: RealTruth | None) -> str:
    """The class of ``pred``'s top-1 V call against ``genotype`` on the read of ``truth`` (required, as in
    ``SupportIndex.classify_v``), in the R1 primary or, with ``thresholds``, the gene-support variant. Build a
    ``SupportIndex`` once to classify many reads."""
    index = SupportIndex(genotype, iuis, merged, reference, support=support, thresholds=thresholds)
    return index.classify_v(pred, truth=truth)


def is_documented(truth: RealTruth | None) -> bool:
    """True for the truth of a documented known answer: a known answer whose allele set is not entirely absent from
    the shared reference (not ``novel_known``). A set may mix documented and absent alleles, whose windows the read
    cannot tell apart (W-97 has 2 such reads, SC-23 10): its observed V is still exactly a documented donor allele's
    window, so no novel allele hides behind a call on it. False without a truth or a known answer."""
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
    top = _top1(pred, "v")
    if "v" not in skip:
        _check(v, truth.read_id, "v")
        observable = index.observably_supported(top, truth)
        scores.update({"real.v.unsupported": float(v != SUPPORTED),
                       "real.v.unsupported_allele": float(v == UNSUPPORTED_ALLELE),
                       "real.v.unsupported_gene": float(v == UNSUPPORTED_GENE),
                       "real.v.no_call": float(v == NO_CALL),
                       "real.v.unsupported_observable": NAN if observable is None else float(v != SUPPORTED
                                                                                             and not observable)})
        if isinstance(calls, ReadCalls):
            scores["real.v.any_listed_supported"] = float(calls.v_any_listed_supported)
    if truth.known is not None:
        if truth.novel_known:
            scores["real.v.novel_known_call"] = float(index.known_gene_top1(top, truth))
        else:
            scores["real.v.known_top1"] = float(index.known_top1(top, truth))
            scores["real.v.known_gene_top1"] = float(index.known_gene_top1(top, truth))
            scores["real.v.known_top1_observable"] = float(index.known_top1_observable(top, truth))
    if truth.ref_known is not None:
        scores["real.v.ref_known_top1"] = float(index.ref_known_top1(top, truth))
        scores["real.v.ref_known_gene_top1"] = float(index.ref_known_gene_top1(top, truth))
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
    scored reads, None without one); ``known_top1`` and ``known_gene_top1`` (shares of all ``reads``);
    ``ref_reads`` and ``ref_known_top1``: the reads with a reference-exact known answer, and the share whose oracle
    call (the set's first shared name) is right there; and ``passed``: every scored read supported, and every read's
    known top-1 and gene right, and every reference-exact one."""
    classes: Counter[str] = Counter()
    genes: dict[str, set[str]] = defaultdict(set)
    reads = scored = supported = top1 = gene_top1 = ref_reads = ref_top1 = 0
    truths = list(truths)
    for truth in truths:
        if truth.ref_known is not None:
            ref_reads += 1
            ref_top1 += index.ref_known_top1(sorted(truth.ref_known)[0], truth)
    for truth in truths:
        pred = oracle_prediction(truth)
        if pred is None:
            continue
        reads += 1
        cls = index.classify_v(pred, truth=truth)
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
            "ref_reads": ref_reads, "ref_known_top1": ref_top1 / ref_reads if ref_reads else None,
            "passed": scored > 0 and supported == scored and top1 == reads and gene_top1 == reads
                      and ref_top1 == ref_reads}


def decoy_calls(index: SupportIndex) -> dict:
    """The decoys of the module docstring for the donor of ``index``, each the first shared V allele in name order
    that qualifies, by the truth alone: ``unsupported_allele``, an allele none of whose genes is excluded (each is
    carried, captured, resolved, not uncalled or dropped, with no allele absent from the shared reference and a
    functional allele, every functional one numbered through IMGT 318) and that equals no donor allele, exactly or
    through 318; ``uncalled``, an allele all of whose genes are uncalled. Each is ``{"call", "genes", "class",
    "expected", "passed"}``, or None when the donor has no such allele; ``passed`` when each found is its class."""
    genotype = index.genotype
    absent = genotype.absent_genes()

    def clean(gene: str) -> bool:
        alleles = genotype.carried(gene)
        functional = [allele for allele in alleles if allele.functional and allele.segment == "v"]
        return (bool(alleles) and gene in genotype.captured and gene in genotype.resolved
                and gene not in genotype.uncalled and gene not in genotype.dropped_genes() and gene not in absent
                and bool(functional) and all(through_318(allele.gapped) is not None for allele in functional))

    found: dict = {"unsupported_allele": None, "uncalled": None}
    for name in sorted(index.reference.alleles.get("v", {})):
        genes = index.genes(name)
        if not genes:
            continue
        if found["unsupported_allele"] is None and all(clean(gene) for gene in genes) and not index._matches(name, "v"):
            found["unsupported_allele"] = _decoy(index, name, genes, UNSUPPORTED_ALLELE)
        if found["uncalled"] is None and set(genes) <= genotype.uncalled:
            found["uncalled"] = _decoy(index, name, genes, "excluded:uncalled")
    found["passed"] = all(entry["passed"] for entry in found.values() if entry is not None)
    return found


def _decoy(index: SupportIndex, name: str, genes: tuple[str, ...], expected: str) -> dict:
    cls = index.classify(name, "v")
    return {"call": name, "genes": list(genes), "class": cls, "expected": expected, "passed": cls == expected}


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
