"""Simulated donors: calibrated diploid genotypes that cohort strata generate reads from.

A donor carries one or two alleles of every reference gene except its homozygously deleted ones. ``sample_donors``
draws each donor independently: a Poisson(``mean_deleted_genes``) number of V/D genes is deleted, drawn uniformly
without replacement (J is never deleted, and every deletable segment keeps at least one gene); every remaining
gene with two or more reference alleles is heterozygous with probability ``het_fraction``; alleles are uniform
within a gene; optionally, some carried V genes are then made homozygous for a novel allele (``suites.novel``).
``Donor.genotype`` builds the GenAIRR ``Genotype`` that ``Experiment.run_cohort`` simulates the donor from;
``donors_to_json`` is the ``donors.json`` form.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from alignair_assay.core.genotype import DonorGenotype, NovelAllele
from alignair_assay.core.reference import Reference, gene_name
from alignair_assay.suites.novel import make_novel_allele

if TYPE_CHECKING:
    from GenAIRR import Genotype

DONORS_KIND = "alignair_assay.donors"
DONORS_VERSION = "1"
# Gene letters in recombination order; the GenAIRR segment of a letter is its upper case.
_LETTERS = ("v", "d", "j")
_DELETABLE = ("v", "d")


@dataclass(frozen=True)
class DonorParams:
    """How donors are drawn. ``het_fraction`` (0 to 1) is the probability that a gene with two or more reference
    alleles is heterozygous. ``mean_deleted_genes`` is the Poisson mean of homozygously deleted genes per donor,
    drawn from the genes of the ``deletable`` segments (``"v"``, ``"d"``; J is never deleted); the count is capped
    so every deletable segment keeps at least one gene. ``novel_v_genes`` carried V genes per donor are made
    homozygous for a novel allele (``suites.novel``), the i-th with ``novel_snp_cycle[i % len(cycle)]`` SNPs.
    ValueError for values outside these ranges."""

    het_fraction: float = 0.40
    mean_deleted_genes: float = 6.7
    deletable: tuple[str, ...] = ("v", "d")
    novel_v_genes: int = 0
    novel_snp_cycle: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        if not 0.0 <= self.het_fraction <= 1.0:
            raise ValueError(f"het_fraction must lie in [0, 1], got {self.het_fraction!r}")
        if not (math.isfinite(self.mean_deleted_genes) and self.mean_deleted_genes >= 0):
            raise ValueError(f"mean_deleted_genes must be a finite number >= 0, got {self.mean_deleted_genes!r}")
        if not set(self.deletable) <= set(_DELETABLE) or len(set(self.deletable)) != len(self.deletable):
            raise ValueError(f"deletable must be distinct segments among {_DELETABLE} (J is never deleted), "
                             f"got {self.deletable!r}")
        if isinstance(self.novel_v_genes, bool) or not isinstance(self.novel_v_genes, int) or self.novel_v_genes < 0:
            raise ValueError(f"novel_v_genes must be an integer >= 0, got {self.novel_v_genes!r}")
        if not self.novel_snp_cycle or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1
                                               for k in self.novel_snp_cycle):
            raise ValueError(f"novel_snp_cycle must be a non-empty sequence of SNP counts >= 1, "
                             f"got {self.novel_snp_cycle!r}")


@dataclass(frozen=True)
class Donor:
    """A simulated donor. ``alleles`` maps every carried gene to its alleles, sorted: one when homozygous, two when
    heterozygous. Homozygously ``deleted`` genes are absent from it. ``novel`` maps a novel allele's name to its
    definition; every novel allele is carried, in ``alleles`` under its base's gene. ValueError for a malformed
    donor, naming the gene or novel allele."""

    subject_id: str
    alleles: dict[str, tuple[str, ...]]
    deleted: frozenset[str]
    novel: dict[str, NovelAllele] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for gene, names in self.alleles.items():
            if len(names) not in (1, 2):
                raise ValueError(f"{self.subject_id}: gene {gene} must carry 1 or 2 alleles, got {names!r}")
            if list(names) != sorted(set(names)):
                raise ValueError(f"{self.subject_id}: alleles of {gene} must be distinct and sorted, got {names!r}")
            foreign = [name for name in names if gene_name(name) != gene]
            if foreign:
                raise ValueError(f"{self.subject_id}: {foreign} are not alleles of gene {gene}")
        both = sorted(set(self.deleted) & set(self.alleles))
        if both:
            raise ValueError(f"{self.subject_id}: deleted genes {both} also carry alleles")
        for name, novel in self.novel.items():
            if novel.name != name or gene_name(novel.base) != gene_name(name):
                raise ValueError(f"{self.subject_id}: novel allele {name!r} must be keyed by its own name and "
                                 "belong to its base allele's gene")
            if name not in self.alleles.get(gene_name(name), ()):
                raise ValueError(f"{self.subject_id}: novel allele {name} is not carried (gene {gene_name(name)} "
                                 f"carries {list(self.alleles.get(gene_name(name), ()))})")

    def view(self) -> DonorGenotype:
        """The plain-data genotype: every carried allele (novel ones included) and the deleted genes."""
        carried = frozenset(name for names in self.alleles.values() for name in names)
        return DonorGenotype(subject_id=self.subject_id, carried=carried, deleted=frozenset(self.deleted))

    def genotype(self, config) -> Genotype:
        """The donor as a strict GenAIRR ``Genotype`` of DataConfig ``config`` (the config its reference was built
        from), tagged with ``subject_id`` so cohort records carry it. Novel alleles are defined from their
        sequences, each carried gene is homozygous or heterozygous (the first sorted allele on haplotype 0), and
        each deleted gene is deleted on both haplotypes. ValueError for a gene ``config`` does not have; GenAIRR
        raises at run time for a gene of ``config`` that is neither carried nor deleted."""
        from GenAIRR import Genotype

        segments = _gene_segments(config)

        def segment(gene: str) -> str:
            if gene not in segments:
                raise ValueError(f"{self.subject_id}: gene {gene} is not in the dataconfig")
            return segments[gene]

        genotype = Genotype.from_dataconfig(config).with_subject(self.subject_id)
        for novel in self.novel.values():
            genotype.add_novel_allele(novel.name, base=novel.base, sequence=novel.sequence,
                                      segment=segment(gene_name(novel.base)))
        for gene, names in self.alleles.items():
            if len(names) == 1:
                genotype.homozygous(gene, names[0], segment=segment(gene))
            else:
                genotype.heterozygous(gene, names[0], names[1], segment=segment(gene))
        for gene in sorted(self.deleted):
            genotype.delete_gene(gene, haplotype="both", segment=segment(gene))
        return genotype

    def tool_genotype(self, reference: Reference) -> dict[str, list[str]]:
        """The genotype a tool is given: per gene letter of ``reference`` (``"v"``, ``"d"``, ``"j"``), the sorted
        reference alleles the donor carries, each novel allele replaced by its base. ValueError for a carried
        allele that is neither in ``reference`` nor novel, and for a novel allele that does not differ from its
        base's reference sequence exactly at its SNPs (``NovelAllele.check_base``)."""
        tool: dict[str, set[str]] = {letter: set() for letter in reference.alleles}
        for names in self.alleles.values():
            for name in names:
                allele = self.novel[name].base if name in self.novel else name
                if not reference.has(allele):
                    raise ValueError(f"{self.subject_id}: carried allele {allele!r} is not in the reference")
                if name in self.novel:
                    self.novel[name].check_base(reference.seq(allele))
                tool[reference.gene_letter(allele)].add(allele)
        return {letter: sorted(names) for letter, names in tool.items()}


@dataclass(frozen=True)
class DonorSpec:
    """A suite's donors: ``n`` donors drawn with ``params``. The count does not scale with ``--scale``. ``tag``
    names the donor set in its seed (``suites.engine``), so a new tag draws new donors; it may not contain ':'.
    ValueError for a count that is not a positive integer or a bad tag."""

    n: int
    tag: str
    params: DonorParams = DonorParams()

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"donors: n must be a positive integer, got {self.n!r}")
        if not self.tag or ":" in self.tag:
            raise ValueError(f"donors: tag must be non-empty and contain no ':', got {self.tag!r}")


def sample_donors(reference: Reference, n: int, seed: int, params: DonorParams = DonorParams(), *,
                  config=None) -> list[Donor]:
    """``n`` donors ``donor00``, ``donor01``, ... drawn from ``reference`` under ``params`` (see the module
    docstring). Each donor has its own random stream seeded by ``seed`` and its id, so a donor does not depend on
    ``n``: a smaller cohort is a prefix of a larger one.

    With ``params.novel_v_genes``, each donor's novel alleles come from a second stream of its own, after the rest
    of its genotype is drawn, so that genotype is the one drawn without novel alleles. The donor's carried
    (non-deleted) V genes are shuffled and taken in turn: the i-th novel allele, ``{gene}*{900 + i}``, derives
    from one of the gene's carried alleles (drawn uniformly) by ``novel_snp_cycle[i % len(cycle)]`` SNPs
    (``suites.novel.make_novel_allele``) and replaces the gene's alleles, homozygously. A gene whose base gives no
    accepted novel allele is passed over. ``config`` is the GenAIRR DataConfig ``reference`` was built from, which
    validates the novel alleles; ValueError without it when novel alleles are asked for, and RuntimeError when a
    donor has fewer eligible V genes than ``novel_v_genes``. ValueError when ``n`` is not a positive integer."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"donors: n must be a positive integer, got {n!r}")
    if params.novel_v_genes and config is None:
        raise ValueError("donors: novel alleles need the GenAIRR DataConfig (config=) that validates them")
    genes = _reference_genes(reference)
    pools = {letter: list(genes[letter]) for letter in params.deletable if letter in genes}
    donors = []
    for index in range(n):
        subject_id = f"donor{index:02d}"
        rng = random.Random(f"{seed}:{subject_id}")
        deleted = _sample_deletions(rng, pools, params.mean_deleted_genes)
        alleles = {}
        for table in genes.values():
            for gene, names in table.items():
                if gene in deleted:
                    continue
                if len(names) >= 2 and rng.random() < params.het_fraction:
                    alleles[gene] = tuple(sorted(rng.sample(names, 2)))
                else:
                    alleles[gene] = (rng.choice(names),)
        novel = {}
        if params.novel_v_genes:
            novel = _novel_v_alleles(random.Random(f"{seed}:{subject_id}:novel"), reference, alleles,
                                     list(genes.get("v", {})), params, config, subject_id)
        donors.append(Donor(subject_id=subject_id, alleles=alleles, deleted=deleted, novel=novel))
    return donors


def _novel_v_alleles(rng: random.Random, reference: Reference, alleles: dict[str, tuple[str, ...]],
                     v_genes: list[str], params: DonorParams, config, subject_id: str) -> dict[str, NovelAllele]:
    """Give ``params.novel_v_genes`` of the carried ``v_genes`` a homozygous novel allele, in place in ``alleles``
    (see ``sample_donors``); returns the novel alleles by name."""
    from GenAIRR import Genotype

    validator = Genotype.from_dataconfig(config)
    candidates = [gene for gene in v_genes if gene in alleles]
    rng.shuffle(candidates)
    novel: dict[str, NovelAllele] = {}
    for gene in candidates:
        if len(novel) == params.novel_v_genes:
            break
        index = len(novel)
        k = params.novel_snp_cycle[index % len(params.novel_snp_cycle)]
        base = rng.choice(alleles[gene])
        try:
            allele = make_novel_allele(reference, base, k, rng, f"{gene}*{900 + index}", genotype=validator)
        except (RuntimeError, ValueError):
            continue  # no functional variant (a base with a stop codon in its coding frame) or too few positions
        novel[allele.name] = allele
        alleles[gene] = (allele.name,)
    if len(novel) < params.novel_v_genes:
        raise RuntimeError(f"{subject_id}: only {len(novel)} of {len(candidates)} carried V genes take a novel "
                           f"allele, {params.novel_v_genes} requested")
    return novel


def donors_to_json(donors: Iterable[Donor]) -> dict:
    """The ``donors.json`` form: ``kind``, ``version`` and per donor its ``subject_id``, ``alleles`` (gene ->
    allele list), ``deleted`` (sorted gene names) and ``novel`` (the ``NovelAllele`` fields, sorted by name)."""
    return {
        "kind": DONORS_KIND,
        "version": DONORS_VERSION,
        "donors": [{
            "subject_id": donor.subject_id,
            "alleles": {gene: list(names) for gene, names in donor.alleles.items()},
            "deleted": sorted(donor.deleted),
            "novel": [{"name": novel.name, "base": novel.base, "sequence": novel.sequence,
                       "snps": list(novel.snps)} for novel in sorted(donor.novel.values(), key=lambda n: n.name)],
        } for donor in donors],
    }


def donors_from_json(data: dict) -> list[Donor]:
    """The donors of a ``donors_to_json`` object; ValueError for another ``kind`` or ``version`` or a malformed
    donor."""
    if data.get("kind") != DONORS_KIND:
        raise ValueError(f"not a donors object: kind {data.get('kind')!r}, expected {DONORS_KIND!r}")
    if data.get("version") != DONORS_VERSION:
        raise ValueError(f"unsupported donors version {data.get('version')!r}, expected {DONORS_VERSION!r}")
    return [Donor(subject_id=item["subject_id"],
                  alleles={gene: tuple(names) for gene, names in item["alleles"].items()},
                  deleted=frozenset(item["deleted"]),
                  novel={novel["name"]: NovelAllele(name=novel["name"], base=novel["base"],
                                                    sequence=novel["sequence"], snps=tuple(novel["snps"]))
                         for novel in item["novel"]})
            for item in data["donors"]]


def read_genotypes(path: str | Path) -> dict[str, DonorGenotype]:
    """The donor genotypes of a ``donors.json`` file (``donors_to_json``) by subject id: each donor's
    ``Donor.view()``. ValueError naming the file for one that is not JSON, not a donors object or a malformed
    one; OSError when it cannot be read."""
    try:
        donors = donors_from_json(json.loads(Path(path).read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{path} is not a valid donors file: {type(error).__name__}: {error}") from error
    return {donor.subject_id: donor.view() for donor in donors}


def donors_summary(donors: Sequence[Donor], reference: Reference) -> dict:
    """The realised donor statistics recorded in ``build.json``: ``donors`` (the count), ``mean_alleles_per_gene``
    per gene letter and over ``all`` letters (carried alleles per carried gene, pooled over donors),
    ``mean_deleted_genes`` per donor and ``heterozygous_share`` (0 to 1) of the carried genes with two or more
    reference alleles. Means are rounded to 4 decimals; a mean over nothing is None. ValueError without donors or
    for a gene that is not in ``reference``."""
    if not donors:
        raise ValueError("no donors to summarise")
    letters = {gene_name(name): letter for letter, table in reference.alleles.items() for name in table}
    allele_counts = Counter(gene_name(name) for table in reference.alleles.values() for name in table)
    per_letter: dict[str, list[int]] = {letter: [] for letter in reference.alleles}
    heterozygous = []
    for donor in donors:
        for gene, names in donor.alleles.items():
            if gene not in letters:
                raise ValueError(f"{donor.subject_id}: gene {gene} is not in the reference")
            per_letter[letters[gene]].append(len(names))
            if allele_counts[gene] >= 2:
                heterozygous.append(len(names) == 2)
    every = [count for counts in per_letter.values() for count in counts]
    return {
        "donors": len(donors),
        "mean_alleles_per_gene": {**{letter: _mean(counts) for letter, counts in per_letter.items()},
                                  "all": _mean(every)},
        "mean_deleted_genes": _mean([len(donor.deleted) for donor in donors]),
        "heterozygous_share": _mean(heterozygous),
    }


def _reference_genes(reference: Reference) -> dict[str, dict[str, list[str]]]:
    """Gene letter (recombination order) -> gene (sorted) -> its reference alleles (sorted)."""
    genes: dict[str, dict[str, list[str]]] = {}
    for letter in _LETTERS:
        if letter not in reference.alleles:
            continue
        table: dict[str, list[str]] = {}
        for name in sorted(reference.alleles[letter]):
            table.setdefault(gene_name(name), []).append(name)
        genes[letter] = dict(sorted(table.items()))
    return genes


def _sample_deletions(rng: random.Random, pools: dict[str, list[str]], mean: float) -> frozenset[str]:
    """A Poisson(``mean``) number of genes drawn uniformly without replacement from the ``pools`` (gene letter ->
    deletable genes), capped and redrawn so every pool keeps at least one gene."""
    pool = [gene for genes in pools.values() for gene in genes]
    count = _poisson(rng, mean, cap=len(pool) - len(pools))
    while True:
        deleted = frozenset(rng.sample(pool, count))
        if all(set(genes) - deleted for genes in pools.values()):
            return deleted


def _poisson(rng: random.Random, mean: float, *, cap: int) -> int:
    """A Poisson(``mean``) draw by inversion of one uniform, at most ``cap``."""
    uniform = rng.random()
    count, term = 0, math.exp(-mean)
    cumulative = term
    while uniform > cumulative and count < cap:
        count += 1
        term *= mean / count
        cumulative += term
    return count


def _gene_segments(config) -> dict[str, str]:
    """GenAIRR segment (``"V"``, ``"D"``, ``"J"``) of every gene of DataConfig ``config``."""
    tables = {"V": config.v_alleles, "D": config.d_alleles, "J": config.j_alleles}
    return {gene: segment for segment, table in tables.items() for gene in (table or {})}


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None
