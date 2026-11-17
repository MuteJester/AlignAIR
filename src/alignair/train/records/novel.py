"""Curriculum streams whose reads come from novel alleles: synthetic SNP variants of the reference's V alleles.

A stream's seeds are cut into epochs of ``NovelAlleles.epoch_records`` seeds. Epoch e (``epoch_alleles``) makes
``round(gene_fraction x eligible V genes)`` genes homozygous for variants drawn from e alone (a seeded generator keyed
by the generator version and e), so a record depends only on its own seed: record k of seed s is record k - a of seed
s + a, and a stream drawn in segments, or by another worker, joins the serial one. ``novel_records`` simulates each
epoch's records from GenAIRR under that genotype; a read whose V is a novel allele carries its definition under
``alignair.reference.variants.NOVEL_FIELD``. GenAIRR is imported only inside functions."""
from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

from ...errors import ConfigurationError
from ...reference.variants import NOVEL_FIELD, NovelAllele, eligible_bases
from .seeds import SEED_BLOCK_RECORDS

GENERATOR = 1


@dataclass(frozen=True)
class NovelAlleles:
    """How novel alleles enter a curriculum: the share of eligible V genes each epoch makes novel, the number of SNPs
    of a variant (``(low, high)``, uniform), and the epoch length in seeds (a power of two, so an epoch never
    straddles two seed blocks)."""
    gene_fraction: float = 0.4
    snps: tuple[int, int] = (1, 6)
    epoch_records: int = 4096

    def __post_init__(self):
        fraction = self.gene_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, Real) or not 0 < fraction <= 1:
            raise ConfigurationError(f"gene_fraction must be a number in (0, 1], got {fraction!r}",
                                     context={"field": "gene_fraction"})
        try:
            low, high = self.snps
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("snps must be (low, high)", context={"field": "snps"}) from exc
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in (low, high)) or not 1 <= low <= high <= 10:
            raise ConfigurationError(f"snps must be integers 1 <= low <= high <= 10, got {self.snps!r}",
                                     context={"field": "snps"})
        records = self.epoch_records
        if (isinstance(records, bool) or not isinstance(records, int) or records < 1
                or records > SEED_BLOCK_RECORDS or records & (records - 1)):
            raise ConfigurationError(f"epoch_records must be a power of two in [1, 2**32], got {records!r}",
                                     context={"field": "epoch_records"})
        object.__setattr__(self, "gene_fraction", float(fraction))
        object.__setattr__(self, "snps", (low, high))

    def to_dict(self) -> dict:
        """The settings as a plain dict, with the generator they draw with (``train_args`` records it)."""
        return {"generator": GENERATOR, "gene_fraction": self.gene_fraction, "snps": list(self.snps),
                "epoch_records": self.epoch_records}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> NovelAlleles:
        """From ``to_dict``'s form: a ConfigurationError for settings of another generator."""
        data = dict(data)
        if data.get("generator") != GENERATOR:
            raise ConfigurationError(f"novel alleles of generator {data.get('generator')!r}; this build draws "
                                     f"generator {GENERATOR}", context={"field": "novel_alleles"})
        return cls(data["gene_fraction"], tuple(data["snps"]), data["epoch_records"])


def epoch_alleles(reference, spec: NovelAlleles, epoch: int) -> tuple[NovelAllele, ...]:
    """Epoch ``epoch``'s novel alleles: one variant each of ``round(gene_fraction x eligible genes)`` eligible V genes
    (at least one), the genes and their bases and SNP counts drawn from the epoch alone; a gene whose variant fails
    passes to the next."""
    from ...genairr.novel import derive_variant
    rng = random.Random(json.dumps(["alignair.novel_alleles", GENERATOR, int(epoch)]))
    eligible = eligible_bases(reference, "V")
    genes = sorted(eligible)
    want = max(1, round(spec.gene_fraction * len(genes)))
    drawn = []
    for gene in rng.sample(genes, len(genes)):
        if len(drawn) == want:
            break
        base, k = rng.choice(eligible[gene]), rng.randint(*spec.snps)
        allele = derive_variant(reference, base, substitutions=k, rng=rng)
        if allele is not None:
            drawn.append(allele)
    return tuple(drawn)


def novel_records(dataconfig, parameters, seed: int, spec: NovelAlleles):
    """The records of the stream that starts at ``seed``, each epoch simulated from its novel alleles' genotype; a
    record's ``sequence_id`` counts from 0 across epochs, and a novel read carries its allele's definition."""
    from ...genairr.experiment import build_experiment
    from ...genairr.novel import novel_genotype
    from ...reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(dataconfig)
    size, k, x = spec.epoch_records, 0, int(seed)
    while True:
        epoch = x // size
        stop = (epoch + 1) * size
        alleles = epoch_alleles(reference, spec, epoch)
        by_name = {allele.name: allele for allele in alleles}
        experiment = build_experiment(dataconfig, {**parameters, "genotype": novel_genotype(dataconfig, alleles)},
                                      allow_curatable=True)
        for record in experiment.stream_records(n=stop - x, seed=x):
            record["sequence_id"] = f"seq{k}"
            k += 1
            if record.get("v_call") in by_name:
                record[NOVEL_FIELD] = {"V": by_name[record["v_call"]].to_dict()}
            yield record
        x = stop


def novel_support(dataconfig, spec: NovelAlleles) -> int:
    """The number of eligible V genes of ``dataconfig``; ConfigurationError naming it when there is none, or when
    GenAIRR refuses epoch 0's genotype (so a refusal shows before any training)."""
    from ...genairr import name_of
    from ...genairr.novel import novel_genotype
    from ...reference.reference_set import ReferenceSet
    reference = ReferenceSet.from_dataconfigs(dataconfig)
    genes = len(eligible_bases(reference, "V"))
    name = name_of(dataconfig) or str(getattr(dataconfig.metadata, "reference_set", "the dataconfig"))
    if genes == 0:
        raise ConfigurationError(f"{name} has no V allele a novel allele can be drawn from (its own anchor on a Cys "
                                 "codon, a gapped form that is its sequence, no stop codon before the anchor)",
                                 context={"field": "novel_alleles", "dataconfig": name})
    novel_genotype(dataconfig, epoch_alleles(reference, spec, 0))
    return genes
