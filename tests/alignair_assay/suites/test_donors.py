import json
import statistics
from collections import Counter
from dataclasses import replace

import pytest

pytest.importorskip("GenAIRR")

from GenAIRR import Experiment, Genotype  # noqa: E402

from alignair_assay.core.genotype import DonorGenotype, NovelAllele  # noqa: E402
from alignair_assay.core.reference import dataconfig, gene_name  # noqa: E402
from alignair_assay.suites.donors import (  # noqa: E402
    Donor,
    DonorParams,
    DonorSpec,
    donors_from_json,
    donors_summary,
    donors_to_json,
    sample_donors,
)

pytestmark = pytest.mark.integration

REFERENCE = "HUMAN_IGH_OGRDB"


def _genes(reference, letter):
    """gene -> sorted reference alleles of segment ``letter``."""
    table = {}
    for name in sorted(reference.alleles[letter]):
        table.setdefault(gene_name(name), []).append(name)
    return table


def _all_genes(reference):
    return {gene for letter in reference.alleles for gene in _genes(reference, letter)}


def _novel_variant(cfg, base, name):
    """A functional one-SNP variant of reference allele ``base`` that GenAIRR accepts, as a ``NovelAllele``."""
    sequence = next(a for alleles in cfg.v_alleles.values() for a in alleles if a.name == base).ungapped_seq.upper()
    for position in range(30, len(sequence)):
        for base_letter in "ACGT":
            if base_letter == sequence[position]:
                continue
            variant = sequence[:position] + base_letter + sequence[position + 1:]
            try:
                Genotype.from_dataconfig(cfg).add_novel_allele(name, base=base, sequence=variant, segment="V")
            except ValueError:
                continue
            return NovelAllele(name=name, base=base, sequence=variant, snps=(position,))
    raise AssertionError(f"no functional variant of {base}")


def test_calibration(reference):
    donors = sample_donors(reference, 300, seed=20261019)
    assert [d.subject_id for d in donors] == [f"donor{i:02d}" for i in range(300)]

    deleted = [len(d.deleted) for d in donors]
    assert 6.3 <= statistics.mean(deleted) <= 7.1
    # Poisson-distributed counts: the variance tracks the mean (a fixed count would have none).
    assert 5.0 <= statistics.variance(deleted) <= 8.5

    multi = {gene for letter in reference.alleles for gene, alleles in _genes(reference, letter).items()
             if len(alleles) >= 2}
    heterozygous = [len(d.alleles[gene]) == 2 for d in donors for gene in multi if gene in d.alleles]
    assert 0.37 <= statistics.mean(heterozygous) <= 0.43
    single = _all_genes(reference) - multi
    assert all(len(d.alleles[gene]) == 1 for d in donors for gene in single if gene in d.alleles)

    # Uniform draws: every allele of every gene is carried, and every V/D gene deleted, by some donor.
    carried = {allele for d in donors for alleles in d.alleles.values() for allele in alleles}
    assert carried == {name for table in reference.alleles.values() for name in table}
    deletable = set(_genes(reference, "v")) | set(_genes(reference, "d"))
    assert set().union(*(d.deleted for d in donors)) == deletable

    summary = donors_summary(donors, reference)
    assert summary["donors"] == 300
    assert summary["mean_deleted_genes"] == pytest.approx(statistics.mean(deleted), abs=1e-4)
    assert summary["heterozygous_share"] == pytest.approx(statistics.mean(heterozygous), abs=1e-4)
    alleles_per_gene = summary["mean_alleles_per_gene"]
    assert set(alleles_per_gene) == {"v", "d", "j", "all"}
    for letter in "vdj":
        genes = set(_genes(reference, letter))
        counts = [len(d.alleles[gene]) for d in donors for gene in genes if gene in d.alleles]
        assert alleles_per_gene[letter] == pytest.approx(statistics.mean(counts), abs=1e-4)
    counts = [len(alleles) for d in donors for alleles in d.alleles.values()]
    assert alleles_per_gene["all"] == pytest.approx(statistics.mean(counts), abs=1e-4)
    assert 1.0 < alleles_per_gene["all"] < 1.5


def test_no_j_deleted_and_alleles_in_reference(reference):
    genes = _all_genes(reference)
    j_genes = set(_genes(reference, "j"))
    for donor in sample_donors(reference, 50, seed=2):
        assert not donor.deleted & j_genes
        assert not donor.deleted & set(donor.alleles) and set(donor.alleles) | donor.deleted == genes
        for gene, alleles in donor.alleles.items():
            assert len(alleles) in (1, 2) and alleles == tuple(sorted(set(alleles)))
            assert all(reference.has(a) and gene_name(a) == gene for a in alleles)

    v_only = sample_donors(reference, 30, seed=2, params=DonorParams(deletable=("v",)))
    assert all(d.deleted and d.deleted <= set(_genes(reference, "v")) for d in v_only)
    plain = sample_donors(reference, 5, seed=2, params=DonorParams(het_fraction=0.0, mean_deleted_genes=0.0))
    assert all(not d.deleted and all(len(a) == 1 for a in d.alleles.values()) for d in plain)
    everything = sample_donors(reference, 5, seed=2, params=DonorParams(het_fraction=1.0))
    multi = {g for letter in reference.alleles for g, a in _genes(reference, letter).items() if len(a) >= 2}
    assert all(len(d.alleles[g]) == 2 for d in everything for g in multi if g in d.alleles)
    # A deletion count beyond the deletable genes still leaves every segment a gene to recombine.
    extreme = sample_donors(reference, 5, seed=2, params=DonorParams(mean_deleted_genes=1000.0))
    for donor in extreme:
        assert len(donor.deleted) == len(set(_genes(reference, "v")) | set(_genes(reference, "d"))) - 2
        for letter in "vdj":
            assert set(_genes(reference, letter)) - donor.deleted


@pytest.mark.parametrize("options, message", [
    (dict(het_fraction=1.5), "het_fraction"),
    (dict(het_fraction=float("nan")), "het_fraction"),
    (dict(mean_deleted_genes=-1.0), "mean_deleted_genes"),
    (dict(mean_deleted_genes=float("inf")), "mean_deleted_genes"),
    (dict(deletable=("v", "j")), "deletable"),
])
def test_invalid_params(options, message):
    with pytest.raises(ValueError, match=message):
        DonorParams(**options)


def test_invalid_spec_and_count(reference):
    with pytest.raises(ValueError, match="n must be"):
        DonorSpec(n=0, tag="S1")
    with pytest.raises(ValueError, match="tag"):
        DonorSpec(n=2, tag="a:b")
    with pytest.raises(ValueError, match="n must be"):
        sample_donors(reference, 0, seed=1)


def test_invalid_donor():
    with pytest.raises(ValueError, match="1 or 2"):
        Donor("d", {"IGHJ1": ("IGHJ1*01", "IGHJ1*02", "IGHJ1*03")}, frozenset())
    with pytest.raises(ValueError, match="sorted"):
        Donor("d", {"IGHJ6": ("IGHJ6*03", "IGHJ6*02")}, frozenset())
    with pytest.raises(ValueError, match="gene"):
        Donor("d", {"IGHJ6": ("IGHJ4*02",)}, frozenset())
    with pytest.raises(ValueError, match="deleted"):
        Donor("d", {"IGHJ6": ("IGHJ6*02",)}, frozenset({"IGHJ6"}))
    with pytest.raises(ValueError, match="novel"):
        Donor("d", {"IGHJ6": ("IGHJ6*02",)}, frozenset(),
              novel={"IGHJ6*90": NovelAllele("IGHJ6*91", "IGHJ6*02", "ACGT", (0,))})


def test_genotype_round_trip(reference):
    cfg = dataconfig(REFERENCE)
    donors = sample_donors(reference, 3, seed=3)
    assert any(len(a) == 2 for a in donors[0].alleles.values()) and all(d.deleted for d in donors)
    for donor in donors:
        genotype = donor.genotype(cfg)
        assert genotype.subject_id == donor.subject_id
        zygosity = {row["gene"]: row["zygosity"] for row in genotype.to_table()}
        for letter in reference.alleles:
            for gene in _genes(reference, letter):
                carried = genotype.carried_alleles(letter.upper(), gene)
                if gene in donor.deleted:
                    assert carried == set() and zygosity[gene] == "deleted"
                else:
                    assert carried == set(donor.alleles[gene])
                    assert zygosity[gene] == ("heterozygous" if len(donor.alleles[gene]) == 2 else "homozygous")


def test_cohort_respects_genotype(reference):
    cfg = dataconfig(REFERENCE)
    donors = sample_donors(reference, 2, seed=4)
    records = (Experiment.on(cfg).recombine().mutate(rate=0.03)
               .run_cohort([d.genotype(cfg) for d in donors], n_per_subject=200, seed=5,
                           expose_provenance=True).records)
    assert Counter(r["subject_id"] for r in records) == {"donor00": 200, "donor01": 200}
    views = {d.subject_id: d.view() for d in donors}
    for record in records:
        view = views[record["subject_id"]]
        # The sampled (first-choice) allele of every segment is one the donor carries, never a deleted gene's.
        for letter in "vdj":
            sampled = record[f"truth_{letter}_call"]
            if sampled:
                assert sampled in view.carried and gene_name(sampled) not in view.deleted
    assert {r["haplotype"] for r in records} == {0, 1}


def test_json_round_trip(reference):
    cfg = dataconfig(REFERENCE)
    donors = sample_donors(reference, 4, seed=6)
    gene, base = next((g, a[0]) for g, a in donors[0].alleles.items() if g.startswith("IGHV") and len(a) == 1)
    novel = _novel_variant(cfg, base, f"{base}_novel")
    donors[0] = replace(donors[0], alleles={**donors[0].alleles, gene: (base, novel.name)},
                        novel={novel.name: novel})

    data = donors_to_json(donors)
    assert data["kind"] == "alignair_assay.donors" and data["version"] == "1"
    assert [d["subject_id"] for d in data["donors"]] == [f"donor{i:02d}" for i in range(4)]
    text = json.dumps(data, indent=2)
    assert donors_from_json(json.loads(text)) == donors
    assert json.dumps(donors_to_json(donors_from_json(json.loads(text))), indent=2) == text
    with pytest.raises(ValueError, match="kind"):
        donors_from_json({**data, "kind": "something.else"})
    with pytest.raises(ValueError, match="version"):
        donors_from_json({**data, "version": "0"})


def test_view_and_tool_genotype(reference):
    cfg = dataconfig(REFERENCE)
    donor = sample_donors(reference, 1, seed=7)[0]
    gene, base = next((g, a[0]) for g, a in donor.alleles.items() if g.startswith("IGHV") and len(a) == 1)
    novel = _novel_variant(cfg, base, f"{base}_novel")
    donor = replace(donor, alleles={**donor.alleles, gene: (base, novel.name)}, novel={novel.name: novel})

    view = donor.view()
    assert isinstance(view, DonorGenotype) and view.subject_id == donor.subject_id
    assert view.carried == {a for alleles in donor.alleles.values() for a in alleles} and novel.name in view.carried
    assert view.deleted == donor.deleted

    tool = donor.tool_genotype(reference)
    assert set(tool) == {"v", "d", "j"}
    for letter, names in tool.items():
        assert names == sorted(set(names)) and all(reference.has(n) for n in names)
        expected = {a for g, alleles in donor.alleles.items() for a in alleles
                    if a != novel.name and reference.gene_letter(a) == letter}
        assert set(names) == expected | ({base} if letter == "v" else set())
    assert novel.name not in tool["v"]

    genotype = donor.genotype(cfg)
    assert genotype.carried_alleles("V", gene) == {base, novel.name}
    assert genotype.novel_allele_names() == {novel.name}

    unknown = replace(donor, alleles={**donor.alleles, gene: (base, f"{gene}*99")}, novel={})
    with pytest.raises(ValueError, match=r"\*99"):
        unknown.tool_genotype(reference)


def test_deterministic(reference):
    a = sample_donors(reference, 5, seed=8)
    assert a == sample_donors(reference, 5, seed=8)
    # A donor depends only on the seed and its index, so a smaller cohort is a prefix of a larger one.
    assert sample_donors(reference, 3, seed=8) == a[:3]
    assert sample_donors(reference, 5, seed=9) != a
    assert sample_donors(reference, 5, seed=8, params=DonorParams(het_fraction=0.9)) != a
    assert donors_summary(a, reference) == donors_summary(sample_donors(reference, 5, seed=8), reference)
