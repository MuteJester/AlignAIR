"""Genotype: diploid-realistic genotype simulator + recovery vs the intended genotype."""
from collections import defaultdict

import GenAIRR.data as gd
import pytest

from alignair.genotype.benchmark import recovery, sample_diploid_genotype
from alignair.genotype.infer import _gene_of
from alignair.reference.reference_set import ReferenceSet


def _ref():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def _per_gene_counts(genotype, gtype):
    c = defaultdict(int)
    for a in genotype[gtype]:
        c[_gene_of(a)] += 1
    return c


def test_default_genotype_is_diploid_at_most_two_per_gene():
    genotype, meta = sample_diploid_genotype(_ref(), seed=0)
    for gtype in ("v", "d", "j"):
        assert all(c <= 2 for c in _per_gene_counts(genotype, gtype).values())
    assert set(meta.values()) <= {"homozygous", "heterozygous"}          # no del/dup by default
    assert "heterozygous" in set(meta.values()) and "homozygous" in set(meta.values())


def test_deletion_knob_produces_empty_genes_marked_deleted():
    genotype, meta = sample_diploid_genotype(_ref(), seed=1, deletion_fraction=0.5)
    deleted = {g for g, z in meta.items() if z == "deleted"}
    assert deleted                                                        # some genes deleted
    counts = _per_gene_counts(genotype, "v")
    assert all(counts.get(g, 0) == 0 for g in deleted)                    # deleted genes contribute no alleles


def test_duplication_knob_produces_three_allele_genes():
    genotype, meta = sample_diploid_genotype(_ref(), seed=2, duplication_fraction=0.6)
    dup = {g for g, z in meta.items() if z == "duplication"}
    assert dup
    counts = _per_gene_counts(genotype, "v")
    counts.update(_per_gene_counts(genotype, "d"))
    counts.update(_per_gene_counts(genotype, "j"))
    assert all(counts[g] == 3 for g in dup)                              # duplication = 3 alleles


def test_recovery_uses_truth_genotype_for_zygosity_and_deletion():
    gs = {"genotype_class_list": [{"documented_alleles": [
        {"label": "IGHV1-2*01"}, {"label": "IGHV1-2*02"}, {"label": "IGHV3-23*01"}],
        "deleted_genes": [{"label": "IGHV7-81"}]}]}
    truth_meta = {"IGHV1-2": "heterozygous", "IGHV3-23": "homozygous", "IGHV7-81": "deleted"}
    truth_records = [{"v_call": "IGHV1-2*01"}, {"v_call": "IGHV1-2*02"}, {"v_call": "IGHV3-23*01"}]
    r = recovery(gs, truth_records, truth_genotype=truth_meta)
    assert r["zygosity_acc"] == 1.0                                      # IGHV1-2 het (2), IGHV3-23 homo (1)
    assert r["deletion_recall"] == 1.0                                   # IGHV7-81 deletion detected
