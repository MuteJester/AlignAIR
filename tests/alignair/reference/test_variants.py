"""Synthetic novel alleles for training banks (``alignair.reference.variants``): SNP variants of a reference's V
alleles, drawn from a seeded generator, that keep their base's anchor codon and frame, equal no reference allele and
have their base as their one nearest allele; ``annotate`` gives each its base's anchor and gap columns, as inference
would."""
import random

import GenAIRR.data as gd
import pytest

from alignair.reference import annotate
from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair.reference.codons import STOP_CODONS
from alignair.reference.variants import NovelAllele, eligible_bases, novel_name, snp_variant

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def igh():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def _draws(reference, seed=11, n=300):
    rng = random.Random(seed)
    alleles = sorted(a for names in eligible_bases(reference).values() for a in names)
    out = []
    for _ in range(n):
        base, k = rng.choice(alleles), rng.randint(1, 6)
        out.append((base, k, snp_variant(reference, base, k, rng)))
    return out


def _stop_before(sequence, anchor):
    return any(sequence[i:i + 3] in STOP_CODONS for i in range(0, anchor - 2, 3))


def test_a_dataconfigs_eligible_bases(igh):
    e = eligible_bases(igh)
    assert len(e) == 51 and sum(map(len, e.values())) == 192
    absent = {"IGHVF2-G4*02", "IGHVF7-G27*02", "IGHVF9-G32*02", "IGHVF9-G32*03", "IGHVF9-G32*04", "IGHVF10-G45*01"}
    assert not absent & {a for names in e.values() for a in names} and "IGHVF10-G45" not in e
    igk = eligible_bases(ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB))
    assert len(igk) == 31 and sum(map(len, igk.values())) == 167
    trb = eligible_bases(ReferenceSet.from_dataconfigs(gd.HUMAN_TCRB_IMGT))
    assert len(trb) == 57 and sum(map(len, trb.values())) == 94
    assert not {"TRBV17*01", "TRBV17*02", "TRBV7-3*02", "TRBV7-3*03"} & {a for names in trb.values() for a in names}
    assert eligible_bases(ReferenceSet.from_dataconfigs(gd.MOUSE_TCRA_IMGT)) == {}


def test_a_snp_variant_keeps_its_bases_anchor_codon_frame_and_nearest_allele(igh):
    v = igh.gene("V")
    draws = _draws(igh)
    assert sum(variant is None for *_, variant in draws) <= 3
    known = set(v.sequences)
    for base, k, variant in draws:
        if variant is None:
            continue
        sequence, anchor = v.sequences[v.index[base]], v.anchors[base]
        assert len(variant.sequence) == len(sequence) and variant.base == base and variant.gene == "V"
        changed = [i for i, (a, b) in enumerate(zip(sequence, variant.sequence)) if a != b]
        assert len(changed) == k and not any(anchor <= i < anchor + 3 for i in changed)
        assert not _stop_before(variant.sequence, anchor) and variant.sequence not in known
        for name, other in zip(v.names, v.sequences):
            if name != base and len(other) == len(sequence):
                assert sum(a != b for a, b in zip(other, variant.sequence)) > k
        assert variant.name == novel_name(base, variant.sequence)


def test_a_variant_is_a_function_of_its_rng(igh):
    base = sorted(a for names in eligible_bases(igh).values() for a in names)[0]
    assert snp_variant(igh, base, 3, random.Random(5)) == snp_variant(igh, base, 3, random.Random(5))
    assert snp_variant(igh, base, 3, random.Random(5)) != snp_variant(igh, base, 3, random.Random(6))


def test_a_base_whose_every_variant_is_known_gives_none():
    base = "TGTAAA"
    variants = [base[:i] + b + base[i + 1:] for i in range(3, 6) for b in "ACGT" if b != base[i]]
    names = ["V1*01", *[f"V1*{i + 2:02d}" for i in range(len(variants))]]
    v = GeneReference(names, [base, *variants], {n: i for i, n in enumerate(names)}, anchors={"V1*01": 0},
                      gapped={n: s for n, s in zip(names, [base, *variants])})
    j = GeneReference(["J1*01"], ["TTTT"], {"J1*01": 0})
    reference = ReferenceSet({"V": v, "J": j}, False)
    reference.gene("V").info = None
    assert snp_variant(reference, "V1*01", 1, random.Random(0)) is None


def test_a_novel_alleles_record_form_round_trips():
    allele = NovelAllele("IGHVF1-G1*01_nv0123456789ab", "IGHVF1-G1*01", "ACGT")
    assert NovelAllele.from_dict(allele.to_dict()) == allele
    with pytest.raises(ValueError):
        NovelAllele.from_dict({"name": "x", "base": "y", "sequence": "ACGT", "gene": "Q"})
    with pytest.raises(ValueError):
        NovelAllele.from_dict({"name": "x", "base": "y"})


def test_annotate_gives_every_variant_its_bases_anchor_and_gap_columns(igh):
    v = igh.gene("V")
    variants = [variant for *_, variant in _draws(igh) if variant is not None]
    genotype = ReferenceSet.from_alleles({"V": {a.name: a.sequence for a in variants}, "J": {}})

    def columns(gapped):
        return [i for i, c in enumerate(gapped) if c == "."]
    for known, check_source in ((igh, True),
                                (ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGH_EXTENDED), False)):
        done = annotate(genotype, known).gene("V")
        for allele in variants:
            assert done.anchors[allele.name] == v.anchors[allele.base]
            assert columns(done.gapped[allele.name]) == columns(v.gapped[allele.base])
            evidence = done.info[allele.name].anchor_evidence
            if check_source:
                assert evidence.method == "nearest_known_allele" and evidence.source == allele.base
