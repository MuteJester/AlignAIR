import pytest

from alignair_assay.reference import Reference, gene_name
from alignair_assay.truth.alleles import observable_alleles
from alignair_assay.truth.segment import ObservedSegment, observe_segment

ga = pytest.importorskip("GenAIRR")

REF = Reference.from_dataconfig("HUMAN_IGH_OGRDB")


def _records(n, seed, e5=None):
    exp = ga.Experiment.on("human_igh").recombine()
    if e5 is not None:
        exp = exp.end_loss_5prime(length=(e5, e5))
    return exp.run_records(n=n, seed=seed)


def _v_truth(rec):
    calls = rec["v_call"].split(",")
    seg = observe_segment(
        rec["sequence"].upper(), REF.seq(calls[0]), calls[0],
        rec["v_sequence_start"], rec["v_germline_start"], rec["v_cigar"],
    )
    if seg is None:
        return None, None
    return seg, observable_alleles(seg, REF, "v", calls)


def test_full_length_unmutated_is_unique_or_identical():
    checked = 0
    for rec in _records(60, 5):
        seg, truth = _v_truth(rec)
        if seg is None:
            continue
        checked += 1
        assert truth.primary == seg.allele
        assert seg.allele in truth.alleles
        full = REF.seq(seg.allele)
        for a in truth.alleles:
            if not REF.has(a):
                continue
            assert seg.observed_germline in REF.seq(a)
            if a != seg.allele and seg.germ_start == 0 and seg.germ_end == len(full):
                assert REF.seq(a) == full
    assert checked > 30


def test_set_grows_with_clipping():
    means = []
    for clip in (0, 200, 260):
        sizes = []
        for rec in _records(100, 9, e5=clip):
            _, truth = _v_truth(rec)
            if truth is not None:
                sizes.append(len(truth.alleles))
        assert sizes
        means.append(sum(sizes) / len(sizes))
    assert means[0] <= means[1] <= means[2]
    assert means[2] >= 3


def test_short_d_matches_many():
    d_alleles = REF.alleles["d"]
    names = list(d_alleles)
    kmer = None
    for name in names:
        s = d_alleles[name]
        for i in range(len(s) - 4):
            k = s[i:i + 5]
            if sum(k in other for other in d_alleles.values()) >= 2:
                kmer, owner = k, name
                break
        if kmer:
            break
    assert kmer is not None
    start = d_alleles[owner].index(kmer)
    seg = ObservedSegment(owner, 0, 5, start, start + 5, "5M", 0, 5, kmer)
    truth = observable_alleles(seg, REF, "d", [owner])
    assert len(truth.alleles) >= 2
    assert owner in truth.alleles


def test_unknown_simulator_call_flags_disagreement_without_crashing():
    name = next(iter(REF.alleles["j"]))
    seq = REF.seq(name)
    seg = ObservedSegment(name, 0, len(seq), 0, len(seq), f"{len(seq)}M", 0, len(seq), seq)
    truth = observable_alleles(seg, REF, "j", [name, "IGHJ99*01"])
    assert truth.disagrees_with_simulator
    assert "IGHJ99*01" not in truth.alleles
    assert truth.primary == name
    assert truth.genes == frozenset(gene_name(a) for a in truth.alleles)
