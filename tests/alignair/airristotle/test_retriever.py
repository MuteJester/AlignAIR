"""Coarse k-mer filter: ranks the true/similar germline high; force_include works."""
import random

from alignair.airristotle.retriever import KmerFilter


def _rand_dna(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def test_exact_query_ranks_its_candidate_first():
    cands = [_rand_dna(300, s) for s in range(20)]
    f = KmerFilter(k=11).fit(cands)
    for true in (0, 7, 19):
        assert f.shortlist(cands[true], k=5)[0] == true


def test_snp_perturbed_query_still_shortlists_the_true_allele():
    cands = [_rand_dna(300, s) for s in range(30)]
    f = KmerFilter(k=11).fit(cands)
    true = 12
    q = list(cands[true])
    for p in (30, 90, 150, 210):                    # 4 SNPs
        q[p] = "A" if q[p] != "A" else "C"
    assert true in f.shortlist("".join(q), k=8)


def test_force_include_guarantees_the_positive():
    cands = [_rand_dna(300, s) for s in range(30)]
    f = KmerFilter(k=11).fit(cands)
    unrelated = _rand_dna(300, 999)                 # matches nothing well
    idx = f.shortlist(unrelated, k=5, force_include=17)
    assert 17 in idx and len(idx) == 5
