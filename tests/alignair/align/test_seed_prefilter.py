from alignair.align.seed_prefilter import SeedPrefilter


class _Gene:
    def __init__(self, names, seqs):
        self.names, self.sequences = names, seqs


class _Ref:
    def __init__(self):
        self._g = {"V": _Gene(
            ["A*01", "A*02", "FAR*01"],
            ["ACGTACGTACGTACGTACGT",         # idx0
             "ACGTACGTACGTACGTACGA",         # idx1: 1 SNP from idx0
             "TTTTTGGGGGCCCCCAAAAA"])}        # idx2: unrelated
    def gene(self, g):
        return self._g[g]


def test_near_neighbor_segment_ranks_related_alleles_first():
    sp = SeedPrefilter(_Ref(), k=5)
    cand = sp.candidates("ACGTACGTACGTACGTACGT", "V", m=2)
    assert set(cand) == {0, 1}                    # the two related alleles, not the far one


def test_divergent_segment_admits_the_far_allele():
    # a read of the FAR allele must be admittable by raw k-mer match even though it shares
    # nothing with alleles 0/1 — this is the swap-robustness guarantee.
    sp = SeedPrefilter(_Ref(), k=5)
    cand = sp.candidates("TTTTTGGGGGCCCCCAAAAA", "V", m=1)
    assert cand == [2]


def test_allowed_restricts_to_genotype():
    sp = SeedPrefilter(_Ref(), k=5)
    cand = sp.candidates("ACGTACGTACGTACGTACGT", "V", m=3, allowed={1})
    assert cand == [1]


def test_segment_shorter_than_k_returns_empty():
    sp = SeedPrefilter(_Ref(), k=5)
    assert sp.candidates("AC", "V", m=3) == []
