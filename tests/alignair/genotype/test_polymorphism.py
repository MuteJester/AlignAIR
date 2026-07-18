"""Genotype task 3: CIGAR->germline observation + SHM-vs-polymorphism resolution."""
from alignair.genotype.observe import germline_observations
from alignair.genotype.polymorphism import polymorphism_profile, resolve


class _Gene:
    def __init__(self, names, seqs):
        self.names = list(names)
        self.sequences = list(seqs)
        self.index = {n: i for i, n in enumerate(names)}


class _Ref:
    def __init__(self, gene):
        self._g = gene

    def gene(self, name):
        return self._g


A = "ACGTACGT"
B = "ACGAACGA"                                      # differs from A at pos 3 (T->A) and 7 (T->A)
REF = _Ref(_Gene(["A", "B"], [A, B]))


def test_germline_observations_walks_cigar():
    obs = germline_observations("ACGAACGT", "8M", 0, 0)
    assert obs[3] == "A"                            # read base aligned to germline pos 3


def test_germline_observations_handles_deletion():
    obs = germline_observations("ACGACGT", "3M1D4M", 0, 0)   # germline pos 3 deleted in the read
    assert obs[3] is None and obs[4] == "A"


def test_systematic_snp_flagged_random_shm_not():
    reads = [("ACGAACGT", "8M", 0, 0) for _ in range(10)]    # ALL mutate germline pos 3 (T->A)
    reads[0] = ("AAGAACGT", "8M", 0, 0)                       # one read also mutates pos 1 (random SHM)
    prof = polymorphism_profile(reads, A)
    assert prof[3]["mismatch_fraction"] > 0.9 and prof[3]["alt"] == "A"   # systematic polymorphism
    assert prof[1]["mismatch_fraction"] < 0.2                             # random SHM, not flagged


def test_reassign_when_diagnostic_sites_covered():
    prof = polymorphism_profile([("ACGAACGA", "8M", 0, 0) for _ in range(10)], A)   # matches B at 3 AND 7
    r = resolve(prof, "A", REF, "v")
    assert r["call"] == "reassign" and r["allele"] == "B"


def test_compatible_when_a_diagnostic_site_is_uncovered():
    prof = polymorphism_profile([("ACGAAC", "6M", 0, 0) for _ in range(10)], A)     # covers 0-5 only
    r = resolve(prof, "A", REF, "v")
    assert r["call"] == "compatible_with" and r["allele"] == "B"          # pos 7 uncovered -> not reassign


def test_novel_candidate_with_provisional_sequence_and_mask():
    prof = polymorphism_profile([("ACGTATGT", "8M", 0, 0) for _ in range(10)], A)   # pos 5 C->T, no known allele
    r = resolve(prof, "A", REF, "v")
    assert r["call"] == "novel" and r["positions"] == [5] and r["sequence"] == "ACGTATGT"
    assert r["source_mask"][5] == "observed"
