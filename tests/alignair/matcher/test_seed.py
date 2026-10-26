
import pytest
from alignair.matcher.seed import kmers, KmerIndex

pytestmark = pytest.mark.component


def test_kmers_basic():
    assert kmers("ACGTA", 3) == {"ACG", "CGT", "GTA"}
    assert kmers("AC", 3) == set()        # shorter than k


def test_shortlist_recalls_true_source_over_unrelated():
    seqs = ["AAAACCCCGGGGTTTTAAAA",   # 0: true source
            "TTTTGGGGCCCCAAAATTTT",   # 1: reverse-ish, unrelated k-mers
            "ACGTACGTACGTACGTACGT"]   # 2: unrelated
    idx = KmerIndex(seqs, k=6)
    read = "AAAACCCCGGtGTTTTAAAA".upper()   # candidate 0 with one substitution
    sl, counts = idx.shortlist(read, topk=2)
    assert sl[0] == 0                          # true source ranks first
    assert counts[0] > counts[1] and counts[0] > counts[2]


def test_best_hits_signals_orientation():
    seqs = ["AAAACCCCGGGGTTTT"]
    idx = KmerIndex(seqs, k=6)
    fwd = idx.best_hits("AAAACCCCGGGGTTTT")
    rc = idx.best_hits("TTTTAAAACCCCGGGG")     # scrambled -> fewer shared k-mers
    assert fwd > rc


def test_anchor_offsets_reports_read_minus_candidate_start():
    seqs = ["AAAACCCCGGGGTTTT", "GGGGAAAACCCC"]
    idx = KmerIndex(seqs, k=4)
    read = "NNNNAAAACCCCGGGGTTTT"
    assert idx.anchor_offsets(read, [0, 1])[0] == 4
