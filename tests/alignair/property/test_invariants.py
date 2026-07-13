"""P0-16 layer 2 — property/invariant tests over randomized inputs: orientation involution,
deterministic prob-sorted selection, and the CIGAR query-consumption bound."""
import numpy as np

from alignair.io.airr import _cigar
from alignair.io.airr_validate import cigar_query_length
from alignair.predict.pipeline import _canonicalize
from alignair.predict.threshold import select_alleles

_BASES = list("ACGTN")


def test_canonicalize_is_an_involution():
    """Each orientation transform (identity/revcomp/complement/reverse) is its own inverse, so
    re-applying the predicted orientation recovers the read — the property the pipeline relies on."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        seq = "".join(rng.choice(_BASES, size=int(rng.integers(1, 80))))
        for oid in (0, 1, 2, 3):
            assert _canonicalize(_canonicalize(seq, oid), oid) == seq


def test_select_alleles_is_prob_sorted_and_deterministic():
    rng = np.random.default_rng(1)
    names = {"v": [f"V{i}" for i in range(20)]}
    probs = rng.random((8, 20))
    calls = select_alleles({"v": probs}, names, cap=3)
    for c in calls["v"]:
        lk = list(c.likelihoods)
        assert lk == sorted(lk, reverse=True)               # kept alleles are sorted by prob desc
    again = select_alleles({"v": probs}, names, cap=3)      # deterministic: identical input -> identical output
    assert [c.names for c in calls["v"]] == [c.names for c in again["v"]]


def test_cigar_query_consumption_never_exceeds_sequence_length():
    """The coordinate-derived per-segment CIGAR never consumes more query bases than the read — the
    invariant `validate-airr` enforces and downstream tools depend on (P0-3/P0-14)."""
    rng = np.random.default_rng(2)
    for _ in range(500):
        L = int(rng.integers(1, 300))
        ss = int(rng.integers(0, L))
        se = int(rng.integers(ss, L + 1))
        gs = int(rng.integers(0, 60))
        ge = int(rng.integers(gs, gs + 100))
        assert cigar_query_length(_cigar(L, ss, se, gs, ge)) <= L
