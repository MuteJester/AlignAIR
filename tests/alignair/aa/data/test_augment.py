import pytest
import numpy as np

from alignair.aa.augment import aa_indel, aa_substitute, aa_truncate, il_ambiguity

pytestmark = pytest.mark.component


def _rng():
    return np.random.default_rng(0)


def test_substitute_changes_about_the_target_fraction():
    seq = "A" * 1000
    out = aa_substitute(seq, 0.1, _rng())
    assert len(out) == len(seq)
    changed = sum(a != b for a, b in zip(seq, out))
    assert 60 <= changed <= 160                       # ~10% of 1000, with slack


def test_substitute_rate_zero_is_identity():
    seq = "QVQLVESGG"
    assert aa_substitute(seq, 0.0, _rng()) == seq


def test_indel_changes_length_by_one():
    seq = "QVQLVESGG"
    lengths = {len(aa_indel(seq, _rng())) for _ in range(20)}
    assert lengths <= {len(seq) - 1, len(seq) + 1} and lengths != {len(seq)}


def test_il_ambiguity_maps_isoleucine_and_leucine_together():
    # mass-spec cannot distinguish I and L; map both to a single symbol deterministically
    assert il_ambiguity("MILK") == il_ambiguity("MLIK")           # I<->L indistinguishable
    assert il_ambiguity("MILK")[1] == il_ambiguity("MILK")[2]     # positions 1,2 collapse
    assert il_ambiguity("QVQ") == "QVQ"                            # non-I/L untouched


def test_truncate_keeps_a_contiguous_window_within_length():
    seq = "ABCDEFGHIJ"
    out = aa_truncate(seq, min_frac=0.4, max_frac=0.8, rng=_rng())
    assert 4 <= len(out) <= 8
    assert out in [seq[i:i + len(out)] for i in range(len(seq) - len(out) + 1)]
