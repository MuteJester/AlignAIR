"""Library flanks for training: non-germline sequence before V and after J (a 5' UTR and leader, a primer masked to N,
a 3' constant region), which GenAIRR does not simulate; every read coordinate of the record moves with the read."""
import pytest

from alignair.data.flanks import READ_COORDINATES, FlankPolicy, add_flanks, flank_records
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.component

RECORD = {"sequence": "ACGTACGTACGTAAA", "sequence_length": 15, "v_sequence_start": 0, "v_sequence_end": 10,
          "v_alignment_start": 0, "v_alignment_end": 10, "v_germline_start": 3, "v_germline_end": 13,
          "v_cigar": "10M", "d_sequence_start": None, "d_sequence_end": None, "d_alignment_start": None,
          "d_alignment_end": None, "j_sequence_start": 11, "j_sequence_end": 15, "j_alignment_start": 11,
          "j_alignment_end": 15, "j_germline_start": 0, "j_germline_end": 4, "junction_start": 6, "junction_end": 15}


def test_add_flanks_moves_every_read_coordinate_and_no_germline_one():
    out = add_flanks(RECORD, prefix="GGG", suffix="TT")
    assert out["sequence"] == "GGGACGTACGTACGTAAATT" and out["sequence_length"] == 20
    assert (out["flank_5_length"], out["flank_3_length"], out["mask_5_length"]) == (3, 2, 0)
    for key in READ_COORDINATES:
        assert out[key] == (None if RECORD[key] is None else RECORD[key] + 3), key
    assert (out["v_germline_start"], out["v_cigar"], out["j_germline_end"]) == (3, "10M", 4)
    assert RECORD["sequence"] == "ACGTACGTACGTAAA"


def test_a_mask_into_v_starts_v_after_it():
    """A primer masked over the first V bases: V starts at the first unmasked base, its germline start and its
    CIGAR moving with it (the bases under N are not observable)."""
    out = add_flanks(RECORD, prefix="GG", mask=4)
    assert out["sequence"] == "NNNNGTACGTACGTAAA" and out["mask_5_length"] == 4
    assert (out["v_sequence_start"], out["v_alignment_start"], out["v_germline_start"]) == (4, 4, 5)
    assert out["v_cigar"] == "8M" and out["v_sequence_end"] == 12 and out["j_sequence_start"] == 13


def test_a_mask_stops_at_v_when_v_opens_with_an_indel():
    out = add_flanks({**RECORD, "v_cigar": "1I9M"}, prefix="GG", mask=4)
    assert out["sequence"].startswith("NNA") and out["mask_5_length"] == 2
    assert (out["v_sequence_start"], out["v_germline_start"], out["v_cigar"]) == (2, 3, "1I9M")


def test_flank_records_follow_the_policy_and_fit_the_window():
    policy = FlankPolicy(p_prefix=1.0, prefix=(5, 9), p_mask=0.0, p_suffix=1.0, suffix=(2, 4), max_length=30)
    out = list(flank_records([RECORD] * 40, policy, seed=3))
    assert all(5 <= r["flank_5_length"] <= 9 and 2 <= r["flank_3_length"] <= 4 for r in out)
    assert all(len(r["sequence"]) == r["sequence_length"] <= 30 for r in out)
    assert all(r["v_sequence_start"] == r["flank_5_length"] for r in out)
    tight = list(flank_records([RECORD] * 10, FlankPolicy(p_prefix=1.0, prefix=(5, 9), p_mask=0.0, p_suffix=1.0,
                                                          suffix=(2, 4), max_length=17), seed=3))
    assert all(len(r["sequence"]) <= 17 and r["sequence"].find("ACGTACGTACGTAAA") >= 0 for r in tight)


def test_flank_records_are_reproducible_per_seed_and_untouched_without_flanks():
    policy = FlankPolicy(p_prefix=0.5, p_suffix=0.5)
    first = [r["sequence"] for r in flank_records([RECORD] * 30, policy, seed=11)]
    assert first == [r["sequence"] for r in flank_records([RECORD] * 30, policy, seed=11)]
    assert first != [r["sequence"] for r in flank_records([RECORD] * 30, policy, seed=12)]
    lengths = [len(s) - 15 for s in first]
    assert 0 in lengths and any(lengths)                    # some reads keep their bare read
    none = list(flank_records([RECORD] * 5, FlankPolicy(p_prefix=0.0, p_suffix=0.0), seed=1))
    assert all(r["sequence"] == RECORD["sequence"] and r["flank_5_length"] == 0 for r in none)


def test_masked_flanks_are_n_and_reach_into_v_within_the_extra():
    policy = FlankPolicy(p_prefix=1.0, prefix=(3, 3), p_mask=1.0, mask_extra=(0, 2), p_suffix=0.0)
    for r in flank_records([RECORD] * 20, policy, seed=5):
        assert r["sequence"][:r["mask_5_length"]] == "N" * r["mask_5_length"]
        assert 3 <= r["mask_5_length"] <= 5 and r["v_sequence_start"] == r["mask_5_length"]


def test_unmasked_flanks_carry_scattered_uncalled_bases_inside_the_flank_only():
    """A 5'RACE consensus read's leader holds uncalled bases (N) here and there, not one masked run: on ``p_flank_n``
    of the unmasked 5' flanks each base turns to N at a rate drawn in ``flank_n_rate``; the read and every
    coordinate are untouched."""
    policy = FlankPolicy(p_prefix=1.0, prefix=(60, 60), p_mask=0.0, p_suffix=0.0, p_flank_n=1.0,
                         flank_n_rate=(0.2, 0.2))
    out = list(flank_records([RECORD] * 20, policy, seed=7))
    assert all(r["sequence"][60:] == RECORD["sequence"] and r["v_sequence_start"] == 60 for r in out)
    assert all(r["mask_5_length"] == 0 and r["sequence"][:60].count("N") < 60 for r in out)
    assert 0.15 < sum(r["sequence"][:60].count("N") for r in out) / (20 * 60) < 0.25
    plain = FlankPolicy(p_prefix=1.0, prefix=(60, 60), p_mask=0.0, p_suffix=0.0, p_flank_n=0.0)
    assert all("N" not in r["sequence"] for r in flank_records([RECORD] * 20, plain, seed=7))
    # on by default, at a share of the unmasked flanks
    flanked = list(flank_records([RECORD] * 200, FlankPolicy(p_prefix=1.0, prefix=(60, 60), p_mask=0.0,
                                                             p_suffix=0.0), seed=7))
    with_n = sum("N" in r["sequence"] for r in flanked)
    assert 0 < with_n < 200


def test_policy_round_trips_and_validates():
    policy = FlankPolicy(p_prefix=0.2, prefix=(10, 40), p_mask=0.5, mask_extra=(0, 6), p_suffix=0.1,
                         suffix=(5, 30), max_length=500, p_flank_n=0.25, flank_n_rate=(0.01, 0.1))
    assert FlankPolicy.from_dict(policy.to_dict()) == policy
    # a policy recorded before the uncalled bases reads back with their defaults
    old = {key: value for key, value in policy.to_dict().items() if key not in ("p_flank_n", "flank_n_rate")}
    restored = FlankPolicy.from_dict(old)
    assert (restored.p_flank_n, restored.flank_n_rate) == (FlankPolicy().p_flank_n, FlankPolicy().flank_n_rate)
    assert restored.prefix == (10, 40) and restored.p_mask == 0.5
    for bad in ({"p_prefix": 1.5}, {"prefix": (9, 3)}, {"suffix": (-1, 4)}, {"mask_extra": (0, -1)},
                {"max_length": 0}, {"p_flank_n": -0.1}, {"flank_n_rate": (0.2, 0.1)},
                {"flank_n_rate": (0.0, 1.5)}):
        with pytest.raises(ConfigurationError):
            FlankPolicy(**bad)
