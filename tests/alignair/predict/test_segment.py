"""Tests for segmentation correction (de-pad, clip, one-directional ordering)."""
import numpy as np

from alignair.predict.segment import correct_segments


def test_floor_clip_and_ordering():
    start = {"v": np.array([0.7]), "d": np.array([50.4]), "j": np.array([90.9])}
    end = {"v": np.array([100.9]), "d": np.array([60.2]), "j": np.array([120.5])}
    seq_lens = np.array([130])
    segs = correct_segments(start, end, seq_lens, max_len=576)
    assert segs.start["v"][0] == 0 and segs.end["v"][0] == 100        # floor
    assert segs.start["d"][0] >= segs.end["v"][0]                     # d_start >= v_end
    assert segs.start["j"][0] >= segs.end["d"][0]                     # j_start >= d_end
    assert segs.end["j"][0] == 120


def test_min_span_and_clip_no_d():
    start = {"v": np.array([5.0]), "j": np.array([5.0])}
    end = {"v": np.array([5.0]), "j": np.array([200.0])}              # end==start; end>len
    seq_lens = np.array([50])
    segs = correct_segments(start, end, seq_lens, max_len=576)
    assert segs.end["v"][0] == segs.start["v"][0] + 1                 # enforced min span
    assert segs.end["j"][0] <= 50                                     # clipped to read length
    assert segs.start["j"][0] >= segs.end["v"][0]                     # ordering with no D


def test_center_pad_mode_shifts():
    start = {"v": np.array([40.0]), "j": np.array([90.0])}
    end = {"v": np.array([80.0]), "j": np.array([110.0])}
    seq_lens = np.array([100])                                        # pad=(576-100)//2=238
    segs = correct_segments(start, end, seq_lens, max_len=576, pad_mode="center")
    # 40-238 < 0 -> clipped to 0 (center padding shifts coords left by pad)
    assert segs.start["v"][0] == 0


# --- Constrained bounded/ordered projection --------------------------------------------

def test_short_read_never_exceeds_length():
    """Reproduction of the audited defect: a V filling a 10-base read must NOT push D/J to 11/12.
    Every coordinate stays within [0, seq_len]; squeezed-out segments become absent, not manufactured."""
    start = {"v": np.array([0.0]), "d": np.array([9.0]), "j": np.array([9.0])}
    end = {"v": np.array([10.0]), "d": np.array([10.0]), "j": np.array([10.0])}
    seq_lens = np.array([10])
    segs = correct_segments(start, end, seq_lens, max_len=576)
    for g in ("v", "d", "j"):
        assert 0 <= segs.start[g][0] <= segs.end[g][0] <= 10, (g, segs.start[g][0], segs.end[g][0])


def test_property_bounds_and_ordering():
    """Random logits and lengths: bounds 0<=s<=e<=L and V<=D<=J ordering hold for every read."""
    rng = np.random.default_rng(0)
    n = 500
    seq_lens = rng.integers(1, 600, size=n)
    start = {g: rng.uniform(-50, 650, size=n) for g in ("v", "d", "j")}
    end = {g: rng.uniform(-50, 650, size=n) for g in ("v", "d", "j")}
    segs = correct_segments(start, end, seq_lens, max_len=576)
    for g in ("v", "d", "j"):
        assert np.all(segs.start[g] >= 0)
        assert np.all(segs.end[g] <= seq_lens)
        assert np.all(segs.start[g] <= segs.end[g])
    assert np.all(segs.end["v"] <= segs.start["d"])          # V ends before D starts
    assert np.all(segs.end["d"] <= segs.start["j"])          # D ends before J starts


def test_length_one_and_two():
    for L in (1, 2):
        start = {"v": np.array([0.0]), "d": np.array([0.0]), "j": np.array([0.0])}
        end = {"v": np.array([float(L)]), "d": np.array([float(L)]), "j": np.array([float(L)])}
        segs = correct_segments(start, end, np.array([L]), max_len=576)
        for g in ("v", "d", "j"):
            assert 0 <= segs.start[g][0] <= segs.end[g][0] <= L


def test_light_chain_ordering_no_d():
    start = {"v": np.array([0.0]), "j": np.array([5.0])}
    end = {"v": np.array([8.0]), "j": np.array([3.0])}       # j predicted before v ends (garbage)
    segs = correct_segments(start, end, np.array([8]), max_len=576)
    assert segs.end["v"][0] <= segs.start["j"][0]
    assert segs.end["j"][0] <= 8


def test_low_quality_flag_when_v_collapses():
    """V squeezed to zero length by a too-short read is flagged, not silently emitted."""
    start = {"v": np.array([9.0]), "j": np.array([9.0])}
    end = {"v": np.array([9.0]), "j": np.array([10.0])}
    segs = correct_segments(start, end, np.array([9]), max_len=576)
    assert segs.low_quality is not None and bool(segs.low_quality[0]) is True


def test_low_quality_flag_when_j_collapses():
    """A collapsed mandatory J (not just V) is flagged low-quality too."""
    # V fills [0,5]; J predicted inside V -> projected forward to [5,5] (zero length)
    start = {"v": np.array([0.0]), "d": np.array([2.0]), "j": np.array([2.0])}
    end = {"v": np.array([5.0]), "d": np.array([4.0]), "j": np.array([4.0])}
    segs = correct_segments(start, end, np.array([5]), max_len=576)
    assert segs.end["j"][0] == segs.start["j"][0]                 # J collapsed
    assert bool(segs.low_quality[0]) is True
