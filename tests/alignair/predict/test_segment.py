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
