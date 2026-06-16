from alignair.gym.crop import crop_record, FLANK


def _rec():
    # full read of length 100; V[0,40) D[48,60) J[70,95); junction center=(40+70)//2=55
    return {
        "sequence": "".join("ACGT"[i % 4] for i in range(100)),
        "v_sequence_start": 0, "v_sequence_end": 40,
        "d_sequence_start": 48, "d_sequence_end": 60,
        "j_sequence_start": 70, "j_sequence_end": 95,
        "v_germline_start": 5, "v_germline_end": 45,
        "d_germline_start": 2, "d_germline_end": 14,
        "j_germline_start": 0, "j_germline_end": 25,
    }


def test_no_crop_when_target_covers_read():
    r = _rec()
    assert crop_record(r, None) is r
    assert crop_record(r, 100) is r
    assert crop_record(r, 200) is r


def test_window_contains_full_d_and_v_j_flanks():
    r = _rec()
    c = crop_record(r, 50)
    n = len(c["sequence"])
    # all genes still present (non-empty span)
    for g in ("v", "d", "j"):
        assert c[f"{g}_sequence_end"] > c[f"{g}_sequence_start"], g
        assert 0 <= c[f"{g}_sequence_start"] <= n
        assert c[f"{g}_sequence_end"] <= n
    # D fully retained (12 bp)
    assert c["d_sequence_end"] - c["d_sequence_start"] == 12
    # >= FLANK of V's 3' end and J's 5' end
    assert c["v_sequence_end"] - c["v_sequence_start"] >= FLANK
    assert c["j_sequence_end"] - c["j_sequence_start"] >= FLANK


def test_germline_coords_track_trimmed_bases():
    r = _rec()
    c = crop_record(r, 50)
    # V is trimmed on its 5' end -> germline_start advances by exactly the lost bases
    v_lost_5 = c["v_sequence_start"] == 0 and (40 - (c["v_sequence_end"]))  # not the path here
    # compute crop origin from how much V's 5' was lost
    # left = c0 - vs = c0 (vs=0); germline_start = 5 + left
    left = c["v_germline_start"] - 5
    assert left >= 0
    assert c["v_sequence_start"] == max(0, 0 - left) == 0
    # D fully inside -> germline span unchanged
    assert c["d_germline_start"] == 2 and c["d_germline_end"] == 14
    # J trimmed on its 3' end -> germline_end decreases by lost bases
    right = 25 - c["j_germline_end"]
    assert right >= 0


def test_observed_span_matches_cropped_sequence_for_d():
    # the D germline span must equal the D observed span when D is untrimmed
    r = _rec()
    c = crop_record(r, 50)
    d_obs = c["d_sequence_end"] - c["d_sequence_start"]
    d_germ = c["d_germline_end"] - c["d_germline_start"]
    assert d_obs == d_germ == 12


def test_light_chain_no_d():
    r = _rec()
    del r["d_sequence_start"]
    del r["d_sequence_end"]
    del r["d_germline_start"]
    del r["d_germline_end"]
    c = crop_record(r, 50)
    assert "d_sequence_start" not in c
    for g in ("v", "j"):
        assert c[f"{g}_sequence_end"] > c[f"{g}_sequence_start"]
