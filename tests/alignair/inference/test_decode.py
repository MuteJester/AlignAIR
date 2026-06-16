import numpy as np
from alignair.inference.decode import extract_positions, correct_segments


def test_extract_positions_argmax():
    L = 8
    pred = {}
    for g in ("v", "j", "d"):
        for b in ("start", "end"):
            logits = np.full((2, L), -10.0, np.float32)
            logits[:, 3] = 10.0  # argmax at 3
            pred[f"{g}_{b}_logits"] = logits
    pos = extract_positions(pred, has_d=True)
    assert pos["v_start"].tolist() == [3, 3]
    assert "d_end" in pos


def test_correct_segments_removes_padding_and_orders():
    # seq len 10, max_length 16 -> padding (16-10)//2 = 3
    sequences = ["A" * 10, "A" * 10]
    positions = {
        "v_start": np.array([3, 3]), "v_end": np.array([8, 8]),
        "d_start": np.array([7, 7]), "d_end": np.array([9, 9]),
        "j_start": np.array([10, 10]), "j_end": np.array([14, 14]),
    }
    corrected = correct_segments(positions, sequences, max_length=16, has_d=True)
    # padding removed: v_start 3-3=0, v_end 8-3=5
    assert corrected["v_start"].tolist() == [0, 0]
    assert corrected["v_end"].tolist() == [5, 5]
    # ordering: v_end <= d_start <= d_end <= j_start <= j_end, all within [0,10]
    assert (corrected["d_start"] >= corrected["v_end"]).all()
    assert (corrected["j_start"] >= corrected["d_end"]).all()
    assert (corrected["j_end"] <= 10).all()
