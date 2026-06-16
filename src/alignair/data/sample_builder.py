"""Shared per-sample (x, y) assembly from a canonical record + tokens."""
import numpy as np


def build_xy(tokens: np.ndarray, rec: dict, encoder, has_d: bool):
    x = {"tokenized_sequence": tokens}
    y = {
        "v_start": np.array([rec["v_start"]], np.float32),
        "v_end": np.array([rec["v_end"]], np.float32),
        "j_start": np.array([rec["j_start"]], np.float32),
        "j_end": np.array([rec["j_end"]], np.float32),
        "v_allele": encoder.encode("V", [rec["v_call_set"]])[0],
        "j_allele": encoder.encode("J", [rec["j_call_set"]])[0],
        "mutation_rate": np.array([rec["mutation_rate"]], np.float32),
        "indel_count": np.array([rec["indel_count"]], np.float32),
        "productive": np.array([rec["productive"]], np.float32),
    }
    if has_d:
        y["d_start"] = np.array([rec["d_start"]], np.float32)
        y["d_end"] = np.array([rec["d_end"]], np.float32)
        y["d_allele"] = encoder.encode("D", [rec["d_call_set"]])[0]
    return x, y
