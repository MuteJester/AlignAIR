import numpy as np
import torch
from alignair.gym.collate import gym_collate
from alignair.nn.heads.region import REGION_INDEX


class _Gene:
    def __init__(self, names):
        self.names = names
        self.index = {n: i for i, n in enumerate(names)}


class _RS:
    def __init__(self):
        self.genes = {"V": _Gene(["v0", "v1", "v2"]), "J": _Gene(["j0", "j1"]),
                      "D": _Gene(["d0", "d1"])}
        self.has_d = True

    def gene(self, g):
        return self.genes[g.upper()]


def _sample(L, vcalls):
    return {
        "tokens": np.ones(L, np.int64),
        "region_labels": np.full(L, REGION_INDEX["V"], np.int64),
        "state_labels": np.zeros(L, np.int64),
        "germline": {"v": (0, 5), "d": (0, 3), "j": (0, 4)},
        "inseq": {"v": (0, L), "d": (1, 2), "j": (2, 3)},
        "calls": {"V": set(vcalls), "J": {"j0"}, "D": {"d0"}},
        "orientation_id": 0, "noise_count": 2.0, "mutation_rate": 0.1,
        "indel_count": 1.0, "productive": 1.0,
    }


def _light_sample(L, vcalls):
    """A light-chain-style sample with NO D gene (no 'd' keys)."""
    return {
        "tokens": np.ones(L, np.int64),
        "region_labels": np.full(L, REGION_INDEX["V"], np.int64),
        "state_labels": np.zeros(L, np.int64),
        "germline": {"v": (0, 5), "j": (0, 4)},
        "inseq": {"v": (0, L), "j": (2, 3)},
        "calls": {"V": set(vcalls), "J": {"j0"}},
        "orientation_id": 0, "noise_count": 2.0, "mutation_rate": 0.1,
        "indel_count": 1.0, "productive": 1.0,
    }


def test_collate_mixed_chain_handles_samples_without_d():
    # regression: a D-less (light-chain) sample in a has_d batch must not crash, and must be
    # excluded from D supervision while keeping its V/J labels.
    rs = _RS()
    batch = [_sample(6, ["v0"]), _light_sample(5, ["v1"])]
    out = gym_collate(batch, rs, has_d=True)                  # previously raised KeyError 'd'
    assert out["d_supervise"].tolist() == [1.0, 0.0]          # heavy supervised, light not
    assert out["d_germline_start"][1].item() == 0             # sentinel for the D-less sample
    assert out["d_allele"][1].tolist() == [0.0, 0.0]          # D multihot zeroed
    assert out["d_allele"][0].tolist() == [1.0, 0.0]          # heavy D intact
    assert out["v_allele"].tolist() == [[1, 0, 0], [0, 1, 0]]  # V/J intact for both
    # all-light batch against a has_d reference also works
    gym_collate([_light_sample(5, ["v0"]), _light_sample(4, ["v1"])], rs, has_d=True)


def test_collate_pads_and_multihot():
    rs = _RS()
    batch = [_sample(6, ["v0"]), _sample(4, ["v1", "v2"])]
    out = gym_collate(batch, rs, has_d=True)
    assert out["tokens"].shape == (2, 6) and out["mask"].shape == (2, 6)
    assert out["mask"][1].tolist() == [True, True, True, True, False, False]
    # region padding label is -100
    assert (out["region_labels"][1, 4:] == -100).all()
    # multi-hot V calls
    assert out["v_allele"].shape == (2, 3)
    assert out["v_allele"][0].tolist() == [1.0, 0.0, 0.0]
    assert out["v_allele"][1].tolist() == [0.0, 1.0, 1.0]   # multi-label row
    assert out["orientation_id"].tolist() == [0, 0]
    assert out["noise_count"].shape == (2, 1)
