from alignair.gym.instrument.task_space import TaskSpace
from alignair.gym.instrument.lattice import LatticeCell, FrozenLattice


def test_standard_lattice_has_hard_cells():
    lat = FrozenLattice.standard(seed=0)
    names = {c.name for c in lat.cells}
    assert {"clean", "heavy_shm", "heavy_shm_fulllen", "junction_boundary"} <= names


def test_fingerprint_is_stable_and_sensitive():
    a = FrozenLattice.standard(seed=0).fingerprint()
    assert a == FrozenLattice.standard(seed=0).fingerprint()      # stable
    assert a != FrozenLattice.standard(seed=1).fingerprint()      # seed changes it


def test_clean_cell_is_actually_clean_and_seed_independent():
    # the reference/easy cell must control ALL axes, incl. the noise axes, and not
    # vary with seed (else the instrument's zero-point is corrupted).
    p0 = FrozenLattice.standard(seed=0).cell_params(
        next(c for c in FrozenLattice.standard(seed=0).cells if c.name == "clean"))
    p7 = FrozenLattice.standard(seed=7).cell_params(
        next(c for c in FrozenLattice.standard(seed=7).cells if c.name == "clean"))
    assert p0 == p7                                  # difficulty point is seed-independent
    assert p0["seq_error_rate"] == 0.0               # truly clean
    assert p0["ambiguous_count"] == (0, 0)
    assert p0["crop_prob"] == 0.0


def test_heavy_shm_fulllen_cell_is_high_shm_and_uncropped():
    lat = FrozenLattice.standard(seed=0)
    cell = next(c for c in lat.cells if c.name == "heavy_shm_fulllen")
    p = lat.cell_params(cell)
    assert p["mutation_rate"] >= 0.25          # hard tail
    assert p["crop_prob"] == 0.0               # full length (the excluded corner)
