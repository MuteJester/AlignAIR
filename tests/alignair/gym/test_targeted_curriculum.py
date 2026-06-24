from alignair.gym.factored import FactoredCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.targeting import TargetedCurriculum


def _tc():
    return TargetedCurriculum(FactoredCurriculum(start_pace=0.3), FrozenLattice.standard(seed=0))


def test_components_sum_to_one_and_delegate_interface():
    tc = _tc()
    comps = tc.components()
    assert abs(sum(w for w, _ in comps) - 1.0) < 1e-9
    assert tc.params()["mutation_count"] == tc.factored.params()["mutation_count"]
    assert tc.advance({"mutation_count": 0.9})                     # delegates axis-pacing to factored


def test_targeted_component_tracks_top_priority_cell():
    tc = _tc()
    tc.update_targets({"clean": {"S": 0.97}, "heavy_shm": {"S": 0.40},
                       "heavy_shm_fulllen": {"S": 0.95}})
    comps = tc.components()
    # the p_alp component should carry heavy_shm's params (lowest competence => top regret)
    lat = FrozenLattice.standard(seed=0)
    hs = lat.cell_params(next(c for c in lat.cells if c.name == "heavy_shm"))
    assert any(abs(w - tc.p_alp) < 1e-9 and p == hs for w, p in comps)


def test_no_targets_folds_alp_into_ramp():
    tc = _tc()
    comps = tc.components()                                         # tracker empty
    ramp_w = sum(w for w, _ in comps[:-1])                          # all but floor
    assert abs(ramp_w - (tc.p_ramp + tc.p_alp)) < 1e-9
