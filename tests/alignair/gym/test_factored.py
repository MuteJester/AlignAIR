from alignair.gym.factored import FactoredCurriculum
from alignair.gym.curriculum import StratifiedCurriculum


def _hard_mass(dist):           # prob mass on the hardest two bins of a (value,prob) list
    return sum(p for _, p in dist[-2:])


def test_advancing_one_axis_only_changes_that_axis():
    fc = FactoredCurriculum(start_pace=0.1)
    before = fc.params()
    fc.pace["mutation_count"] = 1.0          # unlock SHM only
    after = fc.params()
    # SHM distribution shifted toward hard...
    assert _hard_mass(after["mutation_count"]) > _hard_mass(before["mutation_count"])
    # ...while a DECOUPLED axis (indel_count) is unchanged
    assert after["indel_count"] == before["indel_count"]


def test_terminal_equals_full_stratified_distribution():
    fc = FactoredCurriculum()
    for a in fc.axes:
        fc.pace[a] = 1.0
    assert fc.is_terminal()
    # at all-paces-1 the SHM mixture equals StratifiedCurriculum at tau=1
    assert fc.params()["mutation_count"] == StratifiedCurriculum().params(1.0)["mutation_count"]


def test_params_shape_matches_build_experiment_keys():
    p = FactoredCurriculum().params()
    for k in ("mutation_count", "end_loss_5", "end_loss_3", "indel_count",
              "ambiguous_count", "seq_error_rate", "crop_prob", "crop_len_min",
              "crop_len_max", "orient_prob"):
        assert k in p
    assert isinstance(p["mutation_count"], list) and isinstance(p["mutation_count"][0], tuple)
