from alignair.gym.factored import FactoredCurriculum, axis_competence_from_field


def test_advance_only_raises_axes_above_threshold():
    fc = FactoredCurriculum(start_pace=0.2)
    moved = fc.advance({"mutation_count": 0.9, "indel_count": 0.5}, threshold=0.7, step=0.1)
    assert moved == ["mutation_count"]
    assert abs(fc.pace["mutation_count"] - 0.3) < 1e-9
    assert abs(fc.pace["indel_count"] - 0.2) < 1e-9      # below threshold, unmoved


def test_pace_caps_at_one():
    fc = FactoredCurriculum(start_pace=0.95)
    fc.advance({"crop": 1.0}, threshold=0.7, step=0.2)
    assert fc.pace["crop"] == 1.0


def test_axis_competence_mapping_uses_isolated_cells():
    field = {"clean": {"S": 0.8}, "heavy_shm_fulllen": {"S": 0.4}, "fragment": {"S": 0.5},
             "indel": {"S": 0.6}, "trim": {"S": 0.55}}
    ac = axis_competence_from_field(field)
    assert ac["mutation_count"] == 0.4      # SHM axis <- heavy_shm_fulllen
    assert ac["crop"] == 0.5                # crop axis <- fragment
    assert ac["indel_count"] == 0.6         # <- isolated indel cell
    assert ac["end_loss_5"] == 0.55         # <- isolated trim cell
    assert ac["ambiguous_count"] == 0.8     # cell absent -> clean fallback
