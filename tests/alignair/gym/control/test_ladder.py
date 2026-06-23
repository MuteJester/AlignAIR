from alignair.gym.control.ladder import RankLadder


def test_progress_maps_levels_to_unit_interval():
    lad = RankLadder(n_levels=10)
    assert lad.progress(0) == 0.0
    assert lad.progress(9) == 1.0
    assert lad.top == 9
    # clamps out-of-range
    assert lad.progress(-1) == 0.0
    assert lad.progress(99) == 1.0


def test_params_get_harder_with_level():
    lad = RankLadder(n_levels=10)
    easy = lad.params(0)
    hard = lad.params(9)
    assert hard["mutation_rate"] >= easy["mutation_rate"]


def test_each_level_has_a_name():
    lad = RankLadder(n_levels=10)
    assert isinstance(lad.name(0), str) and lad.name(0)
    assert lad.name(0) != lad.name(9)
    # out-of-range still returns a string (no crash)
    assert isinstance(lad.name(99), str)
