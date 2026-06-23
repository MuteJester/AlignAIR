from alignair.gym.control.plateau import PlateauDetector


def test_improving_never_plateaus():
    d = PlateauDetector(patience=3, slope_eps=1e-3)
    for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
        assert d.update(v) is False
    assert d.used == 0


def test_flat_sequence_plateaus_after_patience():
    d = PlateauDetector(patience=3, slope_eps=1e-3)
    assert d.update(0.80) is False        # first obs = new best
    assert d.update(0.8005) is False      # < eps gain -> stall 1
    assert d.update(0.8009) is False      # stall 2
    assert d.update(0.8001) is True       # stall 3 == patience -> ceiling
    assert d.used == 3


def test_reset_clears_stall_counter():
    d = PlateauDetector(patience=2, slope_eps=1e-3)
    d.update(0.80); d.update(0.8001)
    d.reset()
    assert d.used == 0
    assert d.update(0.8002) is False
