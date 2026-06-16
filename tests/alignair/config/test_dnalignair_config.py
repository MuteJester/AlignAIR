from alignair.config.dnalignair_config import DNAlignAIRConfig


def test_defaults_and_roundtrip():
    cfg = DNAlignAIRConfig(d_model=128)
    assert cfg.d_model == 128 and cfg.n_layers >= 1 and cfg.nhead >= 1
    assert cfg.n_regions == 8 and cfg.n_states == 4
    assert DNAlignAIRConfig.from_dict(cfg.to_dict()) == cfg
