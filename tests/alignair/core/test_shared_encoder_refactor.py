from alignair.config.dnalignair_config import DNAlignAIRConfig


def test_config_has_seed_extend_and_ablation_toggles():
    c = DNAlignAIRConfig(aligner="seed_extend", backbone="shared")
    assert c.aligner == "seed_extend"
    for f, default in [("band_features", "full"), ("dp_emissions", "learned"),
                       ("use_learned_reps", True), ("use_reliability", True),
                       ("reader", "dp"), ("encoder_mode", "trained")]:
        assert getattr(c, f) == default


def test_ablation_toggles_roundtrip_through_dict():
    c = DNAlignAIRConfig(aligner="seed_extend", reader="maxsim", encoder_mode="frozen",
                         use_learned_reps=False)
    c2 = DNAlignAIRConfig.from_dict(c.to_dict())
    assert c2.reader == "maxsim" and c2.encoder_mode == "frozen" and c2.use_learned_reps is False
