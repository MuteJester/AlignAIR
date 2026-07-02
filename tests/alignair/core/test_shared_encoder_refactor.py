from alignair.config.dnalignair_config import DNAlignAIRConfig


def test_config_has_ablation_toggles():
    c = DNAlignAIRConfig()
    assert c.band_width == 16
    for f, default in [("band_features", "full"), ("dp_emissions", "learned"),
                       ("use_learned_reps", True), ("use_reliability", True),
                       ("reader", "dp"), ("encoder_mode", "trained")]:
        assert getattr(c, f) == default


def test_ablation_toggles_roundtrip_through_dict():
    c = DNAlignAIRConfig(reader="maxsim", encoder_mode="frozen", use_learned_reps=False)
    c2 = DNAlignAIRConfig.from_dict(c.to_dict())
    assert c2.reader == "maxsim" and c2.encoder_mode == "frozen" and c2.use_learned_reps is False


import torch
from alignair.core.dnalignair import DNAlignAIR


class _Gene:
    sequences = ["ACGTACGTAC", "ACGTTCGTAC"]
    names = ["IGHVx*01", "IGHVx*02"]


class _RS:
    genes = {"V": _Gene()}
    has_d = False
    def gene(self, g): return _Gene()


def _model(d=32):
    cfg = DNAlignAIRConfig(d_model=d, n_layers=1, nhead=2)
    return DNAlignAIR(cfg)


def test_model_has_no_germline_encoder_or_classifier():
    # ONE shared encoder: no separate germline encoder, no allele classifier (dynamic genotype)
    m = _model()
    assert getattr(m, "germline_encoder", None) is None
    assert not hasattr(m, "classifier")


def test_encode_reference_uses_shared_encoder():
    m = _model(d=32)
    ref = m.encode_reference(_RS())
    assert ref["V"]["pos_reps"].shape[0] == 2          # two alleles encoded
    assert ref["V"]["pos_reps"].shape[-1] == 32        # d_model
    assert ref["V"]["embeddings"].shape == (2, 32)


def test_germline_coords_runs():
    m = _model(d=32)
    B, S, Lg = 2, 6, 12
    seg = torch.randn(B, S, 32); germ = torch.randn(B, Lg, 32)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    st = torch.randint(0, 5, (B, S)); gt = torch.randint(0, 5, (B, Lg))
    sl, el = m.germline_coords(seg, sm, germ, gm, seg_tok=st, germ_tok=gt)
    assert sl.shape == (B, Lg) and el.shape == (B, Lg)
