import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR


def test_pointer_aligner_selected_and_coords_run():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, aligner="pointer")
    model = DNAlignAIR(cfg)
    from alignair.nn.aligner.pointer import BandedPointerAligner
    assert isinstance(model.aligner, BandedPointerAligner)
    B, S, Lg, d = 2, 6, 12, 32
    seg = torch.randn(B, S, d); germ = torch.randn(B, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    seg_tok = torch.randint(0, 5, (B, S)); germ_tok = torch.randint(0, 5, (B, Lg))
    rel = torch.rand(B, S)
    sl, el = model.germline_coords(seg, sm, germ, gm, seg_tok=seg_tok,
                                   germ_tok=germ_tok, seg_reliability=rel)
    assert sl.shape == (B, Lg) and el.shape == (B, Lg)
