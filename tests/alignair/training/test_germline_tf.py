import torch
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.training.germline_tf import compute_germline_logits
from alignair.nn.heads.region import REGION_INDEX


class _Gene:
    def __init__(self, n):
        self.names = [f"a{i}" for i in range(n)]
        self.index = {nm: i for i, nm in enumerate(self.names)}
        self.sequences = ["ACGT" * 10 for _ in range(n)]


class _RS:
    def __init__(self):
        self.genes = {"V": _Gene(4), "J": _Gene(3)}
        self.has_d = False

    def gene(self, g):
        return self.genes[g.upper()]


def test_compute_germline_logits_shapes():
    cfg = DNAlignAIRConfig(d_model=32, n_layers=2, nhead=4, dim_feedforward=64)
    model = DNAlignAIR(cfg)
    rs = _RS()
    ref_emb = model.encode_reference(rs)
    B, L = 2, 16
    tokens = torch.randint(0, 6, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    region_labels = torch.zeros(B, L, dtype=torch.long)
    region_labels[:, 2:8] = REGION_INDEX["V"]
    region_labels[:, 8:12] = REGION_INDEX["J"]
    batch = {"region_labels": region_labels,
             "v_allele": torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
             "j_allele": torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])}
    gl = compute_germline_logits(model, tokens, mask, batch, ref_emb, has_d=False)
    Lg_v = ref_emb["V"]["pos_reps"].shape[1]
    assert gl["v"][0].shape == (B, Lg_v) and gl["v"][1].shape == (B, Lg_v)
    assert "j" in gl


def test_compute_germline_logits_seed_extend_reads_off_backbone():
    # seed_extend has NO germline_encoder; segment reps must come from the backbone reps (no re-encode)
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, backbone="shared", aligner="seed_extend")
    model = DNAlignAIR(cfg)
    assert getattr(model, "germline_encoder", None) is None
    rs = _RS()
    ref_emb = model.encode_reference(rs)
    B, L = 2, 16
    tokens = torch.randint(0, 6, (B, L)); mask = torch.ones(B, L, dtype=torch.bool)
    region_labels = torch.zeros(B, L, dtype=torch.long)
    region_labels[:, 2:8] = REGION_INDEX["V"]; region_labels[:, 8:12] = REGION_INDEX["J"]
    batch = {"region_labels": region_labels,
             "v_allele": torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
             "j_allele": torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])}
    out = model.forward_dense(tokens, mask)
    gl = compute_germline_logits(model, out["canon_tokens"], mask, batch, ref_emb,
                                 has_d=False, reps=out["reps"])
    Lg_v = ref_emb["V"]["pos_reps"].shape[1]
    assert gl["v"][0].shape == (B, Lg_v) and gl["v"][1].shape == (B, Lg_v)


def test_compute_germline_logits_seed_extend_without_reps_falls_back_to_backbone():
    # no reps passed -> re-encode the segment through the SHARED backbone (READ), still no GermlineEncoder
    cfg = DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, backbone="shared", aligner="seed_extend")
    model = DNAlignAIR(cfg); rs = _RS(); ref_emb = model.encode_reference(rs)
    B, L = 2, 16
    tokens = torch.randint(0, 6, (B, L)); mask = torch.ones(B, L, dtype=torch.bool)
    region_labels = torch.zeros(B, L, dtype=torch.long); region_labels[:, 2:8] = REGION_INDEX["V"]
    batch = {"region_labels": region_labels, "v_allele": torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
             "j_allele": torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])}
    gl = compute_germline_logits(model, tokens, mask, batch, ref_emb, has_d=False)   # reps=None
    assert gl["v"][0].shape[0] == B
