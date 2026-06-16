import torch
from alignair.config.model_config import ModelConfig
from alignair.core.single_chain import SingleChainAlignAIR
from alignair.core.multi_chain import MultiChainAlignAIR


def _single_cfg():
    return ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                       d_allele_count=4, has_d_gene=True)


def _multi_cfg():
    return ModelConfig(max_seq_length=256, v_allele_count=5, j_allele_count=3,
                       d_allele_count=4, has_d_gene=True, number_of_chains=2,
                       chain_types=["IGH", "IGK"])


def _forward_parity(model_cls, cfg, tmp_path):
    torch.manual_seed(0)
    model = model_cls(cfg).eval()
    x = torch.randint(0, 6, (2, cfg.max_seq_length))
    with torch.no_grad():
        before = model(x).as_dict()
    model.save_pretrained(tmp_path, dataconfig={"ref": "abc"})

    reloaded = model_cls.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(x).as_dict()
    for k in before:
        assert torch.allclose(before[k], after[k], atol=1e-6), f"mismatch in {k}"


def test_single_chain_roundtrip_parity(tmp_path):
    _forward_parity(SingleChainAlignAIR, _single_cfg(), tmp_path)


def test_multi_chain_roundtrip_parity(tmp_path):
    _forward_parity(MultiChainAlignAIR, _multi_cfg(), tmp_path)
    assert "chain_type" in MultiChainAlignAIR.from_pretrained(tmp_path)(
        torch.randint(0, 6, (1, 256))).as_dict()


def test_from_pretrained_picks_class_by_config(tmp_path):
    MultiChainAlignAIR(_multi_cfg()).eval().save_pretrained(tmp_path)
    # Even calling via the single-chain entry, config.is_multi_chain selects MultiChain.
    reloaded = SingleChainAlignAIR.from_pretrained(tmp_path)
    assert isinstance(reloaded, MultiChainAlignAIR)


def test_load_dataconfig(tmp_path):
    SingleChainAlignAIR(_single_cfg()).save_pretrained(tmp_path, dataconfig={"k": 1})
    assert SingleChainAlignAIR.load_dataconfig(tmp_path) == {"k": 1}
