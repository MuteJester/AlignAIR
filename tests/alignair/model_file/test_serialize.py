import torch
import GenAIRR.data as gd
from alignair.core.config import AlignAIRConfig
from alignair.reference.reference_set import ReferenceSet
from alignair.model_file import serialize as S


def test_config_roundtrip_full_fields():
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    back = S.config_from_bytes(S.config_to_bytes(cfg))
    assert back.__dict__ == cfg.__dict__          # field-for-field


def test_state_dict_roundtrip_safetensors():
    sd = {"a.w": torch.randn(3, 4), "b": torch.zeros(2)}
    back = S.state_dict_from_bytes(S.state_dict_to_bytes(sd))
    assert set(back) == set(sd) and torch.equal(back["a.w"], sd["a.w"])


def test_dataconfig_roundtrip():
    back = S.dataconfig_from_bytes(S.dataconfig_to_bytes(gd.HUMAN_IGH_OGRDB))
    assert type(back).__name__ == "DataConfig" and back.metadata.has_d is True


def test_reference_fasta_matches_alleles():
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    fasta = S.reference_fasta(ref)
    names = [ln[1:] for ln in fasta.splitlines() if ln.startswith(">")]
    assert set(ref.gene("V").names).issubset(set(names))


def test_train_state_roundtrip():
    state = {"optimizer": {"x": 1}, "rng": {"torch": torch.get_rng_state()}, "step": 7, "train_args": {"lr": 1e-4}}
    back = S.train_state_from_bytes(S.train_state_to_bytes(state))
    assert back["step"] == 7 and back["train_args"]["lr"] == 1e-4
