import pytest
from dataclasses import replace
import GenAIRR.data as gd
import torch
from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.api import load_model as api_load
from alignair.core import prepare_model
from alignair.errors import ConfigurationError, ModelError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


def test_api_load_model_reads_alignair_without_dataconfig(tmp_path):
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=128, embed_dim=4, trunk_channels=8)
    m = AlignAIR(cfg)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), m, dataconfigs=["HUMAN_IGH_OGRDB"], training={"steps": 1, "batch_size": 1})
    model, reference = api_load(str(p))                 # NO dataconfigs= needed
    assert reference.gene("V").names[0].startswith("IGH")
    assert model.cfg.v_allele_count == cfg.v_allele_count
