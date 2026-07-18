import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.data.genairr import assert_genairr_capable, allele_vocab_from_dataconfig


def test_capability_check_passes():
    assert_genairr_capable()  # must not raise on the installed 2.2.0 code


def test_vocab_from_dataconfig_human_igh():
    vocab = allele_vocab_from_dataconfig(gdata.HUMAN_IGH_OGRDB)
    assert len(vocab["V"]) == 198 and len(vocab["J"]) == 7
    assert vocab["D"][-1] == "Short-D"
    assert len(vocab["D"]) == 34  # 33 real + Short-D
