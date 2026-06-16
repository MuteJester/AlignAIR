import pytest
genairr = pytest.importorskip("GenAIRR")
import GenAIRR.data as gdata
from alignair.data.experiment_presets import full_augmentation, no_corruption, minimal


def test_full_augmentation_streams_a_record():
    exp = full_augmentation(gdata.HUMAN_IGH_OGRDB)
    rec = next(exp.stream_records(n=1, seed=0))
    assert "sequence" in rec and "v_call" in rec
    assert "mutation_rate" in rec and "n_indels" in rec


def test_minimal_and_no_corruption_build():
    assert next(minimal(gdata.HUMAN_IGH_OGRDB).stream_records(n=1, seed=0))["sequence"]
    assert next(no_corruption(gdata.HUMAN_IGH_OGRDB).stream_records(n=1, seed=0))["sequence"]
