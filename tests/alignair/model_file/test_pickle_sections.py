"""A model file is distributable only without pickle sections: one rule for the aligner and the catalog check."""
import pytest

from alignair.model_file import pickle_sections

pytestmark = pytest.mark.component


def test_the_pickle_sections_are_the_dataconfigs_and_the_training_state():
    metadata = {"sections": {"weights": {}, "reference_json": {}, "dataconfig/0": {}, "train_state": {},
                             "dataconfig/1": {}}}
    assert pickle_sections(metadata) == ["dataconfig/0", "train_state", "dataconfig/1"]
    assert pickle_sections({"sections": {"weights": {}}}) == [] and pickle_sections({}) == []
