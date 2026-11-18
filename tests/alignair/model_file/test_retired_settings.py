"""A model file written while the four retired settings existed loads when each holds the value every trained model
used (the model it describes is the one built now), and is refused, naming the setting, when one does not."""
import json

import pytest

from alignair.errors import ModelError
from alignair.model_file.serialize import config_from_bytes

pytestmark = pytest.mark.component

TRAINED = {"use_orientation": True, "anchor_heads": True, "reject_empty_sequences": False}


def _config(has_d=True, **settings):
    return json.dumps({"v_allele_count": 3, "j_allele_count": 2, "d_allele_count": 1 if has_d else 0,
                       "has_d": has_d, "architecture": "chip-v1", **settings}).encode()


@pytest.mark.parametrize("has_d", [True, False])
def test_the_trained_values_load_as_todays_model(has_d):
    cfg = config_from_bytes(_config(has_d, **TRAINED, d_presence_head=has_d))
    assert cfg.has_d is has_d and cfg.architecture == "chip-v1"


@pytest.mark.parametrize("setting, value", [("use_orientation", False), ("anchor_heads", False),
                                            ("reject_empty_sequences", True), ("d_presence_head", False)])
def test_another_value_is_refused_naming_the_setting(setting, value):
    with pytest.raises(ModelError, match=setting) as caught:
        config_from_bytes(_config(**{**TRAINED, "d_presence_head": True, setting: value}))
    assert caught.value.context["settings"] == [setting] and repr(value) in str(caught.value)
