"""The configuration flags of the heads read from a trunk's features: the last fields of ``AlignAIRConfig`` (so
positional construction never shifts when one is added), booleans, and refused on an architecture without a trunk."""
from dataclasses import fields

import pytest

from alignair.core import AlignAIR, AlignAIRConfig
from alignair.core.model import TRUNK_HEAD_FLAGS
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.component


def test_the_trunk_head_flags_are_the_configs_last_fields():
    assert TRUNK_HEAD_FLAGS == ("region_head", "anchor_heads")
    assert tuple(field.name for field in fields(AlignAIRConfig))[-len(TRUNK_HEAD_FLAGS):] == TRUNK_HEAD_FLAGS
    assert all(getattr(AlignAIRConfig(v_allele_count=2, j_allele_count=2), flag) is False for flag in TRUNK_HEAD_FLAGS)


@pytest.mark.parametrize("flag", TRUNK_HEAD_FLAGS)
def test_a_trunk_head_flag_is_a_boolean(flag):
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(v_allele_count=2, j_allele_count=2, **{flag: 1})
    assert caught.value.context["field"] == flag


def test_anchor_heads_without_a_trunk_is_refused():
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, anchor_heads=True)
    with pytest.raises(ConfigurationError) as caught:
        AlignAIR(cfg)
    assert caught.value.context["component"] == "trunk_heads"
    assert caught.value.context["fields"] == ["anchor_heads"]
