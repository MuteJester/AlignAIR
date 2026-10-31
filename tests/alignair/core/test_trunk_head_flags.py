"""The configuration flags of the heads read from a trunk's features: the last fields of ``AlignAIRConfig`` (so
positional construction never shifts when one is added), booleans, and refused on an architecture without a trunk."""
from dataclasses import fields

import pytest

from alignair.core import AlignAIR, AlignAIRConfig
from alignair.core.model import TRUNK_HEAD_FLAGS
from alignair.errors import ConfigurationError

pytestmark = pytest.mark.component


def test_the_trunk_head_flags_are_the_configs_last_fields():
    assert TRUNK_HEAD_FLAGS == ("region_head", "anchor_heads", "d_presence_head")
    assert tuple(field.name for field in fields(AlignAIRConfig))[-len(TRUNK_HEAD_FLAGS):] == TRUNK_HEAD_FLAGS
    assert all(getattr(AlignAIRConfig(v_allele_count=2, j_allele_count=2), flag) is False for flag in TRUNK_HEAD_FLAGS)


@pytest.mark.parametrize("flag", TRUNK_HEAD_FLAGS)
def test_a_trunk_head_flag_is_a_boolean(flag):
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(v_allele_count=2, j_allele_count=2, **{flag: 1})
    assert caught.value.context["field"] == flag


@pytest.mark.parametrize("flag", ["anchor_heads", "d_presence_head"])
def test_a_trunk_head_without_a_trunk_is_refused(flag):
    cfg = AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, **{flag: True})
    with pytest.raises(ConfigurationError) as caught:
        AlignAIR(cfg)
    assert caught.value.context["component"] == "trunk_heads"
    assert caught.value.context["fields"] == [flag]


def test_d_presence_without_d_is_refused():
    """A light chain has no D to be present or absent: the D presence head is refused at configuration."""
    with pytest.raises(ConfigurationError) as caught:
        AlignAIRConfig(v_allele_count=2, j_allele_count=2, has_d=False, d_presence_head=True)
    assert caught.value.context["field"] == "d_presence_head"
    assert AlignAIRConfig(v_allele_count=2, j_allele_count=2, d_allele_count=2, d_presence_head=True).d_presence_head
