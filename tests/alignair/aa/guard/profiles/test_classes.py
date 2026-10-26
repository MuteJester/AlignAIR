import pytest
from alignair.aa.guard.classes import (CLASS_MAP_HASH, COMPLETENESS_CLASSES, DOMAIN_TYPE_CLASSES,
                                       class_map_hash)
from alignair.aa.guard.profiles import RECEPTOR_GROUPS

pytestmark = pytest.mark.component


def test_class_maps_are_ordered_and_hashed():
    assert len(COMPLETENESS_CLASSES) == 5 and len(DOMAIN_TYPE_CLASSES) == 8
    assert COMPLETENESS_CLASSES[0] == "complete"
    assert CLASS_MAP_HASH == class_map_hash() and len(CLASS_MAP_HASH) == 64


def test_vhh_biological_label_is_distinct_from_camelid_igh_evidence():
    # Safeguard 1: 'vhh' is a supervised biological domain-type label; 'camelid_igh' is an HMM-bank
    # evidence group. They must never be conflated.
    assert "vhh" in DOMAIN_TYPE_CLASSES and "vhh" not in RECEPTOR_GROUPS
    assert "camelid_igh" in RECEPTOR_GROUPS and "camelid_igh" not in DOMAIN_TYPE_CLASSES
