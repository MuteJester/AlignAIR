"""P0-14: every emitted AIRR column must be documented in the field map, so no advertised column is an
undocumented placeholder, and `productive` vs `productive_prediction` are distinguished by source."""
from alignair.io.airr import COLUMNS
from alignair.io.airr_field_map import DERIVED, EXTENSION, FIELD_MAP


def test_every_emitted_column_is_documented():
    undocumented = [c for c in COLUMNS if FIELD_MAP.get(c) is None]
    assert undocumented == [], f"undocumented output columns: {undocumented}"


def test_productive_is_derived_prediction_is_extension():
    assert FIELD_MAP["productive"]["source"] == DERIVED
    assert FIELD_MAP["productive_prediction"]["source"] == EXTENSION


def test_identities_are_derived():
    for g in ("v", "d", "j"):
        assert FIELD_MAP[f"{g}_identity"]["source"] == DERIVED
