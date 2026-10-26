"""Tests for calibration_group module (TDD)."""
import pytest

from alignair.aa.product.calibration_group import (
    SUPPLIED_WARNING,
    VALID_GROUPS,
    resolve_group,
)

pytestmark = pytest.mark.component


class TestResolveGroupConstants:
    """Test module constants."""

    def test_valid_groups_tuple(self):
        """VALID_GROUPS should be a tuple of valid group names."""
        assert VALID_GROUPS == ("full", "trunc", "short")

    def test_supplied_warning_text(self):
        """SUPPLIED_WARNING should contain guidance text."""
        assert "Group supplied by caller" in SUPPLIED_WARNING
        assert "calibration-group definition" in SUPPLIED_WARNING
        assert "exchangeable" in SUPPLIED_WARNING


class TestResolveGroupNone:
    """Test resolve_group with None coverage_group (pooled fallback)."""

    def test_none_returns_pooled_fallback(self):
        """coverage_group=None should return pooled fallback."""
        group, source, warning = resolve_group(
            coverage_group=None, sequence_id="seq1", index=0
        )
        assert group == "pooled"
        assert source == "pooled_fallback"
        assert warning is None

    def test_none_with_different_sequence_ids(self):
        """pooled fallback should be the same regardless of sequence_id."""
        for seq_id in ["seqA", "seqB", "arbitrary"]:
            group, source, warning = resolve_group(
                coverage_group=None, sequence_id=seq_id, index=0
            )
            assert (group, source, warning) == ("pooled", "pooled_fallback", None)

    def test_none_with_different_indices(self):
        """pooled fallback should be the same regardless of index."""
        for idx in [0, 1, 5, 100]:
            group, source, warning = resolve_group(
                coverage_group=None, sequence_id="seq1", index=idx
            )
            assert (group, source, warning) == ("pooled", "pooled_fallback", None)


class TestResolveGroupStrValid:
    """Test resolve_group with valid group strings."""

    @pytest.mark.parametrize("group_str", ["full", "trunc", "short"])
    def test_valid_group_string(self, group_str):
        """Valid group strings should return the group with supplied source and warning."""
        group, source, warning = resolve_group(
            coverage_group=group_str, sequence_id="seq1", index=0
        )
        assert group == group_str
        assert source == "supplied"
        assert warning == SUPPLIED_WARNING

    @pytest.mark.parametrize("group_str", ["full", "trunc", "short"])
    def test_valid_group_string_independent_of_sequence_id(self, group_str):
        """Valid group strings should work with any sequence_id."""
        for seq_id in ["seqA", "seqB", "arbitrary"]:
            group, source, warning = resolve_group(
                coverage_group=group_str, sequence_id=seq_id, index=0
            )
            assert group == group_str
            assert source == "supplied"
            assert warning == SUPPLIED_WARNING

    @pytest.mark.parametrize("group_str", ["full", "trunc", "short"])
    def test_valid_group_string_independent_of_index(self, group_str):
        """Valid group strings should work with any index."""
        for idx in [0, 1, 5, 100]:
            group, source, warning = resolve_group(
                coverage_group=group_str, sequence_id="seq1", index=idx
            )
            assert group == group_str
            assert source == "supplied"
            assert warning == SUPPLIED_WARNING


class TestResolveGroupStrInvalid:
    """Test resolve_group with invalid group strings."""

    @pytest.mark.parametrize(
        "invalid_group",
        ["bogus", "FULL", "Full", "truncated", "pool", "unknown", ""],
    )
    def test_invalid_group_string_raises_valueerror(self, invalid_group):
        """Invalid group strings should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_group(
                coverage_group=invalid_group, sequence_id="seq1", index=0
            )
        assert "coverage_group must be one of" in str(exc_info.value)
        assert "VALID_GROUPS" in str(exc_info.value) or "('full', 'trunc', 'short')" in str(exc_info.value)
        assert invalid_group in str(exc_info.value)


class TestResolveGroupDict:
    """Test resolve_group with dict (per-sequence mapping)."""

    def test_dict_lookup_present_sequence(self):
        """Dict lookup should resolve the group for a present sequence."""
        coverage_dict = {"seqA": "trunc", "seqB": "full"}

        group, source, warning = resolve_group(
            coverage_group=coverage_dict, sequence_id="seqA", index=0
        )
        assert group == "trunc"
        assert source == "supplied"
        assert warning == SUPPLIED_WARNING

    def test_dict_lookup_present_sequence_different_groups(self):
        """Dict lookup should resolve different groups for different sequences."""
        coverage_dict = {"seqA": "trunc", "seqB": "full", "seqC": "short"}

        for seq_id, expected_group in [
            ("seqA", "trunc"),
            ("seqB", "full"),
            ("seqC", "short"),
        ]:
            group, source, warning = resolve_group(
                coverage_group=coverage_dict, sequence_id=seq_id, index=0
            )
            assert group == expected_group
            assert source == "supplied"
            assert warning == SUPPLIED_WARNING

    def test_dict_lookup_absent_sequence_falls_back_to_pooled(self):
        """Dict lookup should fall back to pooled for absent sequence."""
        coverage_dict = {"seqA": "trunc"}

        group, source, warning = resolve_group(
            coverage_group=coverage_dict, sequence_id="seqB", index=0
        )
        assert group == "pooled"
        assert source == "pooled_fallback"
        assert warning is None

    def test_dict_lookup_mixed_presence(self):
        """Dict lookup should handle a mix of present and absent sequences."""
        coverage_dict = {"seqA": "trunc", "seqB": "full"}

        # Present sequences
        for seq_id, expected_group in [("seqA", "trunc"), ("seqB", "full")]:
            group, source, warning = resolve_group(
                coverage_group=coverage_dict, sequence_id=seq_id, index=0
            )
            assert group == expected_group
            assert source == "supplied"

        # Absent sequences
        for seq_id in ["seqC", "seqD", "arbitrary"]:
            group, source, warning = resolve_group(
                coverage_group=coverage_dict, sequence_id=seq_id, index=0
            )
            assert group == "pooled"
            assert source == "pooled_fallback"

    def test_dict_value_none_falls_back_to_pooled(self):
        """Dict value of None should fall back to pooled."""
        coverage_dict = {"seqA": None}

        group, source, warning = resolve_group(
            coverage_group=coverage_dict, sequence_id="seqA", index=0
        )
        assert group == "pooled"
        assert source == "pooled_fallback"
        assert warning is None

    def test_dict_invalid_value_raises_valueerror(self):
        """Dict containing invalid group values should raise ValueError."""
        coverage_dict = {"seqA": "bogus"}

        with pytest.raises(ValueError) as exc_info:
            resolve_group(
                coverage_group=coverage_dict, sequence_id="seqA", index=0
            )
        assert "coverage_group must be one of" in str(exc_info.value)


class TestResolveGroupList:
    """Test resolve_group with list/tuple (per-sequence list aligned by index)."""

    def test_list_lookup_valid_index_with_group_str(self):
        """List lookup should resolve a valid group string at a valid index."""
        coverage_list = [None, "short", "full"]

        group, source, warning = resolve_group(
            coverage_group=coverage_list, sequence_id="seq1", index=1
        )
        assert group == "short"
        assert source == "supplied"
        assert warning == SUPPLIED_WARNING

    def test_list_lookup_valid_index_with_none(self):
        """List lookup at index with None should fall back to pooled."""
        coverage_list = [None, "short", "full"]

        group, source, warning = resolve_group(
            coverage_group=coverage_list, sequence_id="seq1", index=0
        )
        assert group == "pooled"
        assert source == "pooled_fallback"
        assert warning is None

    def test_list_lookup_all_indices(self):
        """List lookup should handle all valid indices."""
        coverage_list = [None, "short", "full", "trunc"]

        expected_results = [
            ("pooled", "pooled_fallback", None),
            ("short", "supplied", SUPPLIED_WARNING),
            ("full", "supplied", SUPPLIED_WARNING),
            ("trunc", "supplied", SUPPLIED_WARNING),
        ]

        for idx, (exp_group, exp_source, exp_warning) in enumerate(expected_results):
            group, source, warning = resolve_group(
                coverage_group=coverage_list, sequence_id="seq1", index=idx
            )
            assert group == exp_group
            assert source == exp_source
            assert warning == exp_warning

    def test_list_lookup_out_of_range_raises_valueerror(self):
        """List lookup at out-of-range index should raise ValueError."""
        coverage_list = [None, "short"]

        with pytest.raises(ValueError) as exc_info:
            resolve_group(
                coverage_group=coverage_list, sequence_id="seq1", index=2
            )
        assert "index" in str(exc_info.value).lower() or "out of range" in str(exc_info.value).lower()

    def test_list_lookup_negative_index_out_of_range(self):
        """Negative indices should be out of range (no Python-style negative indexing)."""
        coverage_list = [None, "short"]

        with pytest.raises(ValueError):
            resolve_group(
                coverage_group=coverage_list, sequence_id="seq1", index=-1
            )

    def test_tuple_lookup_works_same_as_list(self):
        """Tuple should work the same as list."""
        coverage_tuple = (None, "short", "full")

        group, source, warning = resolve_group(
            coverage_group=coverage_tuple, sequence_id="seq1", index=1
        )
        assert group == "short"
        assert source == "supplied"
        assert warning == SUPPLIED_WARNING

    def test_list_invalid_value_raises_valueerror(self):
        """List containing invalid group values should raise ValueError."""
        coverage_list = ["bogus"]

        with pytest.raises(ValueError) as exc_info:
            resolve_group(
                coverage_group=coverage_list, sequence_id="seq1", index=0
            )
        assert "coverage_group must be one of" in str(exc_info.value)
